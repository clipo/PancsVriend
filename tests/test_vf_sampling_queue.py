"""sample_into_counts issues one request queue per role (2026-09-05) and must
produce exactly what the per-cell version did: the same prompts with the same
seeds, the same raw records in the same order, the same counts.

The per-cell version is checked out from git (commit 2cdb2c8) at test time as the oracle;
requests are faked, so no server is involved.
"""

import importlib.util
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PR = REPO / "prompt_refinement"
for path in (str(REPO), str(PR), str(REPO / "value_functions" / "sampling")):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_value_function as bvf  # noqa: E402
from ratio_prompt_templates import ALL_COMPOSITIONS, RATIO_CANDIDATES  # noqa: E402


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The pre-queue build_value_function module, pinned to the commit before the change."""
    src = subprocess.run(["git", "show", "2cdb2c8:prompt_refinement/build_value_function.py"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert "for cell in ALL_COMPOSITIONS:\n            n_sim, n_occ = cell\n            k = alloc_fn" in src, \
        "2cdb2c8 must hold the per-cell version (the oracle)"
    path = tmp_path_factory.mktemp("ref") / "bvf_reference.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("bvf_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Raw:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(dict(record))


def _fake_sample_batch(calls):
    """Reply is a pure function of (prompt, seed): the property the queue relies on."""
    def sample_batch(url, model, prompts, temperature, grammar, concurrency,
                     cache_prompt=None, seed_fn=None):
        calls.append(len(prompts))
        out = []
        for i, prompt in enumerate(prompts):
            seed = None if seed_fn is None else seed_fn(i)
            h = zlib.crc32(f"{prompt}|{seed}".encode())
            text = ["MOVE", "STAY", "move it", "??", "MOVE STAY"][h % 5]
            out.append({"text": text, "finish_reason": "stop", "completion_tokens": 1})
        return out
    return sample_batch


def _run(module, style, seed_ctx, monkeypatch):
    calls = []
    monkeypatch.setattr(module, "sample_batch", _fake_sample_batch(calls))
    monkeypatch.setattr(module, "slice_ping", lambda *a, **k: None)
    roles = ["red", "blue"]
    counts = module._blank_counts(roles)
    tpl, fn = RATIO_CANDIDATES[style]
    keywords = bvf.role_keywords("baseline")
    kw_by_role = {role: keywords[role] for role in roles}
    raw = _Raw()
    n_new = module.sample_into_counts(
        counts, lambda role, cell: (cell[0] + cell[1] + (role == "blue")) % 4,
        roles, kw_by_role, style, tpl, fn, "http://x/v1/completions", "m",
        0.3, True, 8, 12345, raw, ping_label="t", ping_ctx="t", seed_ctx=seed_ctx)
    return n_new, counts, raw.records, calls


@pytest.mark.parametrize("style", sorted(RATIO_CANDIDATES))
@pytest.mark.parametrize("seed_ctx", [None, ("topup1", "baseline")], ids=["legacy", "clean"])
def test_queued_sampling_matches_the_per_cell_version(reference, style, seed_ctx, monkeypatch):
    n_ref, counts_ref, records_ref, calls_ref = _run(reference, style, seed_ctx, monkeypatch)
    n_new, counts_new, records_new, calls_new = _run(bvf, style, seed_ctx, monkeypatch)
    assert n_new == n_ref
    assert counts_new == counts_ref
    assert records_new == records_ref                 # same records, same order, same seeds
    assert sum(calls_new) == sum(calls_ref)
    assert len(calls_new) == 2                        # one queue per role ...
    assert len(calls_ref) == sum(1 for role in (0, 1) for cell in ALL_COMPOSITIONS
                                 if (cell[0] + cell[1] + role) % 4)  # ... vs one per cell
