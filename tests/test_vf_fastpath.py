"""The array fast path (vf_fastpath) reproduces the object-grid path bit for bit.

Speed review item 4. The stored-run regeneration is verify_vf_fastpath.py;
these tests pin the pieces and a set of edge cases on small grids.
"""

import random

import numpy as np
import pytest

import config as cfg
import llm_runner
import vf_fastpath


@pytest.fixture
def grid_config():
    saved = (cfg.GRID_SIZE, cfg.NUM_TYPE_A, cfg.NUM_TYPE_B)

    def apply(size, n_a, n_b):
        llm_runner.apply_grid_config({'GRID_SIZE': size, 'NUM_TYPE_A': n_a, 'NUM_TYPE_B': n_b})
    yield apply
    cfg.GRID_SIZE, cfg.NUM_TYPE_A, cfg.NUM_TYPE_B = saved


def _policy(seed, lo=0.0, hi=1.0):
    rng = random.Random(seed)
    return {(key, f"comp:{s},{o}"): rng.uniform(lo, hi)
            for key in ('type_a', 'type_b') for o in range(9) for s in range(o + 1)}


def _run(cls, run_id, policy, max_steps, **kw):
    sim = cls(run_id, 'baseline', 'm', None, None, random_seed=run_id, temperature=0.3,
              value_function_policy=policy, **kw)
    res = sim.run_single_simulation(output_dir=None, max_steps=max_steps)
    sim._collect_llm_counters()
    return sim, res


def _assert_same(a, b):
    sa, sb = np.array(a[0].states), np.array(b[0].states)
    assert sa.dtype == sb.dtype == np.int8
    assert np.array_equal(sa, sb)
    assert a[0].step_log == b[0].step_log
    assert a[0].metrics_history == b[0].metrics_history
    assert a[0].total_llm_calls == b[0].total_llm_calls
    for key in ('converged', 'convergence_step', 'final_step', 'n_states', 'total_agent_moves'):
        assert a[1][key] == b[1][key], key


def test_keyed_uniforms_match_scalar_bit_for_bit():
    rng = random.Random(0)
    keys = [(0, 0), (1, 0), (2**63 - 1, 10**9), (2**64 - 1, 2**64 - 1), (12345, 9999)]
    keys += [(rng.getrandbits(64), rng.getrandbits(40)) for _ in range(50)]
    ids = np.arange(400, dtype=np.uint64)
    for run_id, step in keys:
        u0, u1 = vf_fastpath.keyed_uniforms_np(run_id, step, ids)
        for a in range(400):
            assert u0[a] == llm_runner.keyed_uniform(run_id, step, a, 0)
            assert u1[a] == llm_runner.keyed_uniform(run_id, step, a, 1)
            assert float(u0[a]).hex() == llm_runner.keyed_uniform(run_id, step, a, 0).hex()


def test_policy_table_holds_the_same_floats():
    pol = _policy(1)
    table = vf_fastpath.policy_table(pol)
    for (key, comp), p in pol.items():
        s, o = map(int, comp.split(':')[1].split(','))
        t = 0 if key == 'type_a' else 1
        assert table[t * 81 + s * 9 + o] is p


@pytest.mark.parametrize("size,n_a,n_b,lo,hi,steps", [
    (20, 160, 160, 0.0, 1.0, 60),      # production geometry, heavy movement
    (20, 160, 160, 0.0, 0.05, 200),    # production geometry, converging
    (10, 40, 40, 0.0, 1.0, 80),        # config.py default geometry
    (10, 50, 50, 0.0, 1.0, 10),        # full grid: no empty cell, MOVE becomes a stay
    (10, 50, 49, 0.5, 1.0, 40),        # one empty cell
    (5, 3, 0, 0.0, 1.0, 40),           # one type only
    (3, 2, 2, 0.0, 1.0, 40),           # every cell is an edge cell
])
def test_same_seed_old_vs_new(grid_config, size, n_a, n_b, lo, hi, steps):
    grid_config(size, n_a, n_b)
    for run_id in (0, 1, 7, 2**32 - 1):
        pol = _policy(run_id, lo, hi)
        _assert_same(_run(llm_runner.LLMSimulation, run_id, pol, steps),
                     _run(vf_fastpath.FastVFSimulation, run_id, pol, steps))


def test_resumed_run_matches(grid_config):
    """A resume (initial_int_grid + initial_step) gives the same continuation."""
    grid_config(20, 160, 160)
    pol = _policy(3)
    base, _ = _run(llm_runner.LLMSimulation, 11, pol, 30)
    seed_grid, step = base.states[20], 20
    kw = dict(initial_int_grid=seed_grid, initial_step=step, initial_no_move_steps=0)
    _assert_same(_run(llm_runner.LLMSimulation, 11, pol, 60, **kw),
                 _run(vf_fastpath.FastVFSimulation, 11, pol, 60, **kw))


def test_refuses_what_it_does_not_implement(grid_config, monkeypatch):
    grid_config(10, 40, 40)
    with pytest.raises(ValueError):
        vf_fastpath.FastVFSimulation(0, 'baseline', 'm', None, None, random_seed=0,
                                     temperature=0.3, value_function_policy={})
    monkeypatch.setattr(llm_runner, 'VF_RNG_SCHEME', 'shared')
    with pytest.raises(ValueError):
        vf_fastpath.FastVFSimulation(0, 'baseline', 'm', None, None, random_seed=0,
                                     temperature=0.3, value_function_policy=_policy(0))
