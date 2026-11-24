"""Run the multi-option experiments against a llama-server (design v4).

Three steps, each started by hand and only after the go-ahead:

  probe   checks that must pass before anything else (a few minutes):
          - the token backend's first-token table equals the chat endpoint's
            (decides whether the begin-of-text token is added);
          - which token ids end a reply;
          - how each label is split into tokens;
          - the time one request takes;
          - a 5-state pilot per arm, with the saturation gate.
  stage1  exact reads for every state: path sum, FT-mask (T = 0.3 and T = 1),
          FT-split, depth 1 to 3, with and without grammar, both option orders.
  plan    from Stage 1 values only, chooses census states and draws for each
          primary comparison (no draws are made).
  census  draws the planned replies, one at a time, and runs the tests.

  probe+stage1  the probe, then Stage 1 and the plan, only if the gate passes
          and the probe's time estimate is under --max-hours.

With --serve olmo2 (or qwen, ...) the script starts the model itself with
value_functions.serving.PinnedServer, exactly as the extraction runs did, and
stops it at the end. Requests go one at a time.

Example (from the repo root):
  python value_functions/multi_option/run_multi_option.py probe+stage1 --serve olmo2
  python value_functions/multi_option/run_multi_option.py probe+stage1 --serve qwen --tasks P1
  python value_functions/multi_option/run_multi_option.py probe  --url http://localhost:8085 --tasks C2 T3 P4 O5 A11 B2
  python value_functions/multi_option/run_multi_option.py stage1 --url http://localhost:8085 --tasks C2 T3 P4 O5 A11 B2
  python value_functions/multi_option/run_multi_option.py plan   --stage1 <stage1 file>
  python value_functions/multi_option/run_multi_option.py census --url http://localhost:8085 --plan <plan file>

Every output file starts with a header line that records the settings, so
results can be checked like for like with the simulation.
"""

import argparse  # command-line options
import gzip  # compressed output
import hashlib  # checksums of grammars and prompts
import json  # records
import math  # isnan
import subprocess  # the git commit
import sys  # module search path
import time  # timing
from datetime import datetime, timezone  # time stamps
from pathlib import Path  # file paths

HERE = Path(__file__).resolve().parent  # value_functions/multi_option
REPO = HERE.parents[1]  # the repo root
for sub in ("", "value_functions", "value_functions/logprob", "prompt_refinement"):  # the repo's module folders
    if str(REPO / sub) not in sys.path:
        sys.path.append(str(REPO / sub))
sys.path.insert(0, str(HERE))  # this folder first

import numpy as np  # noqa: E402  averages for the time estimate
import requests  # noqa: E402  talks to the server

import tasks as T  # noqa: E402  the experiments
from multi_option_path_sum import (ChatTextBackend, TokenBackend, discover_end_ids, path_sum,  # noqa: E402
                                   ft_mask, ft_split, depth_read, free_conditional, tv)
import multi_option_census as MC  # noqa: E402  census and tests
from llm_runner import SAMPLER_PARAMS, MOVE_STAY_GRAMMAR  # noqa: E402  the simulation's settings

TEMPERATURE = 0.3  # every campaign
N_PROBS = 64  # top-n table size, as in the extractor
MASS_FLOOR = 1e-7  # as in the extractor
OUT_DIR = REPO / "value_functions" / "multi_option" / "results"  # where results go


# ----------------------------------------------------------------------
# Small helpers.
# ----------------------------------------------------------------------

def sha(text):
    """sha256 of a text, for the record."""
    return hashlib.sha256(text.encode()).hexdigest()


def git_commit():
    """The repo's commit and whether files are changed, for the record."""
    try:
        c = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout
        return {"commit": c, "dirty": bool(d.strip())}
    except OSError:
        return None


def header(args, extra):
    """The settings every output file starts with."""
    url = args.url.rstrip("/")
    props = requests.get(f"{url}/props", timeout=60).json()  # model and build as the server reports them
    rec = {"kind": "header", "time_utc": datetime.now(timezone.utc).isoformat(), "url": url,
           "model": args.model, "temperature": TEMPERATURE, "sampler_params": SAMPLER_PARAMS,
           "n_probs": N_PROBS, "mass_floor": MASS_FLOOR, "max_depth": args.max_depth,
           "build_info": props.get("build_info"), "model_path": props.get("model_path"),
           "chat_template_sha256": sha(props.get("chat_template") or ""),
           "move_stay_grammar_sha256": sha(MOVE_STAY_GRAMMAR), "git": git_commit(), **extra}
    try:  # the extractor's records of how the server was started (best effort)
        import logprob_value_function as lvf
        rec["server_cmdline"] = lvf.server_cmdline(url)  # the launch line
        rec["server_env"] = lvf.server_env(url)  # the pinned clock
        if args.weights and props.get("model_path"):  # the weights' checksum (takes minutes)
            rec["weights"] = lvf.weights_digest(props["model_path"])
    except Exception as e:  # never stop a run for the record
        rec["server_record_error"] = str(e)
    return rec


def open_out(name):
    """A new gzip JSONL file under results/, named with the time."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # make the folder if needed
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # time stamp
    path = OUT_DIR / f"{name}_{stamp}.jsonl.gz"
    return path, gzip.open(path, "wt")


def make_task(name):
    """An experiment by name."""
    return T.ALL_TASKS[name]()


def orders_for(task):
    """Option orders to read: listed and reversed (A11's order is fixed)."""
    return [False] if task.name == "A11" else [False, True]


def backend(args, kind, end_ids, add_special=True):
    """A backend with the run's settings."""
    if kind == "token":
        return TokenBackend(args.url, args.model, TEMPERATURE, N_PROBS, SAMPLER_PARAMS,
                            end_ids=end_ids, add_special=add_special)
    return ChatTextBackend(args.url, args.model, TEMPERATURE, N_PROBS, SAMPLER_PARAMS, end_ids=end_ids)


def root_grammar_for(task, arm, rev, lang):
    """The first request's grammar: the simulation's own text for MOVE/STAY, else generated."""
    if task.name == "B2" and not rev:  # the same request, byte for byte, as the simulation and the extractor
        return MOVE_STAY_GRAMMAR
    return lang.gbnf()


def reads(res, mode):
    """Every read compared with the path sum, from one stored path sum."""
    out = {}
    out["ft_mask"], out["m_off"] = ft_mask(res["root"], res["labels"], mode)  # at the run temperature
    out["ft_mask_t1"], _ = ft_mask(res["root"], res["labels"], mode, power=TEMPERATURE)  # read at T = 1
    out["ft_split"] = ft_split(res["root"], res["labels"], mode)
    out["ft_split_t1"] = ft_split(res["root"], res["labels"], mode, power=TEMPERATURE)
    for d in (1, 2, 3):  # depth-d reads from the stored tree
        out[f"depth{d}"], out[f"m_off{d}"] = depth_read(res, d)
    if mode == "free":  # the on-format share of each option
        out["free_conditional"] = free_conditional(res)
    return out


# ----------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------

def cmd_probe(args):
    url = args.url.rstrip("/")
    end_ids = set(args.end_ids) if args.end_ids else discover_end_ids(url)  # end-of-reply token ids
    print(f"end-of-reply ids: {sorted(end_ids)}")
    path, out = open_out("probe")
    out.write(json.dumps(header(args, {"step": "probe", "end_ids": sorted(end_ids)})) + "\n")
    chat = backend(args, "chat", end_ids)  # the extractor's way
    tok = {s: backend(args, "token", end_ids, add_special=s) for s in (True, False)}  # both begin-of-text choices
    gate = {True: 0.0, False: 0.0}  # largest table difference per choice
    est = []  # (task, arm, grammar requests, free requests, seconds per request) per pilot state
    for name in args.tasks:
        task = make_task(name)
        for arm in task.arms:
            # 1. label tokenisation, with and without a leading space
            for lab in task.labels(arm):
                ids = {v: tok[False].tokenize(v) for v in (lab, " " + lab, lab.lower())}
                out.write(json.dumps({"kind": "label_tokens", "task": name, "arm": arm, "label": lab,
                                      "ids": ids}) + "\n")
                print(f"{name}/{arm:8s} {lab!r:24s} -> {[len(v) for v in ids.values()]} tokens (as is, with space, lower)")
            # 2. root-table equality and the pilot (5 evenly spaced states)
            idx = sorted({round(i * (len(task.states) - 1) / 4) for i in range(5)})
            for i in idx:
                st = task.states[i]
                prompt = task.render(st, arm)
                lang = task.language(arm, "grammar")
                g = root_grammar_for(task, arm, False, lang)
                t0 = time.time()
                a = chat.next_token(prompt, (), "", g)  # chat endpoint
                t_req = time.time() - t0
                pa = {e.id: e.prob for e in a}
                for s in (True, False):
                    b = tok[s].next_token(prompt, (), "", g)  # token endpoint
                    pb = {e.id: e.prob for e in b}
                    diff = max(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb))
                    gate[s] = max(gate[s], diff)
                res = path_sum(tok[True if gate[True] <= gate[False] else False], prompt, lang, "grammar",
                               args.max_depth, MASS_FLOOR, root_grammar=g)
                r = reads(res, "grammar")
                top = max(res["p"])
                out.write(json.dumps({"kind": "pilot", "task": name, "arm": arm, "state": st,
                                      "p": res["p"], "bound": res["bound"], "n_requests": res["n_requests"],
                                      "seconds_one_request": t_req, **r,
                                      "first_tokens": [e._asdict() for e in res["root"][:12]]}) + "\n")
                free_lang = task.language(arm, "free")  # the same state without grammar
                fr = path_sum(tok[True if gate[True] <= gate[False] else False], prompt, free_lang, "free",
                              args.max_depth, MASS_FLOOR)
                out.write(json.dumps({"kind": "pilot_free", "task": name, "arm": arm, "state": st,
                                      "p": fr["p"], "off_format": fr["off_format"], "bound": fr["bound"],
                                      "n_requests": fr["n_requests"], **reads(fr, "free")}) + "\n")
                est.append((name, arm, res["n_requests"], fr["n_requests"], t_req))
                print(f"  pilot {name}/{arm} state {i}: max p {top:.4f}, m_off {r['m_off']:.3g}, "
                      f"TV(path, FT-mask) {tv(res['p'], r['ft_mask']) if not any(map(math.isnan, r['ft_mask'])) else float('nan'):.3g}, "
                      f"{res['n_requests']} req; free: off-format {fr['off_format']:.3g}, B {fr['bound']:.2g}, "
                      f"{fr['n_requests']} req; {t_req:.2f} s/req")
            # 3. what follows a complete label without grammar (shows how the reply ends)
            lab = task.labels(arm)[0]
            b = tok[True].next_token(task.render(task.states[0], arm), tuple(tok[True].tokenize(lab)), lab, None)
            out.write(json.dumps({"kind": "after_label_free", "task": name, "arm": arm, "label": lab,
                                  "top": [e._asdict() for e in b[:12]]}) + "\n")
    choice = True if gate[True] <= gate[False] else False
    ok = gate[choice] <= 1e-6
    # Time estimate for Stage 1: mean requests per state (each task and arm) x states x 2 orders x seconds per request
    sec = float(np.median([e[4] for e in est])) if est else 0.0  # typical seconds for one request
    hours = 0.0
    for name in args.tasks:
        task = make_task(name)
        for arm in task.arms:
            mine = [e for e in est if e[0] == name and e[1] == arm]
            per_state = (np.mean([e[2] for e in mine]) if "grammar" in args.modes else 0) + \
                        (np.mean([e[3] for e in mine]) if "free" in args.modes else 0)
            hours += 2 * len(task.states) * per_state * sec / 3600  # 2 option orders
    print(f"\nStage 1 estimate: about {hours:.1f} hours ({sec:.2f} s per request).")
    out.write(json.dumps({"kind": "gate", "max_diff_add_special_true": gate[True],
                          "max_diff_add_special_false": gate[False], "add_special": choice, "pass": ok,
                          "stage1_hours_estimate": hours, "seconds_per_request": sec}) + "\n")
    out.close()
    print(f"\nGATE: token backend {'PASSES' if ok else 'FAILS'} (largest root-table difference {gate[choice]:.2e}, "
          f"add_special={choice}). Results: {path}")
    if not ok:
        print("Do not run stage1 with --backend token; use --backend chat and report the merged-path mass.")
    return {"pass": ok, "add_special": choice, "file": str(path), "hours": hours}


# ----------------------------------------------------------------------
# stage1
# ----------------------------------------------------------------------

def cmd_stage1(args):
    end_ids = set(args.end_ids) if args.end_ids else discover_end_ids(args.url)
    be = backend(args, args.backend, end_ids, add_special=not args.no_add_special)
    path, out = open_out(f"stage1_{'-'.join(args.tasks)}")
    out.write(json.dumps(header(args, {"step": "stage1", "backend": be.kind, "end_ids": sorted(end_ids),
                                       "add_special": not args.no_add_special, "tasks": args.tasks})) + "\n")
    t0, n = time.time(), 0
    for name in args.tasks:
        task = make_task(name)
        for arm in task.arms:
            for mode in args.modes:
                for rev in orders_for(task):
                    for i, st in enumerate(task.states):
                        prompt = task.render(st, arm, rev)
                        lang = task.language(arm, mode, rev)
                        g = root_grammar_for(task, arm, rev, lang) if mode == "grammar" else None
                        res = path_sum(be, prompt, lang, mode, args.max_depth, MASS_FLOOR, root_grammar=g)
                        rec = {"kind": "state", "task": name, "arm": arm, "mode": mode, "reversed": rev,
                               "i": i, "state": st, "labels": res["labels"], "prompt_sha256": sha(prompt),
                               "grammar_sha256": sha(g) if g else None, "p": res["p"],
                               "off_format": res["off_format"], "bound": res["bound"],
                               "n_requests": res["n_requests"], **reads(res, mode),
                               "root": [e._asdict() for e in res["root"]],
                               "paths": res["paths"][:50], "n_paths": len(res["paths"]),
                               "states": res["states"]}
                        out.write(json.dumps(rec) + "\n")
                        n += 1
                    print(f"{name}/{arm}/{mode}/{'rev' if rev else 'listed'}: {len(task.states)} states, "
                          f"{be.n_requests} requests so far, {time.time() - t0:.0f} s")
    out.close()
    print(f"{n} path sums, {be.n_requests} requests, {time.time() - t0:.0f} s. Results: {path}")
    return str(path)


# ----------------------------------------------------------------------
# plan: census states and draws from Stage 1 values only
# ----------------------------------------------------------------------

def load(path):
    """Header and records of a gzip JSONL file."""
    with gzip.open(path, "rt") as fh:
        rows = [json.loads(l) for l in fh]
    return rows[0], rows[1:]


def _key(r):
    return (r["task"], r["arm"], r["mode"], r["reversed"], r["i"])


def cmd_plan(args):
    hdr, rows = load(args.stage1)
    rows = [r for r in rows if r["kind"] == "state" and not r["reversed"]]  # the census uses the listed order
    by = {_key(r): r for r in rows}
    plans = []
    # Primary 1 (Q1, with grammar): path sum against FT-split, pooled over the test arms.
    pool = {k: r for k, r in by.items() if r["mode"] == "grammar" and r["arm"] == "test"}
    plans.append(_plan("Q1_path_vs_ft_split", pool, lambda r: r["p"], lambda r: r["ft_split"], "grammar", args))
    # Primary 2 (Q2, grammar effect): with-grammar path sum against the free census, B2 (on-format part).
    pool = {k: r for k, r in by.items() if r["task"] == "B2" and r["mode"] == "free"}
    gram = {(r["task"], r["arm"], r["i"]): r for r in rows if r["mode"] == "grammar"}
    plans.append(_plan("Q2_grammar_vs_free_conditional", pool,
                       lambda r: r["free_conditional"], lambda r: gram[(r["task"], r["arm"], r["i"])]["p"], "free", args))
    # Primary 3 (Q3, without grammar): free path sum against the free first-token read, on the
    # test arm whose states carry the most information per draw.
    best, best_info = None, -1.0
    for name in sorted({r["task"] for r in rows}):
        pool = {k: r for k, r in by.items() if r["task"] == name and r["mode"] == "free"
                and r["arm"] in ("test", "single") and not any(map(_nan, r["ft_mask"]))}
        info = sum(MC.chi2_state(r["free_conditional"], r["ft_mask"]) for r in pool.values())
        if pool and info > best_info:
            best, best_info = pool, info
    if best:
        plans.append(_plan("Q3_free_path_vs_free_first_token", best,
                           lambda r: r["free_conditional"], lambda r: r["ft_mask"], "free", args))
    # Validation census of the path sum itself where paths have several tokens: each test arm
    # with grammar (and A11), 10 evenly spaced unsaturated states x 100 draws (a gross-error check).
    for name in sorted({r["task"] for r in rows}):
        arm_rows = sorted([r for r in rows if r["task"] == name and r["arm"] in ("test", "single")
                           and r["mode"] == "grammar" and max(r["p"]) < 0.995],
                          key=lambda r: max(r["p"]))  # ordered by how decided the state is
        if arm_rows and name != "B2":  # MOVE/STAY with grammar is already validated in the paper
            step = max(1, len(arm_rows) // args.validation_states)
            pick = arm_rows[::step][:args.validation_states]
            plans.append({"name": f"validation_{name}", "mode": "grammar",
                          "draws": {json.dumps(_key(r)): 100 for r in pick}, "enough": None})
    out = Path(args.stage1).with_name(Path(args.stage1).name.replace("stage1", "plan").replace(".jsonl.gz", ".json"))
    json.dump({"stage1": str(args.stage1), "plans": plans, "max_draws": args.max_draws}, open(out, "w"), indent=1)
    for p in plans:
        print(f"{p['name']}: {sum(p['draws'].values())} draws on {len(p['draws'])} states, enough={p['enough']}")
    print(f"Plan written to {out}. No draws were made.")


def _nan(x):
    return isinstance(x, float) and math.isnan(x)


def _plan(name, pool, get_p, get_r, mode, args):
    """Draws for one comparison so that I >= 16, from exact values only."""
    unsat = {k: r for k, r in pool.items() if max(get_p(r)) < 0.995 and not any(map(_nan, get_r(r)))}
    gaps = [tv(get_p(r), get_r(r)) for r in unsat.values()]
    plan, info, ok = MC.plan_draws({k: get_p(r) for k, r in unsat.items()},
                                   {k: get_r(r) for k, r in unsat.items()}, cap=args.max_draws)
    return {"name": name, "mode": mode, "draws": {json.dumps(k): n for k, n in plan.items()},
            "information": info, "enough": ok, "n_unsaturated": len(unsat),
            "median_gap_all_unsaturated": float(sorted(gaps)[len(gaps) // 2]) if gaps else None}


# ----------------------------------------------------------------------
# census
# ----------------------------------------------------------------------

def cmd_census(args):
    plan = json.load(open(args.plan))
    _, rows = load(plan["stage1"])
    by = {_key(r): r for r in rows if r["kind"] == "state"}
    smp = MC.Sampler(args.url, args.model, TEMPERATURE, SAMPLER_PARAMS)
    path, out = open_out("census")
    out.write(json.dumps(header(args, {"step": "census", "plan": str(args.plan), "max_tokens": args.max_tokens})) + "\n")
    seed = args.seed0  # every draw gets its own seed
    for p in plan["plans"]:  # each comparison gets its own draws, so the tests are independent
        for kj, n in p["draws"].items():
            k = tuple(json.loads(kj))  # which state
            r = by[k]
            task = make_task(r["task"])
            prompt = task.render(r["state"], r["arm"], r["reversed"])
            lang = task.language(r["arm"], r["mode"], r["reversed"])
            g = root_grammar_for(task, r["arm"], r["reversed"], lang) if r["mode"] == "grammar" else None
            c = MC.census(smp, prompt, lang, r["mode"], n, seed, args.max_tokens, grammar=g)
            seed += n
            out.write(json.dumps({"kind": "census", "plan": p["name"], "key": list(k), "n": n, "seed0": seed - n,
                                  "strict": c["strict"], "loose": c["loose"], "replies": c["replies"]}) + "\n")
            out.flush()
        print(f"{p['name']}: drawn")
    out.close()
    print(f"Census written to {path}. Run the analysis on it; no test is run here.")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["probe", "stage1", "probe+stage1", "plan", "census"],
                    help="probe+stage1 runs Stage 1 and then the plan only if the probe's gate passes")
    ap.add_argument("--serve", default=None,
                    help="start the model with value_functions.serving.PinnedServer (e.g. olmo2, qwen), "
                         "exactly as the extraction runs did, and stop it afterwards")
    ap.add_argument("--port", type=int, default=8085, help="port for --serve")
    ap.add_argument("--max-hours", type=float, default=12.0,
                    help="probe+stage1 stops after the probe if Stage 1 is estimated to take longer than this")
    ap.add_argument("--url", default="http://localhost:8085")
    ap.add_argument("--model", default=None, help="payload model name (default: the served gguf's stem, as the extractor does)")
    ap.add_argument("--tasks", nargs="*", default=["C2", "T3", "P4", "O5", "A11", "B2"])
    ap.add_argument("--modes", nargs="*", default=["grammar", "free"])
    ap.add_argument("--backend", choices=["token", "chat"], default="token")
    ap.add_argument("--no-add-special", action="store_true", help="set if the probe chose add_special=False")
    ap.add_argument("--end-ids", type=int, nargs="*", default=None, help="override the end-of-reply token ids")
    ap.add_argument("--max-depth", type=int, default=16)
    ap.add_argument("--weights", action="store_true", help="also checksum the weights (minutes)")
    ap.add_argument("--stage1", help="stage1 results file (for plan)")
    ap.add_argument("--plan", help="plan file (for census)")
    ap.add_argument("--max-draws", type=int, default=10000, help="cap per comparison")
    ap.add_argument("--validation-states", type=int, default=10, help="states per arm in the validation census")
    ap.add_argument("--max-tokens", type=int, default=24, help="census reply length limit (above the tree depth)")
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()
    if args.step == "plan":  # no server needed
        return cmd_plan(args)
    if args.serve:  # start the model under its pinned clock, as every extraction run did
        from value_functions.serving import PinnedServer
        with PinnedServer(args.serve, port=args.port) as srv:
            args.url = srv.base_url
            return _run(args)
    return _run(args)


def _run(args):
    """Run the chosen step against the server at args.url."""
    if args.model is None:  # the extractor's default: the served file's name without .gguf
        mp = requests.get(f"{args.url.rstrip('/')}/props", timeout=60).json().get("model_path") or "model"
        args.model = Path(mp).stem
    if args.step == "probe+stage1":  # probe, then Stage 1 and the plan only if the gate passes
        gate = cmd_probe(args)
        if not gate["pass"]:
            print("Stopped after the probe: the gate failed. Send the probe file back.")
            return
        if gate["hours"] > args.max_hours:  # too long for one go: stop and let the user decide
            print(f"Stopped after the probe: Stage 1 would take about {gate['hours']:.1f} hours, more than "
                  f"--max-hours {args.max_hours}. Send the probe file back.")
            return
        args.no_add_special = not gate["add_special"]  # the begin-of-text choice the probe found
        args.stage1 = cmd_stage1(args)
        return cmd_plan(args)
    {"probe": cmd_probe, "stage1": cmd_stage1, "census": cmd_census}[args.step](args)


if __name__ == "__main__":
    main()
