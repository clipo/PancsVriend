"""E1: compare first-token readouts with the exact path sum, offline.

What it does, step by step:
  1. Opens each stored extraction record (vflp_*_states.jsonl.gz).
  2. For every (role, cell) line, finds the top-64 table at the EMPTY prefix
     (the very first token the model would write).
  3. Sorts each first token into: starts MOVE ("M"), starts STAY ("S"),
     pure whitespace ("ws"), or not allowed by the grammar ("x").
  4. Computes, per cell:
       m_off        = share of the allowed first-token mass that is pure whitespace
       ft_mask      = P(MOVE) from the first token only, masked to the grammar, at the stored T
       ft_mask_t1   = the same, re-tempered to T = 1
       ft_canon     = P(MOVE) from only the most likely "MOVE"-word token and
                      "STAY"-word token, at the stored T
       ft_canon_t1  = the same at T = 1
       path         = the exact path-sum P(MOVE) already stored in the record
       a_implied    = P(MOVE | reply starts with whitespace), implied by the bound
                      identity in section 2 of the draft (a self-check)
  5. Writes one CSV row per cell and prints a per-model summary.

Run it on the machine that holds the records, for example:
  python3 value_functions/first_token/first_token_vs_path_sum.py --probe value_functions/results/llm_logprob/raw/vflp_X_states.jsonl.gz
  python3 value_functions/first_token/first_token_vs_path_sum.py value_functions/results/llm_logprob/raw/vflp_*_states.jsonl.gz --out e1_cells.csv

--probe prints the structure of the first record so you can check the field
names in the ADAPTER section below. Only standard-library Python is used.

Check before trusting the summary:
  * The glob may also pick up archived race tables or date-study traces.
    Pass only the files the paper uses; duplicates are skipped with a warning.
  * The "unsaturated" rule (--eps) is a guess. Make it match the rule that
    produced the counts in Table 1 of the draft (399 for OLMo, 127 for Qwen, ...).
"""

import argparse  # reads the command-line options
import csv  # writes the per-cell results table
import gzip  # opens the .jsonl.gz files without unpacking them first
import json  # turns each line of text into a Python dictionary
import math  # gives us log() and isnan()
import statistics  # gives us median()

WHITESPACE = " \t\n"  # the whitespace characters the grammar allows before the word
DEFAULT_T = 0.3  # production temperature; used only if a file header does not record one
CANON_MOVE = ("MOVE", " MOVE", "Move", " Move", "move", " move")  # whole-word spellings a naive reader might use
CANON_STAY = ("STAY", " STAY", "Stay", " Stay", "stay", " stay")  # same for STAY


# ---------------------------------------------------------------- ADAPTER --
# The exact field names inside a record were not visible to me. These
# functions try the likely names and stop with a clear message if none fit.
# If --probe shows different names, change only this section.

def get_states(cell):
    """Return the per-state tables of one cell."""
    states = cell.get("states") or cell.get("prefix_states") or cell.get("tables")  # try likely names
    if states is None:  # none of the guesses matched
        raise KeyError(f"No states field found. Cell keys are: {sorted(cell.keys())}")  # show which keys exist
    return states  # a dict {prefix: table} or a list of state records


def get_root_table(cell):
    """Return the top-n list for the empty prefix as [(piece, prob), ...]."""
    states = get_states(cell)  # all prefix states of this cell
    if isinstance(states, dict):  # tables stored as {prefix: table}
        root = states.get("", states.get("root"))  # the empty prefix is the root of the tree
    else:  # tables stored as a list of state records
        root = next((s for s in states if isinstance(s, dict) and s.get("prefix") == ""), None)  # the state whose prefix is empty
        if root is None:  # no state is marked with an empty prefix
            root = next((s for s in states if isinstance(s, dict) and s.get("depth") == 0), None)  # fall back to depth 0
    if root is None:  # still nothing
        raise KeyError("Could not find the empty-prefix state; run --probe and edit get_root_table().")  # clear stop
    if isinstance(root, dict):  # the state is a record holding its table
        table = root.get("top") or root.get("top_probs") or root.get("table")  # try likely names for the token list
    else:  # the state is already the token list
        table = root  # use it as is
    if table is None:  # none of the guesses matched
        raise KeyError(f"No top-n table in the root state. Keys are: {sorted(root.keys())}")  # show which keys exist
    pairs = []  # will hold (piece, prob) tuples
    for entry in table:  # go through each candidate token
        if isinstance(entry, dict):  # entry like {"token": " M", "prob": 0.4}
            piece = entry.get("token", entry.get("piece"))  # the token's text
            prob = entry.get("prob", entry.get("p"))  # its probability after temperature
        else:  # entry like [" M", 0.4]
            piece, prob = entry[0], entry[1]  # unpack the pair
        if piece is None or prob is None:  # skip malformed entries instead of crashing
            continue  # go to the next entry
        pairs.append((str(piece), float(prob)))  # store as a simple tuple
    return pairs  # hand back the cleaned list


def get_path_p_move(cell):
    """Return the stored exact path-sum P(MOVE) for this cell."""
    return float(cell.get("p_move", cell.get("p_move_effective")))  # field named in the extraction docstring


def get_run_params(meta):
    """Read the extraction settings recorded in a file header, so the comparison uses exactly the simulation's settings."""
    lp = meta.get("logprob") or {}  # the logprob block of the header, if present (consumer tables have one)
    # ADAPTER EDIT (real layout): the vflp_*_states.jsonl.gz header is FLAT - n_probs, max_depth and
    # mass_floor sit at the top level next to temperature - so each setting falls back to the top level.
    arm = meta.get("arm")  # recorded in the consumer tables, not in the raw record header
    if arm is None:  # ADAPTER EDIT: the raw header names the endpoint and the grammar hash instead
        endpoint = str(meta.get("endpoint", ""))  # e.g. "/v1/chat/completions (campaign endpoint); ..."
        if endpoint.startswith("/v1/chat/completions") and meta.get("grammar_sha256"):  # chat endpoint with a grammar
            arm = "chat+grammar"  # the same arm name the consumer tables record
    return {  # the settings that matter for a like-for-like comparison
        "temperature": float(lp.get("temperature", meta.get("temperature", DEFAULT_T))),  # temperature the tables were read at
        "n_probs": lp.get("n_probs", meta.get("n_probs")),  # size of each stored top-n table
        "max_depth": lp.get("max_depth", meta.get("max_depth")),  # how deep the path tree went
        "mass_floor": lp.get("mass_floor", meta.get("mass_floor")),  # smallest path mass that was expanded
        "grammar_sha256": meta.get("grammar_sha256"),  # fingerprint of the grammar used
        "sampler_params": json.dumps(meta.get("sampler_params"), sort_keys=True),  # top_k, top_p, penalties
        "arm": arm,  # e.g. chat+grammar
    }


def cell_key(meta, cell, path):
    """A key that identifies one decision state, used to skip duplicates."""
    # ADAPTER EDIT (real layout): "label" is the stable model name (e.g. hermes-4.3-36b-chat-grammar). "model" is the
    # served gguf name, which differs between extraction dates for one model and would split it into two summary rows.
    model = meta.get("label", meta.get("model", path))  # which model
    scenario = meta.get("scenario", path)  # which scenario (falls back to the file name)
    comp = cell.get("cell", cell.get("composition"))  # the layouts guessed before the records were seen
    if comp is None and "n_similar" in cell and "n_occupied" in cell:  # ADAPTER EDIT (real layout): the composition is two flat fields
        comp = [cell["n_similar"], cell["n_occupied"]]  # (like neighbours, occupied neighbours) identifies the cell
    return (model, scenario, cell.get("role"), json.dumps(comp, sort_keys=True))  # tuple key


# --------------------------------------------------------------- ANALYSIS --

def classify(piece):
    """Say what a first token does under the grammar ws* (MOVE|STAY)."""
    if piece and all(ch in WHITESPACE for ch in piece):  # the token is only spaces, tabs or newlines
        return "ws"  # it delays the decision
    rest = piece.lstrip(WHITESPACE).lower()  # drop leading whitespace and ignore case
    if rest and "move".startswith(rest):  # e.g. "M", " Mo", "MOVE"
        return "M"  # this token commits to MOVE
    if rest and "stay".startswith(rest):  # e.g. "S", " ST", "Stay"
        return "S"  # this token commits to STAY
    return "x"  # anything else is not allowed by the grammar


def retemper(pairs, power):
    """Change temperature: p_new is proportional to p_old ** power."""
    raised = [(piece, p ** power) for piece, p in pairs]  # raise every probability to the power
    total = sum(p for _, p in raised)  # add them up to renormalise
    return [(piece, p / total) for piece, p in raised]  # divide so they sum to 1 over the top-n


def canon_ratio(pairs):
    """P(MOVE) from the single most likely whole-word MOVE token vs STAY token."""
    best = {}  # most likely probability seen for each exact spelling
    for piece, p in pairs:  # each candidate token
        best[piece] = max(p, best.get(piece, 0.0))  # keep the largest if a spelling repeats
    pm = max((best[s] for s in CANON_MOVE if s in best), default=None)  # best MOVE spelling present
    ps = max((best[s] for s in CANON_STAY if s in best), default=None)  # best STAY spelling present
    if pm is None or ps is None:  # one word is split into pieces or not in the top 64
        return math.nan  # cannot compute this shortcut here
    return pm / (pm + ps)  # renormalise over the pair


def first_token_readouts(pairs, stored_t=DEFAULT_T):
    """Compute m_off, ft_mask, ft_mask_t1, ft_canon and ft_canon_t1 from one root table read at stored_t."""
    mass = {"M": 0.0, "S": 0.0, "ws": 0.0, "x": 0.0}  # running totals per class
    for piece, p in pairs:  # each candidate first token
        mass[classify(piece)] += p  # add its probability to its class
    allowed = mass["M"] + mass["S"] + mass["ws"]  # what the grammar lets through
    m_off = mass["ws"] / allowed if allowed else math.nan  # share that starts with whitespace
    decided = mass["M"] + mass["S"]  # tokens that already decide the answer
    ft_mask = mass["M"] / decided if decided else math.nan  # P(MOVE) among deciding tokens

    t1 = retemper(pairs, stored_t / 1.0)  # same table at T = 1 (power = stored temperature)
    m1 = sum(p for piece, p in t1 if classify(piece) == "M")  # MOVE-starting mass at T = 1
    s1 = sum(p for piece, p in t1 if classify(piece) == "S")  # STAY-starting mass at T = 1
    ft_mask_t1 = m1 / (m1 + s1) if (m1 + s1) else math.nan  # masked first token at T = 1

    return m_off, ft_mask, ft_mask_t1, canon_ratio(pairs), canon_ratio(t1)  # five numbers per cell


def logit(p):
    """Log-odds, clipped so that 0 and 1 do not give infinity."""
    p = min(max(p, 1e-9), 1 - 1e-9)  # keep p strictly between 0 and 1
    return math.log(p / (1 - p))  # log of the odds


def summarise(values):
    """Median and max of a list, skipping NaN; also how many were NaN."""
    vals = [v for v in values if not math.isnan(v)]  # drop missing values
    missing = len(values) - len(vals)  # count what was dropped
    if not vals:  # nothing left to summarise
        return math.nan, math.nan, missing  # NaN results
    return statistics.median(vals), max(vals), missing  # the three numbers


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # one-line help text
    parser.add_argument("files", nargs="+", help="vflp_*_states.jsonl.gz records")  # input files
    parser.add_argument("--out", default="e1_cells.csv", help="per-cell CSV to write")  # output file
    parser.add_argument("--eps", type=float, default=0.01, help="unsaturated means eps < path < 1 - eps (CHECK against Table 1)")  # saturation rule
    parser.add_argument("--probe", action="store_true", help="print the first record's structure and stop")  # schema check
    args = parser.parse_args()  # read what the user typed

    rows, seen, dupes, run_params = [], set(), 0, {}  # results, keys used, duplicate count, settings per model
    for path in args.files:  # each record file
        with gzip.open(path, "rt") as fh:  # open the gzip file as text
            meta = json.loads(fh.readline())  # line 0 is the header with model and settings
            params = get_run_params(meta)  # the simulation's own settings for this file
            run_params.setdefault(meta.get("model", meta.get("label", path)), set()).add(json.dumps(params, sort_keys=True))  # remember them per model
            for line in fh:  # every later line is one (role, cell)
                cell = json.loads(line)  # parse it
                if args.probe:  # just show the structure
                    print("header keys:", sorted(meta.keys()))  # what the header holds
                    print("header settings:", params)  # the settings this comparison will use
                    print("cell keys:", sorted(cell.keys()))  # what a cell holds
                    states = get_states(cell)  # the per-state tables
                    first = next(iter(states.values())) if isinstance(states, dict) else states[0]  # one example state
                    print("states container:", type(states).__name__, "| example state:", str(first)[:600])  # show its shape
                    return  # stop after the first cell
                key = cell_key(meta, cell, path)  # identify this decision state
                if key in seen:  # already counted from another file
                    dupes += 1  # remember how many we skipped
                    continue  # skip it
                seen.add(key)  # mark as counted
                m_off, ft_mask, ft_mask_t1, ft_canon, ft_canon_t1 = first_token_readouts(get_root_table(cell), params["temperature"])  # first-token numbers at the file's own temperature
                p = get_path_p_move(cell)  # exact value
                a = (p - (1 - m_off) * ft_mask) / m_off if m_off and not math.isnan(ft_mask) else math.nan  # implied P(MOVE | whitespace first)
                rows.append({  # collect one row for the CSV
                    "model": key[0], "scenario": key[1], "role": key[2], "cell": key[3],  # which state
                    "path": p, "m_off": m_off, "ft_mask": ft_mask, "ft_mask_t1": ft_mask_t1,  # main numbers
                    "ft_canon": ft_canon, "ft_canon_t1": ft_canon_t1, "a_implied": a,  # shortcuts and self-check
                })

    if not rows:  # nothing was read
        print("No cells found. Check the file paths, or run --probe.")  # say so instead of crashing
        return  # stop
    if dupes:  # some states appeared in more than one file
        print(f"WARNING: skipped {dupes} duplicate states; check that only the paper's files were passed.")  # tell the user

    with open(args.out, "w", newline="") as fh:  # write the per-cell table
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))  # columns from the first row
        writer.writeheader()  # column names
        writer.writerows(rows)  # all cells

    bad = [r for r in rows if not math.isnan(r["a_implied"]) and not (-1e-6 <= r["a_implied"] <= 1 + 1e-6)]  # self-check: a must be a probability
    if bad:  # the bound identity failed somewhere
        print(f"SELF-CHECK: {len(bad)} cells give an implied a outside [0, 1]; inspect them before using E1.")  # warn loudly

    print("Settings read from the record headers (must match the simulation runs):")  # like-for-like check
    for model, sets in sorted(run_params.items()):  # one line per model
        flag = "" if len(sets) == 1 else "  <-- WARNING: files for this model disagree"  # mixed settings are a problem
        print(f"  {model}: {sorted(sets)[0]}{flag}")  # show the settings
    print("model | unsat | m_off med/max | |path-ft_mask| med/max (share>0.01) | logit diff ft_mask med | |path-ft_canon_t1| med/max (NaN cells)")  # summary header
    for model in sorted({r["model"] for r in rows}):  # one summary line per model
        unsat = [r for r in rows if r["model"] == model and args.eps < r["path"] < 1 - args.eps]  # cells not pinned at 0 or 1
        if not unsat:  # no unsaturated cells for this model
            print(f"{model} | 0 | -")  # nothing to summarise
            continue  # next model
        mo = summarise([r["m_off"] for r in unsat])  # whitespace-first mass
        dm_vals = [abs(r["path"] - r["ft_mask"]) for r in unsat]  # masked first-token errors
        dm = summarise(dm_vals)  # their median and max
        share = sum(1 for v in dm_vals if v > 0.01) / len(dm_vals)  # share of cells off by more than 0.01 (the gate)
        lg = summarise([abs(logit(r["path"]) - logit(r["ft_mask"])) for r in unsat if not math.isnan(r["ft_mask"])])  # logit-scale error
        dc = summarise([abs(r["path"] - r["ft_canon_t1"]) for r in unsat])  # canonical first-token error at T = 1
        print(f"{model} | {len(unsat)} | {mo[0]:.3g}/{mo[1]:.3g} | {dm[0]:.3g}/{dm[1]:.3g} ({share:.0%}) | {lg[0]:.3g} | {dc[0]:.3g}/{dc[1]:.3g} ({dc[2]})")  # one line


if __name__ == "__main__":  # run main() only when the file is executed directly
    main()
