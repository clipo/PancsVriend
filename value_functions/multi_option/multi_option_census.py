"""Sampled check ("census") of the exact option probabilities, and the tests.

The census draws real replies from the server, one at a time, with exactly the
request the simulation sends, and counts how often each option comes out. It
is the unbiased (but noisy) estimate of the target: how often the deployed
sampler plus the parser returns each option.

Two tests are used (design v4, section 9.5):

* Directed likelihood-ratio test between two fully specified reads, e.g. the
  path sum and a first-token read. With counts n[s][o] (state s, option o),
      Lambda = sum_s sum_o n[s][o] * log( P_path[s][o] / P_alt[s][o] ).
  Large Lambda favours the path sum, small Lambda the other read. Its spread
  under each read is found by simulating censuses from that read (parametric
  bootstrap). Decision: "path" if Lambda is above the 99th percentile under
  the other read, "alt" if below the 1st percentile under the path sum, else
  "not separated".
* Deviance G (goodness of fit) of the counts against one read, with a
  bootstrap p-value: the paper's existing test, extended to K options.

How many draws are needed is planned from the exact values before any draw:
the information I = sum_s n_s * chi2_s with chi2_s = sum_o (p - r)^2 / p must
reach 16 for both error rates to be about 2%.
"""

import math  # logarithms
import time  # pauses between retries

import numpy as np  # fast simulated censuses
import requests  # talks to llama-server

from option_language import grammar_language, free_language, parse_loose  # the reply languages and the loose parser


# ----------------------------------------------------------------------
# Drawing replies.
# ----------------------------------------------------------------------

class Sampler:
    """Draws whole replies through the chat endpoint, with the simulation's payload."""

    def __init__(self, url, model, temperature, sampler_params, timeout=120, session=None):
        self.url = url.rstrip("/")  # server address
        self.model = model  # model name as served
        self.temperature = float(temperature)  # sampling temperature (0.3)
        self.sampler_params = dict(sampler_params)  # the simulation's sampler settings
        self.timeout = timeout  # seconds before a request is abandoned
        self.session = session or requests.Session()  # one connection, one request in flight

    def payload(self, user_prompt, grammar, max_tokens, seed):
        """The JSON body of one draw: the simulation's fields, plus a fresh seed."""
        body = {"model": self.model, "stream": False, "temperature": self.temperature,
                "max_tokens": max_tokens, **self.sampler_params, "cache_prompt": False,
                "seed": seed, "messages": [{"role": "user", "content": user_prompt}]}
        if grammar is not None:  # WITH grammar
            body["grammar"] = grammar
        return body

    def draw(self, user_prompt, grammar, max_tokens, seed):
        """One reply: (text, finish_reason). finish_reason 'stop' means the end token was drawn."""
        body = self.payload(user_prompt, grammar, max_tokens, seed)  # build the request
        for attempt in range(6):  # a few tries on network errors
            try:
                r = self.session.post(f"{self.url}/v1/chat/completions", json=body, timeout=self.timeout)
                r.raise_for_status()  # stop on an HTTP error
                ch = r.json()["choices"][0]  # the one reply
                return ch["message"]["content"], ch.get("finish_reason")
            except (requests.RequestException, KeyError, ValueError, IndexError):
                if attempt == 5:  # give up after six tries
                    raise
                time.sleep(2 * (attempt + 1))  # wait longer each time
        return None, None


def classify(labels, text, finish_reason, mode):
    """(strict option or None, loose option or None) for one drawn reply.

    Strict (the event the path sum computes): the reply must have ended by the
    end token ('stop'); with grammar it must spell a label (whitespace allowed),
    without grammar it must match the strict free language. Loose: the one
    label named as a word anywhere in the reply (what a simulation would parse).
    None means off-format."""
    ended = finish_reason == "stop"  # cut off by max_tokens otherwise
    lang = grammar_language(labels) if mode == "grammar" else free_language(labels)  # a list or a ready-made language
    strict = lang.complete(text) if ended else None  # the event the path sum computes
    loose = parse_loose(lang.labels, text)  # whole-word search, longest label wins
    return strict, loose


def census(sampler, user_prompt, labels, mode, n, seed0, max_tokens, grammar=None):
    """n draws, one at a time. Returns counts (K options + off-format) for both parses and the replies."""
    lang = grammar_language(labels) if mode == "grammar" else free_language(labels)  # a list or a ready-made language
    K = len(lang.labels)  # number of options
    strict = [0] * (K + 1)  # last slot counts off-format replies
    loose = [0] * (K + 1)
    replies = []  # every reply, for the record
    if mode == "grammar" and grammar is None:  # the grammar the simulation would send
        grammar = lang.gbnf()
    for i in range(n):  # one draw at a time, never in parallel
        text, fin = sampler.draw(user_prompt, grammar if mode == "grammar" else None, max_tokens, seed0 + i)
        s, l = classify(lang, text, fin, mode)  # parse it both ways
        strict[K if s is None else s] += 1  # count it
        loose[K if l is None else l] += 1
        replies.append({"seed": seed0 + i, "text": text, "finish_reason": fin, "strict": s, "loose": l})
    return {"strict": strict, "loose": loose, "replies": replies}


# ----------------------------------------------------------------------
# Planning: how many draws a comparison needs.
# ----------------------------------------------------------------------

def chi2_state(p, r):
    """chi2_s = sum_o (p_o - r_o)^2 / p_o: information per draw separating read r from read p."""
    out = 0.0  # running sum
    for a, b in zip(p, r):  # each option
        if a > 0:  # an option with zero probability under p adds nothing here
            out += (a - b) ** 2 / a
    return out


def plan_draws(p_by_state, r_by_state, target=16.0, block=100, cap=10000):
    """Choose states and draws so that I = sum n_s chi2_s reaches target, using the fewest draws.

    Draws go in blocks of `block`, always to the state with the largest chi2_s
    that has fewer than 10 blocks (so no single state carries everything).
    Returns (dict state -> draws, I reached, enough?)."""
    chi = {s: chi2_state(p_by_state[s], r_by_state[s]) for s in p_by_state}  # information per draw
    order = sorted(chi, key=lambda s: -chi[s])  # most informative first
    plan, info, used = {}, 0.0, 0  # draws per state, information so far, draws so far
    per_state_cap = 10 * block  # at most 10 blocks per state
    while info < target and used + block <= cap:  # until enough or out of budget
        pick = next((s for s in order if plan.get(s, 0) < per_state_cap and chi[s] > 0), None)  # best state left
        if pick is None:  # every useful state is full
            break
        plan[pick] = plan.get(pick, 0) + block  # one more block there
        info += block * chi[pick]  # its information
        used += block
    return plan, info, info >= target


# ----------------------------------------------------------------------
# The tests.
# ----------------------------------------------------------------------

def _as_arrays(counts, P):
    """Counts and probabilities as numpy arrays of shape (states, options)."""
    return np.asarray(counts, dtype=float), np.asarray(P, dtype=float)


def log_lr(counts, P_path, P_alt):
    """Lambda = sum n log(P_path / P_alt). +inf / -inf when one read is refuted by an observed reply."""
    n, a = _as_arrays(counts, P_path)  # counts and path-sum probabilities
    _, b = _as_arrays(counts, P_alt)  # the other read
    seen = n > 0  # only observed options enter
    if np.any(seen & (b <= 0) & (a > 0)):  # the other read said impossible, yet it happened
        return math.inf
    if np.any(seen & (a <= 0) & (b > 0)):  # the path sum said impossible, yet it happened
        return -math.inf
    ok = seen & (a > 0) & (b > 0)  # cells that contribute
    return float(np.sum(n[ok] * (np.log(a[ok]) - np.log(b[ok]))))


def simulate_counts(P, n_by_state, rng):
    """One simulated census: multinomial draws from P with the given draws per state."""
    return np.array([rng.multinomial(int(n), np.asarray(p) / np.sum(p)) for p, n in zip(P, n_by_state)])


def lr_test(counts, P_path, P_alt, n_boot=10000, seed=12345, alpha=0.01):
    """Directed likelihood-ratio test between the path sum and another read.

    Returns a dict with Lambda, its 1st and 99th percentiles under each read,
    and decision 'path', 'alt', 'not separated', or 'structural (...)' when an
    observed reply had probability zero under one read."""
    counts = np.asarray(counts)  # observed counts, states x options
    lam = log_lr(counts, P_path, P_alt)  # observed statistic
    if math.isinf(lam):  # one read is refuted outright
        return {"Lambda": lam, "decision": "structural (alt refuted)" if lam > 0 else "structural (path refuted)"}
    n_by_state = counts.sum(axis=1)  # draws per state
    rng = np.random.default_rng(seed)  # fixed seed: the result can be repeated
    under_alt = np.array([log_lr(simulate_counts(P_alt, n_by_state, rng), P_path, P_alt) for _ in range(n_boot)])
    under_path = np.array([log_lr(simulate_counts(P_path, n_by_state, rng), P_path, P_alt) for _ in range(n_boot)])
    hi_alt = float(np.quantile(under_alt[np.isfinite(under_alt)], 1 - alpha))  # 99th percentile if the other read were true
    lo_path = float(np.quantile(under_path[np.isfinite(under_path)], alpha))  # 1st percentile if the path sum were true
    if lam > hi_alt and lam >= lo_path:  # clearly too large for the other read
        decision = "path"
    elif lam < lo_path and lam <= hi_alt:  # clearly too small for the path sum
        decision = "alt"
    else:  # both reads would produce a value like this
        decision = "not separated"
    return {"Lambda": lam, "hi_under_alt": hi_alt, "lo_under_path": lo_path,
            "p_under_alt": float(np.mean(under_alt >= lam)), "p_under_path": float(np.mean(under_path <= lam)),
            "decision": decision, "n_boot": n_boot, "seed": seed}


def deviance(counts, P):
    """G = 2 sum n log(n / (N p)): how far the counts are from read P (0 is a perfect fit)."""
    n, p = _as_arrays(counts, P)  # counts and probabilities
    N = n.sum(axis=1, keepdims=True)  # draws per state
    expected = N * p  # what read P expects
    ok = n > 0  # 0 * log 0 = 0
    if np.any(ok & (expected <= 0)):  # an impossible reply was seen
        return math.inf
    return float(2.0 * np.sum(n[ok] * np.log(n[ok] / expected[ok])))


def deviance_test(counts, P, n_boot=10000, seed=12345):
    """Bootstrap p-value of G under read P (the paper's test, K options). High p: no detected disagreement."""
    counts = np.asarray(counts)  # observed counts
    g = deviance(counts, P)  # observed G
    rng = np.random.default_rng(seed)  # fixed seed
    n_by_state = counts.sum(axis=1)  # draws per state
    sims = np.array([deviance(simulate_counts(P, n_by_state, rng), P) for _ in range(n_boot)])  # G under P
    return {"G": g, "null_mean": float(np.mean(sims)), "p": float(np.mean(sims >= g)), "n_boot": n_boot, "seed": seed}


def on_format_split(counts):
    """Split free-census counts (K options + off-format) into (on vs off per state, options among on-format)."""
    counts = np.asarray(counts)  # states x (K + 1)
    on_off = np.stack([counts[:, :-1].sum(axis=1), counts[:, -1]], axis=1)  # [on-format, off-format]
    return on_off, counts[:, :-1]  # the second part is the conditional multinomial


def conditional(P):
    """Probabilities rescaled over the options only (drops an off-format share)."""
    P = np.asarray(P, dtype=float)  # states x K
    return P / P.sum(axis=1, keepdims=True)


def holm(pvalues):
    """Holm-adjusted p-values, in the original order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])  # smallest first
    adj, running = [0.0] * len(pvalues), 0.0  # result, the running maximum
    for rank, i in enumerate(order):  # step down
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adj[i] = running
    return adj
