"""Exact option probabilities for K-option choices, with or without a grammar,
and the first-token reads they are compared with.

This is the K-option version of path_sum() in
value_functions/logprob/logprob_value_function.py.

What a "path sum" is
--------------------
The model writes its reply one token (a word piece) at a time. An option such
as "GIVE A LOT" can be written as many different token sequences
("GIVE"+" A"+" LOT", "G"+"IVE"+" A LOT", ...). The probability that the agent
picks the option is the sum, over every token sequence that spells it, of the
product of the next-token probabilities along that sequence. This module walks
that tree of token sequences, most probable branch first, asking the server for
one next-token table per branch point.

Two ways to ask the server ("backends")
---------------------------------------
* ChatTextBackend: the simulation's chat endpoint, with the text written so far
  sent as an assistant prefill. This is exactly how the existing extractor asks.
  The server turns that text back into tokens its own usual way, so two
  different token paths that spell the same text are read as one. That was
  harmless for MOVE/STAY (every reply is decided by its first token). It is NOT
  harmless once options share a start, so this backend is kept only as a check.
* TokenBackend (primary): the chat prompt is rendered and turned into token ids
  once, and every request sends that id list plus the ids chosen so far to the
  /completion endpoint. Each branch is then a real token sequence, and the sum
  is exact. The probe checks first that its root table equals the chat
  endpoint's table, so it reads the same distribution the simulation samples.

Two modes
---------
* "grammar": only tokens that keep the reply inside the allowed language are
  kept, and their probabilities are rescaled to add to one at every step
  (what llama.cpp's grammar sampler does). Same rule as the existing extractor.
* "free": no grammar field and no rescaling. Every token that leaves the
  strict reply format adds its probability to "off-format". A reply counts as
  an option only if the end-of-reply token comes right after a complete label
  (plus an optional full stop and whitespace).

The error term B
----------------
Every number carries B: probability that fell outside the top-n tables, below
the mass floor, or past the depth limit. The true value lies within +/- B.

The reads compared with the path sum
------------------------------------
* FT-mask:   first token only; each token is credited to the one option it can
             start; tokens that could start several options (and whitespace)
             are dropped; the rest is rescaled to add to one.
* FT-split:  first token only, as a modeller would usually write it: a token
             that could start several options is split equally among them
             (whitespace is still dropped).
* depth-d:   follow the tree only d tokens deep, credit what is decided by
             then, drop the rest, rescale. FT-mask is depth 1.

Only the standard library and `requests` are used.
"""

import time  # pauses between retries
from collections import namedtuple  # light-weight records

import requests  # talks to llama-server

from option_language import grammar_language, free_language  # the reply languages


# One entry of a next-token table: token id (None if the server does not give
# one), the token's text, its probability, and whether it ends the reply.
Entry = namedtuple("Entry", "id piece prob end")


# ----------------------------------------------------------------------
# Shared request helpers.
# ----------------------------------------------------------------------

def _post_json(session, url, body, timeout, tries=6):
    """POST body to url and return the JSON answer, retrying on network errors like the extractor."""
    for attempt in range(tries):  # a few tries, waiting longer each time
        try:
            r = session.post(url, json=body, timeout=timeout)  # send the request
            r.raise_for_status()  # turn an HTTP error into an exception
            return r.json()  # the decoded answer
        except (requests.RequestException, ValueError):  # network trouble or a bad answer
            if attempt == tries - 1:  # out of tries
                raise  # let the caller see the error
            time.sleep(2 * (attempt + 1))  # wait 2, 4, 6, ... seconds
    return None  # never reached


def _entries(items, end_ids, end_pieces):
    """Turn a server's top-n list into Entry records."""
    out = []  # the cleaned table
    for it in items:  # one candidate token at a time
        piece = it.get("token", "")  # the token's text
        if isinstance(piece, list):  # the server sent raw bytes: never part of an answer
            piece = "�"  # the replacement character keeps it out of every label
        if "prob" in it:  # post_sampling_probs gives probabilities directly
            prob = float(it["prob"])
        else:  # otherwise convert the log-probability
            prob = float(2.718281828459045 ** it["logprob"])
        tid = it.get("id")  # the token id, when the server reports it
        if end_ids and tid is not None:  # ids are the safest way to spot the end token
            end = tid in end_ids
        else:  # fall back to the token's text
            end = piece in end_pieces
        out.append(Entry(tid, piece, prob, end))  # one record
    return out


# ----------------------------------------------------------------------
# Backend 1: the chat endpoint with a text prefill (the extractor's way).
# ----------------------------------------------------------------------

class ChatTextBackend:
    """Reads next-token tables through /v1/chat/completions, prefix sent as text."""

    kind = "chat_text"  # written into every record so results say how they were read

    def __init__(self, url, model, temperature, n_probs, sampler_params,
                 end_ids=(), end_pieces=("",), timeout=120, session=None):
        self.url = url.rstrip("/")  # server address without a trailing slash
        self.model = model  # model name as served
        self.temperature = float(temperature)  # sampling temperature (0.3 in every campaign)
        self.n_probs = int(n_probs)  # size of each top-n table (64)
        self.sampler_params = dict(sampler_params)  # the simulation's neutral sampler settings
        self.end_ids = set(end_ids)  # token ids that end the reply
        self.end_pieces = set(end_pieces)  # token texts that end the reply (used when ids are missing)
        self.timeout = timeout  # seconds before a request is abandoned
        self.session = session or requests.Session()  # reuse one connection
        self.n_requests = 0  # how many table requests were made

    def payload(self, user_prompt, prefix_text, grammar):
        """The JSON body of one next-token request: the existing extractor's fields."""
        messages = [{"role": "user", "content": user_prompt}]  # the prompt as the single user message
        if prefix_text:  # text already written goes in as an assistant prefill
            messages.append({"role": "assistant", "content": prefix_text})
        body = {"model": self.model, "messages": messages, "max_tokens": 1,
                "n_probs": self.n_probs, "post_sampling_probs": True,
                "temperature": self.temperature, "cache_prompt": False,
                "stream": False, **self.sampler_params}  # same fields as the extractor
        if grammar is not None:  # WITH grammar: tell the server what may follow
            body["grammar"] = grammar
        return body

    def next_token(self, user_prompt, prefix_ids, prefix_text, grammar):
        """The next-token table after prefix_text (prefix_ids is ignored here)."""
        j = _post_json(self.session, f"{self.url}/v1/chat/completions",
                       self.payload(user_prompt, prefix_text, grammar), self.timeout)  # one request
        cp = j["choices"][0]["logprobs"]["content"][0]  # the one generated position
        items = cp.get("top_probs") or cp.get("top_logprobs") or []  # its top-n table
        self.n_requests += 1  # count it
        return _entries(items, self.end_ids, self.end_pieces)


# ----------------------------------------------------------------------
# Backend 2: token ids through /completion (exact token paths).
# ----------------------------------------------------------------------

class TokenBackend:
    """Reads next-token tables through /completion, prefix sent as token ids."""

    kind = "token_ids"  # written into every record

    def __init__(self, url, model, temperature, n_probs, sampler_params,
                 end_ids=(), end_pieces=("",), add_special=True, timeout=120, session=None):
        self.url = url.rstrip("/")  # server address
        self.model = model  # model name (recorded; /completion does not need it)
        self.temperature = float(temperature)  # sampling temperature
        self.n_probs = int(n_probs)  # size of each top-n table
        self.sampler_params = dict(sampler_params)  # the simulation's sampler settings
        self.end_ids = set(end_ids)  # token ids that end the reply
        self.end_pieces = set(end_pieces)  # token texts that end the reply
        self.add_special = bool(add_special)  # whether /tokenize adds the begin-of-text token (fixed by the probe)
        self.timeout = timeout  # seconds before a request is abandoned
        self.session = session or requests.Session()  # reuse one connection
        self.n_requests = 0  # how many table requests were made
        self._base = {}  # cache: user prompt -> token ids of the rendered chat prompt

    def render(self, user_prompt):
        """The chat prompt exactly as the chat endpoint would build it (/apply-template)."""
        j = _post_json(self.session, f"{self.url}/apply-template",
                       {"messages": [{"role": "user", "content": user_prompt}]}, self.timeout)
        return j["prompt"]  # the rendered text, ending with the assistant turn's opening

    def tokenize(self, text, add_special=False):
        """Token ids of text, with special tokens such as <|im_start|> recognised."""
        j = _post_json(self.session, f"{self.url}/tokenize",
                       {"content": text, "add_special": add_special, "parse_special": True},
                       self.timeout)
        return [t if isinstance(t, int) else t["id"] for t in j["tokens"]]  # plain ids

    def base_ids(self, user_prompt):
        """Token ids of the rendered chat prompt (computed once per prompt)."""
        if user_prompt not in self._base:  # not computed yet
            self._base[user_prompt] = self.tokenize(self.render(user_prompt), self.add_special)
        return self._base[user_prompt]

    def payload(self, user_prompt, prefix_ids, grammar):
        """The JSON body of one /completion request; same sampler fields as the chat request."""
        body = {"prompt": list(self.base_ids(user_prompt)) + list(prefix_ids),  # prompt + tokens so far
                "n_predict": 1, "n_probs": self.n_probs, "post_sampling_probs": True,
                "temperature": self.temperature, "cache_prompt": False,
                "stream": False, **self.sampler_params}
        if grammar is not None:  # WITH grammar: what may follow
            body["grammar"] = grammar
        return body

    def next_token(self, user_prompt, prefix_ids, prefix_text, grammar):
        """The next-token table after the token sequence prefix_ids."""
        j = _post_json(self.session, f"{self.url}/completion",
                       self.payload(user_prompt, prefix_ids, grammar), self.timeout)  # one request
        cp = j["completion_probabilities"][0]  # the one generated position
        items = cp.get("top_probs") or cp.get("top_logprobs") or []  # its top-n table
        self.n_requests += 1  # count it
        return _entries(items, self.end_ids, self.end_pieces)


def discover_end_ids(url, timeout=60):
    """Token ids of the model's end-of-text token, from /props then /tokenize (checked again in the probe)."""
    s = requests.Session()  # a short-lived connection
    props = s.get(f"{url.rstrip('/')}/props", timeout=timeout).json()  # server and model settings
    ids = set()  # what we find
    for key in ("eos_token", "eot_token"):  # names llama.cpp uses for the end tokens
        text = props.get(key)  # the token's text, if reported
        if text:  # tokenise it with special tokens recognised
            j = _post_json(s, f"{url.rstrip('/')}/tokenize",
                           {"content": text, "add_special": False, "parse_special": True}, timeout)
            toks = [t if isinstance(t, int) else t["id"] for t in j["tokens"]]
            if len(toks) == 1:  # a real special token is one id
                ids.add(toks[0])
    return ids


# ----------------------------------------------------------------------
# The path sum.
# ----------------------------------------------------------------------

def path_sum(backend, user_prompt, labels, mode, max_depth, mass_floor, root_grammar=None,
             keep_top=16):
    """Exact probability of each option for one prompt.

    backend:       ChatTextBackend or TokenBackend
    labels:        the option labels, in the order the prompt lists them, or a
                   ready-made OptionLanguage for that mode (e.g. the JSON format)
    mode:          "grammar" (masked and rescaled) or "free" (no grammar)
    max_depth:     the longest token sequence followed
    mass_floor:    branches less likely than this are not followed (added to B)
    root_grammar:  grammar text for the first request (default: built from the
                   labels); pass llm_runner.MOVE_STAY_GRAMMAR to reproduce the
                   simulation's request byte for byte
    Returns a dict:
      p           probability of each option
      off_format  probability of leaving the format (free mode; 0 with grammar)
      bound       the error term B
      n_requests  requests used
      root        the first-token table (list of Entry)
      arrivals    every place the tree reached, for the depth-d reads
      paths       every finished path that counted as an option
      states      every table read (top keep_top entries kept)
    """
    if mode not in ("grammar", "free"):  # guard against typos
        raise ValueError(mode)
    lang = grammar_language(labels) if mode == "grammar" else free_language(labels)  # which replies count
    labels = lang.labels  # labels may have been given as a ready-made language (e.g. JSON)
    K = len(labels)  # number of options
    p = [0.0] * K  # probability of each option
    off = 0.0  # probability of leaving the format (free mode only)
    bound = 0.0  # error term B
    arrivals = []  # where the tree went, with masses, for the depth-d reads
    paths, states, root = [], [], None  # records kept for checking
    n0 = backend.n_requests  # to count this prompt's requests
    # A branch: (token ids so far, text so far, probability of reaching it, tokens written)
    frontier = [((), "", 1.0, 0)]

    def arrive(text, m, depth, parent_text, kind, option=None):
        """Record that mass m reached text after depth tokens."""
        decided = option if option is not None else (lang.decided(text) if kind == "node" else None)
        arrivals.append({"depth": depth, "text": text, "mass": m, "kind": kind,
                         "decided": decided, "parent_decided": lang.decided(parent_text)})

    def settle(ids, text, m, depth, parent_text, by_end):
        """Credit mass m to the option that the finished reply text spells."""
        c = lang.complete(text)  # which option it spells
        p[c] += m  # credit that option
        arrive(text, m, depth, parent_text, "end" if by_end else "node", option=c)  # for the depth reads
        paths.append({"ids": list(ids), "text": text, "option": labels[c], "mass": m})  # keep the path

    while frontier:  # expand the most probable unfinished branch first
        frontier.sort(key=lambda s: -s[2])  # most probable first
        ids, text, mass, depth = frontier.pop(0)  # take it
        if mass < mass_floor or depth >= max_depth:  # too small or too deep to follow
            bound += mass  # counts as unresolved
            continue
        comp = lang.complete(text)  # option spelled if the reply stopped here
        if mode == "grammar":  # what the server's grammar should allow next
            grammar = (root_grammar or lang.gbnf()) if text == "" else lang.remainder_gbnf(text)
        else:  # WITHOUT grammar: no grammar field
            grammar = None
        top = backend.next_token(user_prompt, ids, text, grammar)  # one request
        if depth == 0:  # keep the whole first-token table for the first-token reads
            root = list(top)
        captured = sum(e.prob for e in top)  # probability the top-n table covers
        states.append({"ids": list(ids), "prefix": text, "mass": mass, "captured": captured,
                       "top": [e._asdict() for e in top[:keep_top]]})

        if mode == "grammar":
            allowed = []  # tokens the grammar lets through
            for e in top:  # each listed token
                if e.prob <= 0.0:  # nothing to add
                    continue
                if e.end:  # ending is allowed only right after a complete label
                    if comp is not None:
                        allowed.append(e)
                    continue
                if e.piece and lang.live(text + e.piece):  # the reply can still count
                    allowed.append(e)
            allowed_mass = sum(e.prob for e in allowed)  # what the mask keeps
            if allowed_mass <= 0.0:  # nothing valid in the table
                bound += mass  # unresolved
                continue
            # Probability outside the table could belong to allowed tokens. After
            # rescaling, its share of this branch is at most (1 - captured) / allowed_mass.
            bound += mass * min(1.0, max(0.0, 1.0 - captured) / allowed_mass)
            for e in allowed:  # masked and rescaled
                m = mass * e.prob / allowed_mass  # this token's share of the branch
                if e.end:  # the reply ends here
                    settle(ids, text, m, depth + 1, text, by_end=True)
                    continue
                new_ids = ids + ((e.id,) if e.id is not None else ())  # token path so far
                new = text + e.piece  # text after this token
                if lang.complete(new) is not None and not lang.can_grow(new):  # the grammar forces the end
                    settle(new_ids, new, m, depth + 1, text, by_end=False)
                else:  # keep going
                    arrive(new, m, depth + 1, text, "node")
                    frontier.append((new_ids, new, m, depth + 1))
        else:  # free mode: no mask, no rescaling
            bound += mass * max(0.0, 1.0 - captured)  # probability outside the table
            for e in top:  # each listed token
                if e.prob <= 0.0:
                    continue
                m = mass * e.prob  # the model's own probability of this continuation
                if e.end:  # the reply ends here
                    if comp is not None:  # it spells an option
                        settle(ids, text, m, depth + 1, text, by_end=True)
                    else:  # it ended off-format
                        off += m
                        arrive(text, m, depth + 1, text, "off")
                    continue
                new = text + e.piece  # text after this token
                if e.piece and lang.live(new):  # still a reply that can count
                    new_ids = ids + ((e.id,) if e.id is not None else ())
                    arrive(new, m, depth + 1, text, "node")
                    frontier.append((new_ids, new, m, depth + 1))
                else:  # left the format
                    off += m
                    arrive(new, m, depth + 1, text, "off")
    return {"p": p, "off_format": off, "bound": bound, "n_requests": backend.n_requests - n0,
            "backend": backend.kind, "mode": mode, "labels": list(labels),
            "root": root, "arrivals": arrivals,
            "paths": sorted(paths, key=lambda x: -x["mass"]), "states": states}


# ----------------------------------------------------------------------
# The first-token reads.
# ----------------------------------------------------------------------

UNDEFINED_SHARE = 0.01  # a first-token read is "undefined" when under 1% of its mass decides an option


def _first_token_masses(root, labels, mode, power):
    """For each first token: (re-tempered weight, set of options it can still start)."""
    lang = grammar_language(labels) if mode == "grammar" else free_language(labels)  # which replies count
    out = []  # (weight, options) per useful token
    for e in root:  # each listed first token
        if e.end or not e.piece or e.prob <= 0.0:  # an empty reply never counts
            continue
        if not lang.live(e.piece):  # masked out (grammar) or already off-format (free)
            continue
        out.append((e.prob ** power, lang.possible(e.piece), e.piece))  # power re-tempers the table
    return out


def _labels(labels):
    """The label list, whether labels is a list or an OptionLanguage."""
    return labels.labels if hasattr(labels, "labels") else list(labels)


def ft_mask(root, labels, mode, power=1.0):
    """FT-mask: each first token credited to the one option it can start; the rest dropped.

    power re-tempers the table: power = T_run / T_read (0.3 reads it at T = 1).
    Returns (probabilities, unresolved share m_off). The probabilities are NaN
    when less than UNDEFINED_SHARE of the mass decides an option."""
    res = [0.0] * len(_labels(labels))  # resolved weight per option
    unresolved = 0.0  # weight that does not decide one option
    for w, opts, _ in _first_token_masses(root, labels, mode, power):
        if len(opts) == 1:  # the token decides one option
            res[next(iter(opts))] += w
        else:  # whitespace or a shared start
            unresolved += w
    total = sum(res)  # resolved weight
    m_off = unresolved / (unresolved + total) if unresolved + total > 0 else 1.0  # unresolved share
    if total <= 0.0 or total < UNDEFINED_SHARE * (total + unresolved):  # too little resolves
        return [float("nan")] * len(_labels(labels)), m_off
    return [r / total for r in res], m_off


def ft_split(root, labels, mode, power=1.0):
    """FT-split: a shared first token's weight is split equally among the options it can start.

    Whitespace-only tokens (which can start every option) are dropped, as in
    FT-mask. Returns probabilities (NaN if nothing is credited)."""
    res = [0.0] * len(_labels(labels))  # credited weight per option
    for w, opts, piece in _first_token_masses(root, labels, mode, power):
        if not piece.strip():  # whitespace only: tells us nothing about the option
            continue
        for o in opts:  # split equally among the options this token can start
            res[o] += w / len(opts)
    total = sum(res)  # all credited weight
    if total <= 0.0:  # nothing credited
        return [float("nan")] * len(_labels(labels))
    return [r / total for r in res]


def depth_read(result, d):
    """Depth-d read from a stored path sum: follow d tokens, credit what is decided, drop the rest.

    Uses only the tree already read, so it costs no requests. Depth 1 equals
    FT-mask at the run temperature. Returns (probabilities, unresolved share
    m_off^(d)); probabilities are NaN when less than UNDEFINED_SHARE resolves.
    With grammar: TV(path sum, depth-d read) <= m_off^(d) + 2B (up to rounding),
    because once a reply is decided all of its mass reaches that option.
    Without grammar there is no such bound: a reply that has started an option
    can still leave the format, at a different rate for each option, so the
    read gives the share of replies that START each option, not the share of
    on-format replies that END as it."""
    K = len(result["labels"])  # number of options
    res = [0.0] * K  # mass decided by depth d, per option
    unresolved = 0.0  # mass still undecided at depth d
    for a in result["arrivals"]:  # every place the tree reached
        if a["kind"] == "off":  # left the format (free mode): dropped
            continue
        first = a["decided"] is not None and a["parent_decided"] is None  # the step that decided it
        if first and a["depth"] <= d:  # decided within d tokens
            res[a["decided"]] += a["mass"]
        elif a["decided"] is None and a["depth"] == d:  # still open after exactly d tokens
            unresolved += a["mass"]
    total = sum(res)  # resolved mass
    m_off = unresolved / (unresolved + total) if unresolved + total > 0 else 1.0
    if total <= 0.0 or total < UNDEFINED_SHARE * (total + unresolved):  # too little resolves
        return [float("nan")] * K, m_off
    return [r / total for r in res], m_off


def free_conditional(result):
    """Free mode: probability of each option given that the reply counts (on-format)."""
    tot = sum(result["p"])  # probability of any on-format reply
    if tot <= 0.0:  # never on-format
        return [float("nan")] * len(result["p"])
    return [x / tot for x in result["p"]]


def survival(p_free, p_grammar):
    """c(o) = P_free(o) / P_grammar(o): the share of the with-grammar probability a free reply keeps."""
    return [f / g if g > 0 else float("nan") for f, g in zip(p_free, p_grammar)]


def tv(p, q):
    """Total variation distance: half the summed absolute differences."""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))
