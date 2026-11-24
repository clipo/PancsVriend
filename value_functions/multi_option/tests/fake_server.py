"""A tiny stand-in for llama-server, for the tests. No network, no model.

The "model" has a small vocabulary. Its next-token probabilities depend on
the exact TOKEN path written so far (not only on the text), so two token paths
that spell the same text can continue differently, as in a real model. That
is what lets the tests show that the token backend is exact and the text
backend is not.

The fake answers the same endpoints the code uses:
  /apply-template, /tokenize, /completion, /v1/chat/completions
through an object that looks like requests.Session (post(...) returning an
object with raise_for_status() and json()).
"""

import random  # census draws

EOS = 99  # the end-of-reply token id
BASE = 10000  # prompt token ids start here, so they never clash with the vocabulary


class FakeResponse:
    """What requests returns: status check and JSON body."""

    def __init__(self, body):
        self.body = body  # the decoded JSON

    def raise_for_status(self):
        return None  # the fake never fails

    def json(self):
        return self.body


class FakeModel:
    """Next-token probabilities as a function of the token path."""

    def __init__(self, vocab, eos_weight=3.0, seed=7):
        self.vocab = dict(vocab)  # id -> text
        self.eos_weight = eos_weight  # how much the end token weighs at every step
        self.seed = seed  # changes the made-up probabilities

    def weight(self, path, tid):
        """A made-up positive weight for token tid after path (deterministic)."""
        h = (self.seed * 1000003 + sum((i + 1) * 7919 * t for i, t in enumerate(path)) + 104729 * tid) % 9973
        return 0.2 + (h % 17)  # between 0.2 and 16.2

    def probs(self, path):
        """{token id: probability} after the token path (all tokens, end included)."""
        w = {tid: self.weight(path, tid) for tid in self.vocab}  # ordinary tokens
        w[EOS] = self.eos_weight  # the end token
        tot = sum(w.values())  # normaliser
        return {t: x / tot for t, x in w.items()}

    def text(self, path):
        """The text a token path spells."""
        return "".join(self.vocab[t] for t in path if t != EOS)

    def canonical(self, text):
        """The usual tokenisation of text: greedy, longest piece first (what a server re-tokenising a prefill does)."""
        out, i = [], 0  # token ids, position in text
        pieces = sorted(self.vocab.items(), key=lambda kv: -len(kv[1]))  # longest first
        while i < len(text):
            for tid, s in pieces:
                if s and text.startswith(s, i):  # this piece fits here
                    out.append(tid)
                    i += len(s)
                    break
            else:  # no piece fits: cannot tokenise
                raise ValueError(f"cannot tokenise {text!r}")
        return out


class FakeServer:
    """Answers the endpoints the code calls."""

    def __init__(self, model, mask_language=None):
        self.model = model  # the made-up model
        self.mask_language = mask_language  # if given, census draws with a grammar use it to mask
        self.calls = []  # every (endpoint, body) received, for the tests to inspect

    # The session interface.
    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))  # remember the request
        path = url.split("//", 1)[-1].split("/", 1)[-1]  # endpoint without the host
        if path == "apply-template":
            return FakeResponse({"prompt": "<user>" + json["messages"][0]["content"] + "<assistant>"})
        if path == "tokenize":
            ids = [BASE + (ord(c) % 5000) for c in json["content"]]  # one fake id per character
            return FakeResponse({"tokens": ids})
        if path == "completion":
            prefix = [t for t in json["prompt"] if t < BASE]  # the tokens written after the prompt
            return FakeResponse({"completion_probabilities": [self._table(prefix, json["n_probs"])]})
        if path == "v1/chat/completions":
            msgs = json["messages"]
            prefill = msgs[1]["content"] if len(msgs) > 1 else ""  # assistant prefill text
            if json.get("max_tokens") == 1 and json.get("n_probs"):  # a table request
                prefix = self.model.canonical(prefill)  # the server re-tokenises the text
                tab = self._table(prefix, json["n_probs"])
                return FakeResponse({"choices": [{"logprobs": {"content": [tab]}}]})
            return FakeResponse({"choices": [self._draw(json)]})  # a census draw
        raise ValueError(url)

    def get(self, url, timeout=None):
        return FakeResponse({"eos_token": "<eos>"})

    def _table(self, prefix, n_probs):
        """The top n_probs next-token entries, as llama.cpp reports them with post_sampling_probs."""
        pr = self.model.probs(prefix)  # full distribution
        top = sorted(pr.items(), key=lambda kv: -kv[1])[:n_probs]  # most probable first
        items = [{"id": t, "token": ("" if t == EOS else self.model.vocab[t]), "prob": p} for t, p in top]
        return {"id": top[0][0], "token": items[0]["token"], "prob": top[0][1], "top_probs": items}

    def _draw(self, body):
        """One sampled reply, token by token, with the grammar's mask if one was sent."""
        rng = random.Random(body.get("seed", 0))  # the seed decides the draw
        lang = self.mask_language if body.get("grammar") is not None else None  # mask only with a grammar
        path = []  # tokens drawn
        for _ in range(body.get("max_tokens", 16)):
            pr = self.model.probs(path)  # next-token distribution
            text = self.model.text(path)
            if lang is not None:  # keep only tokens the grammar allows
                pr = {t: p for t, p in pr.items()
                      if (t == EOS and lang.complete(text) is not None)
                      or (t != EOS and lang.live(text + self.model.vocab[t]))}
            toks, ps = zip(*pr.items())
            t = rng.choices(toks, weights=ps)[0]  # draw one token
            if t == EOS:  # the reply ends
                return {"message": {"content": text}, "finish_reason": "stop"}
            path.append(t)
            if lang is not None and lang.complete(self.model.text(path)) is not None \
                    and not lang.can_grow(self.model.text(path)):  # the grammar forces the end
                return {"message": {"content": self.model.text(path)}, "finish_reason": "stop"}
        return {"message": {"content": self.model.text(path)}, "finish_reason": "length"}


def brute_force(model, lang, mode, max_depth):
    """Exact option probabilities by visiting every token path (independent of path_sum's code).

    Returns (p per option, off-format, cut at the depth limit)."""
    K = len(lang.labels)
    p, off, cut = [0.0] * K, [0.0], [0.0]  # lists so the inner function can add to them

    def visit(path, mass):
        text = model.text(path)
        if len(path) >= max_depth:  # past the depth limit
            cut[0] += mass
            return
        pr = model.probs(path)  # every token, no top-n cut
        comp = lang.complete(text)  # option if the reply ended here
        if mode == "grammar":
            ok = {t: q for t, q in pr.items()
                  if (t == EOS and comp is not None) or (t != EOS and lang.live(text + model.vocab[t]))}
            tot = sum(ok.values())
            for t, q in ok.items():
                m = mass * q / tot  # masked and rescaled
                if t == EOS:
                    p[comp] += m
                    continue
                new = text + model.vocab[t]
                if lang.complete(new) is not None and not lang.can_grow(new):  # forced end
                    p[lang.complete(new)] += m
                else:
                    visit(path + [t], m)
        else:
            for t, q in pr.items():
                m = mass * q
                if t == EOS:
                    if comp is not None:
                        p[comp] += m
                    else:
                        off[0] += m
                    continue
                new = text + model.vocab[t]
                if lang.live(new):
                    visit(path + [t], m)
                else:
                    off[0] += m

    visit([], 1.0)
    return p, off[0], cut[0]
