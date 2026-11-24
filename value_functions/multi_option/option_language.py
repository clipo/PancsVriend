"""The set of replies that count as each option, for K-option choices.

A "reply language" says which texts count as which option. It is built from
a short list of SEGMENTS that must appear one after another:

    ("ws",)                       zero or more spaces, tabs or newlines
    ("lit", [s1, s2, ...], ci)    exactly one of the strings s1, s2, ...
                                  (an empty string "" makes the segment optional);
                                  ci=True means upper/lower case do not matter
    ("label",)                    exactly one option label (case does not matter,
                                  unless the language is built with label_ci=False)

Two languages are used:

* WITH a grammar (what the sampler is allowed to write), mirroring the
  simulation's MOVE_STAY_GRAMMAR:       ws label
* WITHOUT a grammar (how a free reply is parsed afterwards), strict and
  finite, so an exact path sum is possible:
                                        ws label ["."] ws

The same object answers the four questions the path sum needs about any text
written so far:

    live(text)       could the text still grow into a reply that counts?
    possible(text)   which options could it still become?
    complete(text)   which option does it spell if the reply ends here (or None)?
    can_grow(text)   could more characters follow and the reply still count?

and it writes the llama.cpp GBNF grammar (whole, or for "what may still
follow this text") so the server's own sampler agrees with the client.

Only the standard library is used.
"""

WS_CHARS = " \t\n"  # the whitespace characters the simulation's grammar allows


class OptionLanguage:
    """A finite-option reply language described by a list of segments."""

    def __init__(self, labels, segments, label_ci=True):
        self.labels = list(labels)  # the option labels, in the order the prompt lists them
        self.segments = list(segments)  # the pattern every reply must follow
        self.label_ci = bool(label_ci)  # True: MOVE, move and Move all count (as in MOVE_STAY_GRAMMAR)
        if len(set(l.lower() for l in self.labels)) != len(self.labels):  # two labels equal up to case
            raise ValueError("option labels must differ ignoring case")  # the language would be ambiguous

    # ------------------------------------------------------------------
    # Character-level matching. A "position" is where we are in the pattern:
    #   (segment index, string index within that segment, characters matched,
    #    which label this reply has chosen so far or None)
    # We keep the SET of all positions the text could have reached, which is
    # how a regular expression engine works, written out by hand.
    # ------------------------------------------------------------------

    def _choices(self, seg):
        """The strings a segment can match, with the case rule and the label index."""
        if seg[0] == "label":  # one of the option labels
            return [(lab, self.label_ci, i) for i, lab in enumerate(self.labels)]  # labels' case rule
        if seg[0] == "lit":  # one of a fixed list of strings
            return [(s, seg[2], None) for s in seg[1]]  # keep the segment's case rule
        raise ValueError(f"unknown segment {seg}")  # "ws" is handled separately

    def _close(self, positions):
        """Add every position reachable without reading a character (skip empty parts)."""
        todo = list(positions)  # positions still to look at
        out = set(positions)  # everything reachable so far
        while todo:  # until nothing new appears
            k, j, n, lab = todo.pop()  # take one position
            if k >= len(self.segments):  # already past the last segment
                continue  # nothing to skip from here
            seg = self.segments[k]  # the segment we are in
            if seg[0] == "ws":  # whitespace may also be empty
                nxt = (k + 1, None, 0, lab)  # so we may move on to the next segment
                if nxt not in out:  # not seen yet
                    out.add(nxt)
                    todo.append(nxt)
                continue
            if j is None:  # at the start of a segment: we may begin any of its strings
                for idx, (s, _, li) in enumerate(self._choices(seg)):  # each string
                    new_lab = lab if li is None else li  # entering a label fixes the option
                    nxt = (k, idx, 0, new_lab)  # inside string idx, 0 characters read
                    if nxt not in out:
                        out.add(nxt)
                        todo.append(nxt)
                continue
            s = self._choices(seg)[j][0]  # the string we are inside
            if n == len(s):  # the whole string has been read
                nxt = (k + 1, None, 0, lab)  # move to the next segment
                if nxt not in out:
                    out.add(nxt)
                    todo.append(nxt)
        return out

    def _step(self, positions, ch):
        """Positions after reading one more character ch."""
        out = set()  # where we can be after ch
        for k, j, n, lab in positions:  # try every position we might be at
            if k >= len(self.segments):  # pattern finished: no more characters allowed
                continue
            seg = self.segments[k]  # current segment
            if seg[0] == "ws":  # inside a whitespace run
                if ch in WS_CHARS:  # another whitespace character
                    out.add((k, None, 0, lab))  # stay in the run
                continue
            if j is None:  # not inside a string yet (handled by _close)
                continue
            s, ci, _ = self._choices(seg)[j]  # the string we are inside
            if n < len(s):  # characters of it are still to come
                want = s[n]  # the next character it needs
                same = (ch.lower() == want.lower()) if ci else (ch == want)  # compare, ignoring case if allowed
                if same:  # the character fits
                    out.add((k, j, n + 1, lab))  # one more character of the string read
        return self._close(out)  # also skip any empty parts that follow

    def _positions(self, text):
        """All positions the pattern can be in after reading text."""
        pos = self._close({(0, None, 0, None)})  # the start of the pattern
        for ch in text:  # read the text one character at a time
            pos = self._step(pos, ch)
            if not pos:  # no way to match: stop early
                break
        return pos

    def _finished(self, pos):
        """True if this position has used up the whole pattern."""
        return pos[0] >= len(self.segments)  # past the last segment

    # ------------------------------------------------------------------
    # The four questions the path sum asks.
    # ------------------------------------------------------------------

    def live(self, text):
        """Could text still grow into (or already be) a reply that counts?"""
        return bool(self._positions(text))  # any surviving position means yes

    def possible(self, text):
        """The set of option indices text could still become."""
        found = set()  # options reachable from here
        for k, j, n, lab in self._positions(text):  # each surviving position
            if lab is not None:  # the label is already decided on this branch
                found.add(lab)
            else:  # the label has not started yet: any option is still possible
                found.update(range(len(self.labels)))
        return found

    def complete(self, text):
        """The option index that text spells if the reply ends right here, else None."""
        done = {lab for (k, j, n, lab) in self._positions(text) if self._finished((k, j, n, lab))}  # finished branches
        if len(done) > 1:  # two options would be spelled by the same reply: a design error
            raise ValueError(f"reply {text!r} counts as more than one option")
        return next(iter(done)) if done else None  # the one option, or None

    def can_grow(self, text):
        """Could at least one more character follow text inside the language?"""
        for pos in self._positions(text):  # each surviving position
            k, j, n, lab = pos
            if k >= len(self.segments):  # finished: nothing may follow on this branch
                continue
            seg = self.segments[k]
            if seg[0] == "ws":  # whitespace can always take one more space
                return True
            if j is not None and n < len(self._choices(seg)[j][0]):  # a string is half read
                return True
        return False  # every branch is finished

    def decided(self, text):
        """The option index if text can only become one option, else None."""
        opts = self.possible(text)  # options still reachable
        return next(iter(opts)) if len(opts) == 1 else None

    # ------------------------------------------------------------------
    # GBNF grammars for llama.cpp.
    # ------------------------------------------------------------------

    @staticmethod
    def _gbnf_string(s, ci):
        """One string as GBNF: letters as [Aa] classes when case does not matter."""
        parts = []  # pieces of the rule
        for ch in s:  # one character at a time
            if ci and ch.isalpha():  # a letter whose case is free
                parts.append(f"[{ch.upper()}{ch.lower()}]")  # e.g. [Mm], as in MOVE_STAY_GRAMMAR
            else:  # anything else must appear exactly
                esc = ch.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")  # escape it
                parts.append(f'"{esc}"')  # a quoted literal
        return " ".join(parts) if parts else '""'  # an empty string becomes ""

    def _gbnf_segment(self, seg):
        """One segment as a GBNF expression."""
        if seg[0] == "ws":  # a whitespace run
            return "[ \\t\\n]*"  # the same rule as MOVE_STAY_GRAMMAR's ws
        alts = [self._gbnf_string(s, ci) for (s, ci, _) in self._choices(seg) if s]  # each non-empty choice
        optional = any(not s for (s, _, _) in self._choices(seg))  # an empty choice makes the segment optional
        return _group(alts, optional)  # GBNF has no empty literal, so use ( ... )? instead

    def gbnf(self):
        """The whole grammar, to send in the request's grammar field."""
        body = " ".join(self._gbnf_segment(s) for s in self.segments)  # segments in order
        return f"root ::= {body}\n"  # one rule is enough

    def remainder_gbnf(self, text):
        """Grammar for what may still be written after text (for an assistant prefill).

        The server's grammar starts afresh after a prefill, so it must be told
        only what may follow. Built as a choice over every position text can be
        in: the rest of the current string, then the remaining segments."""
        alts = []  # one alternative per surviving position
        for k, j, n, lab in sorted(self._positions(text), key=str):  # each position, in a fixed order
            if k >= len(self.segments):  # finished: the reply may end here
                alts.append('""')  # nothing more
                continue
            seg = self.segments[k]
            if seg[0] == "ws":  # inside whitespace: the rest is this run and what follows
                rest = [self._gbnf_segment(seg)] + [self._gbnf_segment(s) for s in self.segments[k + 1:]]
            elif j is None:  # at a segment start (already expanded by _close, so skip)
                continue
            else:  # inside a string: its remaining characters, then later segments
                s, ci, _ = self._choices(seg)[j]
                rest = ([self._gbnf_string(s[n:], ci)] if n < len(s) else []) + \
                       [self._gbnf_segment(x) for x in self.segments[k + 1:]]
            rest = [r for r in rest if r]  # drop empty pieces
            alts.append(" ".join(rest) if rest else '""')  # this branch's continuation
        alts = sorted(set(alts))  # remove duplicates, fixed order
        if not alts:  # text is not in the language at all
            raise ValueError(f"no reply can start with {text!r}")
        optional = '""' in alts  # the reply may also end right here
        alts = [a for a in alts if a != '""']  # the non-empty continuations
        if not alts:  # nothing may follow at all
            return None  # no request is needed: the grammar forces the reply to end
        return "root ::= " + _group(alts, optional) + "\n"


def _group(alts, optional):
    """GBNF for a choice between alts, optionally allowed to be empty."""
    if not alts:  # only the empty string
        return ""  # nothing to write
    body = alts[0] if len(alts) == 1 else "( " + " | ".join(alts) + " )"  # one or a group
    if optional:  # may also be skipped
        body = (body if body.startswith("(") else "( " + body + " )") + "?"  # GBNF's optional mark
    return body


# ----------------------------------------------------------------------
# The two standard languages.
# ----------------------------------------------------------------------

def grammar_language(labels):
    """What the sampler may write WITH a grammar: optional whitespace, then one label."""
    if isinstance(labels, OptionLanguage):  # already a language: use it as it is
        return labels
    return OptionLanguage(labels, [("ws",), ("label",)])  # same shape as MOVE_STAY_GRAMMAR


def free_language(labels):
    """How a FREE reply is parsed: optional whitespace, a label, an optional full stop, optional whitespace.

    Strict on purpose: the set of replies that count is finite, so the free
    path sum is exact. Anything else is "off-format"."""
    if isinstance(labels, OptionLanguage):  # already a language: use it as it is
        return labels
    return OptionLanguage(labels, [("ws",), ("label",), ("lit", ["", "."], False), ("ws",)])


def parse_free(labels, reply):
    """Option index a finished free reply counts as under the strict rule, else None.

    labels may also be an OptionLanguage (for formats such as JSON)."""
    return free_language(labels).complete(reply)  # the same rule the free path sum uses


def parse_loose(labels, reply):
    """Secondary count only: the one label that appears as a word in the reply, else None.

    Used to report how many off-format replies a looser parser would have
    accepted; never used for the exact numbers."""
    import re  # regular expressions for the word search
    if isinstance(labels, OptionLanguage):  # a language was passed: use its labels
        labels = labels.labels
    hits = set()  # labels found in the reply
    for i, lab in enumerate(labels):  # look for each label
        if re.search(r"(?<![A-Za-z])" + re.escape(lab) + r"(?![A-Za-z])", reply, flags=re.I):  # whole-word match
            hits.add(i)
    if len(hits) == 1:  # exactly one label named
        return hits.pop()
    longest = [i for i in hits if not any(labels[i].lower() in labels[j].lower() and i != j for j in hits)]  # drop labels inside longer ones
    return longest[0] if len(longest) == 1 else None  # still ambiguous: no answer
