"""Optional LM decoding: CTC prefix beam search + character 5-gram LM (interpolated Kneser-Ney), pure python + numpy.

BPE vocabulary rule: only single CJK units are scored by the LM (alpha * log P_lm + beta each). Every other unit (English
pieces, accented letters, ...) is LM-neutral: no LM term, no beta. A maximal run of neutral units enters the LM context as ONE
placeholder token PH, so the CJK char after an English word is still conditioned sensibly.
The end-of-sentence term is always scored.
LM file: gzip JSON of the n-gram count tables (format "eg2-charlm-kn/1"; no pickle, safe to load)."""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from .normalize import CJK
from .tokenizer import Tok

BOS, EOS, PH, NEG = "\x02", "\x03", "\x04", -1e30
_CJK1, _NONCJK = re.compile(f"[{CJK}]"), re.compile(f"[^{CJK}]+")


def lse(a, b):
    return a if b <= NEG else b if a <= NEG else (max(a, b) + math.log1p(math.exp(-abs(a - b))))


class CharLM:
    def __init__(self, sents, order, D=0.75, counts=None):
        self.N, self.D = order, D
        if counts is not None:
            self.cnt = [None] + counts
        else:
            raw = [None] + [collections.Counter() for _ in range(order)]
            for s in sents:
                s = BOS * (order - 1) + s + EOS
                for n in range(1, order + 1):
                    raw[n].update(s[i : i + n] for i in range(len(s) - n + 1))
            self.cnt = [None] * (order + 1)
            self.cnt[order] = raw[order]
            for n in range(order - 1, 0, -1):  # continuation counts
                c = collections.Counter()
                for y in raw[n + 1]:
                    c[y[1:]] += 1
                self.cnt[n] = c
        self.tot, self.typ = {}, {}
        for n in range(1, order + 1):
            for g, c in self.cnt[n].items():
                h = g[:-1]
                self.tot[h] = self.tot.get(h, 0) + c
                self.typ[h] = self.typ.get(h, 0) + 1
        self.V = len(self.cnt[1]) + 1
        self.cache = {}

    def save(self, path) -> str:
        """Deterministic gzip JSON of the count tables; returns sha256 of the file."""
        d = {
            "format": "eg2-charlm-kn/1",
            "order": self.N,
            "D": self.D,
            "BOS": BOS,
            "EOS": EOS,
            "PH": PH,
            "counts": [dict(sorted(c.items())) for c in self.cnt[1:]],
        }
        b = gzip.compress(json.dumps(d, separators=(",", ":")).encode(), compresslevel=9, mtime=0)
        Path(path).write_bytes(b)
        return hashlib.sha256(b).hexdigest()

    @classmethod
    def load(cls, path):
        d = json.loads(gzip.decompress(Path(path).read_bytes()))
        if d.get("format") != "eg2-charlm-kn/1" or (d["BOS"], d["EOS"], d["PH"]) != (BOS, EOS, PH):
            raise ValueError(f"{path} is not an eg2-charlm-kn/1 language model")
        return cls(None, d["order"], d["D"], counts=d["counts"])

    def p(self, h, w):  # h: context string (any length, truncated to N-1)
        h = h[-(self.N - 1) :] if self.N > 1 else ""
        k = (h, w)
        v = self.cache.get(k)
        if v is None:
            if len(self.cache) > 4_000_000:
                self.cache.clear()  # ponytail: crude bound for long-lived processes; LRU if it matters
            v = self.cache[k] = self._p(h, w)
        return v

    def _p(self, h, w):
        t = self.tot.get(h)
        lower = (1.0 / self.V) if not h else self._p(h[1:], w)
        if not t:
            return lower
        c = self.cnt[len(h) + 1].get(h + w, 0)
        return max(c - self.D, 0) / t + self.D * self.typ[h] / t * lower

    def logp(self, prefix, w):
        return math.log(self.p((BOS * (self.N - 1) + prefix)[-(self.N - 1) :] if self.N > 1 else "", w))


def is_bpe(itos) -> bool:
    return any("▁" in u for u in itos)


def lm_text(s: str) -> str:
    """Corpus side of the BPE rule: each non-CJK run -> one placeholder."""
    return _NONCJK.sub(PH, s)


def lm_units(itos):
    """Per unit id: string the LM scores, or None = LM-neutral. Char vocab: every unit. BPE: single CJK chars only."""
    if not is_bpe(itos):
        return list(itos)
    return [u.replace("臺", "台") if len(u) == 1 and _CJK1.match(u) else None for u in itos]


def beam_decode(lp, tok: Tok, lm: CharLM, alpha, beta, beam=16, topk=8, units=None) -> str:
    """lp [T,V] float log-probs (numpy). Score = log P_ctc + alpha*sum log P_lm(scored chars) + beta*#scored chars
    (+ alpha*log P(EOS) at the end). Prefixes are unit-id tuples, so multi-char units and CTC repeats are exact."""
    units = units if units is not None else lm_units(tok.itos)
    st = {(): ("", 0.0, 0)}  # prefix -> (LM context, sum log P_lm, #scored chars)

    def state(pf):
        v = st.get(pf)
        if v is None:
            ctx, s, n = state(pf[:-1])
            u = units[pf[-1]]
            if u is None:
                if not ctx.endswith(PH):
                    s += lm.logp(ctx, PH)  # zh->en switch is an LM token; English content is never scored
                    ctx += PH
                v = (ctx, s, n)
            else:
                for ch in u:
                    s += lm.logp(ctx, ch)
                    ctx += ch
                v = (ctx, s, n + len(u))
            st[pf] = v
        return v

    def score(pf, pb, pnb):
        _, s, n = state(pf)
        return lse(pb, pnb) + alpha * s + beta * n

    beams = {(): (0.0, NEG)}
    for t in range(lp.shape[0]):
        row = lp[t]
        cand = np.argpartition(-row, topk)[:topk] if row.shape[0] > topk else range(row.shape[0])
        cand = [int(c) for c in cand if row[c] > -14 or c == 0]
        new = collections.defaultdict(lambda: [NEG, NEG])
        for pf, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            for c in cand:
                p = float(row[c])
                if c == 0:
                    e = new[pf]
                    e[0] = lse(e[0], tot + p)
                    continue
                if pf and c == pf[-1]:
                    e = new[pf + (c,)]
                    e[1] = lse(e[1], pb + p)
                    e = new[pf]
                    e[1] = lse(e[1], pnb + p)
                else:
                    e = new[pf + (c,)]
                    e[1] = lse(e[1], tot + p)
        beams = dict(sorted(((k, tuple(v)) for k, v in new.items()), key=lambda kv: -score(kv[0], *kv[1]))[:beam])

    def final(kv):
        ctx = state(kv[0])[0]
        return score(kv[0], *kv[1]) + alpha * lm.logp(ctx, EOS)

    return tok.decode(max(beams.items(), key=final)[0]).replace("臺", "台")


def lm_decoder(lm: CharLM, alpha=0.25, beta=1.5, beam=16):
    """-> DECODERS-style fn(logits [B,T,V], lengths [B], tok) -> list[str]."""

    def decode(logits: torch.Tensor, lengths: torch.Tensor, tok: Tok) -> list[str]:
        lp = logits.float().log_softmax(-1).cpu().numpy()
        units = lm_units(tok.itos)
        return [beam_decode(lp[i, :n], tok, lm, alpha, beta, beam, units=units) for i, n in enumerate(lengths.tolist())]

    return decode
