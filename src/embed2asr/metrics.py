"""CER / WER / MER, identical counts to the research score.py (corpus level: sum of errors / sum of reference units).
CER: characters of norm(text) without spaces. WER: whitespace words of norm(text) (word Levenshtein = jiwer S+D+I).
MER: each CJK character or each run of non-space non-CJK characters (latin word) is one unit. No torch import."""

from __future__ import annotations

import re

from .normalize import CJK, norm

_TOK = re.compile(rf"[{CJK}]|[^\s{CJK}]+")
METRIC_BY_LANG = {"zh-TW": "CER", "en-US": "WER", "mix": "MER", "zh-en": "MER"}


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def mer_tokens(s: str) -> list[str]:
    return _TOK.findall(norm(s))


def cer_counts(ref: str, hyp: str):
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return edit_distance(r, h), len(r)


def wer_counts(ref: str, hyp: str):
    r, h = norm(ref).split(), norm(hyp).split()
    return edit_distance(r, h), len(r)


def mer_counts(ref: str, hyp: str):
    r = mer_tokens(ref)
    return edit_distance(r, mer_tokens(hyp)), len(r)


COUNTS = {"CER": cer_counts, "WER": wer_counts, "MER": mer_counts}


def corpus(pairs, metric: str) -> dict:
    """pairs: iterable of (ref, hyp). -> {metric, value, errors, ref_units, utts}."""
    e = n = u = 0
    for ref, hyp in pairs:
        a, b = COUNTS[metric](ref, hyp)
        e, n, u = e + a, n + b, u + 1
    return {"metric": metric, "value": e / max(n, 1), "errors": e, "ref_units": n, "utts": u}


def score(rows) -> dict:
    """rows: dicts with ref, hyp, lang. Per-language corpus score using METRIC_BY_LANG (as research score.py)."""
    rows = list(rows)
    return {
        lang: corpus(((r["ref"], r["hyp"]) for r in rows if r["lang"] == lang), m)
        for lang, m in METRIC_BY_LANG.items()
        if any(r["lang"] == lang for r in rows)
    }
