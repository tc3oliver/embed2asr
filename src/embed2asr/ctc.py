"""CTC helpers."""


def collapse_ctc(ids, blank=0):
    """Greedy CTC collapse: merge repeats, drop blanks."""
    out, prev = [], blank
    for t in ids:
        if t != blank and t != prev:
            out.append(t)
        prev = t
    return out


def ctc_min_len(target):
    """Minimum number of frames CTC needs for a label sequence (repeats need a blank in between)."""
    return len(target) + sum(a == b for a, b in zip(target, target[1:]))
