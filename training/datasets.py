"""Feature-cache datasets for reader training: index loading, GPU-resident feature store, bucketed batches, rotate sampler.

Ported from the research trainer (scripts/train_reader.py + scripts/train_fast.py); the sampling is kept bit-identical so a
run with the same data, seed and pool specs draws exactly the same rows per epoch."""

import json
import random
from pathlib import Path

import numpy as np
import torch

D = 1536  # EmbeddingGemma 2 audio feature width


def load_index(paths):
    """Concatenate index.jsonl files. Feature paths are resolved next to their index when they exist there."""
    rows = []
    for p in paths:
        p = Path(p)
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            r["text"] = r["text"].replace("臺", "台")  # same variant merge as normalize.norm (older caches predate it)
            local = p.parent / r["path"]
            if not Path(r["path"]).is_absolute() and local.exists():
                r["path"] = str(local)
            rows.append(r)
    return rows


def nodigit(rows):
    """Drop rows whose label has digits: the audio says the number in words, the label does not."""
    return [r for r in rows if not any(c.isdigit() for c in r["text"])]


def parse_pool(spec):
    """'PATH:HOURS' -> (PATH, HOURS per epoch)."""
    path, hours = spec.rsplit(":", 1)
    return path, float(hours)


def bucket_batches(rows, bs, rng):
    """Shuffle, sort chunks of 50 batches by length (less padding), cut into batches, shuffle batch order."""
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    chunks = [sorted(idx[i : i + bs * 50], key=lambda j: rows[j]["frames"]) for i in range(0, len(idx), bs * 50)]
    batches = [c[i : i + bs] for c in chunks for i in range(0, len(c), bs)]
    rng.shuffle(batches)
    return batches


def split_chunks(rows, k, rng):
    """Same epoch rows split at random into k parts (every part mixes all pools); k=1 consumes no randomness."""
    if k <= 1:
        return [rows]
    perm = list(range(len(rows)))
    rng.shuffle(perm)
    return [[rows[i] for i in perm[c::k]] for c in range(k)]


class RotateSampler:
    """Per epoch, take up to HOURS of the NEXT unused rows from each pool's shuffled order; reshuffle only after a full pass.

    The order of pass n of pool k is random.Random(seed * 1009 + k * 101 + n), so epoch e's rows are a pure function of
    (seed, pools, e): resume needs no sampler state. No row repeats within an epoch, also across a pass boundary."""

    def __init__(self, pools, seed):
        self.pools, self.seed = pools, seed  # pools: [(rows, hours_per_epoch)]
        self._orders, self._state = {}, {0: [(0, 0)] * len(pools)}  # state[e] = per-pool (pass, pos) after epoch e

    def _order(self, k, n):
        if (k, n) not in self._orders:
            o = list(range(len(self.pools[k][0])))
            random.Random(self.seed * 1009 + k * 101 + n).shuffle(o)
            self._orders[k, n] = o
        return self._orders[k, n]

    def _take(self, k, n, pos):
        rows, want = self.pools[k]
        take, seen, h = [], set(), 0.0
        while h < want and len(take) < len(rows):
            o = self._order(k, n)
            if pos >= len(o):
                n, pos = n + 1, 0
                continue
            i = o[pos]
            pos += 1
            if i in seen:
                continue
            take.append(i)
            seen.add(i)
            h += rows[i]["seconds"] / 3600
        return take, n, pos

    def epoch(self, ep):
        """Rows of epoch ep (1-based) and the passes completed per pool after it (logged as pool_passes)."""
        e = max(x for x in self._state if x < ep)
        while e < ep:
            takes = [self._take(k, n, pos) for k, (n, pos) in enumerate(self._state[e])]
            e += 1
            self._state[e] = [(n, pos) for _, n, pos in takes]
        got = [self.pools[k][0][i] for k, (take, _, _) in enumerate(takes) for i in take]
        cov = {k: round(n + pos / len(self.pools[k][0]), 3) for k, (n, pos) in enumerate(self._state[ep])}
        return got, cov


class Store:
    """Features of `rows` in ONE flat bf16 tensor [N+1, D] on `device`; row N is zeros (gather target for padding)."""

    def __init__(self, rows, device="cuda", max_chunk=1 << 16):
        n = sum(r["frames"] for r in rows)
        self.device = device
        self.flat = torch.empty(n + 1, D, dtype=torch.bfloat16, device=device)
        self.flat[n].zero_()
        self.pad, self.lens = n, torch.tensor([r["frames"] for r in rows], dtype=torch.long)
        starts, pos, buf, nb, mms = [0] * len(rows), 0, [], 0, {}

        def flush():
            nonlocal pos, buf, nb
            if buf:
                a = np.concatenate(buf) if len(buf) > 1 else buf[0]
                self.flat[pos : pos + len(a)].copy_(torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16))
                pos += len(a)
            buf, nb = [], 0

        for i in sorted(range(len(rows)), key=lambda i: (rows[i]["path"], rows[i]["off"])):  # sequential reads
            r = rows[i]
            starts[i] = pos + nb
            if r["path"] not in mms:
                mms[r["path"]] = np.memmap(r["path"], dtype=np.int16, mode="r").reshape(-1, D)
            assert r["off"] % (2 * D) == 0
            buf.append(mms[r["path"]][r["off"] // (2 * D) : r["off"] // (2 * D) + r["frames"]])
            nb += r["frames"]
            if nb >= max_chunk:
                flush()
        flush()
        assert pos == n
        self.starts = torch.tensor(starts, dtype=torch.long)

    def gather(self, idx, T):
        """x [B, T, D] bf16 for row indices idx (CPU LongTensor), zero-padded to T."""
        s = self.starts[idx].to(self.device, non_blocking=True)
        L = self.lens[idx].to(self.device, non_blocking=True)
        ar = torch.arange(T, device=self.device)[None]
        return self.flat[torch.where(ar < L[:, None], s[:, None] + ar, self.pad)]
