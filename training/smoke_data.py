#!/usr/bin/env python3
"""Synthetic feature caches (random bf16 frames, random zh labels) for configs/smoke.yaml and the tests.
Same on-disk format as prepare_features.py; exercises the training code without audio, encoder or GPU.

usage: python -m training.smoke_data [--root runs/smoke]   -> ROOT/{train,pool,dev}/index.jsonl"""

import argparse
import json
import random
import zlib
from pathlib import Path

import numpy as np

DIM = 1536


def make_cache(d, n, seed):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    r, g = random.Random(seed), np.random.default_rng(seed)
    with (d / "features.bin").open("wb") as fb, (d / "index.jsonl").open("w", encoding="utf-8") as fi:
        for i in range(n):
            frames = r.randint(20, 60)
            data = (g.standard_normal((frames, DIM), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()  # bf16
            text = "".join(r.choice("天地玄黃宇宙") for _ in range(r.randint(2, 6)))
            fi.write(
                json.dumps(
                    {
                        "i": i,
                        "path": "features.bin",
                        "off": fb.tell(),
                        "frames": frames,
                        "seconds": frames / 25,
                        "text": text,
                        "lang": "mix" if i % 2 == 0 else "zh-TW",
                        "speaker_id": "",
                        "audio": f"synthetic/{seed}/{i}",
                        "crc32": zlib.crc32(data),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fb.write(data)
    return str(d / "index.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("runs/smoke"))
    a = p.parse_args()
    for name, n, seed in (("train", 40, 1), ("pool", 30, 2), ("dev", 8, 3)):
        print(make_cache(a.root / name, n, seed))


if __name__ == "__main__":
    main()
