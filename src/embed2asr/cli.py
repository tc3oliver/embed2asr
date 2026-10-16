"""embed2asr transcribe FILE... --model DIR [--device cuda|cpu] [--json] | embed2asr info --model DIR_OR_REPO"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict


def _parser():
    p = argparse.ArgumentParser(prog="embed2asr", description="Speech recognition with frozen EmbeddingGemma 2 + GRU-CTC reader")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcribe", help="transcribe audio files")
    t.add_argument("files", nargs="+")
    t.add_argument("--model", required=True, help="release directory or Hugging Face repo id")
    t.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lm", default=None, help="character n-gram LM file (path or relative to --model); default greedy CTC")
    t.add_argument("--lm-alpha", type=float, default=0.25)
    t.add_argument("--lm-beta", type=float, default=1.5)
    t.add_argument("--beam", type=int, default=16)
    t.add_argument("--json", action="store_true", help="one JSON object per file (text, duration, chunks, rtf)")
    i = sub.add_parser("info", help="show release config and manifest")
    i.add_argument("--model", required=True, help="release directory or Hugging Face repo id")
    return p


def main(argv=None, encoder=None) -> int:
    """`encoder` lets tests inject a stub audio encoder (no model download)."""
    a = _parser().parse_args(argv)
    if a.cmd == "info":
        from .pipeline import load_release, resolve_release

        d = resolve_release(a.model)
        _, _, itos, _, cfg = load_release(d)
        info = {"config": cfg, "units": len(itos)}
        if (d / "manifest.json").is_file():
            info["manifest"] = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        print(json.dumps(info, indent=1, ensure_ascii=False))
        return 0
    from .pipeline import Embed2ASR

    asr = Embed2ASR.from_pretrained(a.model, device=a.device, encoder=encoder, lm=a.lm, lm_alpha=a.lm_alpha, lm_beta=a.lm_beta, beam=a.beam)
    rc = 0
    for f in a.files:
        t0 = time.perf_counter()
        try:
            r = asr.transcribe(f, batch_size=a.batch_size)
        except (OSError, ValueError) as e:
            print(f"embed2asr: {f}: {e}", file=sys.stderr)
            rc = 1
            continue
        dt = time.perf_counter() - t0
        if a.json:
            print(json.dumps({"audio": f, **asdict(r), "rtf": dt / max(r.duration, 1e-9)}, ensure_ascii=False))
        else:
            print(r.text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
