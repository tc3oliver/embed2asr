#!/usr/bin/env python3
"""Audio manifest -> frozen EmbeddingGemma 2 feature cache for training (port of the research scripts/cache_features.py).

input  : JSONL manifest, one utterance per line: {"audio": path, "text": ..., "lang": ..., "speaker_id": ...} (+ optional "seconds")
output : OUT/features.bin  raw bf16 frames [frames, 1536], appended per utterance
         OUT/index.jsonl   {"i", "path", "off", "frames", "seconds", "text", "lang", "speaker_id", "audio", "crc32"} per kept row
         OUT/cache.json    encoder id + resolved revision, dtype, feature config, manifest sha256
Keeps 0.4 s <= duration <= --max-seconds. Labels are normalized with embed2asr.normalize.norm (idempotent).

Resume (rerun the same command) validates the cache before appending; it refuses (exit, nothing written) when
  * cache.json differs: other manifest content, encoder, revision, dtype or feature config -> use a new --out;
  * an index line other than the last is unreadable or inconsistent (row id, audio path, offsets not contiguous);
  * features.bin is shorter than the index claims, or a checked row's crc32 does not match (last 64 rows; all with --verify-all).
It repairs only what an interrupted write can leave: a torn last index line (dropped) and bytes in features.bin past the last
indexed row (cut). Features are flushed before their index line, so both repairs lose at most the in-flight batch.

usage: python -m training.prepare_features --manifest train.jsonl --out cache/train
TODO(PR #1): switch decode/encode to embed2asr.audio / embed2asr.encoder once they are on main."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from embed2asr.normalize import norm

MODEL_ID = "google/embeddinggemma-2"
DIM, MIN_SECONDS, ROW = 1536, 0.4, 1536 * 2  # ROW = bytes per frame (bf16)
FIELDS = {"i", "path", "off", "frames", "seconds", "text", "audio", "crc32"}


class CacheError(ValueError):
    pass


def ffmpeg_exe():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg") or "ffmpeg"


def decode_16k(path):
    """Any ffmpeg-readable file -> mono float32 16 kHz."""
    p = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
        capture_output=True,
        check=True,
    )
    a = np.frombuffer(p.stdout, dtype="<f4").copy()
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"invalid audio: {path}")
    return a


def resolved_revision(model_id, revision):
    """Hub commit hash the files were loaded from (local directories: None)."""
    if Path(model_id).exists():
        return None
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, revision=revision, local_files_only=True)).name


def open_cache(out, meta, manifest, verify_all=False):
    """Validate (and minimally repair) an existing cache in `out`; returns the manifest row ids already cached.
    Raises CacheError when the cache does not belong to (meta, manifest) or is damaged beyond an interrupted write."""
    out.mkdir(parents=True, exist_ok=True)
    meta_p, idx_p, bin_p = out / "cache.json", out / "index.jsonl", out / "features.bin"
    if meta_p.exists():
        old = json.loads(meta_p.read_text())
        diff = sorted(k for k in set(old) | set(meta) if old.get(k) != meta.get(k))
        if diff:
            raise CacheError(f"{out} was built with different {diff}: {[(k, old.get(k), meta.get(k)) for k in diff]}; use a new --out")
    elif idx_p.exists() or bin_p.exists():
        raise CacheError(f"{out} has index/features but no cache.json (unknown provenance); use a new --out")
    raw = idx_p.read_bytes() if idx_p.exists() else b""
    torn = bool(raw) and not raw.endswith(b"\n")
    if torn:  # interrupted while writing the last line
        raw = raw[: raw.rfind(b"\n") + 1]
    rows = []
    for n, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
        try:
            r = json.loads(line)
            assert FIELDS <= set(r), f"missing {sorted(FIELDS - set(r))}"
            assert 0 <= r["i"] < len(manifest) and r["audio"] == manifest[r["i"]]["audio"], "row id / audio does not match the manifest"
            assert r["frames"] > 0 and r["path"] == "features.bin"
        except Exception as e:
            raise CacheError(f"{idx_p} line {n} corrupt ({type(e).__name__}: {e}); not appending")
        rows.append(r)
    if len({r["i"] for r in rows}) != len(rows):
        raise CacheError(f"{idx_p}: duplicate row ids")
    end = 0
    for r in rows:  # rows are appended in write order: offsets must be contiguous
        if r["off"] != end:
            raise CacheError(f"{idx_p}: row i={r['i']} offset {r['off']} != expected {end}")
        end += r["frames"] * ROW
    size = bin_p.stat().st_size if bin_p.exists() else 0
    if size < end:
        raise CacheError(f"{bin_p} is {size} bytes but the index needs {end}: features truncated; rebuild the cache")
    if rows:  # size >= end > 0, so features.bin exists
        with bin_p.open("rb") as f:
            for r in rows if verify_all else rows[-64:]:
                f.seek(r["off"])
                if zlib.crc32(f.read(r["frames"] * ROW)) != r["crc32"]:
                    raise CacheError(f"{bin_p}: crc32 mismatch for row i={r['i']} (features corrupted); rebuild the cache")
    if torn:
        idx_p.write_bytes(raw)
        print(f"[resume] dropped a torn last line of {idx_p}", flush=True)
    if size > end:  # features of a batch whose index lines were never written
        with bin_p.open("r+b") as f:
            f.truncate(end)
        print(f"[resume] cut {size - end} unindexed bytes from {bin_p}", flush=True)
    meta_p.write_text(json.dumps(meta, indent=1))
    return {r["i"] for r in rows}


def append_row(bin_f, rec, data):
    """Write one row's features; returns its index record (written by the caller after flushing features.bin)."""
    rec = {**rec, "path": "features.bin", "off": bin_f.tell(), "crc32": zlib.crc32(data)}
    bin_f.write(data)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--revision", default=None, help="encoder revision (branch/tag/commit); the resolved commit is recorded")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seconds", type=float, default=29.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--verify-all", action="store_true", help="on resume, crc-check every cached row (reads all of features.bin)")
    p.add_argument("--min-free-gib", type=float, default=16.0, help="shared GPU: exit (resumable) when free memory drops below")
    a = p.parse_args()
    import torch
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(a.model, revision=a.revision)
    fe = getattr(proc, "feature_extractor", proc).to_dict()
    meta = {
        "encoder": a.model,
        "revision": resolved_revision(a.model, a.revision),
        "dtype": "bfloat16",
        "dim": DIM,
        "output": "get_audio_features().last_hidden_state",
        "feature_extractor_sha256": hashlib.sha256(json.dumps(fe, sort_keys=True, default=str).encode()).hexdigest(),
        "min_seconds": MIN_SECONDS,
        "max_seconds": a.max_seconds,
        "text_norm": "embed2asr.normalize.norm",
        "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
    }
    rows = [json.loads(line) for line in a.manifest.open(encoding="utf-8")]
    try:
        done = open_cache(a.out, meta, rows, a.verify_all)
    except CacheError as e:
        raise SystemExit(f"cache check failed: {e}")
    model = AutoModel.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16).to(a.device).eval()
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("seconds", 0))  # similar lengths per batch = less padding
    batches = [order[i : i + a.batch_size] for i in range(0, len(order), a.batch_size)]
    idx_f, bin_f = (a.out / "index.jsonl").open("a", encoding="utf-8"), (a.out / "features.bin").open("ab")
    pool, t0, skipped, cuda = ThreadPoolExecutor(8), time.time(), 0, a.device.startswith("cuda")
    for bi, b in enumerate(batches):
        b = [i for i in b if i not in done]
        if not b:
            continue
        if cuda and torch.cuda.mem_get_info()[0] / 2**30 < a.min_free_gib:
            raise SystemExit(f"PAUSE: free device memory < {a.min_free_gib} GiB; rerun later (resumable)")
        auds = list(pool.map(lambda i: decode_16k(rows[i]["audio"]), b))
        keep = [(i, x) for i, x in zip(b, auds) if MIN_SECONDS <= len(x) / 16000 <= a.max_seconds]
        skipped += len(b) - len(keep)
        if not keep:
            continue
        inp = proc(audio=[x for _, x in keep], return_tensors="pt")
        with torch.inference_mode():
            o = model.get_audio_features(
                input_features=inp["input_features"].to(a.device, torch.bfloat16),
                input_features_mask=inp["input_features_mask"].to(a.device),
            )
        lens, hid = o.attention_mask.sum(1).tolist(), o.last_hidden_state.to("cpu", torch.bfloat16)
        assert hid.shape[-1] == DIM, hid.shape
        recs = []
        for k, (i, x) in enumerate(keep):
            v = int(lens[k])
            f = hid[k, :v].contiguous()
            if v <= 0 or not torch.isfinite(f.float()).all():
                print(f"BAD features {rows[i]['audio']}", flush=True)
                skipped += 1
                continue
            recs.append(
                append_row(
                    bin_f,
                    {
                        "i": i,
                        "frames": v,
                        "seconds": len(x) / 16000,
                        "text": norm(rows[i]["text"]),
                        "lang": rows[i].get("lang"),
                        "speaker_id": rows[i].get("speaker_id", ""),
                        "audio": rows[i]["audio"],
                    },
                    f.view(torch.int16).numpy().tobytes(),
                )
            )
        bin_f.flush()  # features before index: an index line never points past the end of features.bin
        idx_f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        idx_f.flush()
        if bi % 10 == 0:
            print(f"batch {bi}/{len(batches)} {time.time() - t0:.0f}s skipped={skipped}", flush=True)
    print(f"done {time.time() - t0:.0f}s skipped={skipped} -> {a.out / 'index.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
