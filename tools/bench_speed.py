#!/usr/bin/env python3
"""End-to-end speed and memory of Embed2ASR (optionally Whisper large-v3) on your own hardware.

Same protocol as benchmarks/speed_v0.1.0.json: timed scope = audio file -> decode -> features -> model -> text, model
loading excluded; 1 untimed warm-up; median RTF of --passes passes. Peak memory is torch peak allocated on CUDA/ROCm,
not measured on MPS, and process max RSS on CPU. Run one process per (system, batch size) so the peak is clean.

usage: python tools/bench_speed.py ours|whisper --list clips.txt [--lang en|zh] [--bs 1] [--device cuda|mps|cpu] --out r.json
clips.txt: one audio path per line, one language per run (see docs/benchmark_contributing.md for the suggested clip set).
Whisper needs `pip install transformers` (already a dependency) and downloads openai/whisper-large-v3."""

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import torch

from embed2asr import __version__
from embed2asr.audio import load_audio

ZH_PROMPT = "以下是繁體中文的句子。"


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def make_runner(system, dev, lang, model):
    if system == "ours":
        from embed2asr import Embed2ASR

        asr = Embed2ASR.from_pretrained(model, device=dev)

        def run(paths):
            if len(paths) == 1:
                return [asr.transcribe(paths[0]).text]
            return [r.text for r in asr.transcribe(paths, batch_size=len(paths), cross_file=True)]

        return run
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", dtype=dtype).to(dev).eval()
    kw = dict(language=lang, task="transcribe", num_beams=1)
    if lang == "zh":
        kw["prompt_ids"] = proc.get_prompt_ids(ZH_PROMPT, return_tensors="pt").to(dev)

    @torch.inference_mode()
    def run(paths):
        f = proc.feature_extractor([load_audio(p) for p in paths], sampling_rate=16000, return_tensors="pt").input_features
        return proc.batch_decode(wm.generate(f.to(dev, dtype), **kw), skip_special_tokens=True)

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["ours", "whisper"])
    ap.add_argument("--list", required=True, help="text file, one audio path per line")
    ap.add_argument("--lang", default="en", choices=["en", "zh"], help="Whisper language (Embed2ASR ignores it)")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default="tc3oliver/Embed2ASR-v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = [p.strip() for p in Path(a.list).read_text(encoding="utf-8").splitlines() if p.strip()]
    secs = [len(load_audio(p)) / 16000 for p in paths]  # untimed; also warms the page cache
    batches = [(paths[i : i + a.bs], sum(secs[i : i + a.bs])) for i in range(0, len(paths), a.bs)]

    run = make_runner(a.system, dev, a.lang, a.model)
    run(batches[0][0])  # warm-up, untimed
    sync(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rtfs, lat = [], []
    for _ in range(a.passes):
        t0 = time.perf_counter()
        for b, _sec in batches:
            tb = time.perf_counter()
            run(b)
            sync(dev)
            lat.append(time.perf_counter() - tb)
        rtfs.append((time.perf_counter() - t0) / sum(secs))
    lat.sort()
    if dev.type == "cuda":
        peak = {"peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30, "device_name": torch.cuda.get_device_name(0)}
    elif dev.type == "cpu":
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux, bytes on macOS
        peak = {
            "max_rss_gib": rss / (2**30 if sys.platform == "darwin" else 2**20),
            "device_name": platform.processor() or platform.machine(),
        }
    else:
        peak = {"device_name": platform.machine()}
    res = {
        "system": a.system,
        "bs": a.bs,
        "device": dev.type,
        **peak,
        "n_clips": len(paths),
        "audio_s": sum(secs),
        "rtf": statistics.median(rtfs),
        "rtf_passes": rtfs,
        "latency_p50_s": lat[len(lat) // 2],
        "latency_p95_s": lat[int(len(lat) * 0.95)],
        "torch": torch.__version__,
        "embed2asr": __version__,
        "python": platform.python_version(),
        "os": platform.platform(),
    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
