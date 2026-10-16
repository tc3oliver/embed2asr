"""Fill the speed/memory numbers in README.md, README.zh-TW.md and docs/benchmarks.md from benchmarks/speed_v0.1.0.json.

usage: python scripts/fill_perf.py [--check]

Regions are delimited by <!-- PERF:<NAME>:BEGIN --> / <!-- PERF:<NAME>:END -->. Ratios and percentages are computed from the
unrounded JSON values (same `derive` as make_perf_svg.py). --check fails if any region or assets/performance.svg is stale."""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_perf_svg import JSON, ROOT, derive, rtf  # noqa: E402

T = {
    "en": {
        "hero": "**Up to {sp:.1f}× faster than Whisper large-v3 (batch 1, our setup); over {mem}% lower peak GPU memory.**",
        "head": ["", "Embed2ASR-v1", "Whisper large-v3", "Difference"],
        "rtf": "RTF, batch {n} (lower is better)",
        "mem": "Peak GPU memory, batch {n}",
        "cold": "Cold start (median of 3)",
        "faster": "{x:.2f}× faster",
        "wfaster": "Whisper faster",
        "cond": (
            "Setup: one MI300X, idle before the run; ROCm 7.1, torch 2.10; bf16 for both systems. {n} clips ({m}), {a:,.0f} s of audio. "
            "Timing covers audio decode, features, model and text, not model loading; median of 3 passes after 1 warm-up. "
            "Whisper uses Hugging Face `generate`, greedy (`num_beams=1`). Embed2ASR at batch 16 uses `cross_file=True` "
            "(chunks of different files are batched together; about 4% of short clips can change by one unit; off by default). "
            "Memory is torch peak allocated, including the full audio encoder and the reader, not whole-process memory. "
            'Cold start is process start to first output. "Up to" refers to these two batch sizes on this hardware only. '
            "Transcripts can differ by +-1 unit across transformers versions and GPU models (identical within one environment); "
            "this benchmark ran on torch 2.10.0+rocm7.1.1, transformers 5.19.0, MI300X."
        ),
        "clips": "100 Mandarin + 100 English",
        "d_head": ["", "Embed2ASR-v1", "Whisper large-v3"],
        "d_lat": "Latency p50 / p95, batch {n} ({u})",
        "d_res": "Peak reserved memory, batch {n}",
        "d_pass": "RTF of the 3 passes, batch {n}",
        "u1": "per clip",
        "u16": "per batch of 16",
    },
    "zh": {
        "hero": "**最高比 Whisper large-v3 快 {sp:.1f} 倍（batch 1，本測試環境）；GPU 峰值記憶體低 {mem}% 以上。**",
        "head": ["", "Embed2ASR-v1", "Whisper large-v3", "差異"],
        "rtf": "RTF，batch {n}（越低越好）",
        "mem": "GPU 峰值記憶體，batch {n}",
        "cold": "冷啟動（3 次中位數）",
        "faster": "快 {x:.2f} 倍",
        "wfaster": "Whisper 較快",
        "cond": (
            "測試條件：單張 MI300X，測前閒置；ROCm 7.1、torch 2.10；兩者都用 bf16。{n} 段音訊（{m}），共 {a:,.0f} 秒。"
            "計時包含音檔解碼、特徵、模型與文字輸出，不含模型載入；暖機 1 次後取 3 輪中位數。"
            "Whisper 使用 Hugging Face `generate`，greedy（`num_beams=1`）。Embed2ASR 在 batch 16 使用 `cross_file=True`"
            "（不同檔案的片段混批；約 4% 的短句可能差 1 個單位；預設關閉）。"
            "記憶體是 torch 的 peak allocated，含完整 audio encoder 與 reader，不是整個 process 的用量。"
            "冷啟動是從程序啟動到第一個輸出。「最高」只代表這兩個 batch size 與這組硬體。"
            "轉寫結果在不同 transformers 版本與 GPU 型號之間可能有 ±1 個單位的差異（同一環境內結果一致）；"
            "本次 benchmark 環境為 torch 2.10.0+rocm7.1.1、transformers 5.19.0、MI300X。"
        ),
        "clips": "中文 100 + 英文 100",
        "d_head": ["", "Embed2ASR-v1", "Whisper large-v3"],
        "d_lat": "延遲 p50 / p95，batch {n}（{u}）",
        "d_res": "峰值 reserved 記憶體，batch {n}",
        "d_pass": "3 輪 RTF，batch {n}",
        "u1": "每段",
        "u16": "每個 16 段 batch",
    },
}


def row(c):
    return "| " + " | ".join(c) + " |"


def regions(d: dict, lang: str) -> dict:
    t, v, c = T[lang], derive(d), d["conditions"]
    e, w = v["e"], v["w"]
    tab = [row(t["head"]), row(["---"] * 4)]
    for bs, n in (("bs1", 1), ("bs16", 16)):
        tab.append(row([t["rtf"].format(n=n), rtf(e[bs]["rtf"]), rtf(w[bs]["rtf"]), t["faster"].format(x=v[f"speedup_{bs}"])]))
    for bs, n in (("bs1", 1), ("bs16", 16)):
        tab.append(
            row(
                [
                    t["mem"].format(n=n),
                    f"{e[bs]['peak_alloc_gib']:.2f} GiB",
                    f"{w[bs]['peak_alloc_gib']:.2f} GiB",
                    f"-{v[f'mem_cut_{bs}'] * 100:.1f}%",
                ]
            )
        )
    tab.append(row([t["cold"], f"{e['cold_start_s']['median']:.1f} s", f"{w['cold_start_s']['median']:.1f} s", t["wfaster"]]))
    det = [row(t["d_head"]), row(["---"] * 3)]
    for bs, n in (("bs1", 1), ("bs16", 16)):
        det.append(
            row(
                [
                    t["d_lat"].format(n=n, u=t[f"u{n}"]),
                    *(f"{x[bs]['latency_s']['p50']:.3f} / {x[bs]['latency_s']['p95']:.3f} s" for x in (e, w)),
                ]
            )
        )
        det.append(row([t["d_res"].format(n=n), *(f"{x[bs]['peak_reserved_gib']:.2f} GiB" for x in (e, w))]))
        det.append(row([t["d_pass"].format(n=n), *(", ".join(rtf(p) for p in x[bs]["rtf_passes"]) for x in (e, w))]))
    return {
        "HERO": t["hero"].format(sp=v["speedup_bs1"], mem=math.floor(v["mem_cut_bs1"] * 100)),
        "TABLE": "\n".join(tab),
        "COND": t["cond"].format(n=c["clips"], m=t["clips"], a=c["audio_seconds"]),
        "DETAIL": "\n".join(det),
    }


def apply(text: str, vals: dict) -> str:
    for k, val in vals.items():
        pat = re.compile(rf"(<!-- PERF:{k}:BEGIN -->)\n.*?\n?(<!-- PERF:{k}:END -->)", re.S)
        text = pat.sub(lambda m: f"{m.group(1)}\n{val}\n{m.group(2)}", text)
    return text


FILES = {"README.md": "en", "README.zh-TW.md": "zh", "docs/benchmarks.md": "en"}

if __name__ == "__main__":
    d = json.loads(JSON.read_text(encoding="utf-8"))
    stale = []
    for f, lang in FILES.items():
        p = ROOT / f
        old = p.read_text(encoding="utf-8")
        new = apply(old, regions(d, lang))
        if new != old:
            stale.append(f)
            if "--check" not in sys.argv:
                p.write_text(new, encoding="utf-8")
    if "--check" in sys.argv:
        r = subprocess.run([sys.executable, str(Path(__file__).with_name("make_perf_svg.py")), "--check"])
        if stale or r.returncode:
            sys.exit(f"stale: {stale or ''} {'performance.svg' if r.returncode else ''}")
        print("perf numbers consistent")
    else:
        print("updated:", stale or "nothing")
