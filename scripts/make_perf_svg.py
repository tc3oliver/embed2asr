"""Generate assets/performance.svg from benchmarks/speed_v0.1.0.json.

usage: python scripts/make_perf_svg.py [--check]

--check regenerates in memory and fails if assets/performance.svg differs. `derive` is shared with fill_perf.py, so the figure,
the README tables and the hero line all come from the same unrounded values."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "benchmarks" / "speed_v0.1.0.json"
OUT = ROOT / "assets" / "performance.svg"


def derive(d: dict) -> dict:
    e, w = d["results"]["embed2asr"], d["results"]["whisper_large_v3"]
    o = {"e": e, "w": w}
    for bs in ("bs1", "bs16"):
        o[f"speedup_{bs}"] = w[bs]["rtf"] / e[bs]["rtf"]
        o[f"mem_cut_{bs}"] = 1 - e[bs]["peak_alloc_gib"] / w[bs]["peak_alloc_gib"]
    return o


def rtf(v: float) -> str:
    return f"{v:.3g}"


STYLE = """  <style>
    text { font-family: -apple-system, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif; }
    .card { fill: #ffffff; stroke: #d0d7de; }
    .t { fill: #1f2328; }
    .s { fill: #57606a; }
    .bo { fill: #1a7f37; }
    .bw { fill: #8c959f; }
    .win { fill: #1a7f37; }
    @media (prefers-color-scheme: dark) {
      .card { fill: #0d1117; stroke: #30363d; }
      .t { fill: #e6edf3; }
      .s { fill: #9da7b3; }
      .bo { fill: #3fb950; }
      .bw { fill: #6e7681; }
      .win { fill: #3fb950; }
    }
  </style>"""

X0, WMAX = 120, 400


def text(x, y, s, cls, size, weight=None, anchor="start"):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}"{w} class="{cls}">{s}</text>'


def bars(y, rows, vmax):
    """rows: (label, value, label_text, class). Returns svg lines; each row is 40 px high."""
    o = []
    for i, (lab, v, vt, cls) in enumerate(rows):
        yy = y + 40 * i
        w = max(3, WMAX * v / vmax)
        o.append(text(24, yy + 20, lab, "t", 16))
        o.append(f'<rect x="{X0}" y="{yy + 2}" width="{w:.1f}" height="28" rx="4" class="{cls}"/>')
        o.append(text(X0 + w + 10, yy + 22, vt, "t", 16, 600))
    return o


def build(d: dict) -> str:
    v = derive(d)
    e, w = v["e"], v["w"]
    c = d["conditions"]
    b = [text(24, 36, "Embed2ASR-v1 vs Whisper large-v3: speed and memory", "t", 20, 700)]
    b.append(text(24, 58, f"{c['clips']} clips, {c['audio_seconds']:,.0f} s of audio, bf16, 1x MI300X. Lower is better.", "s", 15))
    y = 90
    for bs, name in (("bs1", "Batch 1"), ("bs16", "Batch 16")):
        b.append(text(24, y, f"{name}: real-time factor (RTF)", "t", 17, 600))
        b.append(text(616, y, f"{v['speedup_' + bs]:.2f}x faster", "win", 17, 700, "end"))
        b += bars(
            y + 14, [("Embed2ASR", e[bs]["rtf"], rtf(e[bs]["rtf"]), "bo"), ("Whisper", w[bs]["rtf"], rtf(w[bs]["rtf"]), "bw")], w[bs]["rtf"]
        )
        y += 124
    b.append(text(24, y, "Peak GPU memory (GiB, torch max allocated)", "t", 17, 600))
    vmax = w["bs16"]["peak_alloc_gib"]
    for k, (bs, name) in enumerate((("bs1", "Batch 1"), ("bs16", "Batch 16"))):
        yy = y + 30 + 112 * k
        b.append(text(24, yy, name, "s", 16, 600))
        b.append(text(616, yy, f"-{v['mem_cut_' + bs] * 100:.1f}%", "win", 17, 700, "end"))
        b += bars(
            yy + 8,
            [
                ("Embed2ASR", e[bs]["peak_alloc_gib"], f"{e[bs]['peak_alloc_gib']:.2f}", "bo"),
                ("Whisper", w[bs]["peak_alloc_gib"], f"{w[bs]['peak_alloc_gib']:.2f}", "bw"),
            ],
            vmax,
        )
    h = y + 30 + 112 + 96 + 40
    b.append(
        text(
            24,
            h - 14,
            f"Cold start (median of 3): Embed2ASR {e['cold_start_s']['median']:.1f} s, Whisper {w['cold_start_s']['median']:.1f} s (Whisper is faster).",
            "s",
            15,
        )
    )
    body = "\n  ".join(b)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{h}" viewBox="0 0 640 {h}" role="img" aria-labelledby="t d">\n'
        f'  <title id="t">Embed2ASR-v1 vs Whisper large-v3: speed and memory</title>\n'
        f'  <desc id="d">Batch 1 RTF {rtf(e["bs1"]["rtf"])} vs {rtf(w["bs1"]["rtf"])} ({v["speedup_bs1"]:.2f}x faster). Batch 16 RTF {rtf(e["bs16"]["rtf"])} vs {rtf(w["bs16"]["rtf"])} ({v["speedup_bs16"]:.2f}x faster). '
        f"Peak GPU memory {e['bs1']['peak_alloc_gib']:.2f} vs {w['bs1']['peak_alloc_gib']:.2f} GiB at batch 1 and {e['bs16']['peak_alloc_gib']:.2f} vs {w['bs16']['peak_alloc_gib']:.2f} GiB at batch 16. "
        f"Cold start {e['cold_start_s']['median']:.1f} s vs {w['cold_start_s']['median']:.1f} s. Lower is better.</desc>\n"
        f'{STYLE}\n  <rect x="1" y="1" width="638" height="{h - 2}" rx="16" class="card" stroke-width="2"/>\n  {body}\n</svg>\n'
    )


if __name__ == "__main__":
    svg = build(json.loads(JSON.read_text(encoding="utf-8")))
    if "--check" in sys.argv:
        sys.exit(0 if OUT.read_text(encoding="utf-8") == svg else f"{OUT} is out of date; run scripts/make_perf_svg.py")
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT}")
