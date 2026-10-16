"""Fill the README BENCH placeholders from benchmarks/v0.1.0.json.

usage: python scripts/fill_bench.py [benchmarks/v0.1.0.json] [README.md ...]

Besides the BENCH markers, an `<!-- ACC:BEGIN -->...<!-- ACC:END -->` region is rebuilt from the JSON `accuracy` array.
The JSON holds a flat object {"readme": {"<key>": "<formatted cell text>"}} with the keys used in
`<!-- BENCH:<key> -->...<!-- /BENCH -->`. Only text between the markers is replaced, so the script can be re-run.
Keys missing from the JSON stay unchanged and are reported."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"(<!-- BENCH:(\w+) -->)(.*?)(<!-- /BENCH -->)")


def fill(text: str, values: dict) -> tuple[str, set]:
    missing = set()

    def rep(m):
        k = m.group(2)
        if k not in values:
            missing.add(k)
            return m.group(0)
        return f"{m.group(1)}{values[k] if values[k] is not None else '–'}{m.group(4)}"

    return PAT.sub(rep, text), missing


def acc_table(rows: list) -> str:
    def pct(x):
        return "–" if x is None else f"{x * 100:.1f}%"

    out = ["| Test set | Subset | Metric | Utterances | Embed2ASR greedy | + zh LM | Whisper large-v3 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['name']} | {r['lang']} | {r['metric']} | {r['utts']:,} | {pct(r['greedy'])} | {pct(r['lm'])} | {pct(r['whisper_large_v3'])} |"
        )
    return "\n".join(out)


def main(argv):
    src = Path(argv[1]) if len(argv) > 1 else ROOT / "benchmarks" / "v0.1.0.json"
    targets = [Path(p) for p in argv[2:]] or [ROOT / "README.md", ROOT / "README.zh-TW.md"]
    data = json.loads(src.read_text(encoding="utf-8"))
    values = data["readme"]
    acc = re.compile(r"(<!-- ACC:BEGIN -->)\n.*?\n?(<!-- ACC:END -->)", re.S)
    for t in targets:
        out, missing = fill(t.read_text(encoding="utf-8"), values)
        out = acc.sub(lambda m: f"{m.group(1)}\n{acc_table(data['accuracy'])}\n{m.group(2)}", out)
        t.write_text(out, encoding="utf-8")
        print(f"{t.name}: filled, {len(missing)} key(s) not in JSON" + (f": {sorted(missing)}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
