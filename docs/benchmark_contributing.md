# Contributing benchmark results

All published speed numbers come from one AMD Instinct MI300X (ROCm). We have not measured CUDA GPUs, Apple Silicon or CPUs.
If you run Embed2ASR on other hardware, a report helps other users more than anything we can write. Slow results and
results where Whisper wins are just as useful.

Open a *Benchmark report* issue and attach the JSON files. Please do not edit `benchmarks/*.json`; those are the v0.1.0 run.

## What to run

Install from a checkout (any torch build for your device: CUDA, ROCm, MPS or CPU):

```bash
git clone https://github.com/tc3oliver/embed2asr.git && cd embed2asr
pip install -e ".[eval]"
```

Clip set: LibriSpeech test-clean ([openslr.org/12](https://www.openslr.org/12)), first 200 files in sorted order.

```bash
find LibriSpeech/test-clean -name '*.flac' | sort | head -200 > clips.txt
```

Speed, one process per run so the memory peak is clean:

```bash
python tools/bench_speed.py ours    --list clips.txt --bs 1  --out ours.bs1.json
python tools/bench_speed.py ours    --list clips.txt --bs 16 --out ours.bs16.json
python tools/bench_speed.py whisper --list clips.txt --bs 1  --out whisper.bs1.json
python tools/bench_speed.py whisper --list clips.txt --bs 16 --out whisper.bs16.json
```

Add `--device cpu` or `--device mps` if needed. On CPU, batch 1 alone is enough. The Whisper runs are optional, but a
comparison on the same machine says much more than an absolute number. For Mandarin, pass any public Mandarin test set
with `--lang zh`. Name it in the report.

What the script measures (same protocol as [benchmarks.md](benchmarks.md#protocol)):

- Timed: audio file → decode → features → model → text. Model loading is excluded. 1 warm-up, then the median RTF of 3 passes.
- Batch 16 for Embed2ASR uses `cross_file=True`, as in our run.
- Memory: torch peak allocated on CUDA/ROCm (includes the full audio encoder). On CPU, process max RSS. On MPS, no peak is recorded.
- Precision: bf16 on CUDA/ROCm, fp32 on CPU and MPS (package default). Results across precisions are not directly comparable.

## Accuracy check (optional)

This checks that your environment produces the same transcripts as ours. On the full LibriSpeech test-clean we measured
9.0% WER (greedy). Transcripts can differ by about one unit across GPU models and transformers versions, so expect a value
within a few tenths of that, not an exact match.

```bash
find LibriSpeech/test-clean -name '*.flac' | sort > all.txt
embed2asr transcribe $(cat all.txt) --model tc3oliver/Embed2ASR-v1 --json > hyp.jsonl
```

```python
import json, pathlib
from embed2asr.metrics import score

refs = {}
for t in pathlib.Path("LibriSpeech/test-clean").rglob("*.trans.txt"):
    for line in t.read_text().splitlines():
        k, txt = line.split(" ", 1)
        refs[k] = txt
rows = [{"ref": refs[pathlib.Path(h["audio"]).stem], "hyp": h["text"], "lang": "en-US"} for h in map(json.loads, open("hyp.jsonl"))]
rows = [r for r in rows if not any(c.isdigit() for c in r["ref"])]
print(score(rows))
```

## What to put in the report

- Hardware: GPU or CPU model, memory, driver and CUDA / ROCm version, or macOS version for Apple Silicon.
- Software: OS, Python, torch, transformers, embed2asr version (all except the driver are in the JSON).
- Whether anything else was running on the device.
- The JSON files, unedited.
- Any deviation from the commands above.

We will add reproducible reports to a community results table, credited to you, together with the hardware and versions.
