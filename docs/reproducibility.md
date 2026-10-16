# Reproducibility

> **Skeleton.** This page lists what v0.1.0 intends to make reproducible and how. Sections marked TBD are filled in when the training and benchmark code lands.

## What will be reproducible

| Item | Reproducible from this repo | Notes |
|---|---|---|
| Inference on a local checkpoint | yes | Requires EmbeddingGemma 2 from the official source and a reader checkpoint. |
| Formal benchmark tables in the README | yes | One script, fixed manifests, fixed normaliser; results in `benchmarks/v0.1.0.json`, tables filled by `scripts/fill_bench.py`. |
| Evaluation data preparation | yes | Users download evaluation corpora themselves under their own licences. We never publish audio or transcripts. TBD. |
| Feature caching | yes | Encoder run once per utterance; bf16 features. TBD. |
| Reader training | yes, up to hardware nondeterminism | Training method: [method.md](method.md#4-training). TBD. |
| Exact bitwise checkpoints | no | GPU kernels (MIOpen GRU, HIP graphs) are not bitwise deterministic; expect small metric differences. |

## Environment

- Python 3.10+.
- Research runs: AMD MI300X, ROCm 7.1, PyTorch 2.10 (ROCm build), bf16 encoder features, fp32 reader. TBD: pinned lock file and container digest.
- CUDA is expected to work for inference but was not used for the research runs. TBD: verified matrix.

## Steps (TBD)

1. Obtain EmbeddingGemma 2 and accept its terms.
2. Cache encoder features.
3. Train the reader with the published config.
4. Run the benchmark script; compare with `benchmarks/results`.

## Seeds and determinism

- Sampling order is a pure function of (seed, pass), so resumed runs see the same data order.
- Default seed: 42.
- Known sources of nondeterminism: GPU kernels, shared-GPU contention (affects timing only).

## Reporting rules

- Official test splits are evaluated once, for the final selected checkpoint.
- Greedy and LM results are reported separately.
- CER, WER and MER are never averaged into one number.
- Speed numbers state hardware, batch size, precision and whether the GPU was shared.
