# Embed2ASR v0.1.0 release notes

## Summary

First research release of Embed2ASR: speech recognition built on Google's EmbeddingGemma 2 audio encoder, kept frozen, with a 9.66M-parameter GRU-CTC reader trained on top. Parameters loaded for inference: ~314.5M. Languages: Taiwanese Mandarin, English, and Mandarin-English code-switching.

## What's included

- Inference package (greedy CTC; optional Chinese character 5-gram LM).
- Training code for the reader.
- Benchmark script and results: [benchmarks.md](benchmarks.md), data in `benchmarks/`
- Documentation: architecture, method, limitations, reproducibility, roadmap.

## Benchmark

Test-set results, speed and memory: [benchmarks.md](benchmarks.md). CER, WER and MER are reported separately; greedy and LM separately.

## Weights

Hugging Face: https://huggingface.co/tc3oliver/Embed2ASR-v1


Embed2ASR-v1 artifacts (reader weights, tokenizer, unit list, optional LM) are released under Apache-2.0. Checkpoint: Embed2ASR-v1, epoch 74, sha256 `90d1861dde754574ba0d8cc4a116fb7a7a7a5bf94d48a72c474becc2798aee29`. The EmbeddingGemma 2 encoder is not included; download it from Google.

## Known limitations

See [limitations.md](limitations.md). Highlights: frozen-encoder ceiling, homophone errors from CTC, English behind Whisper large-v3, narrow code-switching evidence, offline only.

## Credits

EmbeddingGemma 2 was created by Google.
