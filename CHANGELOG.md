# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- README: removed the per-utterance demo transcripts; an audio demo will follow with recordings cleared for publication.
- Model card images use absolute Hugging Face URLs, so `docs/model_card.md` renders on GitHub as well.
- CI checks that the README, benchmark docs, performance figure and model card match the benchmark JSON.
- Added a guide for contributing speed and accuracy results from other hardware ([docs/benchmark_contributing.md](docs/benchmark_contributing.md)).

## [0.1.0] - 2026-10-16

### Added

- Inference with the frozen EmbeddingGemma 2 audio encoder and a 9.66M-parameter GRU-CTC reader (greedy CTC).
- Optional Chinese character 5-gram LM with CTC prefix beam search.
- Shared output units: CJK characters plus English BPE pieces, for Mandarin, English and code-switching.
- Research documentation: architecture, method, limitations, reproducibility, roadmap.
- Architecture and overview figures.
- Formal benchmark on the test splits of 14 evaluation subsets, with Whisper large-v3 as reference, and end-to-end speed and memory on one MI300X.
