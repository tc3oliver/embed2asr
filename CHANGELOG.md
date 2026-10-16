# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - unreleased

### Added

- Inference with the frozen EmbeddingGemma 2 audio encoder and a 9.66M-parameter GRU-CTC reader (greedy CTC).
- Optional Chinese character 5-gram LM with CTC prefix beam search.
- Shared output units: CJK characters plus English BPE pieces, for Mandarin, English and code-switching.
- Research documentation: architecture, method, limitations, reproducibility skeleton, roadmap.
- Architecture and overview figures.

### Planned

- Formal benchmark on official test splits.
- Demo recording.
