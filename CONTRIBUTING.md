# Contributing

Thanks for your interest. Embed2ASR is a small research project; the most useful contributions are bug reports, reproduction reports and careful benchmark reports.

## Before you open an issue

- Bug: use the *Bug report* template. Include the exact command, environment (OS, Python, PyTorch, GPU / ROCm or CUDA version) and the full error.
- Reproduction: use the *Reproduction report* template if you re-ran data preparation, training or evaluation and your numbers differ from ours (or match them).
- Benchmark: use the *Benchmark report* template. State the dataset, split, metric (CER / WER / MER), decoding mode (greedy or LM), hardware and version. Do not combine CER, WER and MER into one number.

## Pull requests

1. Open an issue first for anything beyond a small fix.
2. Keep changes small and focused; one topic per pull request.
3. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages, e.g. `fix(decode): handle empty input`.
4. Run the tests before pushing (`python -m pytest`).
5. Do not add new dependencies without discussing them in the issue first.

## Data and weights

- Never commit audio, transcripts, feature caches, checkpoints or any data covered by a third-party licence.
- Do not attach model weights to issues or pull requests. Link to the official release instead.
- Respect the licence and terms of any dataset you use. In particular, do not attempt to identify speakers.

## Reporting results

- Name the checkpoint and epoch for every number.
- Report greedy and LM-decoded results separately.
- Say whether a number comes from a dev set or a test set.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
