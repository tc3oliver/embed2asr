# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes (once released) |
| < 0.1 | no |

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use GitHub's private vulnerability reporting ("Report a vulnerability" under the repository's Security tab) and include:

- affected version or commit,
- steps to reproduce,
- impact as you understand it.

We aim to acknowledge reports within 7 days.

## Scope notes

- **Checkpoints are pickled PyTorch files.** Loading an untrusted `.pt` file can execute arbitrary code. Only load checkpoints from sources you trust, and prefer `torch.load(..., weights_only=True)` where supported.
- Audio decoding relies on third-party libraries (e.g. ffmpeg); treat untrusted audio input accordingly.
- Privacy issues with training data (e.g. a speaker requesting removal) should be reported to the original dataset maintainers; we do not redistribute audio or transcripts.
