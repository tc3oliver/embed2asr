---
license: apache-2.0
language:
  - zh
  - en
pipeline_tag: automatic-speech-recognition
base_model: google/embeddinggemma-2
library_name: embed2asr
tags:
  - automatic-speech-recognition
  - speech-recognition
  - asr
  - ctc
  - code-switching
  - mandarin
  - embedding
  - frozen-encoder
  - representation-reuse
model-index:
  - name: Embed2ASR-v1
    results:
      - task: {type: automatic-speech-recognition, name: Automatic Speech Recognition}
        dataset: {name: FLEURS en_us, type: google/fleurs, config: en_us, split: test}
        metrics: [{type: wer, value: 24.5, name: WER (greedy, our normalization)}]
        source: {name: Embed2ASR benchmark protocol, url: https://github.com/tc3oliver/embed2asr/blob/main/docs/benchmarks.md}
      - task: {type: automatic-speech-recognition, name: Automatic Speech Recognition}
        dataset: {name: LibriSpeech test-clean, type: openslr/librispeech_asr, config: clean, split: test}
        metrics: [{type: wer, value: 9.0, name: WER (greedy, our normalization)}]
        source: {name: Embed2ASR benchmark protocol, url: https://github.com/tc3oliver/embed2asr/blob/main/docs/benchmarks.md}
      - task: {type: automatic-speech-recognition, name: Automatic Speech Recognition}
        dataset: {name: LibriSpeech test-other, type: openslr/librispeech_asr, config: other, split: test}
        metrics: [{type: wer, value: 19.7, name: WER (greedy, our normalization)}]
        source: {name: Embed2ASR benchmark protocol, url: https://github.com/tc3oliver/embed2asr/blob/main/docs/benchmarks.md}
      - task: {type: automatic-speech-recognition, name: Automatic Speech Recognition}
        dataset: {name: VoxPopuli en, type: facebook/voxpopuli, config: en, split: test}
        metrics: [{type: wer, value: 15.3, name: WER (greedy, our normalization)}]
        source: {name: Embed2ASR benchmark protocol, url: https://github.com/tc3oliver/embed2asr/blob/main/docs/benchmarks.md}
      - task: {type: automatic-speech-recognition, name: Automatic Speech Recognition}
        dataset: {name: AMI IHM, type: edinburghcstr/ami, config: ihm, split: test}
        metrics: [{type: wer, value: 23.6, name: WER (greedy, our normalization)}]
        source: {name: Embed2ASR benchmark protocol, url: https://github.com/tc3oliver/embed2asr/blob/main/docs/benchmarks.md}
---

# Embed2ASR-v1 — Turning Audio Embeddings into Speech Recognition

**A frozen embedding encoder. A 9.66M trainable reader. Speech transcription without encoder fine-tuning.**

> Can an embedding model transcribe speech without fine-tuning its encoder?

Embed2ASR reuses the representations of an audio embedding model for speech recognition. The audio tower of Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) stays frozen and unmodified. We take its pre-pooling
`last_hidden_state` (1536-d temporal features, 25 frames per second; not the final pooled embedding vector) and decode it with a
GRU-CTC reader of 9,655,172 trainable parameters. About 314.5M parameters are loaded for inference (304,824,608 frozen
encoder + 9,655,172 reader). Languages: Taiwanese Mandarin, English and Mandarin-English code-switching.

This is representation reuse, not a new ASR backbone trained from scratch. The base model listed in the metadata is the
**frozen** encoder: it is not fine-tuned, and its weights are not in this repository. They are downloaded from the base model
repository (pinned revision) at load time. This repository holds the reader, its vocabulary and an optional language model.

Code: https://github.com/tc3oliver/embed2asr · Benchmark protocol: [docs/benchmarks.md](https://github.com/tc3oliver/embed2asr/blob/main/docs/benchmarks.md)

## Architecture

<p align="center"><img src="assets/overview.svg" alt="Audio goes into the frozen EmbeddingGemma 2 audio encoder. Its pre-pooling last_hidden_state (1536-d, 25 frames per second) is fed to a trainable 9,655,172-parameter GRU-CTC reader that outputs the transcript." width="880"></p>

```
16 kHz audio -> EmbeddingGemma 2 audio tower (frozen, bf16 on GPU) -> last_hidden_state, 25 frames/s x 1536
             -> reader (fp32) -> CTC logits over 4,996 units -> greedy decode (or LM beam search)
```

Reader:

- `LayerNorm(1536) -> Linear(1536, 512) -> GELU`
- 3 residual blocks: `Conv1d(512, 512, k=5) -> LayerNorm -> GELU`
- 2-layer bidirectional GRU, hidden size 256 per direction
- `Linear(512, 4996)`, CTC loss, blank id 0

Output units (4,996): blank, one unit per CJK character, and an English SentencePiece BPE vocabulary (400 pieces) for ASCII runs,
plus a few non-ASCII letters and symbols. Mandarin and English share one vocabulary, so code-switching needs no language tag.

| | parameters |
|---|---|
| reader (trained) | 9,655,172 |
| audio tower (frozen) | 304,824,608 |
| total loaded for inference | 314,479,780 |

Encoder: `google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, module `audio_tower`, `last_hidden_state`.

## Performance and memory

**4.16× faster inference and 76.5% lower peak GPU memory than Whisper large-v3 at batch 1, measured on the same MI300X and the same clips.**

<p align="center"><img src="assets/performance.svg" alt="Bar charts: Embed2ASR-v1 vs Whisper large-v3 on one MI300X. Real-time factor at batch 1 and batch 16, peak GPU memory, lower is better." width="640"></p>

|  | Embed2ASR-v1 | Whisper large-v3 | Difference |
| --- | --- | --- | --- |
| RTF, batch 1 (lower is better) | 0.0138 | 0.0576 | 4.16× faster |
| RTF, batch 16 (lower is better) | 0.00345 | 0.0106 | 3.06× faster |
| Peak GPU memory, batch 1 | 0.75 GiB | 3.20 GiB | -76.5% |
| Peak GPU memory, batch 16 | 1.74 GiB | 6.88 GiB | -74.7% |
| Cold start (median of 3) | 7.9 s | 6.0 s | Whisper faster |

Setup: one MI300X, idle before the run; ROCm 7.1, torch 2.10; bf16 for both systems. 200 clips (100 Mandarin + 100 English), 1,086 s of audio. Timing covers audio decode, features, model and text, not model loading; median of 3 passes after 1 warm-up. Whisper uses Hugging Face `generate`, greedy (`num_beams=1`). Embed2ASR at batch 16 uses `cross_file=True` (chunks of different files are batched together; about 4% of short clips can change by one unit; off by default). Memory is torch peak allocated, including the full audio encoder and the reader, not whole-process memory. Cold start is process start to first output. The figures apply to these two batch sizes on this hardware only. Transcripts can differ by +-1 unit across transformers versions and GPU models (identical within one environment); this benchmark ran on torch 2.10.0+rocm7.1.1, transformers 5.19.0, MI300X.

This compares two different systems. The gap comes from several things we did not ablate (loaded model size, CTC versus
autoregressive decoding, this test setup), so read it as a measurement of this pipeline, not as a property of embedding models.
It says nothing about accuracy: see the next sections. Embed2ASR cold start is slower because it loads the full EmbeddingGemma 2
checkpoint before keeping only the audio tower.

## Quick start

The package is not a Transformers ASR pipeline: this repository cannot be loaded with `pipeline("automatic-speech-recognition")`.
Use the `embed2asr` package. Install PyTorch for your platform first, then:

```bash
pip install "embed2asr @ git+https://github.com/tc3oliver/embed2asr.git"
```

```python
from embed2asr import Embed2ASR

asr = Embed2ASR.from_pretrained("tc3oliver/Embed2ASR-v1")  # or a local directory
print(asr.transcribe("audio.wav").text)  # greedy CTC

# optional character 5-gram LM (CPU beam search, slower than greedy; Mandarin and mixed audio)
asr = Embed2ASR.from_pretrained("tc3oliver/Embed2ASR-v1", lm="lm/charlm5.json.gz", lm_alpha=0.25, lm_beta=1.5, beam=16)
```

```bash
embed2asr transcribe audio.wav --model tc3oliver/Embed2ASR-v1
embed2asr transcribe audio.wav --model tc3oliver/Embed2ASR-v1 --lm lm/charlm5.json.gz --lm-alpha 0.25 --lm-beta 1.5 --beam 16
```

The first run downloads the `audio_tower` of `google/embeddinggemma-2`. Audio longer than 29.5 s is cut into consecutive chunks
(no overlap, no VAD). Output is normalized text: lowercase, no punctuation, no spaces next to CJK characters.

## Accuracy vs Whisper large-v3

Each test set was evaluated once with the final checkpoint and the frozen decoding settings. CER for Mandarin, WER for English,
MER for code-switching (CJK character + Latin word units). Reference and hypothesis go through the same normalization, Whisper
is scored the same way, and rows whose reference contains digits are dropped. Whisper large-v3 runs with `num_beams=1` and a
Traditional Chinese prompt. "+ zh LM" is the character 5-gram below (it does not apply to English subsets). Error rates, lower is better.

| Test set | Subset | Metric | Utterances | Embed2ASR greedy | + zh LM | Whisper large-v3 |
|---|---|---|---|---|---|---|
| Common Voice zh-TW 27.0 test | zh | CER | 4,422 | 15.6% | 14.7% | 9.6% |
| Parliament test (set A) | zh | CER | 959 | 19.2% | 19.0% | 22.7% |
| Parliament test (set B) | zh | CER | 5,342 | 19.3% | 19.2% | 23.8% |
| ASCEND test | zh | CER | 628 | 22.7% | 21.4% | 17.8% |
| ASCEND test | zh-en mixed | MER | 358 | 23.9% | 22.9% | 27.8% |
| ASCEND test | en | WER | 229 | 44.6% | – | 26.3% |
| ML2021_ASR_ST test | zh | CER | 8,669 | 16.0% | 15.3% | 7.1% |
| ML2021_ASR_ST test | zh-en mixed | MER | 5,792 | 29.5% | 28.9% | 16.0% |
| ML2021_ASR_ST test | en | WER | 98 | 91.5% | – | 56.6% |
| FLEURS en-US test | en | WER | 621 | 24.5% | – | 4.5% |
| LibriSpeech test-clean | en | WER | 2,610 | 9.0% | – | 4.0% |
| LibriSpeech test-other | en | WER | 2,930 | 19.7% | – | 5.7% |
| VoxPopuli en test | en | WER | 1,829 | 15.3% | – | 11.0% |
| AMI IHM test | en | WER | 10,603 | 23.6% | – | 18.8% |

Embed2ASR-v1 is more accurate on Parliament (both test sets) and on the mixed-language ASCEND sentences. Whisper large-v3 is
more accurate on Common Voice zh-TW, on NTUML2021 (mixed sentences by a wide margin), on FLEURS, LibriSpeech and the other English
sets, and on ASCEND when all sentences are taken together.

The advantage on Parliament and ASCEND is measured on test sets from the same domain as the training distribution. NTUML2021
(ML2021_ASR_ST) is the only out-of-domain test set, and Whisper large-v3 is clearly more accurate on it. Do not read these results
as cross-domain superiority. Scores use our normalization and drop digit rows, so they are not directly comparable with
numbers computed under other conventions. Speed does not imply equal accuracy.

## Training method

Only the reader is trained, on bf16 encoder features cached once per utterance (the encoder is never run during training).
CTC loss, AdamW, cosine learning-rate decay with 500 warmup steps, batch size 32, light time masking, seed 42. Training continues
across stages, each initialised from the previous stage's selected checkpoint; checkpoint selection uses development sets only,
and official test splits are not used for training or selection. Training and evaluation audio, transcripts and feature caches are
not distributed with this model, and the training data is not described here. Method details:
[docs/method.md](https://github.com/tc3oliver/embed2asr/blob/main/docs/method.md); training code and a synthetic smoke recipe:
[docs/training.md](https://github.com/tc3oliver/embed2asr/blob/main/docs/training.md).

## Language model (optional)

`--lm` switches decoding from greedy CTC to CTC prefix beam search with a character 5-gram LM. Only single-CJK-character units
are scored (`alpha * log P_lm + beta` per character); English pieces are LM-neutral, and a run of them is one placeholder token in
the LM context. `alpha=0.25`, `beta=1.5`, `beam=16` were chosen on a single Mandarin development set and then frozen; they were
not tuned on any test set. The LM file is a table of n-gram counts (up to 5 characters), not running text.

## Intended use and limitations

Intended use: research on reusing embedding representations, and offline transcription of Mandarin, English and Mandarin-English speech.

- Offline only. There is no native streaming mode; audio is processed in chunks cut at fixed 29.5 s boundaries, so a word on a boundary can be split.
- English accuracy is clearly behind Whisper large-v3 (see the table above).
- No punctuation, casing or number formatting in the output; digit-bearing references are excluded from the scores.
- Latin letters with diacritics are single non-ASCII units, so word boundaries around them can be lost.
- The encoder is not padding-invariant: batching clips of different lengths together (`cross_file=True`) can change roughly 4% of short clips by about one unit. It is off by default.
- Transcripts can differ by one unit across transformers versions and GPU models; they are identical within one environment.
- LM decoding runs in pure Python on CPU and is much slower than greedy; its parameters were tuned for Mandarin and change nothing on English-only audio.
- Quality drops on noisy, far-field or overlapping speech and on out-of-domain vocabulary.

## Files and checksums

| file | content |
|---|---|
| `model.safetensors` | reader weights, fp32 |
| `config.json` | reader and encoder configuration |
| `units.json`, `tokenizer_bpe.model` | output units, English SentencePiece model |
| `lm/charlm5.json.gz` | optional character 5-gram language model (interpolated Kneser-Ney count tables, gzip JSON) |
| `manifest.json`, `SHA256SUMS` | parameter counts, checksums |
| `assets/*.svg` | figures used by this card |

Verify with `sha256sum -c SHA256SUMS` after downloading. The reader checkpoint (epoch 74) was exported to safetensors with all 34
tensors bit-identical. Reproduction notes: [docs/reproducibility.md](https://github.com/tc3oliver/embed2asr/blob/main/docs/reproducibility.md).
The encoder is not redistributed here.

## License and citation

Code and the files in this repository are released under Apache-2.0. EmbeddingGemma 2 is a separate Google release, downloaded from its own repository.

```bibtex
@misc{embed2asr2026,
  title  = {Embed2ASR: Can an Embedding Model Become a Speech Recognition Model?},
  author = {tc3oliver},
  year   = {2026},
  url    = {https://github.com/tc3oliver/embed2asr}
}
```
