---
# Hugging Face model card metadata (draft)
license: apache-2.0
language:
  - zh
  - en
pipeline_tag: automatic-speech-recognition
tags:
  - ctc
  - code-switching
  - embeddinggemma
base_model: google/embeddinggemma-2   # frozen, not modified; verify exact repo id before publishing
---

# Embed2ASR reader

## Model summary

- What this is: a 9,655,172-parameter GRU-CTC reader that transcribes speech from the frozen EmbeddingGemma 2 audio encoder's temporal features (`last_hidden_state` of the audio tower, `[T × 1536]`, 25 fps, before `embed_audio` and pooling).
- What this is not: the encoder. EmbeddingGemma 2 was created by Google and is not included or modified; download it from the official source under its own terms (Apache-2.0; see the Gemma Prohibited Use Policy linked from its model card).
- Parameters at inference: ~314.5M (304.8M frozen encoder + 9.66M reader).
- Outputs: 4,996 CTC units (blank + 4,596 CJK / non-ASCII characters + 399 English BPE pieces); lowercase, unpunctuated, spoken-form numbers.
- Languages: Taiwanese Mandarin (zh-TW), English, Mandarin-English code-switching.
- Checkpoint: Embed2ASR-v1, epoch 74, sha256 `90d1861dde754574ba0d8cc4a116fb7a7a7a5bf94d48a72c474becc2798aee29`.

## Intended use

Offline transcription research for Taiwanese Mandarin, English and code-switched speech. Not intended for streaming, safety-critical use, or speaker identification.

## How to use

See the repository README Quick start. Requires the EmbeddingGemma 2 encoder and this checkpoint.

## Training

Reader-only training on cached encoder features: AdamW, cosine learning-rate decay, batch size 32, light time masking, continued across stages. Checkpoint selection uses development sets only. Details: repository `docs/method.md`.

## Evaluation

| Dataset | Split | Metric | Embed2ASR greedy | + zh LM | Whisper large-v3 |
|---|---|---|---|---|---|
| Common Voice zh-TW 27.0 | test | CER | <!-- BENCH:cv_zh_tw_cer -->15.6%<!-- /BENCH --> | <!-- BENCH:cv_zh_tw_cer_lm -->14.7%<!-- /BENCH --> | <!-- BENCH:cv_zh_tw_cer_whisper -->9.6%<!-- /BENCH --> |
| Parliament test (set A) | test | CER | <!-- BENCH:parl_cer -->19.2%<!-- /BENCH --> | <!-- BENCH:parl_cer_lm -->19.0%<!-- /BENCH --> | <!-- BENCH:parl_cer_whisper -->22.7%<!-- /BENCH --> |
| ASCEND (mixed) | test | MER | <!-- BENCH:ascend_mix_mer -->23.9%<!-- /BENCH --> | <!-- BENCH:ascend_mix_mer_lm -->22.9%<!-- /BENCH --> | <!-- BENCH:ascend_mix_mer_whisper -->27.8%<!-- /BENCH --> |
| NTUML2021 (mixed) | test | MER | <!-- BENCH:ntuml_mer -->29.5%<!-- /BENCH --> | <!-- BENCH:ntuml_mer_lm -->28.9%<!-- /BENCH --> | <!-- BENCH:ntuml_mer_whisper -->16.0%<!-- /BENCH --> |
| FLEURS en_us | test | WER | <!-- BENCH:fleurs_en_wer -->24.5%<!-- /BENCH --> | – | <!-- BENCH:fleurs_en_wer_whisper -->4.5%<!-- /BENCH --> |
| LibriSpeech test-clean | test | WER | <!-- BENCH:libri_clean_wer -->9.0%<!-- /BENCH --> | – | <!-- BENCH:libri_clean_wer_whisper -->4.0%<!-- /BENCH --> |

Full table: repository `docs/benchmarks.md`. Lower is better.

## Limitations

Frozen-encoder ceiling; homophone substitutions; English function words and proper nouns; narrow code-switching evidence; non-verbatim references in one evaluation set; no streaming. Details: repository `docs/limitations.md`.

## License

Apache-2.0. The EmbeddingGemma 2 encoder is a separate download from Google under its own terms.

## Citation

See repository `CITATION.cff`. Please also cite EmbeddingGemma 2 per Google's model card.
