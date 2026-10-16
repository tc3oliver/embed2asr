# Launch copy

Draft text for the release. Nothing here has been posted anywhere. Numbers are from the formal test run (`benchmarks/v0.1.0.json`).

## GitHub description

Speech recognition from a frozen EmbeddingGemma 2 audio encoder and a 9.66M-parameter GRU-CTC reader.

## Technical summary (about 100 words)

Embed2ASR reads the temporal states of Google's EmbeddingGemma 2 audio encoder (last_hidden_state, 1536-d, 25 frames per second, taken before pooling) with a small GRU-CTC reader. The encoder stays frozen; only the 9,655,172-parameter reader is trained. One 4,996-unit CTC output covers Chinese characters and English BPE pieces, so the model handles Taiwanese Mandarin, English and code-switching. On our test sets it beats Whisper large-v3 on the Parliament sets and on ASCEND mixed sentences and is clearly behind on English, Common Voice zh-TW and NTUML2021. Inference loads about 314.5M parameters. Code and weights are Apache-2.0.

## Topics

`speech-recognition`, `asr`, `ctc`, `embeddings`, `embeddinggemma`, `code-switching`, `mandarin`, `taiwanese-mandarin`, `pytorch`, `rocm`

## Show HN title

Show HN: Embed2ASR, speech recognition on top of a frozen audio embedding model

## Introduction for the ML/ASR community

EmbeddingGemma 2 is an embedding model, and its audio path pools encoder states into one vector per clip. We asked whether the states before pooling are enough for transcription. We froze the audio tower, took `last_hidden_state` (1536-d, 25 frames/s), and trained a GRU-CTC reader on top: LayerNorm and a linear projection, three residual Conv1d blocks, a 2-layer BiGRU, and a linear layer over 4,996 CTC units (blank, 4,596 CJK and other non-ASCII characters, 399 English BPE pieces). The reader has 9,655,172 parameters; the encoder has 304,824,608.

Training ran on cached encoder features (AdamW, cosine decay, batch size 32, time masking), continued across stages. Every tracked dev metric improved per stage, but the gain per added unit of data shrank. On large pools, error on seen sentences stays close to dev error, which we read as a limit of the frozen representation. A partial fine-tuning probe on a small subset did not help, but it was not conclusive. Encoder LoRA is the obvious next experiment and is not part of v0.1.0.

Greedy CTC is the default. An optional Chinese character 5-gram LM with prefix beam search recovers part of the homophone errors.

Code and weights are Apache-2.0. The encoder is not redistributed.

## Main results and limits

Test sets, Embed2ASR-v1 greedy / + zh LM / Whisper large-v3 (lower is better):

| Dataset | Metric | Embed2ASR | + LM | Whisper large-v3 |
|---|---|---|---|---|
| Common Voice zh-TW 27.0 | CER | <!-- BENCH:cv_zh_tw_cer -->15.6%<!-- /BENCH --> | <!-- BENCH:cv_zh_tw_cer_lm -->14.7%<!-- /BENCH --> | <!-- BENCH:cv_zh_tw_cer_whisper -->9.6%<!-- /BENCH --> |
| Parliament test (set A) | CER | <!-- BENCH:parl_cer -->19.2%<!-- /BENCH --> | <!-- BENCH:parl_cer_lm -->19.0%<!-- /BENCH --> | <!-- BENCH:parl_cer_whisper -->22.7%<!-- /BENCH --> |
| ASCEND (mixed) | MER | <!-- BENCH:ascend_mix_mer -->23.9%<!-- /BENCH --> | <!-- BENCH:ascend_mix_mer_lm -->22.9%<!-- /BENCH --> | <!-- BENCH:ascend_mix_mer_whisper -->27.8%<!-- /BENCH --> |
| NTUML2021 (mixed) | MER | <!-- BENCH:ntuml_mer -->29.5%<!-- /BENCH --> | <!-- BENCH:ntuml_mer_lm -->28.9%<!-- /BENCH --> | <!-- BENCH:ntuml_mer_whisper -->16.0%<!-- /BENCH --> |
| FLEURS en_us | WER | <!-- BENCH:fleurs_en_wer -->24.5%<!-- /BENCH --> | – | <!-- BENCH:fleurs_en_wer_whisper -->4.5%<!-- /BENCH --> |
| LibriSpeech test-clean | WER | <!-- BENCH:libri_clean_wer -->9.0%<!-- /BENCH --> | – | <!-- BENCH:libri_clean_wer_whisper -->4.0%<!-- /BENCH --> |

The Parliament and ASCEND results are on test sets from the same domain as the training distribution; NTUML2021 is the only out-of-domain set, and Whisper is clearly better there. On all ASCEND sentences Whisper is slightly ahead.

Limits:

- English, Common Voice zh-TW and NTUML2021 are clearly behind Whisper large-v3.
- Mandarin errors are mostly homophones; CTC has no output-side context.
- Parliament references are not verbatim, so part of its CER is reference noise.
- The code-switching evidence is narrow: the mixed-language dev set is small and recorded in Hong Kong.
- Offline only. Inference runs the full 304.8M-parameter encoder, so it is not light.
- Test-set numbers are not published yet.
