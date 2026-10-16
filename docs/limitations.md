# Limitations

Known limitations of Embed2ASR v0.1.0, with the evidence behind each. None of this page is a benchmark claim.

## Frozen encoder ceiling

The encoder is never updated, so the reader can only use what the EmbeddingGemma 2 audio states already encode.

Evidence: we score training sentences the model has already seen ("seen", with exposure counts replayed from the sampler) and compare with dev.

- On large, diverse training pools, seen-sentence error stays close to dev error even after 8–26 passes per sentence. The reader is not memorising these sets; it is under-fitting them.
- On small sets, seen error goes to near zero while dev error stays far higher: plain overfitting from too little data.

The under-fitting pattern on large pools is the main reason we believe the frozen encoder, not the reader size or data volume alone, is the current limit. A partial fine-tuning probe on a small subset did not help, but it was run in an overfitting regime and is not conclusive. Encoder adaptation is on the [roadmap](roadmap.md), outside v0.1.0.

## CTC conditional independence: homophones

CTC predicts each frame independently given the audio, with no output-side context. In Mandarin the dominant error type is substitution between same-sounding characters (店/電, 他/她, 再/在, 的/得, 市/是). Deletions and insertions are rare on read speech. The optional Chinese 5-gram LM recovers part of this, at the cost of sometimes replacing rare correct words with common ones.

## English: function words and proper nouns

English is weaker than Mandarin and clearly weaker than Whisper large-v3 on dev. The most frequent errors are short function words (and/in, the/a, of/a) and proper nouns (javanese → japanese, filipino → philippino). The 399-piece BPE vocabulary and lack of a word-level LM both contribute. Latin letters with diacritics (ç, ü, ñ) are separate units and can merge neighbouring words (about 1% of English dev rows).

## Code-switching evidence is narrow

- The mixed-language dev set is small: 3 dev speakers and 2 test speakers, so variance is high.
- Seen-sentence error on that data is near zero while dev error is far higher.
- It was recorded in Hong Kong with speakers from Hong Kong, Taiwan and mainland China. It does not show that the model handles Taiwanese-accented code-switching. NTUML2021 (evaluation only) is the Taiwanese-accent check.
- Annotation style differs between corpora: filler words (uh / 呃 / 嗯 / um) and 它/他 are a large share of errors.

## Reference transcripts

The Parliament evaluation references come from official proceedings, not verbatim transcription. Deletions and insertions on its dev set are each around 7%, concentrated on 的 / 是 / 這 / 個. Part of the measured CER is therefore reference noise, not recognition error.

## Evaluation practice

- Model selection and LM tuning use dev sets only. Official test splits are evaluated once, for the final selected checkpoint.
- Only exact-match sentence overlap was removed from training; near-duplicates were not.
- Speaker independence differs between evaluation sets, and some sets have no speaker IDs. A dev score on a set with speaker overlap is optimistic.

## Language model effect

- The LM is Chinese only: BPE pieces are treated as characters, so it does not help English or the English part of mixed sentences.
- LM weights were tuned on one Mandarin dev set only, and the best weight sat at the edge of the grid. Expect the dev gain to be optimistic.
- Gains on other domains are expected to be smaller.
- Greedy and LM results are always reported separately.

## No native streaming

The encoder runs on whole segments (tested up to about 29.5 s) and the reader uses a bidirectional GRU. There is no streaming or low-latency partial output. Long audio must be segmented externally.

## Training cost vs inference cost

- Training is cheap only because encoder features are cached once: the reader trains on stored `[T × 1536]` features. The cache is large (about 0.26 GB of disk per audio hour in bf16) and has to be rebuilt if the encoder or its preprocessing changes.
- Inference cannot use a cache: every request runs the full 304.8M-parameter encoder plus the reader (~314.5M loaded). "9.66M" describes trainable parameters only.
- Cold start (loading the encoder) is slower than for some alternatives.
- Training was done on AMD MI300X (ROCm) GPUs shared with other workloads; timing figures from those runs are noisy.

## ROCm GRU dropout

On the ROCm PyTorch builds used (torch 2.10 with ROCm 7.1), `nn.GRU(num_layers>1, dropout>0)` produced corrupted outputs in training mode (output collapse from the second call on; evaluation mode unaffected). Upstream has a fix (pytorch/pytorch#189285, ROCm/ROCm#6339) that is not in those builds. The reader avoids the bug by building two single-layer GRUs with explicit dropout in between. All research checkpoints were trained with GRU dropout 0, so they are not affected either way.

## Other

- The output vocabulary is closed: characters not seen in training cannot be produced.
- The model outputs normalised text: lowercase, no punctuation, spoken-form numbers.
- Robustness to noise, far-field audio, children's speech and Taiwanese Hokkien or Hakka has not been evaluated.
