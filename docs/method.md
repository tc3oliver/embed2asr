# Method: what was tested, what worked, what didn't

This page is the research narrative. Numbers here are development-set observations only. Formal test results are published separately.

## 1. The question

EmbeddingGemma 2 is trained to produce one semantic vector per input. Speech recognition needs a time-aligned sequence. The question was whether the audio encoder inside an embedding model already carries enough phonetic detail that a small reader, trained on top of a frozen encoder, can transcribe speech.

## 2. Frozen-encoder hypothesis

Hypothesis. The audio tower's temporal states (before pooling) are rich enough that only a small CTC reader needs training.

What we did.

- Cached `last_hidden_state` (`[T × 1536]`, 25 fps) for every training utterance once, then trained only the reader on the cached features.
- Kept the encoder frozen in every reported run.

Probes run early (small subsets, single seed, bounded "bet first" rules):

| Probe | Result | Decision |
|---|---|---|
| Intermediate layers (L2–L12, learned weighted sum) vs `last_hidden_state` | Best intermediate layer within seed noise of the tap; lower layers clearly worse; weighted sum worse | Keep `last_hidden_state` |
| Partial fine-tuning (last 2 conformer layers + `output_proj` trainable, +51.9M params) | Faster memorisation of the training set, no better dev CER at the same epoch; ~4.6× training time | Stay frozen; LoRA variants not run |
| Feature-space augmentation (time mask, Gaussian noise on cached features) | Small gains, below the pre-set 1-point threshold | Keep light time masking only |
| Waveform speed perturbation | Promising on one seed, not confirmed on a second seed | Not adopted (would require re-caching) |
| Output head tied to text embeddings ("embedding CTC") | Pipeline worked; not pursued | Dropped |

What we learned. A frozen encoder plus a ~10M reader does learn to transcribe Mandarin, English and mixed speech. The limit shows up later: on large training pools the error on seen sentences stays close to the dev error (see [limitations.md](limitations.md#frozen-encoder-ceiling)). We read that as a representation ceiling of the frozen encoder. It is the reason encoder adaptation (e.g. LoRA) is on the [roadmap](roadmap.md), outside v0.1.0.

## 3. Output units: characters vs characters + BPE

The one architecture comparison was the output unit set, with the same reader and the same training budget:

- char: one unit per character, including every English letter.
- char + BPE: one unit per CJK / non-ASCII character, plus 399 English BPE pieces.

At the same epoch, Mandarin CER was essentially tied, while English WER was much better with BPE on both English dev sets. With BPE an English word needs fewer CTC units per second (about 5.7 vs 14.6), which suits a 25 fps encoder.

Decision: char + BPE.

## 4. Training

- Reader only; encoder features are cached bf16 `[T × 1536]`.
- AdamW, cosine learning-rate decay with 500 warmup steps, batch size 32, light time masking (0.1), dropout 0, seed 42.
- Training is continued across stages: each stage initialises from the previous stage's selected checkpoint (`--init`), and the output layer is remapped when the vocabulary changes. Runs can resume from the last checkpoint.
- Checkpoint selection uses development sets only. Per-metric dev scores are normalised against the previous stage's baseline and averaged; the best combined checkpoint is kept, and per-metric best checkpoints are kept separately. Embed2ASR-v1 is epoch 74 of the final stage.
- Evaluation every 2 epochs. Official test splits are not used for training or selection.

Observations across stages (dev only):

- Every tracked dev metric improved from one stage to the next, including English.
- Gains per added unit of training data shrank.
- Small training sets are memorised: seen-sentence error goes near zero while dev error stays far higher.
- Large sets are not memorised: seen-sentence error stays close to dev error even after many passes (under-fitting). Adding more of the same kind of data has diminishing returns under a frozen encoder.

## 5. Decoding with a language model

A Chinese character 5-gram LM (Kneser-Ney, text with every dev/test sentence removed) with CTC prefix beam search lowered Mandarin dev CER at the selected checkpoint. It fixes many homophone errors but also "corrects" rare correct words into common ones. Tuning was done on dev only, so the gain is optimistic. It does not apply to English (BPE pieces are treated as characters). LM results are always reported apart from greedy.

## 6. What we deliberately did not do

- No training or fine-tuning of the EmbeddingGemma 2 encoder in any reported run.
- No reader-size, dropout or seed sweeps; one reader configuration throughout.
- No multi-GPU training (no speed-up measured at this model size).
- No use of any official test split for training or model selection.
