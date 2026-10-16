# Roadmap

## v0.1.0

- Inference package: frozen EmbeddingGemma 2 audio encoder + GRU-CTC reader, greedy CTC decoding, optional Chinese character 5-gram LM.
- Training code for the reader.
- One formal benchmark run on the official test splits, with Whisper large-v3 as a reference, and end-to-end speed and memory numbers.
- Documentation: architecture, method, limitations, reproducibility.
- Embed2ASR-v1 artifacts (reader weights, tokenizer, unit list, optional LM) released under Apache-2.0.

## Later (explicitly outside v0.1.0)

These are research directions, not commitments:

- **Encoder LoRA.** Low-rank adaptation of the audio encoder, to test whether the frozen-encoder ceiling (see [limitations.md](limitations.md#frozen-encoder-ceiling)) can be lifted at modest cost.
- **More English data.** More speakers, accents and spontaneous speech, aimed at the function-word and proper-noun errors.
- **Word-level LM.** A language model that handles English words and mixed sentences, which the current Chinese character LM cannot.

Not planned: streaming, new model families, or non-speech tasks.
