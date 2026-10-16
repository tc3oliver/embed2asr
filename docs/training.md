# Training

How the reader is trained and how to run the trainer. The recipe config is `configs/recipe_v0.1.0.yaml`; the training data is not distributed, so this document describes the method, not a data-for-data reproduction.

## What is reproducible

| Claim | Status |
|---|---|
| Method: model, loss, optimizer, schedule, sampler, selection and stopping rules | Implemented in `training/train.py`, single GPU. |
| Same model shape as the release | With `configs/recipe_v0.1.0.yaml` the reader has the release `state_dict` layout (`embed2asr.reader.Reader`, d 512, GRU 256, 3 blocks). |
| Training forward = inference forward | Verified: `research/parity_pr3.md` (logits, CTC loss, gradients, padding, graphs). |
| Exact resume | On CPU, weights are bit-identical. On GPU, a resumed epoch gives the same loss and dev metrics, and weights differ by at most 1.5e-7 (non-deterministic GPU kernels, graphs captured again). |
| Bit-identical `reader.pt` | No. GPU kernels are non-deterministic, and the released weights also depend on data that is not redistributed. |

## Pipeline

```
audio manifest (JSONL: audio, text, lang, speaker_id)
  -> training/prepare_features.py   decode 16 kHz mono (ffmpeg), keep 0.4-29.5 s, normalize text,
                                    frozen EmbeddingGemma 2 audio tower -> last_hidden_state [T, 1536] @ 25 fps
  -> feature cache                  features.bin (raw bf16 rows) + index.jsonl (offset, frames, crc32, text, ...) + cache.json
  -> training/train.py              GRU-CTC reader on the cached features (the encoder is never loaded)
  -> checkpoints                    best.pt / best_<metric>.pt / last.pt / resume.pt
```

| config | purpose | runs where |
|---|---|---|
| `configs/smoke.yaml` | End-to-end check of the training code on synthetic features (`python -m training.smoke_data`). Runs in CI. | Anywhere; CPU is enough. |
| `configs/recipe_v0.1.0.yaml` | Hyperparameters of the release recipe. Paths are `${E2A_CACHE}` / `${E2A_CKPT}`. | A GPU and your own feature caches. |

```bash
pip install -e ".[train]"

# smoke test: no data, no encoder, no GPU
python -m training.smoke_data && python -m training.train --config configs/smoke.yaml --device cpu

# your own data
python -m training.prepare_features --manifest data/train.jsonl --out cache/train      # resumable; one GPU
python -m training.prepare_features --manifest data/dev.jsonl   --out cache/dev
export E2A_CACHE=$PWD/cache E2A_CKPT=$PWD/runs   # expects $E2A_CACHE/train and $E2A_CACHE/dev
python -m training.train --config configs/recipe_v0.1.0.yaml
```

- Run from the repository root.
- `${VAR}` in a config expands from the environment; an unset variable is an error.
- `prepare_features.py` writes feature paths relative to the index file, so its caches work from any directory.

## Frozen encoder

- The encoder (`google/embeddinggemma-2`, audio tower, bf16) is used only in `prepare_features.py`, under `torch.inference_mode()`. The trainer reads cached features and never loads the encoder, so no encoder weight can change.
- `cache.json` records:
  - encoder id and the resolved Hub commit (`--revision` pins it; resolved from the local Hub snapshot);
  - dtype, feature width and output tensor;
  - a hash of the feature-extractor config;
  - duration filters and the text normalizer;
  - the manifest sha256.
- To check a cache against an encoder, encode a few manifest rows into a fresh directory with the recorded revision and compare them with the cached rows. Close values are expected, not identical bytes: bf16 numerics depend on batch padding (the same rows encoded as one batch vs one at a time differ by about 1%).

## Feature cache integrity

When `prepare_features.py` resumes, it validates the existing cache before appending anything (`open_cache`; tests in `tests/test_prepare_features.py`):

- `cache.json` must match: encoder, revision, dtype, feature config, filters and **manifest sha256**. If the manifest changed, the old cache is never reused silently: the run refuses and asks for a new `--out`. Index or features files without a `cache.json` are refused too.
- Every index line except a torn last line must parse and match the manifest: row id in range, same audio path, unique ids, contiguous offsets.
- If `features.bin` is shorter than the index requires, the cache is truncated and the run refuses.
- The crc32 of the last 64 rows (all rows with `--verify-all`) must match; a mismatch means corrupted features and the run refuses.
- Repairs cover only what an interrupted write can leave behind. Features are flushed before their index lines, so a crash can leave a torn last index line (dropped) or features past the last indexed row (cut). Both repairs lose at most the in-flight batch, which is recomputed. A refused cache is not modified.

## Data preparation

Build one feature cache per training and dev set from a manifest with `prepare_features.py`. Labels are normalized with `embed2asr.normalize.norm`. The trainer also drops pool rows with digits and CTC-infeasible rows.

## Reader architecture

`embed2asr.reader.Reader` keeps the same `state_dict` layout as the research checkpoints:

- `pre`: LayerNorm(1536) -> Linear(1536→512) -> GELU (790,016 params)
- 3 residual blocks: masked Conv1d(512, k=5) -> LayerNorm -> GELU (3,936,768)
- 2-layer bidirectional GRU, 256 per direction (2,365,440)
- `out`: Linear(512→4,996) CTC head; unit 0 = blank (2,562,948)
- Total 9,655,172 params, fp32, dropout 0. Units = every CJK char + SentencePiece BPE(400) pieces for ASCII runs (`embed2asr.tokenizer`).

Training uses a padded GRU forward (`PaddedCore`). The reverse direction is flipped within each sequence's own length, so valid frames match `pack_padded_sequence`. Its shapes are static, so it can be graph-captured. Evaluation uses the packed `Reader.forward`.

`PaddedCore` calls the private `torch._VF.gru`. It is verified on torch 2.10.0+rocm7.1.1 (MI300X) and on CPU; see `research/parity_pr3.md` and `tests/test_parity.py`. A torch release can change this private signature or its behaviour without notice. Mitigations:

- Every start (unless `--no-graph`) runs `padded_selfcheck`, which compares `PaddedCore` with the packed Reader on a random batch and exits if they differ by more than 1e-4.
- `--no-graph` trains with the packed `Reader.forward` only: public API, no graphs, slower. It is covered by `tests/test_train_smoke.py::test_no_graph_packed_path`.
- After a torch upgrade, run `pytest tests/test_parity.py` on the target device.

## Loss

- CTC loss (blank 0) is computed per utterance with `reduction="none"`, divided by the target length, and averaged over **finite** losses. Non-finite losses are counted (`inf_frac` in the log), not hidden. If every loss in a batch is non-finite, the run aborts.
- Utterances whose target needs more frames than they have (`ctc_min_len`) are dropped at load time.
- Pool rows whose label contains digits are dropped too, because the speech says the number in words. The main `--train` set and the dev set are not digit-filtered (same as research).
- Time masking 0.1: about 2.5% of frames start a 4-frame zeroed span, so about 10% of frames are masked.

## Optimizer and schedule

- AdamW, weight decay 0.01, peak LR 5e-4, gradient clipping at norm 5.
- Linear warmup over `min(500, total/10)` steps, then cosine decay to 5% of the peak LR, with `total = steps_per_epoch(epoch 1) × epochs`. Chunk tails add a few steps per epoch, so the last steps run at the floor LR.
- Effective batch: **32 utterances per optimizer step** (one GPU, no gradient accumulation, no DDP).
- Batches are bucketed: shuffle, sort each window of 50 batches by length, cut, shuffle the batch order. The time axis is padded to a multiple of 50 frames.
- Full batches use a CUDA/HIP graph (`torch.cuda.make_graphed_callables`, forward + backward, one graph per (B, T) shape). Partial batches run eagerly, because a graph per odd batch size leaks memory. `--no-graph` runs every batch eagerly with the packed Reader.
- `--bf16` enables autocast for pre/conv/out. The recipe does not use it: compute is fp32 on bf16-stored features.

## Sampling (rotate, hours per pool)

Each source is a pool with an hours-per-epoch budget: `--train-hours` for the main set, `--pool PATH:HOURS` for the others.

- Each epoch, every pool contributes the **next** unused rows of its shuffled order until the budget is reached.
- A pool is reshuffled only after a full pass, and no row repeats within an epoch.
- Pass *n* of pool *k* is shuffled with `random.Random(seed*1009 + k*101 + n)`. The rows of epoch *e* therefore depend only on (data, seed, *e*), and resume needs no sampler state.
- `pool_passes` in the log reports the passes completed per pool.

Each epoch's rows are uploaded to the GPU in `--pool-chunks 8` random parts, so feature memory is about 1/8 of the epoch, and each part mixes all pools. The chunk split and the batch order draw from one `random.Random(seed)` stream, which is saved in `resume.pt`.

## Evaluation, checkpoint selection, stopping

- Every `--eval-every 2` epochs, greedy decoding is scored on the main dev set, each `--dev-extra` set, and a fixed random subset of 2,000 training rows. Metrics:
  - CER (spaces removed);
  - WER;
  - MER (units: a CJK char or a Latin word);
  - MER on code-switched rows only (`lang` = `mix`/`zh-en` → `*_mixmer`).
- `best_<metric>.pt` keeps the per-metric best for each `--select` key.
- `best.pt` keeps the combined best. The combined score is mean(metric / base), where each `--select KEY:BASE` base is a reference value for that metric (default 1). A candidate counts only if no selected metric is worse than (1 + 0.05) × its own best (`--select-tol`), so one language cannot be traded away.
- Patience: the run stops after 8 evals with no relative improvement above 0.3% (`--min-delta`) in any selected metric or in the combined score, but never before epoch 20 (`--min-epochs`).
- `--max-hours 14` is a wall-clock budget summed over resumes.
- The regression guard stops the run if a selected metric exceeds 1.10 × this run's own best for 4 evals in a row, from epoch 20 on (`--regress-ref best`).
- Every stop writes `stop_reason.json`.

## Seed, resume, checkpoints

- One seed (42) sets the model init, the augmentation and the data order. `torch.manual_seed(seed)` runs before the model is built and again afterwards, as in the research trainer.
- All checkpoints are written atomically (tmp file + `os.replace`). `resume.pt` is saved at every epoch boundary, and the previous one is kept as `resume.prev.pt`. It contains:
  - model, optimizer, scheduler, selection and stopping state;
  - Python/torch/CUDA/NumPy RNG states and the wall time;
  - the arguments, vocabulary, tokenizer hash, code version and data signature.
- `--resume auto` continues from `resume.pt`. It falls back to `resume.prev.pt` if the newest file is corrupt, and starts fresh if neither exists.
- If free GPU memory drops below `--min-free-gib` on a shared GPU, the run exits, and the next `--resume` repeats that epoch.
- Resume compatibility: the run stops and lists every problem if any of the following differ from the saved run.
  - An argument other than `resume`, `max_hours`, `min_free_gib`, `config`, `out` or `init`. Changing `epochs`, for example, would silently change the LR schedule.
  - The vocabulary (`itos`), the reader config or the BPE model hash.
  - The data signature: sha256 of every index file plus the size of every referenced `features.bin`. Feature bytes are not hashed at resume time (hundreds of GB); the crc32 values from `prepare_features.py` cover their integrity.
  - The code version: a hash of `train.py`, `datasets.py`, `reader.py`, `tokenizer.py` and `ctc.py`. `--resume-allow-code-change` accepts a new version explicitly.
  - RNG states that are missing or come from another device type.
  - Optimizer layout, `initial_lr`, weight decay or state shapes.
  - A scheduler whose `last_epoch` differs from the saved step, or whose `base_lrs` differ from `--lr`.
- `--init CKPT` warm-starts a new run. It copies every non-head tensor (the reader shape must match) and maps output-head rows by unit string, so new units keep a fresh init. It reuses the checkpoint's BPE model and starts a fresh optimizer and schedule.

## Resources

- Resident features take about 0.28 GB of GPU memory per audio hour (1536 x bf16 x 25 fps); `--pool-chunks K` divides the per-epoch footprint by about K.
- Software: ROCm 7.1.1, PyTorch 2.10.0. The code is plain PyTorch and also runs on CUDA; the tests use the CPU path.
