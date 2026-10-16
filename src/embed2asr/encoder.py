"""Frozen EmbeddingGemma 2 audio encoder: only `audio_tower` is kept (vision tower, language model and embed_* projections
are dropped right after loading). Features = audio_tower(...).last_hidden_state, 1536-d at 25 frames/s, before embed_audio.

Kept module: 304,824,608 parameters (+480 scalar clipping buffers) (google/embeddinggemma-2 @ ENCODER_REVISION; vs ~744M for the full checkpoint).
Nothing is downloaded at import time."""

from __future__ import annotations

import gc

import numpy as np
import torch

ENCODER_ID = "google/embeddinggemma-2"
ENCODER_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
FEATURE_DIM = 1536
FRAME_RATE = 25  # 40 ms per frame


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"  # ROCm builds also report "cuda"


class FrozenAudioEncoder:
    """Callable: list of float32 16 kHz waveforms -> (features [B,T,1536] in `dtype`, lengths [B] int64 on CPU)."""

    def __init__(self, model_id: str = ENCODER_ID, revision: str = ENCODER_REVISION, device: str | None = None, dtype=None):
        from transformers import AutoFeatureExtractor, AutoModel

        self.device = torch.device(device or default_device())
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
        # ponytail: the full checkpoint is materialised on CPU once (~1.5 GB bf16) and everything but audio_tower is freed
        # before moving to the device; load only audio_tower.* tensors if host RAM ever matters.
        full = AutoModel.from_pretrained(model_id, revision=revision, dtype=self.dtype)
        self.tower = full.audio_tower
        del full
        gc.collect()
        self.tower.requires_grad_(False).eval().to(self.device)
        assert not self.tower.training and not any(p.requires_grad for p in self.tower.parameters())
        self.model_id, self.revision = model_id, revision

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.tower.parameters())

    @torch.inference_mode()
    def __call__(self, wavs: list[np.ndarray]):
        inp = self.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt")
        o = self.tower(inp["input_features"].to(self.device, self.dtype), inp["input_features_mask"].to(self.device), return_dict=True)
        return o.last_hidden_state, o.attention_mask.sum(1).cpu()
