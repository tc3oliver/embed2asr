import math

import numpy as np
import pytest
import torch

from embed2asr.pipeline import save_release
from embed2asr.reader import Reader
from embed2asr.tokenizer import Tok

TINY = dict(in_dim=16, d=8, gru=4, blocks=1)


class StubEncoder:
    """Deterministic fake audio encoder: 25 frames/s, TINY in_dim features. No download."""

    def __call__(self, wavs):
        lens = torch.tensor([max(1, math.ceil(len(w) / 640)) for w in wavs])
        g = torch.Generator().manual_seed(0)
        return torch.randn(len(wavs), int(lens.max()), TINY["in_dim"], generator=g), lens


@pytest.fixture(scope="session")
def bpe_tok(tmp_path_factory):
    texts = ["hello world 你好", "the cat sat on the mat", "good morning 早安", "we love speech recognition"] * 20
    return Tok.build("bpe", texts, tmp_path_factory.mktemp("bpe"), bpe_size=40)


@pytest.fixture(scope="session")
def tiny_release(tmp_path_factory, bpe_tok):
    torch.manual_seed(0)
    reader = Reader(vocab_size=len(bpe_tok.itos), **TINY)
    out = tmp_path_factory.mktemp("release")
    save_release(out, reader.state_dict(), reader.cfg, bpe_tok.itos, bpe_tok.sp_bytes)
    return out


@pytest.fixture
def wav_file(tmp_path):
    import soundfile as sf

    sr = 22050
    t = np.arange(int(sr * 1.5)) / sr
    p = tmp_path / "tone.wav"
    sf.write(p, np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1), sr)
    return p
