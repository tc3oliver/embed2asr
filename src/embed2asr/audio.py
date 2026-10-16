"""Audio loading: any file/array -> float32 mono 16 kHz.

Files are decoded with ffmpeg (the binary from `imageio-ffmpeg` if installed - that is what the research benchmarks used -
else the system `ffmpeg`), falling back to soundfile + scipy.signal.resample_poly when no ffmpeg is available.
Multi-channel: the fallback (and array input) averages channels; ffmpeg `-ac 1` mixes stereo as (L+R)/sqrt(2), i.e. +3 dB.
The reference models were evaluated on mono files, where both paths agree up to resampler differences."""

from __future__ import annotations

import os
import shutil
import subprocess
from fractions import Fraction

import numpy as np

SAMPLE_RATE = 16000


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return shutil.which("ffmpeg")


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return x
    from scipy.signal import resample_poly

    f = Fraction(SAMPLE_RATE, int(sr))
    return resample_poly(x, f.numerator, f.denominator).astype(np.float32)


def _check(a: np.ndarray, what: str) -> np.ndarray:
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"invalid audio (empty or non-finite samples): {what}")
    return a


def _from_array(a: np.ndarray, sr: int | None) -> np.ndarray:
    if sr is None:
        raise ValueError("sr is required when passing a raw array")
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:  # (frames, channels), as returned by soundfile
        a = a.mean(axis=1)
    elif a.ndim != 1:
        raise ValueError(f"expected 1-D or (frames, channels) audio, got shape {a.shape}")
    return _check(_resample(a, sr), "array")


def _ffmpeg(exe: str, path: str) -> np.ndarray:
    cmd = [exe, "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise ValueError(f"ffmpeg could not decode {path}: {p.stderr.decode(errors='replace').strip()[-500:]}")
    return np.frombuffer(p.stdout, dtype="<f4").copy()


def _soundfile(path: str) -> np.ndarray:
    import soundfile as sf

    try:
        a, sr = sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError as e:  # unsupported container/codec or corrupt file
        raise ValueError(f"soundfile could not decode {path} ({e}); install ffmpeg or imageio-ffmpeg for more formats") from e
    return _resample(a.mean(axis=1), sr)


def load_audio(src: str | os.PathLike | np.ndarray, sr: int | None = None) -> np.ndarray:
    """Return float32 mono 16 kHz samples. `sr` is required for arrays and ignored for files."""
    if isinstance(src, np.ndarray):
        return _from_array(src, sr)
    path = os.fspath(src)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    exe = ffmpeg_exe()
    return _check(_ffmpeg(exe, path) if exe else _soundfile(path), path)
