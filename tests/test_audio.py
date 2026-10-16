import numpy as np
import pytest

import embed2asr.audio as audio
from embed2asr.audio import load_audio


def test_array_resample_and_mono():
    x = np.ones((44100, 2), dtype=np.float64) * 0.5
    y = load_audio(x, sr=44100)
    assert y.dtype == np.float32 and y.ndim == 1 and len(y) == 16000
    assert np.allclose(y[2000:-2000], 0.5, atol=1e-3)
    assert len(load_audio(np.zeros(800, np.float32) + 0.1, sr=16000)) == 800
    with pytest.raises(ValueError):
        load_audio(np.zeros(10), sr=None)
    with pytest.raises(ValueError):
        load_audio(np.array([np.nan]), sr=16000)


@pytest.mark.parametrize("use_ffmpeg", [True, False])
def test_file(wav_file, monkeypatch, use_ffmpeg):
    if use_ffmpeg and audio.ffmpeg_exe() is None:
        pytest.skip("no ffmpeg")
    if not use_ffmpeg:
        monkeypatch.setattr(audio, "ffmpeg_exe", lambda: None)
    y = load_audio(wav_file)
    assert y.dtype == np.float32 and abs(len(y) - 24000) <= 1
    # stereo 0.3 tone: channel mean -> 0.3; ffmpeg -ac 1 downmixes stereo as (L+R)/sqrt(2) -> 0.42
    assert abs(np.abs(y).max() - (0.3 * 2**0.5 if use_ffmpeg else 0.3)) < 0.02


@pytest.mark.parametrize("use_ffmpeg", [True, False])
def test_bad_files(tmp_path, monkeypatch, use_ffmpeg):
    if use_ffmpeg and audio.ffmpeg_exe() is None:
        pytest.skip("no ffmpeg")
    if not use_ffmpeg:
        monkeypatch.setattr(audio, "ffmpeg_exe", lambda: None)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF not really audio")
    with pytest.raises(ValueError):
        load_audio(bad)
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "missing.wav")


def test_ffmpeg_matches_soundfile_on_16k_mono(tmp_path, monkeypatch):
    import soundfile as sf

    if audio.ffmpeg_exe() is None:
        pytest.skip("no ffmpeg")
    x = (0.1 * np.sin(np.arange(16000) / 7)).astype(np.float32)
    sf.write(tmp_path / "m.wav", x, 16000, subtype="FLOAT")
    a = load_audio(tmp_path / "m.wav")
    monkeypatch.setattr(audio, "ffmpeg_exe", lambda: None)
    assert np.array_equal(a, load_audio(tmp_path / "m.wav")) and np.array_equal(a, x)
