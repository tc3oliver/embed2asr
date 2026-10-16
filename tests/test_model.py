import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import TINY, StubEncoder

from embed2asr import cli
from embed2asr.encoder import FEATURE_DIM
from embed2asr.pipeline import Embed2ASR, load_release, make_config, validate_config
from embed2asr.reader import Reader


def _export_tool():
    spec = importlib.util.spec_from_file_location("export_release", Path(__file__).parents[1] / "tools" / "export_release.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reader_shape_and_research_checkpoint_export(tmp_path, bpe_tok):
    torch.manual_seed(0)
    r = Reader(vocab_size=len(bpe_tok.itos), **TINY).eval()
    x, lens = torch.randn(2, 7, TINY["in_dim"]), torch.tensor([7, 4])
    y = r(x, lens)
    assert y.shape == (2, 7, len(bpe_tok.itos))
    src = tmp_path / "research"
    src.mkdir()
    ck = {"model": r.state_dict(), "itos": bpe_tok.itos, "cfg": r.cfg, "tok": {"mode": "bpe", "sp": bpe_tok.sp_bytes}, "epoch": 3}
    torch.save(ck, src / "reader.pt")
    (src / "units.json").write_text(json.dumps(bpe_tok.itos, ensure_ascii=False), encoding="utf-8")
    (src / "tokenizer_bpe.model").write_bytes(bpe_tok.sp_bytes)
    m = _export_tool().export(src, tmp_path / "rel", encoder_params=123)
    assert m["params"] == {
        "reader_trainable": sum(v.numel() for v in r.parameters()),
        "audio_encoder_frozen": 123,
        "total_loaded_for_inference": sum(v.numel() for v in r.parameters()) + 123,
    }
    for line in (tmp_path / "rel" / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((tmp_path / "rel" / name).read_bytes()).hexdigest() == digest
    sd, cfg, itos, sp, _ = load_release(tmp_path / "rel")
    r2 = Reader(**cfg).eval()
    r2.load_state_dict(sd)  # strict
    assert torch.equal(r2(x, lens), y) and sp == bpe_tok.sp_bytes and itos == bpe_tok.itos
    with pytest.raises(ValueError, match="export_release"):
        load_release(src / "reader.pt")  # inference never unpickles research checkpoints
    assert Reader(**{**cfg, "drop_gru": 0.1}).state_dict().keys() != sd.keys()  # rnn1/rnn2 variant differs


def test_release_architecture_numbers():
    """v0.1.0 release reader: 9,655,172 params, 4,996 units, blank 0, 1536-d input (values from release/config.json)."""
    rc = {"in_dim": 1536, "vocab_size": 4996, "d": 512, "gru": 256, "blocks": 3, "dropout": 0.0, "drop_block": 0.0, "drop_gru": 0.0}
    assert sum(p.numel() for p in Reader(**rc).parameters()) == 9_655_172
    cfg = make_config(rc, 4996, True)
    validate_config(cfg, 4996)
    assert cfg["blank_id"] == 0 and cfg["frame_rate_hz"] == 25 and cfg["encoder"]["feature_dim"] == FEATURE_DIM == 1536
    assert cfg["encoder"]["revision"] == "914f7f89142e33e77833254d9c9b90c3cef7303b"


def test_release_roundtrip_bit_identical(tiny_release):
    from safetensors.torch import load_file

    sd = load_file(str(tiny_release / "model.safetensors"))
    r = Reader(**json.loads((tiny_release / "config.json").read_text())["reader"])
    r.load_state_dict(sd)
    assert all(torch.equal(sd[k], v) for k, v in r.state_dict().items())


def test_pipeline_cpu_stub(tiny_release, wav_file):
    asr = Embed2ASR.from_pretrained(tiny_release, device="cpu", encoder=StubEncoder())
    one = asr.transcribe(wav_file)
    assert isinstance(one.text, str) and abs(one.duration - 1.5) < 1e-3 and len(one.chunks) == 1
    many = asr.transcribe([wav_file, np.zeros(16000 * 5, np.float32)], sr=16000, chunk_seconds=2.0)
    assert [len(t.chunks) for t in many] == [1, 3] and many[1].chunks[-1].end == 5.0
    packed = asr.transcribe([wav_file, np.zeros(16000 * 5, np.float32)], sr=16000, chunk_seconds=2.0, cross_file=True, batch_size=2)
    assert [[(c.start, c.end) for c in t.chunks] for t in packed] == [[(c.start, c.end) for c in t.chunks] for t in many]
    assert asr.transcribe(np.zeros(800, np.float32), sr=16000).chunks == []  # < 0.1 s: nothing to decode
    with pytest.raises(ValueError, match="unknown decoder"):
        Embed2ASR.from_pretrained(tiny_release, device="cpu", encoder=StubEncoder(), decoder="beam")


def test_config_validation():
    good = make_config({"in_dim": 1536, "vocab_size": 5, "d": 8, "gru": 4, "blocks": 1}, 5, False)
    validate_config(good, 5)
    bad = [
        ({**good, "format": "x"}, "format"),
        ({**good, "blank_id": 3}, "blank_id"),
        ({**good, "vocab_size": 6}, "vocab_size"),
        ({**good, "reader": {"in_dim": 1536}}, "reader needs"),
        ({**good, "encoder": {**good["encoder"], "feature_dim": 1024}}, "in_dim"),
    ]
    for cfg, msg in bad:
        with pytest.raises(ValueError, match=msg):
            validate_config(cfg)
    with pytest.raises(ValueError, match="units.json"):
        validate_config(good, 4)


def test_cli(tiny_release, wav_file, capsys, tmp_path):
    assert cli.main(["transcribe", str(wav_file), "--model", str(tiny_release), "--device", "cpu", "--json"], encoder=StubEncoder()) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["audio"] == str(wav_file) and {"text", "duration", "chunks", "rtf"} <= out.keys()
    assert cli.main(["transcribe", str(tmp_path / "nope.wav"), "--model", str(tiny_release), "--device", "cpu"], encoder=StubEncoder()) == 1
    assert cli.main(["info", "--model", str(tiny_release)]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["format"] == "embed2asr"


def test_info_accepts_hf_repo_id(tiny_release, capsys, monkeypatch):
    import huggingface_hub

    seen = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision=None: seen.append(repo) or str(tiny_release))
    assert cli.main(["info", "--model", "someone/embed2asr-test"]) == 0
    assert seen == ["someone/embed2asr-test"]
    assert json.loads(capsys.readouterr().out)["units"] > 0


@pytest.mark.slow
def test_real_encoder():
    from embed2asr.encoder import FrozenAudioEncoder

    enc = FrozenAudioEncoder(device="cpu")
    assert type(enc.tower).__name__ == "Gemma4AudioModel"  # audio tower only, not the 744.4M EmbeddingGemma2Model
    assert enc.num_parameters() == 304_824_608
    assert not enc.tower.training and not any(p.requires_grad for p in enc.tower.parameters())
    f, lens = enc([np.zeros(16000, np.float32), np.zeros(32000, np.float32)])
    assert f.dtype == torch.float32 and f.shape == (2, 50, 1536) and lens.tolist() == [25, 50]  # 25 fps
