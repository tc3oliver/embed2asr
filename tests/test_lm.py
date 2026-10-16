import numpy as np
import torch
from conftest import StubEncoder

from embed2asr import cli
from embed2asr.lm import CharLM, beam_decode, lm_decoder, lm_text, lm_units
from embed2asr.pipeline import Embed2ASR
from embed2asr.tokenizer import Tok

ITOS = ["<blank>", "你", "好", "壞", "▁the", "▁cat"]
TOK = Tok(ITOS)
TOK.sp = object()  # BPE-style decode: "▁" -> space; only the flag matters here


def _lp(*rows):  # one-hot-ish frames over ITOS (blank=0), log-probs
    x = np.full((len(rows), len(ITOS)), 1e-3)
    for t, r in enumerate(rows):
        for u, p in r.items():
            x[t, u] = p
    return np.log(x / x.sum(1, keepdims=True))


def _lm(sents, order=3):
    return CharLM([lm_text(s) for s in sents], order)


def test_units_only_cjk_scored():
    assert lm_units(ITOS) == [None, "你", "好", "壞", None, None]
    assert lm_text("你好 the cat 好") == "你好\x04好"


def test_lm_flips_acoustically_ambiguous_char():
    lm = _lm(["你好"] * 30)
    lp = _lp({1: 0.9}, {2: 0.45, 3: 0.5})  # acoustics prefer 壞, LM knows 你好
    assert beam_decode(lp, TOK, lm, 0.0, 0.0) == "你壞"
    assert beam_decode(lp, TOK, lm, 1.0, 0.0) == "你好"


def test_english_units_are_lm_neutral():
    lm = _lm(["你好"] * 30)
    lp = _lp({4: 0.9}, {5: 0.9})
    assert beam_decode(lp, TOK, lm, 1.0, 0.0) == beam_decode(lp, TOK, lm, 0.0, 0.0) == "the cat"


def test_save_load_roundtrip(tmp_path):
    lm = _lm(["你好", "你壞好"])
    sha = lm.save(tmp_path / "a.gz")
    assert sha == lm.save(tmp_path / "b.gz")  # deterministic
    lm2 = CharLM.load(tmp_path / "a.gz")
    assert lm.logp("你", "好") == lm2.logp("你", "好")


def test_decoder_batch_respects_lengths():
    lm = _lm(["你好"] * 30)
    lp = torch.from_numpy(np.stack([_lp({1: 0.9}, {2: 0.9}), _lp({1: 0.9}, {2: 0.9})]))
    out = lm_decoder(lm, 0.5, 1.0, 8)(lp, torch.tensor([2, 1]), TOK)
    assert out == ["你好", "你"]


def test_cli_lm_matches_pipeline(tiny_release, wav_file, tmp_path, capsys, bpe_tok):
    lm_file = tmp_path / "lm.json.gz"
    _lm(["你好"] * 5).save(lm_file)
    asr = Embed2ASR.from_pretrained(tiny_release, device="cpu", encoder=StubEncoder(), lm=lm_file, beam=4)
    want = asr.transcribe(wav_file).text
    assert (
        cli.main(
            ["transcribe", str(wav_file), "--model", str(tiny_release), "--device", "cpu", "--lm", str(lm_file), "--beam", "4"],
            StubEncoder(),
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == want
