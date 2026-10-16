import pytest

from embed2asr.ctc import collapse_ctc, ctc_min_len
from embed2asr.metrics import cer_counts, corpus, mer_counts, mer_tokens, score, wer_counts
from embed2asr.normalize import norm
from embed2asr.tokenizer import Tok


def test_bpe_roundtrip(bpe_tok):
    for s in ["hello world", "你好 hello world", "早安the cat"]:
        assert bpe_tok.decode(bpe_tok.encode(norm(s))) == norm(s)
    assert bpe_tok.itos[0] == "<blank>" and 0 not in bpe_tok.encode("hello 你好")


def test_char_roundtrip():
    tok = Tok(["<blank>", "你", "好"])
    assert tok.decode(tok.encode("你好嗎")) == "你好"


def test_ctc():
    assert collapse_ctc([0, 1, 1, 0, 1, 2, 2]) == [1, 1, 2]
    assert collapse_ctc([3, 3, 3], blank=3) == []
    assert ctc_min_len([1, 1, 2]) == 4


def test_norm():
    assert norm("部長 早 。 大型 災後 ，4 月 OK  hello World！") == "部長早大型災後4月ok hello world"
    assert norm("今天  天氣 very good") == "今天天氣very good" and norm("臺灣") == "台灣"


def test_metrics_match_research_selftest():
    assert cer_counts("今天 天氣 很好", "今天天汽很好") == (1, 6)
    assert mer_tokens("我想 buy a 手機 now") == ["我", "想", "buy", "a", "手", "機", "now"]
    assert mer_counts("我想 buy 手機", "我想 by 手機") == (1, 5)
    assert wer_counts("The cat sat.", "the cat sit") == (1, 3)
    assert wer_counts("a b", "") == (2, 2) and wer_counts("", "x y") == (2, 0)
    rows = [
        {"ref": "你好嗎", "hyp": "你好吗", "lang": "zh-TW"},
        {"ref": "hello world", "hyp": "hello word", "lang": "en-US"},
        {"ref": "我 love 你", "hyp": "我 love 你", "lang": "mix"},
    ]
    r = score(rows)
    assert abs(r["zh-TW"]["value"] - 1 / 3) < 1e-9 and r["en-US"]["value"] == 0.5 and r["mix"]["value"] == 0
    assert corpus([("a b c", "a c"), ("d", "e")], "WER") == {"metric": "WER", "value": 0.5, "errors": 2, "ref_units": 4, "utts": 2}


def test_wer_counts_insertions_deletions():
    assert wer_counts("one two three", "one three four five") == (3, 3)  # del two, ins four five (S+D+I minimal)


def test_numnorm_zh():
    from embed2asr.numnorm import norm_nn, zh

    assert zh("2023年10月有2個人")[0] == "二零二三年十月有兩個人"
    assert zh("成長5.8%")[0] == "成長百分之五點八"
    assert zh("9:30 開會") == ("9:30 開會", 2)
    assert norm_nn("共 5 , 000 元", "zh") == "共五千元"


def test_numnorm_en():
    pytest.importorskip("num2words")
    from embed2asr.numnorm import en, norm_nn

    assert en("in 1990 we paid $5 for the 3rd")[0] == "in nineteen ninety we paid five dollars for the third"
    assert norm_nn("It rose 12%", "en") == "it rose twelve percent"
    assert en("COVID-19")[1] == 1


def test_metrics_without_torch():
    import subprocess
    import sys

    code = "import sys, embed2asr.metrics; assert 'torch' not in sys.modules; from embed2asr import Embed2ASR"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cjk_range_excludes_private_use_and_includes_compat_ideographs():
    import re

    from embed2asr.normalize import CJK

    assert re.fullmatch(f"[{CJK}]", "豈") and re.fullmatch(f"[{CJK}]", "你")
    assert not re.fullmatch(f"[{CJK}]", "")
