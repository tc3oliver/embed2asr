"""Output units for Embed2ASR. char: one unit per character. bpe: every non-ASCII char (CJK) is its own unit, ASCII runs (English words+spaces)
are SentencePiece BPE pieces -> shared zh-char + en-subword vocabulary that also covers intra-sentence code-switching."""

import re
from pathlib import Path

_RUN = re.compile(r"[\x00-\x7f]+|[^\x00-\x7f]")
# Known limitation (logs/check_bpe.log): Latin-extended letters (ç ü ñ) are single non-ASCII units and every ASCII run gets a
# SentencePiece dummy "▁", so word boundaries around them are lost ("erkoç and" -> "erkoçand"; rare in English text).
# Fix = ASCII-fold Latin diacritics in English text normalization (vocab-compatible), applied with the numnorm round.
_GAP = re.compile(r"(?<=[^\x00-\x7f]) | (?=[^\x00-\x7f])")  # spaces next to non-ASCII carry no information


class Tok:
    def __init__(self, itos, sp_bytes=None):
        self.itos, self.stoi, self.sp_bytes = itos, {c: i for i, c in enumerate(itos)}, sp_bytes
        self.sp = None
        if sp_bytes:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor(model_proto=sp_bytes)

    @classmethod
    def build(cls, mode, texts, out: Path, bpe_size=400):
        chars = sorted({c for t in texts for c in t})
        if mode == "char":
            return cls(["<blank>"] + chars)
        import sentencepiece as spm

        out.mkdir(parents=True, exist_ok=True)
        mf = out / "bpe.model"
        if not mf.exists():
            en = out / "bpe_train.txt"
            en.write_text("\n".join(s for t in texts for s in re.findall(r"[\x00-\x7f]+", t) if s.strip()))
            spm.SentencePieceTrainer.train(
                input=str(en),
                model_prefix=str(out / "bpe"),
                vocab_size=bpe_size,
                model_type="bpe",
                character_coverage=1.0,
                bos_id=-1,
                eos_id=-1,
                unk_id=0,
                input_sentence_size=2_000_000,
                shuffle_input_sentence=True,
            )
        b = mf.read_bytes()
        sp = spm.SentencePieceProcessor(model_proto=b)
        return cls(["<blank>"] + [c for c in chars if ord(c) > 127] + [sp.id_to_piece(i) for i in range(1, sp.get_piece_size())], b)

    def encode(self, text):
        if self.sp is None:
            return [self.stoi[c] for c in text if c in self.stoi]
        ids = []
        for seg in _RUN.findall(text):
            ids += (
                [self.stoi[p] for p in self.sp.encode(seg, out_type=str) if p in self.stoi]
                if seg.isascii()
                else ([self.stoi[seg]] if seg in self.stoi else [])
            )
        return ids

    def decode(self, ids):
        s = "".join(self.itos[i] for i in ids)
        return _GAP.sub("", " ".join(s.replace("▁", " ").split())) if self.sp else s
