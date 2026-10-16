"""End-to-end inference: audio -> frozen audio encoder -> GRU-CTC reader -> greedy CTC -> text.

Release directory layout: config.json, model.safetensors (reader weights, fp32), units.json (itos), tokenizer_bpe.model
(SentencePiece; absent for char-only vocabularies). Only safetensors is read here; research *.pt checkpoints are converted
once with tools/export_release.py (torch.load(weights_only=True)), never unpickled at inference time."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .audio import SAMPLE_RATE, load_audio
from .ctc import collapse_ctc
from .encoder import ENCODER_ID, ENCODER_REVISION, FRAME_RATE, default_device
from .reader import Reader
from .tokenizer import Tok

FORMAT = "embed2asr"
FORMAT_VERSION = 1
BLANK_ID = 0
CHUNK_SECONDS = 29.5  # encoder feature extractor truncates at 30 s
MIN_CHUNK_SAMPLES = 1600  # 0.1 s; shorter tail chunks are dropped (as in research transcribe.py)
REQUIRED_READER_KEYS = ("in_dim", "vocab_size", "d", "gru", "blocks")


def greedy(logits: torch.Tensor, lengths: torch.Tensor, tok: Tok) -> list[str]:
    ids = logits.argmax(-1).cpu().tolist()
    return [tok.decode(collapse_ctc(s[:n], BLANK_ID)) for s, n in zip(ids, lengths.tolist())]


# name -> fn(logits [B,T,V], lengths [B], tok) -> list[str]. The LM decoder is built per instance (see Embed2ASR.__init__).
DECODERS = {"greedy": greedy}


@dataclass
class Chunk:
    start: float  # seconds
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    duration: float  # audio seconds
    chunks: list[Chunk] = field(default_factory=list)


def make_config(reader_cfg: dict, vocab_size: int, has_sp: bool, encoder_revision: str = ENCODER_REVISION) -> dict:
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "reader": reader_cfg,
        "vocab_size": vocab_size,
        "blank_id": BLANK_ID,
        "sample_rate": SAMPLE_RATE,
        "frame_rate_hz": FRAME_RATE,
        "chunk_seconds": CHUNK_SECONDS,
        "encoder": {
            "id": ENCODER_ID,
            "revision": encoder_revision,
            "module": "audio_tower",
            "feature": "last_hidden_state",
            "feature_dim": reader_cfg["in_dim"],  # FEATURE_DIM for the real encoder
            "dtype": "bfloat16 on GPU, float32 on CPU; reader runs in float32",
        },
        "tokenizer": {"units": "units.json", "sentencepiece": "tokenizer_bpe.model" if has_sp else None},
    }


def validate_config(cfg: dict, n_units: int | None = None) -> None:
    def need(ok, msg):
        if not ok:
            raise ValueError(f"invalid embed2asr config: {msg}")

    need(cfg.get("format") == FORMAT, f"format must be {FORMAT!r}")
    need(cfg.get("format_version") == FORMAT_VERSION, f"unsupported format_version {cfg.get('format_version')!r}")
    r = cfg.get("reader")
    need(isinstance(r, dict) and all(k in r for k in REQUIRED_READER_KEYS), f"reader needs {REQUIRED_READER_KEYS}")
    need(r["vocab_size"] == cfg.get("vocab_size"), "reader.vocab_size != vocab_size")
    need(cfg.get("blank_id") == BLANK_ID, "blank_id must be 0")
    need(r["in_dim"] == cfg.get("encoder", {}).get("feature_dim"), "reader.in_dim != encoder.feature_dim")
    need(n_units is None or n_units == cfg["vocab_size"], f"units.json has {n_units} units, config says {cfg['vocab_size']}")


def save_release(out: Path, state_dict: dict, reader_cfg: dict, itos: list[str], sp_bytes: bytes | None) -> dict:
    """Write config.json, model.safetensors, units.json (+ tokenizer_bpe.model). Returns the config."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_config(reader_cfg, len(itos), bool(sp_bytes))
    validate_config(cfg, len(itos))
    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "units.json").write_text(json.dumps(itos, ensure_ascii=False), encoding="utf-8")
    if sp_bytes:
        (out / "tokenizer_bpe.model").write_bytes(sp_bytes)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return cfg


def load_release(path: Path):
    """-> (state_dict, reader_cfg, itos, sp_bytes, config) from a release directory."""
    path = Path(path)
    if not (path / "config.json").is_file():
        raise ValueError(f"{path} is not an embed2asr release directory (no config.json); convert *.pt with tools/export_release.py")
    from safetensors.torch import load_file

    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    itos = json.loads((path / cfg.get("tokenizer", {}).get("units", "units.json")).read_text(encoding="utf-8"))
    validate_config(cfg, len(itos))
    spf = cfg["tokenizer"].get("sentencepiece")
    sp = (path / spf).read_bytes() if spf else None
    return load_file(str(path / "model.safetensors")), cfg["reader"], itos, sp, cfg


def resolve_release(path_or_repo: str | os.PathLike, revision: str | None = None) -> Path:
    """A local release directory as is; anything else is treated as a Hugging Face repo id and downloaded (cached)."""
    path = Path(path_or_repo)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo), revision=revision))


class Embed2ASR:
    def __init__(self, reader: Reader, tok: Tok, encoder, config: dict, device=None, decoder: str = "greedy", lm=None):
        if decoder not in DECODERS:
            raise ValueError(f"unknown decoder {decoder!r}; available: {sorted(DECODERS)}")
        self.reader, self.tok, self.encoder, self.config = reader, tok, encoder, config
        self.device = torch.device(device or default_device())
        self.decode = DECODERS[decoder]
        if lm is not None:  # (CharLM, alpha, beta, beam)
            from .lm import lm_decoder

            self.decode = lm_decoder(*lm)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | os.PathLike,
        device=None,
        encoder_revision: str | None = None,
        encoder=None,
        revision: str | None = None,
        decoder: str = "greedy",
        lm: str | os.PathLike | None = None,
        lm_alpha: float = 0.25,
        lm_beta: float = 1.5,
        beam: int = 16,
    ) -> Embed2ASR:
        """`path_or_repo`: release directory or a Hugging Face repo id (downloaded with `revision`).
        `lm`: character n-gram LM file (a path, or relative to the release directory, e.g. "lm/<name>.charlm.json.gz");
        switches greedy CTC to prefix beam search with LM weight `lm_alpha`, per-char bonus `lm_beta`, `beam` width.
        `encoder`: inject any callable(wavs) -> (features, lengths) instead of loading EmbeddingGemma 2 (tests, stubs)."""
        path = resolve_release(path_or_repo, revision)
        sd, rcfg, itos, sp, cfg = load_release(path)
        device = torch.device(device or default_device())
        reader = Reader(**rcfg)
        reader.load_state_dict(sd)  # strict: architecture must match exactly
        reader.to(device).eval().requires_grad_(False)
        if encoder is None:
            from .encoder import FrozenAudioEncoder

            enc = cfg["encoder"]
            encoder = FrozenAudioEncoder(enc["id"], encoder_revision or enc["revision"], device=device)
        lm_args = None
        if lm is not None:
            from .lm import CharLM

            lm_path = Path(lm) if Path(lm).is_file() else path / lm
            lm_args = (CharLM.load(lm_path), lm_alpha, lm_beta, beam)
        return cls(reader, Tok(itos, sp), encoder, cfg, device, decoder, lm_args)

    @torch.inference_mode()
    def transcribe(self, audio, sr: int | None = None, batch_size: int = 16, chunk_seconds: float | None = None, cross_file: bool = False):
        """audio: path, float array (needs `sr`), or a list of those. Returns a Transcript (or a list for list input).
        Audio longer than chunk_seconds is cut into consecutive fixed-length chunks (no overlap, no VAD: a word on a
        boundary can be split); chunk texts are joined with "" for CJK-only output, else with " " (as research transcribe.py).
        Batching: by default each input is encoded on its own (its chunks batched up to batch_size), which reproduces
        research transcribe.py. cross_file=True packs chunks of all inputs into length-sorted batches: ~4x throughput on
        short clips, but the encoder is not padding-invariant, so ~4% of short clips change by ~1 unit (corpus CER/WER
        within 0.1 pt on the dev sets)."""
        single = not isinstance(audio, (list, tuple))
        wavs = [load_audio(a, sr) for a in ([audio] if single else audio)]
        n = int((chunk_seconds or self.config.get("chunk_seconds", CHUNK_SECONDS)) * SAMPLE_RATE)
        pieces = [(k, i) for k, w in enumerate(wavs) for i in range(0, len(w), n) if len(w[i : i + n]) >= MIN_CHUNK_SAMPLES]
        segs = [wavs[k][i : i + n] for k, i in pieces]
        if cross_file:
            order = sorted(range(len(segs)), key=lambda j: len(segs[j]))  # length-sorted: less padding
            batches = [order[b : b + batch_size] for b in range(0, len(order), batch_size)]
        else:
            per = [[j for j, (k, _) in enumerate(pieces) if k == f] for f in range(len(wavs))]
            batches = [p[b : b + batch_size] for p in per for b in range(0, len(p), batch_size)]
        texts = [""] * len(segs)
        for idx in batches:
            feats, lens = self.encoder([segs[j] for j in idx])
            for j, t in zip(idx, self.decode(self.reader(feats.to(self.device).float(), lens), lens, self.tok)):
                texts[j] = t
        out = [Transcript("", len(w) / SAMPLE_RATE) for w in wavs]
        for (k, i), t in zip(pieces, texts):
            out[k].chunks.append(Chunk(i / SAMPLE_RATE, min(i + n, len(wavs[k])) / SAMPLE_RATE, t))
        for t in out:
            parts = [c.text for c in t.chunks]
            t.text = " ".join(parts) if " " in "".join(parts) else "".join(parts)
        return out[0] if single else out
