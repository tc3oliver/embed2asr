#!/usr/bin/env python3
"""Convert a frozen research release (reader.pt + units.json + tokenizer_bpe.model) into an embed2asr release directory:
model.safetensors, config.json, units.json, tokenizer_bpe.model, manifest.json, SHA256SUMS.
Verifies the exported tensors are bit-identical to reader.pt and that units/tokenizer match the checkpoint.
usage: python tools/export_release.py SRC_DIR OUT_DIR   (OUT_DIR must not exist)"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from embed2asr import __version__
from embed2asr.encoder import ENCODER_ID, ENCODER_REVISION
from embed2asr.pipeline import save_release


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def audio_tower_params(revision: str) -> int:
    """nn.Parameter count of the audio tower, built on the meta device from the pinned config (no weights loaded).
    The checkpoint's audio_tower.* tensors hold 480 more elements: scalar clipping buffers (input/output min/max)."""
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(ENCODER_ID, revision=revision).audio_config
    with torch.device("meta"):
        return sum(p.numel() for p in AutoModel.from_config(cfg).parameters())


def export(src: Path, out: Path, encoder_params: int | None = None) -> dict:
    """reader.pt (+ units.json, tokenizer_bpe.model) in `src` -> release directory `out`. Returns the manifest.
    `encoder_params` skips the (network/cache) audio-tower count, for tests."""
    src, out = Path(src), Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    ck_path = src / "reader.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)  # tensors + plain containers only, no arbitrary pickle
    itos, sp = ck["itos"], ck.get("tok", {}).get("sp")
    assert itos == json.loads((src / "units.json").read_text(encoding="utf-8")), "units.json != checkpoint itos"
    if sp:
        assert sp == (src / "tokenizer_bpe.model").read_bytes(), "tokenizer_bpe.model != checkpoint SentencePiece model"

    save_release(out, ck["model"], ck["cfg"], itos, sp)
    re = load_file(str(out / "model.safetensors"))
    assert re.keys() == ck["model"].keys()
    for k, v in ck["model"].items():
        assert re[k].dtype == v.dtype and re[k].shape == v.shape and torch.equal(re[k], v), f"tensor mismatch: {k}"

    reader_n = sum(v.numel() for v in ck["model"].values())
    enc_n = audio_tower_params(ENCODER_REVISION) if encoder_params is None else encoder_params
    files = sorted(f.name for f in out.iterdir())
    manifest = {
        "name": out.name,
        "embed2asr_version": __version__,
        "source_checkpoint": {"file": ck_path.name, "sha256": sha256(ck_path), "epoch": ck.get("epoch")},
        "encoder": {"id": ENCODER_ID, "revision": ENCODER_REVISION, "module": "audio_tower"},
        "params": {"reader_trainable": reader_n, "audio_encoder_frozen": enc_n, "total_loaded_for_inference": reader_n + enc_n},
        "verified_bit_identical_tensors": len(re),
        "files": {f: sha256(out / f) for f in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "SHA256SUMS").write_text("".join(f"{sha256(out / f)}  {f}\n" for f in sorted(files + ["manifest.json"])))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    m = export(a.src, a.out)
    print(json.dumps(m["params"]), f"-> {a.out}")


if __name__ == "__main__":
    main()
