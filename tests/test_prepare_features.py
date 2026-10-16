"""Feature-cache resume validation (no encoder needed): interrupted writes are repaired, anything else is refused."""

import json
import zlib

import pytest

from training.prepare_features import ROW, CacheError, append_row, open_cache

MANIFEST = [{"audio": f"a{i}.wav", "text": "x"} for i in range(6)]
META = {"encoder": "e", "revision": "r1", "dtype": "bfloat16", "dim": 1536, "manifest_sha256": "m1"}


def build(out, ids=(0, 1, 2), frames=3):
    open_cache(out, META, MANIFEST)
    with (out / "features.bin").open("ab") as fb, (out / "index.jsonl").open("a", encoding="utf-8") as fi:
        for i in ids:
            rec = append_row(
                fb, {"i": i, "frames": frames, "seconds": 1.0, "text": "x", "audio": MANIFEST[i]["audio"]}, bytes([i]) * frames * ROW
            )
            fi.write(json.dumps(rec) + "\n")
    return out


def test_fresh_and_clean_resume(tmp_path):
    assert open_cache(tmp_path / "c", META, MANIFEST) == set()
    assert open_cache(build(tmp_path / "d"), META, MANIFEST, verify_all=True) == {0, 1, 2}


def test_interrupted_write_is_repaired(tmp_path):
    out = build(tmp_path / "c")
    end = (out / "features.bin").stat().st_size
    with (out / "features.bin").open("ab") as f:
        f.write(b"\1" * (ROW + 7))  # features of a batch whose index lines never landed (and a torn frame)
    with (out / "index.jsonl").open("a") as f:
        f.write('{"i": 3, "path": "featu')  # torn last line
    assert open_cache(out, META, MANIFEST) == {0, 1, 2}
    assert (out / "features.bin").stat().st_size == end and (out / "index.jsonl").read_bytes().endswith(b"}\n")
    build(out, ids=(3,))  # appending afterwards gives a valid cache again
    assert open_cache(out, META, MANIFEST, verify_all=True) == {0, 1, 2, 3}


@pytest.mark.parametrize(
    "damage",
    [
        "corrupt_middle_line",
        "truncated_features",
        "flipped_feature_byte",
        "offset_gap",
        "manifest_changed",
        "manifest_audio_changed",
        "revision_changed",
        "no_cache_json",
    ],
)
def test_refuses_damaged_or_foreign_cache(tmp_path, damage):
    out = build(tmp_path / "c")
    idx, fb = out / "index.jsonl", out / "features.bin"
    lines = idx.read_text().splitlines(keepends=True)
    meta, manifest = META, MANIFEST
    if damage == "corrupt_middle_line":
        idx.write_text(lines[0] + '{"i": 1, garbage\n' + lines[2])
    elif damage == "truncated_features":
        with fb.open("r+b") as f:
            f.truncate(fb.stat().st_size - 1)
    elif damage == "flipped_feature_byte":
        with fb.open("r+b") as f:
            f.seek(2 * 3 * ROW + 5)
            f.write(b"\xff")
    elif damage == "offset_gap":
        r = json.loads(lines[2])
        r["off"] += ROW
        lines[2] = json.dumps(r) + "\n"
        idx.write_text("".join(lines))
    elif damage == "manifest_changed":
        meta = {**META, "manifest_sha256": "m2"}
    elif damage == "manifest_audio_changed":
        manifest = [{**MANIFEST[0], "audio": "other.wav"}, *MANIFEST[1:]]
    elif damage == "revision_changed":
        meta = {**META, "revision": "r2"}
    elif damage == "no_cache_json":
        (out / "cache.json").unlink()
    before = (idx.read_bytes(), fb.read_bytes())
    with pytest.raises(CacheError):
        open_cache(out, meta, manifest, verify_all=True)
    assert (idx.read_bytes(), fb.read_bytes()) == before, "a refused cache must not be modified"


def test_crc_recorded(tmp_path):
    out = build(tmp_path / "c", ids=(4,), frames=2)
    r = json.loads((out / "index.jsonl").read_text())
    assert r["off"] == 0 and r["crc32"] == zlib.crc32(bytes([4]) * 2 * ROW)
