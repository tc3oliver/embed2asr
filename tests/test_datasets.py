import json
import random

from training.datasets import RotateSampler, bucket_batches, load_index, split_chunks


def research_rot_rows(pools, seed, ep):
    """Verbatim logic of the research trainer's rot_rows (scripts/train_fast.py) -> per-epoch row ids per pool."""
    orders, out = {}, []
    for k, (pr, want) in enumerate(pools):
        n = pos = 0
        for _ in range(ep):
            take, seen, h = [], set(), 0.0
            while h < want and len(take) < len(pr):
                if (k, n) not in orders:
                    o = list(range(len(pr)))
                    random.Random(seed * 1009 + k * 101 + n).shuffle(o)
                    orders[k, n] = o
                o = orders[k, n]
                if pos >= len(o):
                    n, pos = n + 1, 0
                    continue
                i = o[pos]
                pos += 1
                if i in seen:
                    continue
                take.append(i)
                seen.add(i)
                h += pr[i]["seconds"] / 3600
        out += [pr[i]["id"] for i in take]
    return out


def make_pools(seed=0):
    r = random.Random(seed)
    pools = []
    for k, (n, hours) in enumerate([(50, 0.02), (13, 0.05), (7, 0.004)]):  # pool 1/2: epoch wants more/less than a pass
        pools.append(([{"id": f"{k}-{i}", "seconds": r.uniform(1, 20), "frames": r.randint(10, 500)} for i in range(n)], hours))
    return pools


def test_rotate_matches_research_and_resumes():
    pools = make_pools()
    s = RotateSampler(pools, seed=42)
    for ep in range(1, 9):
        rows, cov = s.epoch(ep)
        ids = [r["id"] for r in rows]
        assert ids == research_rot_rows(pools, 42, ep)
        assert len(ids) == len(set(ids)), "no row twice within an epoch"
    fresh = RotateSampler(pools, seed=42)  # resume = new process, jump straight to epoch 6
    assert fresh.epoch(6) == RotateSampler(pools, seed=42).epoch(6) == s.epoch(6)
    assert RotateSampler(pools, seed=43).epoch(1)[0] != s.epoch(1)[0]


def test_rotate_hours_and_coverage():
    pools = make_pools()
    s = RotateSampler(pools, seed=1)
    rows, cov = s.epoch(1)
    for k, (pr, want) in enumerate(pools):
        got = [r for r in rows if r["id"].startswith(f"{k}-")]
        h = sum(r["seconds"] for r in got) / 3600
        assert h >= want or len(got) == len(pr)  # stops once HOURS are reached, or the whole pool is used
    assert cov[1] == 1.0 and cov[0] < 1  # pool 1 (13 rows, 0.05 h) is used in full every epoch


def test_bucket_batches():
    rows = [{"frames": random.Random(i).randint(1, 999)} for i in range(1000)]
    a, b = bucket_batches(rows, 16, random.Random(3)), bucket_batches(rows, 16, random.Random(3))
    assert a == b
    flat = [i for x in a for i in x]
    assert sorted(flat) == list(range(1000)) and all(len(x) <= 16 for x in a)
    assert all([rows[i]["frames"] for i in x] == sorted(rows[i]["frames"] for i in x) for x in a)


def test_split_chunks():
    rows = list(range(10))
    assert split_chunks(rows, 1, random.Random(0)) == [rows]
    parts = split_chunks(rows, 3, random.Random(0))
    assert sorted(sum(parts, [])) == rows and len(parts) == 3


def test_load_index(tmp_path):
    (tmp_path / "features.bin").write_bytes(b"")
    (tmp_path / "index.jsonl").write_text(
        json.dumps({"path": "features.bin", "off": 0, "frames": 1, "text": "臺北"}) + "\n", encoding="utf-8"
    )
    [r] = load_index([tmp_path / "index.jsonl"])
    assert r["text"] == "台北" and r["path"] == str(tmp_path / "features.bin")
