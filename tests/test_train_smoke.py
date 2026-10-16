"""Tiny CPU training runs on synthetic features: train, evaluate, select, checkpoint, exact resume, resume refusal, warm start."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from training import smoke_data, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def args(tmp, out, epochs, *extra):
    return [
        "--train",
        tmp["train"],
        "--train-hours",
        "0.005",
        "--pool",
        f"{tmp['pool']}:0.003",
        "--pool-chunks",
        "2",
        "--dev",
        tmp["dev"],
        "--dev-extra",
        f"x={tmp['dev']}",
        "--select",
        "dev_cer:0.9,dev_x_mixmer:0.9",
        "--vocab-mode",
        "char",
        "--d",
        "32",
        "--gru",
        "16",
        "--blocks",
        "1",
        "--batch-size",
        "4",
        "--lr",
        "3e-3",
        "--epochs",
        str(epochs),
        "--eval-every",
        "1",
        "--round-to",
        "8",
        "--device",
        "cpu",
        "--out",
        str(out),
        *extra,
    ]


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    t = tmp_path_factory.mktemp("cache")
    return {
        "train": smoke_data.make_cache(t / "train", 40, 1),
        "pool": smoke_data.make_cache(t / "pool", 30, 2),
        "dev": smoke_data.make_cache(t / "dev", 8, 3),
        "root": t,
    }


def resume_from_epoch2(tmp, src, name, *extra):
    d = tmp["root"] / name
    d.mkdir()
    shutil.copy(src / "resume.prev.pt", d / "resume.pt")
    train.main(args(tmp, d, 3, "--resume", "auto", *extra))
    return d


@pytest.fixture(scope="module")
def trained(tmp):
    a = tmp["root"] / "a"
    train.main(args(tmp, a, 3))
    return a


def test_train_resume_init(tmp, trained):
    a = trained
    log = [json.loads(line) for line in (a / "train_log.jsonl").open()]
    assert [r["epoch"] for r in log] == [1, 2, 3] and all(np.isfinite(r["loss"]) and r["inf_frac"] == 0 for r in log)
    for f in ("best.pt", "last.pt", "best_dev_cer.pt", "best_dev_x_mixmer.pt", "resume.pt", "resume.prev.pt", "config.json"):
        assert (a / f).exists(), f
    # exact resume: continue from the epoch-2 state -> identical epoch-3 weights
    b = resume_from_epoch2(tmp, a, "b")
    wa, wb = torch.load(a / "resume.pt", weights_only=False), torch.load(b / "resume.pt", weights_only=False)
    assert wa["epoch"] == wb["epoch"] == 3 and wa["step"] == wb["step"]
    assert all(torch.equal(wa["model"][k], wb["model"][k]) for k in wa["model"])
    # warm start into a new run (head rows mapped by unit)
    c = tmp["root"] / "c"
    train.main(args(tmp, c, 1, "--init", str(a / "best.pt")))
    assert (c / "best.pt").exists()


@pytest.mark.parametrize("change", ["lr", "epochs", "data", "code", "optimizer", "rng"])
def test_resume_refuses_incompatible(tmp, trained, change):
    a = trained
    d = tmp["root"] / f"r_{change}"
    d.mkdir()
    r = torch.load(a / "resume.prev.pt", weights_only=False)
    extra = []
    if change == "lr":
        extra = ["--lr", "1e-3"]
    elif change == "epochs":
        extra = ["--epochs", "4"]  # would silently change the LR schedule
    elif change == "data":
        r["data_sig"] = {**r["data_sig"], next(iter(r["data_sig"])): "0" * 64}
    elif change == "code":
        r["code"] = "0" * 16
    elif change == "optimizer":
        r["opt"]["param_groups"][0]["weight_decay"] = 0.1
    elif change == "rng":
        r["torch_rng"] = None
    torch.save(r, d / "resume.pt")
    argv = args(tmp, d, 3, "--resume", "auto")
    if change == "epochs":
        argv = args(tmp, d, 4, "--resume", "auto")
    elif extra:
        argv += extra
    with pytest.raises(SystemExit, match="incompatible"):
        train.main(argv)
    if change == "code":  # explicit override is allowed
        train.main(argv + ["--resume-allow-code-change"])


def test_no_graph_packed_path(tmp):
    d = tmp["root"] / "packed"
    train.main(args(tmp, d, 1, "--no-graph"))
    assert np.isfinite(json.loads((d / "train_log.jsonl").read_text())["loss"])


def test_smoke_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, n, seed in (("train", 40, 1), ("pool", 30, 2), ("dev", 8, 3)):
        smoke_data.make_cache(Path("runs/smoke") / name, n, seed)
    train.main(["--config", str(ROOT / "configs/smoke.yaml"), "--device", "cpu", "--epochs", "1"])
    assert Path("runs/smoke/ckpt/best.pt").exists()


def test_recipe_config_needs_env(monkeypatch):
    cfg = str(ROOT / "configs/recipe_v0.1.0.yaml")
    monkeypatch.delenv("E2A_CACHE", raising=False)
    with pytest.raises(SystemExit):
        train.parse_args(["--config", cfg])
    monkeypatch.setenv("E2A_CACHE", "/c")
    monkeypatch.setenv("E2A_CKPT", "/k")
    a = train.parse_args(["--config", cfg])
    assert a.train[0] == "/c/train/index.jsonl" and str(a.out) == "/k/reader" and a.epochs == 80
