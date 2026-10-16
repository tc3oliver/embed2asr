"""PaddedCore (training forward, torch._VF.gru on a padded batch) vs the packed embed2asr Reader (inference forward).

Same weights, a batch with different lengths and padding: FP32 logits, per-sample CTC loss and parameter gradients must
match, and garbage in padded frames must not change valid outputs or gradients. On CUDA the graph-captured callable is
compared too. `python tests/test_parity.py` prints the max differences (research/parity_pr3.md)."""

import json
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from embed2asr.reader import Reader  # noqa: E402
from training.train import PaddedCore, padded_selfcheck  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
LENS, T, V = [50, 33, 7, 1], 56, 11
TL = [10, 8, 3, 1]


def setup(dev, drop_gru=0.0, size=(32, 16, 2, V), lens=LENS, T=T, tl=TL):
    d, gru, blocks, vocab = size
    torch.manual_seed(0)
    m = Reader(1536, vocab, d=d, gru=gru, blocks=blocks, dropout=0.0, drop_gru=drop_gru).to(dev).train()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(len(lens), T, 1536, generator=g)
    lens = torch.tensor(lens)
    x[torch.arange(T)[None] >= lens[:, None]] = 0.0  # clean padding, as Store.gather produces
    tg = torch.randint(1, vocab, (sum(tl),), generator=g)
    return m, x.to(dev), lens, tg.to(dev), torch.tensor(tl)


def run(m, fwd, x, lens, tg, tl):
    m.zero_grad(set_to_none=True)
    logits = fwd(x)
    per = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), tg, lens, tl, blank=0, reduction="none", zero_infinity=False)
    (per / tl.to(x.device)).mean().backward()
    return logits.detach(), per.detach(), {k: p.grad.clone() for k, p in m.named_parameters()}


def max_diff(a, b):
    return max((a[k] - b[k]).abs().max().item() for k in a) if isinstance(a, dict) else (a - b).abs().max().item()


def report(dev, drop_gru=0.0, eval_mode=False, **kw):
    m, x, lens, tg, tl = setup(dev, drop_gru, **kw)
    T = x.shape[1]
    m.train(not eval_mode)
    ld = lens.to(dev)
    packed = run(m, lambda x: m(x, lens), x, lens, tg, tl)
    padded = run(m, lambda x: PaddedCore(m)(x, ld), x, lens, tg, tl)
    junk = x.clone()
    junk[(torch.arange(T)[None] >= lens[:, None]).to(dev)] = 1e3  # garbage in padded frames
    junked = run(m, lambda x: PaddedCore(m)(x, ld), junk, lens, tg, tl)
    valid = (torch.arange(T)[None] < lens[:, None]).to(dev)
    r = {
        "logits": max_diff(packed[0], padded[0]),
        "ctc_loss": max_diff(packed[1], padded[1]),
        "grad": max_diff(packed[2], padded[2]),
        "grad_rel": max_diff(packed[2], padded[2]) / max(g.abs().max().item() for g in packed[2].values()),
        "padding_junk_valid_logits": (junked[0] - padded[0])[valid].abs().max().item(),
        "padding_junk_grad": max_diff(junked[2], padded[2]),
    }
    if dev == "cuda" and not eval_mode:
        graphed = torch.cuda.make_graphed_callables(PaddedCore(m), (x, ld))
        g = run(m, lambda x: graphed(x, ld), x, lens, tg, tl)
        r.update(graph_logits=max_diff(g[0], padded[0]), graph_grad=max_diff(g[2], padded[2]))
    return r


@pytest.mark.parametrize("dev", DEVICES)
def test_parity_two_layer_gru(dev):
    r = report(dev)
    tol = 1e-5 if dev == "cpu" else 1e-4
    assert r["logits"] < tol and r["ctc_loss"] < tol and r["grad_rel"] < tol, r
    assert r["padding_junk_valid_logits"] < tol and r["padding_junk_grad"] < tol, r
    if dev == "cuda":
        assert r["graph_logits"] < tol and r["graph_grad"] < tol, r


@pytest.mark.parametrize("dev", DEVICES)
def test_parity_split_gru_eval(dev):
    """rnn1 + dropout + rnn2 layout (drop_gru > 0); eval mode so dropout is off. cuDNN/MIOpen RNN backward needs train mode,
    so gradients are compared on CPU only."""
    if dev != "cpu":
        m, x, lens, *_ = setup(dev, 0.1)
        m.eval()
        with torch.no_grad():
            assert (m(x, lens) - PaddedCore(m)(x, lens.to(dev))).abs().max().item() < 1e-4
        return
    r = report(dev, 0.1, eval_mode=True)
    assert r["logits"] < 1e-5 and r["grad_rel"] < 1e-5 and r["padding_junk_valid_logits"] < 1e-5, r


@pytest.mark.parametrize("dev", DEVICES)
def test_selfcheck(dev):
    m = Reader(1536, V, d=32, gru=16, blocks=1).to(dev)
    assert padded_selfcheck(m, torch.device(dev)) < 1e-4


if __name__ == "__main__":
    out = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
    v1 = dict(size=(512, 256, 3, 4996), lens=[740, 512, 300, 37, 1], T=750, tl=[120, 90, 40, 8, 1])  # release shape, ~30 s utterance
    for dev in DEVICES:
        out[f"{dev}/small"] = report(dev)
        out[f"{dev}/v1_shape"] = report(dev, **v1)
    out["cpu/small_split_gru_eval"] = report("cpu", 0.1, eval_mode=True)
    json.dump(out, sys.stdout, indent=1)
