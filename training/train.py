#!/usr/bin/env python3
"""Train the GRU-CTC reader on cached frozen-encoder features (port of the research scripts/train_fast.py, single GPU).

  * every pool (the --train set included) is sampled with the rotate sampler: --train-hours / PATH:HOURS per epoch;
  * the epoch's rows are uploaded to the device in --pool-chunks random parts (GPU-resident bf16 features, no DataLoader);
  * full batches run a CUDA/HIP-graph-captured padded forward+backward (make_graphed_callables), partial batches eager;
    CTC loss, grad clip, AdamW and the LR schedule are eager;
  * eval every --eval-every epochs: greedy CER / WER / MER, per-metric best_<key>.pt, combined best.pt (normalized mean
    with --select-tol), patience with --min-epochs/--min-delta, --max-hours, regression guard;
  * atomic resume.pt (+ resume.prev.pt) at epoch boundaries; --resume auto makes a re-launched job continue exactly.

usage: python -m training.train --config configs/smoke.yaml [--key value ...]   (CLI flags override the config)
       ${VAR} in config strings is expanded from the environment (configs/recipe_v0.1.0.yaml uses E2A_CACHE / E2A_CKPT).

Private API: the padded forward calls torch._VF.gru (verified on torch 2.10.0+rocm7.1.1 and CPU builds; see
research/parity_pr3.md). A start-up self-check compares it with the packed Reader and stops on mismatch; --no-graph
trains with the packed Reader.forward only (public API, no graphs)."""

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from embed2asr import ctc as ctc_mod
from embed2asr import reader as reader_mod
from embed2asr import tokenizer as tokenizer_mod
from embed2asr.ctc import collapse_ctc, ctc_min_len
from embed2asr.metrics import edit_distance
from embed2asr.reader import Reader
from embed2asr.tokenizer import Tok
from training import datasets as datasets_mod
from training.datasets import D, RotateSampler, Store, bucket_batches, load_index, nodigit, parse_pool, split_chunks

# MER units on already-normalized text (embed2asr.metrics.mer_tokens would normalize again)
MER_UNIT = re.compile(r"[⺀-鿿豈-﫿㐀-䶿]|[^\s⺀-鿿豈-﫿㐀-䶿]+")  # each CJK char, or a latin word


def round_up(n, k):
    return -(-n // k) * k


@torch.no_grad()
def evaluate(model, store, texts, tok, round_to, bs=256, mixed=()):
    """Greedy decode with the packed Reader.forward in eval mode. Returns (CER no-space, WER, MER, MER on `mixed` rows, ex)."""
    model.eval()
    idx = sorted(range(len(texts)), key=lambda i: int(store.lens[i]))
    ce = cn = we = wn = me = mn = xe = xn = 0
    ex = []
    for k in range(0, len(idx), bs):
        b = torch.tensor(idx[k : k + bs])
        lens = store.lens[b]
        ids = model(store.gather(b, round_up(int(lens.max()), round_to)).float(), lens).argmax(-1).cpu().tolist()
        for seq, L, i in zip(ids, lens.tolist(), b.tolist()):
            hyp, ref = tok.decode(collapse_ctc(seq[:L])), texts[i]
            r = ref.replace(" ", "")
            ce += edit_distance(hyp.replace(" ", ""), r)
            cn += len(r)
            rw = ref.split()
            we += edit_distance(hyp.split(), rw)
            wn += len(rw)
            rm = MER_UNIT.findall(ref)
            e = edit_distance(MER_UNIT.findall(hyp), rm)
            me += e
            mn += len(rm)
            if i in mixed:
                xe += e
                xn += len(rm)
            if len(ex) < 3:
                ex.append((ref, hyp))
    model.train()
    return ce / max(cn, 1), we / max(wn, 1), me / max(mn, 1), xe / max(xn, 1), ex


# ---- padded (graph-capturable) training forward ----
def gru_padded(rnn, x, lens):
    """Bidirectional nn.GRU on a PADDED batch, equal at valid frames to pack_padded_sequence: the reverse direction runs
    on each sequence flipped within its own length, so padding never leaks into valid outputs."""
    B, T, _ = x.shape
    ar = torch.arange(T, device=x.device)[None]
    rev = lens[:, None] - 1 - ar
    rev = torch.where(rev >= 0, rev, ar)  # involution: valid part reversed, padding fixed in place

    def flip(t):
        return t.gather(1, rev[..., None].expand(-1, -1, t.size(-1)))

    h0 = x.new_zeros(1, B, rnn.hidden_size)
    for layer in range(rnn.num_layers):
        outs = []
        for sfx in ("", "_reverse"):
            w = [getattr(rnn, f"{n}_l{layer}{sfx}") for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
            o = torch._VF.gru(flip(x) if sfx else x, h0, w, True, 1, 0.0, rnn.training, False, True)[0]
            outs.append(flip(o) if sfx else o)
        x = torch.cat(outs, -1)
    return x


class PaddedCore(nn.Module):
    """Reader.forward with the padded GRU; shares parameters with the Reader. Padded output frames are zeroed."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, lens):
        m = self.m
        mask = (torch.arange(x.size(1), device=x.device)[None] < lens[:, None]).unsqueeze(-1).to(x.dtype)
        h = m.pre(x)
        for b in m.blocks:
            h = b(h, mask)
        h = (h * mask).float()  # GRU always fp32 (also under --bf16 autocast)
        if m.drop_gru > 0:
            h = gru_padded(m.rnn2, F.dropout(gru_padded(m.rnn1, h, lens), m.drop_gru, m.training), lens)
        else:
            h = gru_padded(m.rnn, h, lens)
        return m.out(h * mask)


@torch.no_grad()
def padded_selfcheck(model, dev, atol=1e-4):
    """PaddedCore (torch._VF.gru) must equal the packed Reader at valid frames; guards against private-API drift."""
    was = model.training
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev.type == "cuda" else []):
        torch.manual_seed(0)
        model.eval()
        lens = torch.tensor([60, 37, 5])
        x = torch.randn(3, 64, D, device=dev)
        ref, got = model(x, lens), PaddedCore(model)(x, lens.to(dev))
    model.train(was)
    valid = (torch.arange(64)[None] < lens[:, None]).to(dev)
    err = (ref - got)[valid].abs().max().item()
    if not err <= atol:
        raise SystemExit(
            f"padded GRU forward differs from the packed Reader by {err:.2e} (torch {torch.__version__}); "
            "torch._VF.gru may have changed: train with --no-graph"
        )
    return err


def code_version():
    """Hash of the code that defines a training step; resume refuses a different version unless explicitly allowed."""
    files = [Path(__file__), *(Path(m.__file__) for m in (datasets_mod, reader_mod, tokenizer_mod, ctc_mod))]
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]


def file_sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def data_signature(index_paths, row_lists):
    """Content hash of every index file + size of every features file it references."""
    sig = {f"index:{x}": file_sha256(x) for x in index_paths}
    sig.update({f"features:{x}": os.path.getsize(x) for x in sorted({r["path"] for rows in row_lists for r in rows})})
    return sig


RESUME_FORMAT = 2
RESUME_MUTABLE = {"resume", "resume_allow_code_change", "max_hours", "min_free_gib", "config", "out", "init"}  # may change between launches


def plain(d):
    return json.loads(json.dumps(d, default=str))


def resume_problems(r, a, ctx, opt_params):
    """Every reason the saved run cannot continue under the current code/config/data (empty list = compatible)."""
    p = []
    if r.get("format") != RESUME_FORMAT:
        return [f"resume format {r.get('format')} != {RESUME_FORMAT}"]
    old, new = ({k: v for k, v in plain(x).items() if k not in RESUME_MUTABLE} for x in (r["args"], vars(a)))
    p += [f"arg {k}: saved {old.get(k)!r} now {new.get(k)!r}" for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
    for k in ("itos", "cfg", "tok_sha", "data_sig"):
        if r[k] != ctx[k]:
            detail = sorted(x for x in set(r[k]) | set(ctx[k]) if r[k].get(x) != ctx[k].get(x)) if k == "data_sig" else ""
            p.append(f"{k} differs {detail}".rstrip())
    if r["code"] != ctx["code"] and not a.resume_allow_code_change:
        p.append(f"code version {r['code']} != {ctx['code']} (pass --resume-allow-code-change to accept)")
    if not (
        isinstance(r["rng"], tuple)
        and isinstance(r["torch_rng"], torch.Tensor)
        and isinstance(r["np_rng"], tuple)
        and (isinstance(r["cuda_rng"], torch.Tensor) if ctx["cuda"] else r["cuda_rng"] is None)
    ):
        p.append("RNG state missing or from another device type")
    groups, state = r["opt"]["param_groups"], r["opt"]["state"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(opt_params):
        p.append("optimizer parameter layout differs")
    else:
        g = groups[0]
        if g.get("initial_lr") != a.lr or g["weight_decay"] != 0.01:
            p.append(f"optimizer hyper-parameters differ (initial_lr {g.get('initial_lr')}, wd {g['weight_decay']})")
        for i, q in zip(g["params"], opt_params):
            if i in state and tuple(state[i]["exp_avg"].shape) != tuple(q.shape):
                p.append(f"optimizer state shape differs for param {i}")
                break
    if r["sched"]["last_epoch"] != r["step"] or r["sched"]["base_lrs"] != [a.lr]:
        p.append(
            f"scheduler state inconsistent (last_epoch {r['sched']['last_epoch']}, step {r['step']}, base_lrs {r['sched']['base_lrs']})"
        )
    return p


def save(obj, path):
    """Atomic torch.save: a crash never leaves a truncated checkpoint under the final name."""
    torch.save(obj, f"{path}.tmp")
    os.replace(f"{path}.tmp", path)


# ---- config ----
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="YAML with any of the options below (underscored names); CLI overrides it")
    g = p.add_argument_group("data")
    g.add_argument("--train", nargs="+", help="index.jsonl file(s) of the main train set (pool 0, no digit filter)")
    g.add_argument("--train-hours", type=float, default=10.0, help="hours of --train rows per epoch (rotate sampler)")
    g.add_argument("--pool", nargs="*", default=[], help="PATH:HOURS ; each epoch takes the next HOURS of distinct rows of that pool")
    g.add_argument("--pool-chunks", type=int, default=8, help="upload each epoch's rows in K random parts (~1/K feature memory)")
    g.add_argument("--dev", nargs="*", default=[], help="main dev set (dev_cer)")
    g.add_argument("--dev-extra", nargs="*", default=[], help="NAME=index.jsonl, reported as dev_NAME_{cer,wer,mer,mixmer}")
    g = p.add_argument_group("model / vocabulary")
    g.add_argument("--vocab-mode", choices=["char", "bpe"], default="bpe")
    g.add_argument("--bpe-size", type=int, default=400)
    g.add_argument("--d", type=int, default=512)
    g.add_argument("--gru", type=int, default=256)
    g.add_argument("--blocks", type=int, default=3)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument(
        "--init",
        type=Path,
        help="warm start: reader weights from this checkpoint, head rows mapped by unit, its BPE model; fresh optimizer and schedule",
    )
    g = p.add_argument_group("optimization")
    g.add_argument("--epochs", type=int, default=80)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--time-mask", type=float, default=0.1, help="fraction of frames zeroed (spans of 4)")
    g.add_argument("--bf16", action="store_true", help="bf16 autocast for pre/conv/out (GRU, CTC, eval stay fp32)")
    g.add_argument("--no-graph", action="store_true", help="train with the packed Reader.forward (public API, no graphs, slower)")
    g.add_argument("--round-to", type=int, default=50, help="pad time axis to a multiple (bounds distinct graph shapes)")
    g.add_argument("--seed", type=int, default=42)
    g = p.add_argument_group("evaluation / selection / stopping")
    g.add_argument("--eval-every", type=int, default=2)
    g.add_argument("--train-cer-n", type=int, default=2000, help="train CER on a fixed random subset of this size")
    g.add_argument("--select", default="dev_cer", help="KEY[:BASE],... combined = mean(metric/BASE) -> best.pt; best_KEY.pt per key")
    g.add_argument("--select-tol", type=float, default=0.05, help="best.pt only if no selected metric is > (1+tol) x its own best")
    g.add_argument("--patience", type=int, default=8, help="evals without improvement before early stop")
    g.add_argument("--min-epochs", type=int, default=0, help="no early stop before this epoch")
    g.add_argument("--min-delta", type=float, default=0.0, help="relative margin that counts as an improvement")
    g.add_argument("--max-hours", type=float, default=0, help="wall-time budget summed over resumes (0 = none)")
    g.add_argument("--regress-guard", type=float, default=0, help="stop if a selected metric > REF x this for --regress-evals evals")
    g.add_argument("--regress-evals", type=int, default=4)
    g.add_argument("--regress-ref", choices=["base", "best"], default="best", help="REF = the --select BASE or this run's own best")
    g.add_argument("--regress-from", type=int, default=20)
    g = p.add_argument_group("run")
    g.add_argument("--out", type=Path, help="checkpoint dir")
    g.add_argument("--save-every", type=int, default=1, help="write resume.pt every N epochs")
    g.add_argument("--resume", nargs="?", const="yes", choices=["yes", "auto"], help="continue from <out>/resume.pt; auto = if one exists")
    g.add_argument("--resume-allow-code-change", action="store_true", help="resume although the training code version changed")
    g.add_argument("--min-free-gib", type=float, default=2.0, help="shared GPU: save and exit (resumable) when free memory drops below")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        import yaml

        cfg = yaml.safe_load(pre.config.read_text(encoding="utf-8")) or {}
        if isinstance(cfg.get("pool"), list):
            cfg["pool"] = [x if isinstance(x, str) else f"{x['path']}:{x['hours']}" for x in cfg["pool"]]
        if isinstance(cfg.get("dev_extra"), dict):
            cfg["dev_extra"] = [f"{k}={v}" for k, v in cfg["dev_extra"].items()]
        if isinstance(cfg.get("select"), dict):
            cfg["select"] = ",".join(f"{k}:{v}" if v else k for k, v in cfg["select"].items())

        def expand(v):
            if isinstance(v, str):
                e = os.path.expandvars(v)
                if re.search(r"\$\{?\w+", e):
                    p.error(f"config value {v!r}: environment variable not set")
                return e
            return [expand(x) for x in v] if isinstance(v, list) else v

        cfg = {k: expand(v) for k, v in cfg.items()}
        known = {a.dest for a in p._actions}
        bad = set(cfg) - known
        if bad:
            p.error(f"unknown config keys {sorted(bad)}")
        p.set_defaults(**{k: Path(v) if k in ("out", "init") and v else v for k, v in cfg.items()})
    a = p.parse_args(argv)
    if not a.train or not a.out:
        p.error("--train and --out are required (CLI or config)")
    return a


def main(argv=None):
    a = parse_args(argv)
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    if cuda:
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free < a.min_free_gib:
            raise SystemExit(f"PAUSE: free device memory {free:.1f} GiB < {a.min_free_gib}")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)  # chunk split + batch order; the rotate sampler has its own per-(pool, pass) seeds

    tr, dv = load_index(a.train), load_index(a.dev)
    dev_extra = {kv.split("=", 1)[0]: nodigit(load_index([kv.split("=", 1)[1]])) for kv in a.dev_extra}
    pools = [(nodigit(load_index([path])), h) for path, h in map(parse_pool, a.pool)]
    a.out.mkdir(parents=True, exist_ok=True)
    init = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
    if init and init.get("tok", {}).get("sp") and not (a.out / "bpe.model").exists():
        assert a.vocab_mode == "bpe", "--init checkpoint uses bpe"
        (a.out / "bpe.model").write_bytes(init["tok"]["sp"])
    tok = Tok.build(a.vocab_mode, [r["text"] for r in tr] + [r["text"] for pr, _ in pools for r in pr], a.out, a.bpe_size)
    itos = tok.itos

    def feasible(rows):
        return [r for r in rows if ctc_min_len(tok.encode(r["text"])) <= r["frames"]]  # CTC needs enough frames

    n_tr, tr = len(tr), feasible(tr)
    pools = [(tr, a.train_hours)] + [(feasible(pr), h) for pr, h in pools]
    print(f"vocab {a.vocab_mode}: {len(itos)} units; train {len(tr)}/{n_tr} CTC-feasible, dev {len(dv)}", flush=True)
    for k, (pr, h) in enumerate(pools):
        print(f"pool {k}: {len(pr)} feasible rows, {sum(r['seconds'] for r in pr) / 3600:.1f} h available, {h:.1f} h/epoch", flush=True)
    sampler = RotateSampler(pools, a.seed)

    t0 = time.time()
    sub = sorted(random.Random(a.seed + 1).sample(range(len(tr)), a.train_cer_n)) if len(tr) > a.train_cer_n else list(range(len(tr)))
    sub_rows = [tr[i] for i in sub]
    evals = {"train": (Store(sub_rows, dev), [r["text"] for r in sub_rows], set())}
    if dv:
        evals["dev"] = (Store(dv, dev), [r["text"] for r in dv], set())
    for k, rows in dev_extra.items():
        evals[f"dev_{k}"] = (
            Store(rows, dev),
            [r["text"] for r in rows],
            {i for i, r in enumerate(rows) if r.get("lang") in ("mix", "zh-en")},
        )
    en_major = {f"dev_{k}" for k, rows in dev_extra.items() if sum(r.get("lang") == "en-US" for r in rows) > len(rows) / 2}
    print(
        f"eval features resident: {sum(s.flat.numel() * 2 for s, _, _ in evals.values()) / 2**30:.2f} GiB in {time.time() - t0:.0f}s",
        flush=True,
    )

    steps_per_ep = (len(sampler.epoch(1)[0]) + a.batch_size - 1) // a.batch_size  # schedule length (chunk tails add a few steps)
    total = steps_per_ep * a.epochs
    warm = max(1, min(500, total // 10))

    def lam(s):
        return min(1, (s + 1) / warm) * (0.5 * (1 + torch.cos(torch.tensor(min(s / total, 1.0) * 3.14159265)).item()) * 0.95 + 0.05)

    torch.manual_seed(a.seed)
    model = Reader(D, len(itos), d=a.d, gru=a.gru, blocks=a.blocks, dropout=a.dropout).to(dev).train()
    if init:  # every non-head tensor must match; output-head rows copied by unit string, new units keep their fresh init

        def same(c):
            return {k: v for k, v in c.items() if k != "vocab_size"}

        assert same(model.cfg) == same(init["cfg"]), f"--init: reader config differs {model.cfg} vs {init['cfg']}"
        sd, o_stoi = model.state_dict(), {u: i for i, u in enumerate(init["itos"])}
        hit = torch.tensor([o_stoi.get(u, -1) for u in itos])
        for k, t in init["model"].items():
            if k.startswith("out.") and t.shape[0] == len(init["itos"]):
                sd[k][(hit >= 0).to(sd[k].device)] = t[hit[hit >= 0]].to(sd[k].device)
            else:
                sd[k].copy_(t)
        model.load_state_dict(sd)
        print(f"[init] {a.init}: {int((hit >= 0).sum())}/{len(itos)} output units carried over", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    (a.out / "config.json").write_text(
        json.dumps({**vars(a), "model_cfg": model.cfg, "vocab_size": len(itos), "trainer": "embed2asr"}, default=str, indent=1)
    )
    print(f"params {sum(x.numel() for x in model.parameters())} | {steps_per_ep} steps/epoch, warmup {warm}, total {total}", flush=True)
    torch.manual_seed(a.seed)  # augmentation noise stream (same reseed as the research trainer)

    def amp():
        return torch.autocast(dev.type, torch.bfloat16, enabled=a.bf16, cache_enabled=False)

    sel = [(k, float(b or 1)) for k, _, b in (x.partition(":") for x in a.select.split(","))] if dv else [("train_cer", 1.0)]

    S = dict(best=9.0, best_k={}, bad=0, done=False, reg={})  # selection / stopping state (saved in resume.pt)
    graphs, step, start, wall0, t_run = {}, 0, 1, 0.0, time.time()
    ctx = {
        "itos": itos,
        "cfg": model.cfg,
        "cuda": cuda,
        "code": code_version(),
        "tok_sha": hashlib.sha256(tok.sp_bytes or b"").hexdigest(),
        "data_sig": data_signature(
            [*a.train, *a.dev, *(parse_pool(s)[0] for s in a.pool), *(kv.split("=", 1)[1] for kv in a.dev_extra)],
            [tr, dv, *(pr for pr, _ in pools), *dev_extra.values()],
        ),
    }
    if not a.no_graph:
        print(f"[selfcheck] padded GRU vs packed Reader max |diff| {padded_selfcheck(model, dev):.1e}", flush=True)

    def stop(why):
        print(f"STOP: {why}", flush=True)
        S["done"] = True
        (a.out / "stop_reason.json").write_text(json.dumps({"reason": why, "at": time.strftime("%F %T")}, ensure_ascii=False))

    def save_resume(ep):
        st = {
            "format": RESUME_FORMAT,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": ep,
            "step": step,
            **S,
            "wall": wall0 + time.time() - t_run,
            "rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if cuda else None,
            "np_rng": np.random.get_state(),
            **{k: v for k, v in ctx.items() if k != "cuda"},
            "saved_at": time.strftime("%F %T"),
            "args": plain(vars(a)),
        }
        torch.save(st, a.out / "resume.pt.tmp")  # atomic; the previous valid one stays as resume.prev.pt
        if (a.out / "resume.pt").exists():
            os.replace(a.out / "resume.pt", a.out / "resume.prev.pt")
        os.replace(a.out / "resume.pt.tmp", a.out / "resume.pt")

    def load_resume():
        for f in ("resume.pt", "resume.prev.pt"):
            try:
                r = torch.load(a.out / f, map_location="cpu", weights_only=False)
                assert all(k in r for k in ("model", "opt", "sched", "rng", "step", "epoch"))
                print(f"[resume] loaded {f} (saved {r.get('saved_at')}, code {r.get('code')})", flush=True)
                return r
            except FileNotFoundError:
                pass
            except Exception as e:  # truncated / corrupt file: fall back to the previous one
                print(f"[resume] {f} unusable ({type(e).__name__}: {e}); trying previous", flush=True)
        return None

    if a.resume == "auto" and not any((a.out / f).exists() for f in ("resume.pt", "resume.prev.pt")):
        print("[resume] auto: no checkpoint, fresh start", flush=True)
        a.resume = None
    if a.resume:
        r = load_resume()
        if r is None:
            raise SystemExit("resume: no valid resume.pt / resume.prev.pt")
        bad = resume_problems(r, a, ctx, list(model.parameters()))
        if bad:
            raise SystemExit("resume: incompatible checkpoint, not continuing:\n  " + "\n  ".join(bad))
        model.load_state_dict(r["model"])
        opt.load_state_dict(r["opt"])
        sched.load_state_dict(r["sched"])
        S.update({k: r[k] for k in S})
        rng.setstate(r["rng"])
        torch.set_rng_state(r["torch_rng"])
        np.random.set_state(r["np_rng"])
        if cuda and r["cuda_rng"] is not None:
            torch.cuda.set_rng_state(r["cuda_rng"])
        start, step, wall0 = r["epoch"] + 1, r["step"], r["wall"]
        print(f"resumed after epoch {r['epoch']} (step {step})", flush=True)
        if S["done"]:
            print("run already finished (see stop_reason.json)", flush=True)
            return

    def train_step(x, lens, tg, tl, acc, diag):
        if a.time_mask > 0:
            m = torch.rand(x.shape[:2], device=x.device) < a.time_mask / 4
            for s in range(1, 4):
                m |= torch.roll(m, s, 1)
            x = x.masked_fill(m.unsqueeze(-1), 0.0)
        lens_d = lens.to(dev, non_blocking=True)
        key = tuple(x.shape[:2])
        if a.no_graph:  # public-API path: packed GRU (lengths on CPU)

            def fwd(x, _):
                return model(x, lens)
        elif not cuda or x.shape[0] != a.batch_size:  # a graph per partial-batch shape would leak memory
            fwd = PaddedCore(model)
        else:
            if key not in graphs:
                with (
                    torch.random.fork_rng(devices=[torch.cuda.current_device()]),
                    amp(),
                ):  # capture warm-ups must not consume RNG (exact resume)
                    graphs[key] = torch.cuda.make_graphed_callables(PaddedCore(model), (x, lens_d))
            fwd = graphs[key]
        with amp():
            logits = fwd(x, lens_d).float()
        lp = logits.log_softmax(-1).transpose(0, 1)
        per = F.ctc_loss(lp, tg, lens, tl, blank=0, reduction="none", zero_infinity=False)  # CPU lengths: no host sync
        fin = torch.isfinite(per)
        loss = torch.where(fin, per / tl.to(dev).clamp(min=1), 0.0).sum() / fin.sum().clamp(min=1)  # mean over finite samples
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        acc["loss"] += loss.detach()
        acc["gn"] += gn
        acc["inf"] += (~fin).sum()
        acc["allbad"] |= ~fin.any()
        if diag:  # blank probability / logit spread on valid frames only
            with torch.no_grad():
                vm = torch.arange(logits.size(1), device=dev)[None] < lens_d[:, None]
                acc["bl"] += lp.transpose(0, 1).exp()[vm][:, 0].mean()
                acc["ls"] += logits[vm].std()
                acc["nd"] += 1

    for ep in range(start, a.epochs + 1):
        rows, cov = sampler.epoch(ep)
        acc = {k: torch.zeros((), device=dev) for k in ("loss", "gn", "inf", "bl", "ls")}
        acc.update(allbad=torch.zeros((), dtype=torch.bool, device=dev), nd=0)
        n_seen = n_steps = 0
        ts, load_sec, store = time.time(), 0.0, None
        for part in split_chunks(rows, a.pool_chunks, rng):
            tl0 = time.time()
            store = None
            if cuda:
                torch.cuda.empty_cache()  # free the previous chunk first
            store = Store(part, dev)
            load_sec += time.time() - tl0
            for b in bucket_batches(part, a.batch_size, rng):
                b = torch.tensor(b)
                lens = store.lens[b]
                x = store.gather(b, round_up(int(lens.max()), a.round_to)).float()
                tg = [torch.tensor(tok.encode(part[i]["text"]), dtype=torch.long) for i in b.tolist()]
                tl = torch.tensor([len(t) for t in tg])
                step += 1
                n_steps += 1
                n_seen += len(b)
                train_step(x, lens, torch.cat(tg).to(dev, non_blocking=True), tl, acc, step % 50 == 0)
        store = None
        if cuda:
            torch.cuda.synchronize()
        train_sec = time.time() - ts - load_sec
        if acc["allbad"].item():
            raise SystemExit("all CTC losses non-finite in some batch (infeasible alignment)")
        hours = sum(r["seconds"] for r in rows) / 3600
        rec = {
            "epoch": ep,
            "step": step,
            "loss": acc["loss"].item() / max(n_steps, 1),
            "lr": sched.get_last_lr()[0],
            "inf_frac": acc["inf"].item() / max(n_seen, 1),
            "grad_norm": acc["gn"].item() / max(n_steps, 1),
            "blank_p": acc["bl"].item() / max(acc["nd"], 1),
            "logit_std": acc["ls"].item() / max(acc["nd"], 1),
            "epoch_hours": round(hours, 2),
            "epoch_train_sec": round(train_sec, 1),
            "load_sec": round(load_sec, 1),
            "audio_h_per_h": round(hours * 3600 / max(train_sec, 1e-9), 1),
            "n_graphs": len(graphs),
            "pool_passes": cov,
            "sec": round(time.time() - t_run, 1),
        }
        if cuda:
            free = torch.cuda.mem_get_info()[0] / 2**30
            rec.update(free_gib=round(free, 1), max_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            if free < a.min_free_gib:  # this epoch is redone after --resume (resume.pt is from the previous epoch)
                raise SystemExit(f"PAUSE: free device memory {free:.1f} GiB < {a.min_free_gib}; continue with --resume")
        if ep % a.eval_every == 0 or ep == a.epochs:
            for name, (st, texts, mixed) in evals.items():
                cer, wer, mer, mixmer, ex = evaluate(model, st, texts, tok, a.round_to, mixed=mixed)
                rec[f"{name}_cer"] = cer
                if name in en_major:
                    rec[f"{name}_wer"] = wer
                if mixed:
                    rec[f"{name}_mer"], rec[f"{name}_mixmer"] = mer, mixmer  # all rows / code-switched rows
                if name == "dev" or (name == "train" and not dv):
                    shown = ex[0] if ex else None
            print(json.dumps(rec, ensure_ascii=False), "|", shown, flush=True)
            cur = sum(rec[k] / b for k, b in sel) / len(sel)
            ck = {
                "model": model.state_dict(),
                "itos": itos,
                "cfg": model.cfg,
                "epoch": ep,
                "metrics": rec,
                "seed": a.seed,
                "tok": {"mode": a.vocab_mode, "sp": tok.sp_bytes},
            }
            save(ck, a.out / "last.pt")
            improved = False
            for k, _ in sel:  # per-metric best (e.g. best_dev_cer.pt = best zh, best_dev_en_wer.pt = best en)
                if rec[k] < S["best_k"].get(k, 9.0):
                    improved |= rec[k] < S["best_k"].get(k, 9.0) * (1 - a.min_delta)
                    S["best_k"][k] = rec[k]
                    save(ck, a.out / f"best_{k}.pt")
            if cur < S["best"] and all(rec[k] <= S["best_k"][k] * (1 + a.select_tol) for k, _ in sel):  # don't trade a language away
                improved |= cur < S["best"] * (1 - a.min_delta)
                S["best"] = cur
                save(ck, a.out / "best.pt")
            S["bad"] = 0 if improved else S["bad"] + 1
            with (a.out / "train_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if S["bad"] >= a.patience and ep >= a.min_epochs:
                stop(f"early stop at epoch {ep}: {S['bad']} evals without >{a.min_delta:.1%} improvement; best combined {S['best']:.4f}")
            for k, b in sel if a.regress_guard else []:  # sustained regression of one selected metric
                ref = b if a.regress_ref == "base" else S["best_k"][k]
                S["reg"][k] = S["reg"].get(k, 0) + 1 if ep >= a.regress_from and rec[k] > ref * a.regress_guard else 0
                if S["reg"][k] >= a.regress_evals:
                    stop(
                        f"REGRESSION STOP at epoch {ep}: {k} {rec[k]:.4f} > {a.regress_guard} x {a.regress_ref} {ref:.4f} for {S['reg'][k]} evals"
                    )
        else:
            print(json.dumps(rec, ensure_ascii=False), flush=True)
        if a.max_hours and not S["done"] and wall0 + time.time() - t_run >= a.max_hours * 3600:
            stop(f"wall-time budget {a.max_hours} h reached after epoch {ep}")
        if ep % a.save_every == 0 or S["done"] or ep == a.epochs:
            save_resume(ep)
        if S["done"]:
            break
    mem = f" max_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB" if cuda else ""
    print(f"done {time.time() - t_run:.0f}s step {step}{mem}", flush=True)


if __name__ == "__main__":
    main()
