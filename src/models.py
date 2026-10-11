"""1D-CNN quantifier, samplers and training loops (CPU). Hyper-parameters are fixed a priori (not tuned on test folds)."""
import os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
_R = _os.path.dirname(_SRC) + "/"   # repository root
_CHEMICAM = _os.environ.get("CHEMICAM_DIR", _R + "data/chemicam_open_database/")
import sys, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, _SRC)
from data import NPIX, TARGETS, to_log, CROP, apply_crop

torch.set_num_threads(2)
NT = len(TARGETS)
SCALE = float(np.sqrt(NPIX))     # unit-L2 spectra -> unit RMS inputs


class Net(nn.Module):
    def __init__(self, n_out=NT):
        super().__init__()

        def blk(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, stride=2, padding=k // 2), nn.BatchNorm1d(o), nn.GELU())
        self.f = nn.Sequential(blk(1, 16, 9), blk(16, 32, 7), blk(32, 64, 5), blk(64, 96, 5), nn.AvgPool1d(2))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(96 * 32, 128), nn.GELU(), nn.Dropout(0.1),
                               nn.Linear(128, n_out))

    def forward(self, x):
        return self.h(self.f(x))


# ------------------------------------------------------------------ samplers: return (x[B,1024], y[B,NT], mask[B,NT])
class RealSampler:
    """Draws blocks uniformly over a set of training SAMPLES (each sample equally likely, regardless of #blocks)."""
    def __init__(self, R, samples):
        self.R, self.samples = R, np.asarray(samples)

    def __call__(self, rng, bs):
        s = rng.choice(self.samples, bs)
        b = self.R["start"][s] + (rng.random(bs) * self.R["count"][s]).astype(int)
        y = self.R["Ylog"][s]
        return self.R["Y"][b] * SCALE, np.nan_to_num(y), (~np.isnan(y)).astype(np.float32)


class SynthSampler:
    def __init__(self, Y, L):
        self.Y, self.y = (apply_crop(Y) if CROP else Y), to_log(L).astype(np.float32)

    def __call__(self, rng, bs):
        i = rng.integers(0, len(self.Y), bs)
        return self.Y[i] * SCALE, self.y[i], np.ones((bs, NT), np.float32)


class MixSampler:
    def __init__(self, a, b, frac_a=0.5):
        self.a, self.b, self.fa = a, b, frac_a

    def __call__(self, rng, bs):
        na = int(round(bs * self.fa))
        xa, ya, ma = self.a(rng, na); xb, yb, mb = self.b(rng, bs - na)
        return np.concatenate([xa, xb]), np.concatenate([ya, yb]), np.concatenate([ma, mb])


def augment(x, rng):
    bs = len(x)
    sh = rng.integers(-2, 3, bs)
    idx = np.clip(np.arange(NPIX)[None, :] + sh[:, None], 0, NPIX - 1)
    x = x[np.arange(bs)[:, None], idx]
    return x + rng.normal(0, 0.05, x.shape).astype(np.float32)


def train(model, sampler, steps, lr, bs=64, wd=1e-4, seed=0, log=None, warm=30, freeze_f=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if freeze_f:
        for p in model.f.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=min(0.3, warm / steps + 0.05),
                                                anneal_strategy="cos", div_factor=10, final_div_factor=50)
    model.train()
    if freeze_f:
        model.f.eval()          # frozen feature extractor: weights AND BatchNorm statistics fixed
    run = 0.0
    for t in range(steps):
        x, y, m = sampler(rng, bs)
        x = torch.from_numpy(augment(x, rng).astype(np.float32))[:, None, :]
        y = torch.from_numpy(y.astype(np.float32)); m = torch.from_numpy(m)
        p = model(x)
        loss = (F.huber_loss(p, y, reduction="none", delta=1.0) * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if t else loss.item()
        if log and (t + 1) % max(1, steps // 5) == 0:
            log(f"    step {t+1}/{steps} train-loss(ema) {run:.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict_blocks(model, X, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy((X[i:i + bs] * SCALE).astype(np.float32))[:, None, :]
        out.append(model(x).numpy())
    return np.concatenate(out)


def predict_samples(model, R, samples, cap=120, seed=0):
    """Mean (log10) prediction over (up to cap random) blocks of each sample."""
    rng = np.random.default_rng(seed)
    res = {}
    for s in samples:
        n = R["count"][s]
        idx = R["start"][s] + (rng.choice(n, min(cap, n), replace=False))
        res[s] = predict_blocks(model, R["Y"][idx]).mean(axis=0)
    return res
