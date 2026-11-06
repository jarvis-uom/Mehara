"""Module A on Kaggle: U-Net water maps trained on the Sen1Floods11 hand labels.

Steps
1. Get the data: 446 hand-labelled chips (512 x 512 pixels, 10 m) from the public Sen1Floods11 bucket.
   Each chip has Sentinel-1 radar (VV, VH, in dB), Sentinel-2 L1C (13 bands) and a label
   (-1 = no data, 0 = not water, 1 = water). The official lists split them into
   train 252, valid 89, test 90 and Bolivia 15 (a country kept out of training).
2. Scale every band with the mean and spread of the training chips, so all inputs have a similar range.
3. Train three U-Nets with the same settings: radar only, optical only, radar + optical.
   Training uses random 256 x 256 crops with flips and turns. Pixels labelled -1 are ignored.
4. Keep the weights from the epoch with the best water IoU on the valid chips.
5. Score them on the test chips, the Bolivia chips and the 9 Sri Lankan test chips: water IoU, the water-area
   error, and the share of real water found by water-body size (under 1 ha up to over 100 ha).
6. Score simple rules the same way: the dataset's own radar threshold map, NDWI > 0, MNDWI > 0, and NDWI,
   MNDWI and radar VV with their best thresholds, tuned on the training chips (a fair comparison for RQ1).
7. Save the scores, timings, weights and example pictures in the output folder.

Run 1 (1 Nov 2026): 40 epochs and fixed thresholds. Run 2: 80 epochs, tuned rules, size scores, per-chip scores.
Run 3: the radar + optical model trained twice, as before and with fake clouds pasted over random patches of the
optical bands (labels unchanged), so that it learns to use radar under cloud. Both are also scored on the test
chips with fake clouds added ("test with clouds"), overall and on the cloud-covered pixels only.
Run 4: also fake sun glint (bright patches on the water only, as on Mahakanadarawa on 30 Mar 2025), and a test
with fake glint ("water found under glint").
Run 5: the normal and the fake-cloud model, three runs each with different random seeds, reported as mean +/- spread.
Run 6: the radar-only model and the fake-cloud model again, and a new model with stronger fake clouds. In that new
model, 35% of the cloudy crops are covered completely (our cloudy-day check showed that run 3 fails when the whole
tank is under cloud), half get a dark cloud shadow, and every crop gets random brightness changes. A new test covers
the whole test chips with thick cloud ("full cloud"), so only radar is left. The size scores add precision (the share
of the map's water that is real water). Weights are saved for every seed.

On Kaggle it runs with no arguments. Quick local check on a CPU (a few seconds, 4 chips per split):
    python -I train_unet.py --dry-run --data <folder that holds v1.1> --out <folder>
"""
import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


try:
    import rasterio
except ImportError:
    pip_install("rasterio")
    import rasterio
try:
    import segmentation_models_pytorch as smp
except ImportError:
    pip_install("--no-deps", "segmentation-models-pytorch==0.6.0")
    import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1/"
SPLITS = ["train", "valid", "test", "bolivia"]
EVAL = ["valid", "test", "bolivia"]
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]  # order in S2Hand
S2_USE = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]  # 10 m and 20 m bands; L1C and L2A both have them
CHANNELS = ["VV", "VH"] + S2_USE
# model variants: input bands, the shares of training crops that get fake clouds and fake glint, and run 6's stronger
# clouds: the share of cloudy crops covered completely, the share with a shadow, and random brightness changes
VARIANTS = {"radar": (["VV", "VH"], 0.0, 0.0, {}), "optical": (S2_USE, 0.0, 0.0, {}),
            "radar+optical": (CHANNELS, 0.0, 0.0, {}), "radar+optical+clouds": (CHANNELS, 0.5, 0.0, {}),
            "radar+optical+clouds+glint": (CHANNELS, 0.5, 0.3, {}),
            "radar+optical+fullclouds": (CHANNELS, 0.6, 0.0, {"full": 0.35, "shadow": 0.5, "jitter": 0.15})}
DEFAULT_VARIANTS = "radar,radar+optical+clouds,radar+optical+fullclouds"  # run 6
DEFAULT_SEEDS = "0,1,2"  # three runs of each, to see how much scores vary between identical runs
# rough top-of-atmosphere values of a bright cloud in Sentinel-2 L1C (x 10000) for B2 ... B12
CLOUD_DN = np.array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 4000, 3000], np.float32)
LAYERS = {"s1": "S1Hand", "s2": "S2Hand", "label": "LabelHand", "otsu": "S1OtsuLabelHand"}
# water-body sizes in pixels of 10 m x 10 m (100 pixels = 1 ha); our village tanks are 6 to 16 ha
SIZE_BINS = [(0, 100, "under 1 ha"), (100, 500, "1 to 5 ha"), (500, 2000, "5 to 20 ha"),
             (2000, 10000, "20 to 100 ha"), (10000, None, "over 100 ha")]
# rules tuned on the training chips: water if the value is at or above (+1) or below (-1) the threshold
TUNED = {"NDWI": (+1, np.linspace(-1, 1, 201)), "MNDWI": (+1, np.linspace(-1, 1, 201)),
         "VV": (-1, np.arange(-40, 10.01, 0.25))}


# ---------- 1. data ----------

def fetch(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, part)
            part.replace(dest)
            return
        except OSError:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def chip_ids(root, split):
    rel = f"splits/flood_handlabeled/flood_{split}_data.csv"
    fetch(BUCKET + rel, root / rel)
    lines = (root / rel).read_text().split()
    return [ln.split(",")[0].replace("_S1Hand.tif", "") for ln in lines if ln.strip()]


def layer_rel(layer, cid):
    return f"data/flood_events/HandLabeled/{layer}/{cid}_{layer}.tif"


def download_all(root, ids):
    rels = [layer_rel(layer, c) for c in ids for layer in LAYERS.values()]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda rel: fetch(BUCKET + rel, root / rel), rels))


def read(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def load_split(root, ids):
    """Raw inputs (N, 12, 512, 512) float32, labels and the dataset's radar threshold map (N, 512, 512) int8,
    and the values the simple rules use (NDWI, MNDWI, radar VV in dB) as float16."""
    s2_index = [S2_BANDS.index(b) for b in S2_USE]
    x = np.empty((len(ids), len(CHANNELS), 512, 512), np.float32)
    y, otsu = (np.empty((len(ids), 512, 512), np.int8) for _ in range(2))
    values = {k: np.empty((len(ids), 512, 512), np.float16) for k in TUNED}
    for i, c in enumerate(ids):
        s1, s2 = read(root / layer_rel(LAYERS["s1"], c)), read(root / layer_rel(LAYERS["s2"], c))
        s2[s2 == 0] = np.nan  # Sentinel-2 marks "no data" with 0
        x[i, :2], x[i, 2:] = s1, s2[s2_index]
        x[i][~np.isfinite(x[i])] = np.nan  # missing pixels: left out of the scaling, filled later
        y[i] = read(root / layer_rel(LAYERS["label"], c))[0]
        otsu[i] = read(root / layer_rel(LAYERS["otsu"], c))[0]
        green, nir, swir = s2[S2_BANDS.index("B3")], s2[S2_BANDS.index("B8")], s2[S2_BANDS.index("B11")]
        with np.errstate(divide="ignore", invalid="ignore"):
            values["NDWI"][i] = (green - nir) / (green + nir)
            values["MNDWI"][i] = (green - swir) / (green + swir)
        values["VV"][i] = s1[0]
    return x, y, otsu, values


# ---------- 2. simple rules ----------

def tune_threshold(values, y, sign, edges):
    """The threshold with the best water IoU on these chips, read from two histograms (water and not water)."""
    ok = (y >= 0) & np.isfinite(values)
    v = np.clip(values[ok].astype(np.float32), edges[0], edges[-1])
    water = y[ok] == 1
    hw, hl = np.histogram(v[water], bins=edges)[0], np.histogram(v[~water], bins=edges)[0]
    if sign > 0:  # water if value >= edges[k]
        tp, fp = hw[::-1].cumsum()[::-1], hl[::-1].cumsum()[::-1]
    else:  # water if value < edges[k + 1]
        tp, fp = hw.cumsum(), hl.cumsum()
    iou = tp / np.maximum(hw.sum() + fp, 1)  # tp + fp + fn = all water pixels + fp
    k = int(np.argmax(iou))
    return round(float(edges[k] if sign > 0 else edges[k + 1]), 3)


def apply_rule(values, sign, t):
    with np.errstate(invalid="ignore"):
        return ((values >= t) if sign > 0 else (values < t)).astype(np.int8)


# ---------- 3. fake clouds ----------

def cloud_alpha(h, w, rng, cover=(0.1, 0.7)):
    """A soft cloud mask (0 = clear, 1 = thick cloud): smoothed noise above a random cover, with a random opacity."""
    noise = ndimage.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma=rng.uniform(8, 32))
    thr = np.quantile(noise, 1 - rng.uniform(*cover))
    return np.clip((noise - thr) / (0.3 * noise.std() + 1e-6), 0, 1) * rng.uniform(0.6, 1.0)


def add_clouds(x, alpha, opt_pos, mean_opt, std_opt, rng, brightness=(0.7, 1.2)):
    """Blend the scaled optical bands of x (bands, h, w) towards bright cloud values where alpha > 0."""
    cloud = (CLOUD_DN * rng.uniform(*brightness) - mean_opt) / std_opt
    x[opt_pos] = x[opt_pos] * (1 - alpha) + cloud[:, None, None] * alpha
    return x


def glint_alpha(y, rng):
    """Sun glint: patchy brightness on the water only (label 1), with soft edges."""
    water = ndimage.gaussian_filter((y == 1).astype(np.float32), 2)
    return cloud_alpha(*y.shape, rng, cover=(0.3, 0.9)) * water


def add_glint(x, y, opt_pos, mean_opt, std_opt, rng):
    """Glint is bright in every optical band, like a thin cloud, but it sits on the water, not over the land."""
    return add_clouds(x, glint_alpha(y, rng), opt_pos, mean_opt, std_opt, rng, brightness=(0.3, 1.0))


def add_shadow(x, alpha, opt_pos, mean_opt, std_opt, rng):
    """A cloud's shadow: the cloud's shape moved 10 to 40 pixels, darkening the optical bands by 40 to 80%."""
    dy, dx = rng.randint(10, 41, size=2) * rng.choice([-1, 1], size=2)
    shade = np.roll(alpha, (int(dy), int(dx)), axis=(0, 1)) * rng.uniform(0.4, 0.8)
    m, s = mean_opt[:, None, None], std_opt[:, None, None]
    x[opt_pos] = ((x[opt_pos] * s + m) * (1 - shade) - m) / s
    return x


def jitter_brightness(x, opt_pos, mean_opt, std_opt, rng, size):
    """Random brightness: the whole crop by up to +/- size, and each band by up to 3% more (haze, sun angle)."""
    f = (rng.uniform(1 - size, 1 + size) * rng.uniform(0.97, 1.03, len(opt_pos))).astype(np.float32)
    x[opt_pos] = x[opt_pos] * f[:, None, None] + (mean_opt * (f - 1) / std_opt)[:, None, None]
    return x


# ---------- 4. training ----------

class Chips(Dataset):
    """One item = one chip (valid/test) or one random crop of a chip with flips and turns (train).
    With cloud_p > 0, that share of training crops gets fake clouds on its optical bands. Run 6's options (extra):
    full = the share of those clouds that cover the whole crop, shadow = the share with a cloud shadow,
    jitter = random brightness changes on every crop."""

    def __init__(self, x, y, channels, crop=None, repeats=1, cloud_p=0.0, opt_pos=(), mean_opt=None, std_opt=None,
                 glint_p=0.0, extra=None):
        self.x, self.y, self.ch, self.crop, self.repeats = x, y, channels, crop, repeats
        self.cloud_p, self.opt_pos, self.mean_opt, self.std_opt = cloud_p, list(opt_pos), mean_opt, std_opt
        self.glint_p, self.extra = glint_p, extra or {}

    def __len__(self):
        return len(self.x) * self.repeats

    def __getitem__(self, i):
        k = i % len(self.x)
        x, y = self.x[k, self.ch], self.y[k]
        if self.crop:
            r, c = np.random.randint(0, x.shape[1] - self.crop + 1, size=2)
            x, y = x[:, r:r + self.crop, c:c + self.crop], y[r:r + self.crop, c:c + self.crop]
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[::-1]
            turns = np.random.randint(4)
            x, y = np.rot90(x, turns, (1, 2)), np.rot90(y, turns)
            if self.extra.get("jitter"):
                x = jitter_brightness(np.array(x, dtype=np.float32), self.opt_pos, self.mean_opt, self.std_opt,
                                      np.random, self.extra["jitter"])
            if self.glint_p and np.random.rand() < self.glint_p:
                x = np.array(x, dtype=np.float32)
                x = add_glint(x, np.asarray(y), self.opt_pos, self.mean_opt, self.std_opt, np.random)
            if self.cloud_p and np.random.rand() < self.cloud_p:
                x = np.array(x, dtype=np.float32)
                if self.extra.get("full") and np.random.rand() < self.extra["full"]:
                    alpha = np.full(x.shape[1:], np.random.uniform(0.85, 1.0), np.float32)  # thick cloud everywhere
                else:
                    alpha = cloud_alpha(*x.shape[1:], np.random, cover=(0.1, 0.9) if self.extra else (0.1, 0.7))
                if self.extra.get("shadow") and np.random.rand() < self.extra["shadow"]:
                    x = add_shadow(x, alpha, self.opt_pos, self.mean_opt, self.std_opt, np.random)
                x = add_clouds(x, alpha, self.opt_pos, self.mean_opt, self.std_opt, np.random)
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)))


def seed_worker(_):
    np.random.seed(torch.initial_seed() % 2 ** 32)  # each loader worker gets its own crops


# ---------- 5. scores ----------

def scores(pred, y):
    """Water IoU, F1 and water-area error over all valid pixels (labels -1 are left out), in %.
    IoU is None when neither map has water; area error is None when the label has no water."""
    ok = y >= 0
    p, t = (pred == 1) & ok, (y == 1) & ok
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    return {"iou": round(100 * tp / (tp + fp + fn), 1) if tp + fp + fn else None,
            "f1": round(100 * 2 * tp / (2 * tp + fp + fn), 1) if tp + fp + fn else None,
            "area_error": round(100 * (fp - fn) / (tp + fn), 1) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn}


def size_scores(pred, y):
    """By water-body size (8-connected pixels): the share of real water found (recall; bodies sized in the hand label)
    and the share of the map's water that is real (precision; bodies sized in the map, labels -1 left out).
    A body counts as found when the map marks at least half of its pixels as water."""
    # pixels found, pixels, bodies found, bodies, map pixels that are real water, map pixels with a label
    tot = {name: np.zeros(6, np.int64) for _, _, name in SIZE_BINS}
    for k in range(len(y)):
        ok = y[k] >= 0
        lab, n = ndimage.label(y[k] == 1, structure=np.ones((3, 3), bool))
        if n:
            sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            hits = np.bincount(lab.ravel(), weights=(pred[k] == 1).ravel().astype(np.float64), minlength=n + 1)[1:]
            for lo, hi, name in SIZE_BINS:
                m = (sizes >= lo) & (sizes < (hi or np.inf))
                tot[name][:4] += [int(hits[m].sum()), int(sizes[m].sum()), int((hits[m] >= 0.5 * sizes[m]).sum()),
                                  int(m.sum())]
        lab_p, n_p = ndimage.label(pred[k] == 1, structure=np.ones((3, 3), bool))
        if n_p:
            sizes_p = np.bincount(lab_p.ravel(), minlength=n_p + 1)[1:]
            real = np.bincount(lab_p.ravel(), weights=((y[k] == 1) & ok).ravel().astype(np.float64), minlength=n_p + 1)[1:]
            known = np.bincount(lab_p.ravel(), weights=ok.ravel().astype(np.float64), minlength=n_p + 1)[1:]
            for lo, hi, name in SIZE_BINS:
                m = (sizes_p >= lo) & (sizes_p < (hi or np.inf))
                tot[name][4:] += [int(real[m].sum()), int(known[m].sum())]
    out = {}
    for name, t in tot.items():
        rec = 100 * t[0] / t[1] if t[1] else None
        prec = 100 * t[4] / t[5] if t[5] else None
        out[name] = {"water_found": None if rec is None else round(rec), "bodies": int(t[3]),
                     "bodies_found": round(100 * t[2] / t[3]) if t[3] else None,
                     "precision": None if prec is None else round(prec),
                     "f1": round(2 * rec * prec / (rec + prec)) if rec and prec else None}
    return out


def per_chip(pred, y, ids):
    out = {}
    for k, cid in enumerate(ids):
        s, ok = scores(pred[k], y[k]), y[k] >= 0
        out[cid] = {"water_share": round(100 * float((y[k] == 1).sum()) / max(int(ok.sum()), 1), 1),
                    "iou": s["iou"], "area_error": s["area_error"]}
    return out


@torch.no_grad()
def predict(model, x, ch, device, batch=8):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch][:, ch].astype(np.float32)).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(xb).argmax(1).to(torch.int8).cpu().numpy())
    return np.concatenate(out)


def train_one(name, ch, data, args, device, log, cloud_p=0.0, mean=None, std=None, glint_p=0.0, seed=0, extra=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = smp.Unet(args.encoder, encoder_weights=None if args.dry_run else "imagenet",
                     in_channels=len(ch), classes=2).to(device)
    opt_pos = [i for i, c in enumerate(ch) if CHANNELS[c] in S2_USE]
    optical = [c for c in ch if CHANNELS[c] in S2_USE]
    chips = Chips(data["train"]["x"], data["train"]["y"], ch, args.crop, args.crops_per_chip, cloud_p, opt_pos,
                  None if mean is None else mean[optical], None if std is None else std[optical], glint_p, extra)
    loader = DataLoader(chips,
                        batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                        worker_init_fn=seed_worker, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    steps = args.max_steps or len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    ce, dice = nn.CrossEntropyLoss(ignore_index=-1), smp.losses.DiceLoss("multiclass", ignore_index=-1)
    best, history, train_secs, crops = {"iou": -1.0}, [], 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, done = time.time(), 0.0, 0
        for step, (xb, yb) in enumerate(loader):
            if step >= steps:
                break
            if (yb >= 0).sum() == 0:  # every pixel in the batch is "no data"
                continue
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(xb)
            loss = ce(logits.float(), yb) + dice(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, done, crops = total + loss.item(), done + 1, crops + len(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_secs += time.time() - t0
        val = scores(predict(model, data["valid"]["x"], ch, device), data["valid"]["y"])
        val_iou = val["iou"] or 0.0
        history.append({"epoch": epoch, "loss": round(total / max(done, 1), 4), "valid_iou": val_iou,
                        "secs": round(time.time() - t0, 1)})
        log(f"{name} epoch {epoch}/{args.epochs}: loss {history[-1]['loss']}, valid IoU {val_iou}%, "
            f"{history[-1]['secs']} s")
        if val_iou > best["iou"]:
            best = {"iou": val_iou, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model, best, history, {"train_secs": round(train_secs), "crops_per_sec": round(crops / max(train_secs, 1e-9), 1)}


# ---------- 6. pictures ----------

def save_examples(path, data, preds, ids, mean, std, picks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["Sentinel-2 (true colour)", "Sentinel-1 VV", "hand label"] + [f"U-Net: {n}" for n in preds]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.6 * len(cols), 2.6 * len(picks)), squeeze=False)
    for r, k in enumerate(picks):
        raw = data["x"][k].astype(np.float32) * std[:, None, None] + mean[:, None, None]
        rgb = np.stack([raw[CHANNELS.index(b)] for b in ["B4", "B3", "B2"]], -1)
        rgb = np.clip(rgb / np.nanpercentile(rgb, 98), 0, 1)
        label = np.where(data["y"][k] < 0, np.nan, data["y"][k])
        panels = [rgb, raw[0], label] + [p[k] for p in preds.values()]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=None if c == 0 else ("gray" if c == 1 else "Blues"), vmin=None if c < 2 else 0,
                      vmax=None if c < 2 else 1)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
        axes[r, 0].set_ylabel(ids[k], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/sen1floods11")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--crops-per-chip", type=int, default=4)
    ap.add_argument("--encoder", default="efficientnet-b0")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS, help=f"comma list from: {', '.join(VARIANTS)}")
    ap.add_argument("--seeds", default=DEFAULT_SEEDS, help="comma list of random seeds; each variant is trained once per seed")
    ap.add_argument("--dry-run", action="store_true", help="CPU check: 4 chips per split, 1 epoch, 2 steps")
    args = ap.parse_args()
    if args.dry_run:
        args.epochs, args.max_steps, args.workers, args.crop, args.batch = 1, 2, 0, 64, 2
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lines = []

    def log(msg):
        line = f"[{time.time() - started:6.0f} s] {msg}"
        print(line, flush=True)
        lines.append(line)
        (out / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    env = {"torch": torch.__version__, "smp": smp.__version__, "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    log(f"start: {env}")
    torch.backends.cudnn.benchmark = True

    ids = {s: chip_ids(root, s) for s in SPLITS}
    if args.dry_run:
        ids = {s: v[:4] for s, v in ids.items()}
    download_all(root, [c for v in ids.values() for c in v])
    log("data ready: " + ", ".join(f"{s} {len(v)}" for s, v in ids.items()))

    data, otsu, values = {}, {}, {}
    for s in SPLITS:
        x, y, otsu[s], values[s] = load_split(root, ids[s])
        data[s] = {"x": x, "y": y}
    mean = np.array([np.nanmean(data["train"]["x"][:, c]) for c in range(len(CHANNELS))], np.float32)
    std = np.array([np.nanstd(data["train"]["x"][:, c]) for c in range(len(CHANNELS))], np.float32)
    for s in SPLITS:  # scale, fill missing pixels with the mean (0 after scaling), store as float16
        x = (data[s]["x"] - mean[:, None, None]) / std[:, None, None]
        data[s]["x"] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
    log("bands scaled: " + ", ".join(f"{b} {m:.1f}/{d:.1f}" for b, m, d in zip(CHANNELS, mean, std)))

    thresholds = {k: tune_threshold(values["train"][k], data["train"]["y"], sign, edges)
                  for k, (sign, edges) in TUNED.items()}
    log(f"thresholds tuned on the training chips: {thresholds}")
    rule_names = {"NDWI": "NDWI", "MNDWI": "MNDWI", "VV": "radar VV (dB)"}
    rules = {s: {"radar threshold (dataset's Otsu map)": otsu[s],
                 "NDWI > 0": apply_rule(values[s]["NDWI"], +1, 0.0),
                 "MNDWI > 0": apply_rule(values[s]["MNDWI"], +1, 0.0),
                 **{f"{rule_names[k]} tuned ({'>=' if sign > 0 else '<'} {thresholds[k]:g})":
                    apply_rule(values[s][k], sign, thresholds[k]) for k, (sign, _) in TUNED.items()}}
             for s in EVAL}
    del values, otsu

    lanka = {s: [i for i, c in enumerate(ids[s]) if c.startswith("Sri-Lanka")] for s in SPLITS}
    report = {"settings": vars(args), "env": env, "chips": {s: len(v) for s, v in ids.items()},
              "sri_lanka_chips": {s: len(v) for s, v in lanka.items()},
              "scaling": {"bands": CHANNELS, "mean": mean.round(3).tolist(), "std": std.round(3).tolist()},
              "size_bins_pixels": [[lo, hi, name] for lo, hi, name in SIZE_BINS], "tuned_thresholds": thresholds,
              "simple_rules": {}, "unet": {}}

    def split_scores(pred_by_split):
        res = {}
        for s in EVAL:
            res[s] = scores(pred_by_split[s], data[s]["y"])
            if s != "valid":
                res[s]["by_size"] = size_scores(pred_by_split[s], data[s]["y"])
        if lanka["test"]:
            k = lanka["test"]
            res["sri_lanka_test"] = {**scores(pred_by_split["test"][k], data["test"]["y"][k]),
                                     "by_size": size_scores(pred_by_split["test"][k], data["test"]["y"][k])}
        res["per_chip"] = {s: per_chip(pred_by_split[s], data[s]["y"], ids[s]) for s in ["test", "bolivia"]}
        return res

    def summary(r):
        village, area = r["test"]["by_size"]["5 to 20 ha"], r["test"]["area_error"]
        return (f"test IoU {r['test']['iou']}%, area {'n/a' if area is None else f'{area:+}'}%, Bolivia {r['bolivia']['iou']}%"
                + (f", Sri Lanka test {r['sri_lanka_test']['iou']}%" if "sri_lanka_test" in r else "")
                + f", 5-20 ha bodies: water found {village['water_found']}%, precision {village['precision']}%")

    for rule in rules["test"]:
        report["simple_rules"][rule] = split_scores({s: rules[s][rule] for s in EVAL})
        log(f"rule {rule}: {summary(report['simple_rules'][rule])}")

    # the test chips with fake clouds, the same for every model (30 to 70% cover, thick or thin)
    rng = np.random.RandomState(123)
    opt_all = [CHANNELS.index(b) for b in S2_USE]
    test_alpha = np.stack([cloud_alpha(512, 512, rng, cover=(0.3, 0.7)) for _ in ids["test"]])
    test_cloudy = data["test"]["x"].astype(np.float32)
    for k in range(len(test_cloudy)):
        add_clouds(test_cloudy[k], test_alpha[k], opt_all, mean[opt_all], std[opt_all], rng)
    test_cloudy = test_cloudy.astype(np.float16)
    under = np.where(test_alpha >= 0.5, data["test"]["y"], -1).astype(np.int8)  # score only pixels under thick cloud
    log(f"fake clouds on the test chips: {100 * (test_alpha >= 0.5).mean():.0f}% of pixels under thick cloud")
    # the test chips with fake glint on the water, the same for every model
    glint_a = np.stack([glint_alpha(data["test"]["y"][k], rng) for k in range(len(ids["test"]))])
    test_glint = data["test"]["x"].astype(np.float32)
    for k in range(len(test_glint)):
        add_clouds(test_glint[k], glint_a[k], opt_all, mean[opt_all], std[opt_all], rng, brightness=(0.3, 1.0))
    test_glint = test_glint.astype(np.float16)
    under_glint = np.where(glint_a >= 0.3, data["test"]["y"], -1).astype(np.int8)  # bright water pixels only
    log(f"fake glint on the test chips: {100 * (under_glint == 1).sum() / max((data['test']['y'] == 1).sum(), 1):.0f}% "
        "of the water pixels")
    # run 6: the test chips under thick cloud everywhere, so only radar is left (scored on all pixels)
    full_cloud, rng_full = data["test"]["x"].astype(np.float32), np.random.RandomState(456)
    for k in range(len(full_cloud)):
        add_clouds(full_cloud[k], np.ones((512, 512), np.float32), opt_all, mean[opt_all], std[opt_all], rng_full)
    full_cloud = full_cloud.astype(np.float16)

    test_preds, cloudy_preds = {}, {}
    seeds = [int(s) for s in args.seeds.split(",")]
    for variant in args.variants.split(","):
        bands, cloud_p, glint_p, extra = VARIANTS[variant]
        ch = [CHANNELS.index(b) for b in bands]
        for seed in seeds:
            name = variant if len(seeds) == 1 else f"{variant} seed {seed}"
            model, best, history, speed = train_one(name, ch, data, args, device, log, cloud_p, mean, std, glint_p, seed,
                                                    extra)
            preds = {s: predict(model, data[s]["x"], ch, device) for s in EVAL}
            cloudy = predict(model, test_cloudy, ch, device)
            if seed == seeds[0]:
                test_preds[variant], cloudy_preds[variant] = preds["test"], cloudy
            g = scores(predict(model, test_glint, ch, device), under_glint)
            report["unet"][name] = {"variant": variant, "seed": seed, "bands": bands, "cloud_share": cloud_p,
                                    "glint_share": glint_p, "extra": extra, "best_epoch": best["epoch"],
                                    **split_scores(preds),
                                    "test_with_clouds": scores(cloudy, data["test"]["y"]),
                                    "under_clouds": scores(cloudy, under),
                                    "test_full_cloud": scores(predict(model, full_cloud, ch, device), data["test"]["y"]),
                                    "water_found_under_glint": round(100 * g["tp"] / max(g["tp"] + g["fn"], 1)),
                                    "speed": speed, "history": history}
            tag = "" if seed == seeds[0] else f"_seed{seed}"  # every seed is kept, to test all of them on our tanks
            torch.save({"state_dict": best["state"], "encoder": args.encoder, "bands": bands,
                        "mean": mean[ch].tolist(), "std": std[ch].tolist()},
                       out / f"unet_{variant.replace('+', '_')}{tag}_best.pt")
            r = report["unet"][name]
            log(f"{name}: best epoch {best['epoch']}, {summary(r)}; with fake clouds: test IoU "
                f"{r['test_with_clouds']['iou']}%, under the clouds {r['under_clouds']['iou']}%, under full cloud "
                f"{r['test_full_cloud']['iou']}%; water found under fake glint {r['water_found_under_glint']}%")
            report["total_secs"] = round(time.time() - started)
            (out / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        if len(seeds) > 1:  # mean and spread over the seeds
            runs = [v for v in report["unet"].values() if v.get("variant") == variant]
            pick = {"test IoU": lambda v: v["test"]["iou"], "Bolivia IoU": lambda v: v["bolivia"]["iou"],
                    "Sri Lanka IoU": lambda v: v.get("sri_lanka_test", {}).get("iou"),
                    "under clouds IoU": lambda v: v["under_clouds"]["iou"],
                    "full cloud IoU": lambda v: v["test_full_cloud"]["iou"],
                    "5-20 ha water found": lambda v: v["test"]["by_size"]["5 to 20 ha"]["water_found"],
                    "5-20 ha precision": lambda v: v["test"]["by_size"]["5 to 20 ha"]["precision"]}
            def spread(f):
                vals = [x for x in (f(v) for v in runs) if x is not None]
                return {"mean": round(float(np.mean(vals)), 1), "sd": round(float(np.std(vals)), 1)} if vals else None

            report.setdefault("over_seeds", {})[variant] = {k: spread(f) for k, f in pick.items()}
            log(f"{variant} over {len(seeds)} seeds: " + ", ".join(
                f"{k} {s['mean']} +/- {s['sd']}" for k, s in report["over_seeds"][variant].items() if s))
            (out / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    picks = (lanka["test"][:3] + [i for i in range(len(ids["test"])) if i not in lanka["test"]][:3])[:6]
    save_examples(out / "examples_test_chips.png", data["test"], test_preds, ids["test"], mean, std, picks)
    save_examples(out / "examples_test_chips_with_fake_clouds.png", {"x": test_cloudy, "y": data["test"]["y"]},
                  cloudy_preds, ids["test"], mean, std, picks)
    log(f"done in {round((time.time() - started) / 60)} min")


if __name__ == "__main__":
    main()
