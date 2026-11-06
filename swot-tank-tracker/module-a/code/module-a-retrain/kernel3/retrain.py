r"""Module A retraining on Kaggle, run 3 (3 Nov 2026), under ../PROTOCOL-3.md (fixed before the run).

What changed from run 2 (../kernel2/retrain.py), with the reason:
- It starts from run 6 (Bolivia 66.0; seed s from run 6's seed s), not run 2, to keep flood-mapping skill.
- Every 4th crop (4 of each batch of 16) is a Sen1Floods11 hand-labelled training chip: expert water is water of
  either kind, the rest land. The model keeps practising floods. Bolivia and the test split are never trained on.
- A fair control ("photo"): photo labels and no storage, on clear days and on every radar pass within 5 days of a
  clear photo. Run 2's control saw clear days only.
- labels.py version 2.1 (the dry-bed fix).
- Sampling picks a reservoir first, then one of its chips, so that the big reservoirs don't dominate.
- No downloads during the run: Sen1Floods11 comes in a private dataset.
- Three Kaggle jobs, each far below the 12-hour limit (run 2 took 659 of 720 minutes):
  - PART "prep" (CPU): the label inputs for the 103 reservoirs, saved as res.pkl;
  - PART "A" (GPU, Kaggle): folds 1-3; "B1" (GPU, Colab): fold 4; "B2" (GPU, Kaggle): fold 5. The tables are
    written after every fold. (Colab's free session lasts about 5 hours on 2 CPUs, too short for folds 4 and 5.)
- Only the two new models are tested. Run 2, run 6, the tuned threshold and Otsu give the same maps as in run 2, so
  ../scripts/merge_run3.py takes their areas from run 2's output, adds the new ones and scores everything.
Same reservoirs and folds as run 2 (its folds.csv), and the same training: 10 epochs of 300 batches of 16, lr 3e-4,
the last epoch kept.
On Kaggle it runs with no arguments (PART is set by ../scripts/make_run3_kernels.py). Quick local check on a CPU,
all parts in one go (a few reservoirs, a few steps):
    python -I retrain.py --dry-run --input <project folder> --out <folder>
"""
import argparse
import csv
import json
import pickle
import re
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import ndimage, stats


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
import torch.nn.functional as F
from rasterio.warp import transform_bounds
from torch.utils.data import DataLoader, Dataset

PART = "A"  # "prep", "A", "B1" or "B2"; set by ../scripts/make_run3_kernels.py
FOLDS_OF = {"A": (0, 1, 2), "B": (3, 4), "B1": (3,), "B2": (4,)}  # B1 on Colab, B2 on Kaggle
ACFT_TO_MCM = 1233.48 / 1e6
OLD, R6, NEW, NEW_OPEN, NEW_PLANTS, CTRL = ("radar U-Net r2", "radar U-Net r6 (3 seeds)", "retrained, water",
                                            "retrained, open water", "retrained, water under plants",
                                            "photo-label control, water")
R2NEW, R2CTRL = "run 2 retrained, water", "run 2 photo-label control, water"
TUNED, OTSU = "VV threshold tuned to the photo", "Otsu radar threshold"
METHODS = (OLD, R6, R2NEW, R2CTRL, NEW, NEW_OPEN, CTRL, TUNED, OTSU)  # scored by merge_run3.py
PIXEL = (OLD, R6, R2NEW, NEW, CTRL, TUNED, OTSU)  # methods scored pixel by pixel
BASELINES = ("no model", "seasonal")
MIX = {"rule": {"clear": 0.4, "cloudy": 0.1, "pass": 0.5}, "photo": {"clear": 0.4, "pass": 0.6}}
FLOOD_EVERY = 4  # every 4th crop is a Sen1Floods11 chip: 4 of each batch of 16
MIN_PASSES = 20
TRANSFER_DAYS, TRANSFER_STORAGE = 5, 0.01
SIZES = (("all", 0, 1e9), ("under 5 million m3", 0, 5), ("5 to 30 million m3", 5, 30),
         ("30 million m3 and more", 30, 1e9))
MEASURES = ("spearman_all", "spearman_clear", "spearman_cloudy", "later_error_pct", "later_full_error_pct")
BANDS_STORAGE = ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.85), (0.85, 9))
L = None  # labels.py, imported once its folder is known


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------- 1. data ----------

def find(base, pattern, up, must=""):
    hits = sorted(p for p in Path(base).rglob(pattern) if must in p.as_posix())
    if not hits:
        raise FileNotFoundError(f"{pattern} ({must}) under {base}")
    p = hits[0]
    for _ in range(up):
        p = p.parent
    return p


def paths(args):
    """Where each input is. The prep job needs no weights and no res.pkl."""
    if args.dry_run:  # the project folder on the laptop
        base = Path(args.input)
        w6 = base / "temp/module-a-tanks/weights/run-6"
        return {"tables": base / "temp/module-a-retrain/kaggle-data/tables",
                "first": base / "temp/module-a-retrain/kaggle-data/first-check",
                "passes": base / "temp/module-a-retrain/kaggle-data/passes",
                "code": base / "temp/module-a-retrain/scripts",
                "folds2": base / "temp/module-a-retrain/runs/2026-11-03-retrain-2/folds.csv",
                "s1f": base / "temp/module-a-kaggle/downloads/sen1floods11/v1.1",
                "prep": Path(args.out),
                "r6": [w6 / "unet_radar_best.pt", w6 / "unet_radar_seed1_best.pt", w6 / "unet_radar_seed2_best.pt"]}
    i = args.input
    p = {"tables": find(i, "storage_daily.csv", 1), "first": find(i, "Akathimuruppu_2023-07-04.tif", 2),
         "passes": find(i, "Akathimuruppu/2020-10-11.tif", 2), "code": find(i, "labels.py", 1),
         "folds2": find(i, "run2_folds.csv", 0), "s1f": find(i, "flood_train_data.csv", 3)}
    if args.part != "prep":
        r6 = find(i, "unet_radar_seed1_best.pt", 1, "run-6")
        p["r6"] = [r6 / "unet_radar_best.pt", r6 / "unet_radar_seed1_best.pt", r6 / "unet_radar_seed2_best.pt"]
        p["prep"] = find(i, "res.pkl", 1)
    return p


def clean(s, cap):
    """Drops one-day spikes (over 10% of capacity from both neighbours, which agree within 2%) and flat runs of 60
    days or more below 98% full (stale entries). Returns the cleaned series and what was dropped."""
    days = sorted(s)
    v = np.array([s[d] for d in days]) / cap
    keep = np.ones(len(v), bool)
    spikes = 0
    for i in range(1, len(v) - 1):
        if (days[i + 1] - days[i - 1]).days == 2 and abs(v[i - 1] - v[i + 1]) <= 0.02 \
                and min(abs(v[i] - v[i - 1]), abs(v[i] - v[i + 1])) > 0.10:
            keep[i], spikes = False, spikes + 1
    flat, i = 0, 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i] and (days[j + 1] - days[j]).days == 1:
            j += 1
        if j - i + 1 >= 60 and v[i] < 0.98:
            keep[i:j + 1], flat = False, flat + j - i + 1
        i = j + 1
    over = float((v > 1.05).mean()) if len(v) else 0.0
    return {d: s[d] for d, k in zip(days, keep) if k}, {"spikes": spikes, "flat_days": flat,
                                                          "share_above_105": round(over, 3)}


def sheet(tables):
    info, raw = {}, {}
    with open(tables / "storage_daily.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["reservoir"]:
                continue
            info.setdefault(r["reservoir"], {"kind": r["kind"], "cap": float(r["capacity_acft"]) * ACFT_TO_MCM})
            if r["storage_acft"]:
                raw.setdefault(r["reservoir"], {})[date.fromisoformat(r["date"])] = \
                    float(r["storage_acft"]) * ACFT_TO_MCM
    series, report = {}, []
    for n, s in raw.items():
        series[n], rep = clean(s, info[n]["cap"])
        report.append({"reservoir": n, **rep, "capacity_doubtful": rep["share_above_105"] > 0.2})
    return info, series, report


def nearest(s, d, days=2):
    for k in range(days + 1):
        for dd in (d - timedelta(days=k), d + timedelta(days=k)):
            if dd in s:
                return s[dd]
    return None


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")  # as export_reservoirs.safe


def read_full(path):
    """VV, VH (dB) and 10 Sentinel-2 bands (NaN = no data); JRC occurrence; Cloud Score+."""
    with rasterio.open(path) as src:
        a = src.read().astype(np.float32)
    s1, s2 = a[:2], a[2:12]
    s1[s1 == -32768] = np.nan
    s1 /= 100
    s2[s2 == 0] = np.nan
    return np.concatenate([s1, s2]), a[12], a[13] / 100


def read_radar(path):
    with rasterio.open(path) as src:
        a = src.read([1, 2]).astype(np.float32)
    a[a == -32768] = np.nan
    return a / 100


def tank_mask(occ):
    """The JRC water body the export centred the chip on (the largest body whose box is centred within 5 pixels of the
    chip centre), widened by 30 m."""
    lab, n = ndimage.label(occ > 0)
    h, w = occ.shape
    if n == 0:
        return np.zeros(occ.shape, bool)
    sizes = ndimage.sum(np.ones(occ.shape), lab, index=np.arange(1, n + 1))
    off = [np.hypot((s[0].start + s[0].stop - 1) / 2 - (h - 1) / 2, (s[1].start + s[1].stop - 1) / 2 - (w - 1) / 2)
           for s in ndimage.find_objects(lab)]
    near = [i for i in range(n) if off[i] <= 5]
    k = 1 + (max(near, key=lambda i: sizes[i]) if near else int(np.argmin(off)))
    return ndimage.binary_dilation(lab == k, iterations=3)


def prepare(name, info, series, p):
    """The reservoir's outline, cover range, label context and chips with their storage. A radar pass gets "transfer"
    (the rule's photo label: a clear photo within 5 days, storage within 1% of capacity) and "photo_near" (the control's:
    the nearest clear photo within 5 days, no storage)."""
    cap, s, sf = info[name]["cap"], series.get(name, {}), safe(name)
    with open(p["tables"] / "manifests" / f"{sf}.csv", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    first = p["first"] / sf / man[0]["file"]
    _, occ, _ = read_full(first)
    with rasterio.open(first) as src:
        lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    inside = tank_mask(occ)
    clear, cloudy, photos, s_photo = [], [], [], []
    for r in man:
        sp, sr = nearest(s, date.fromisoformat(r["s2_date"])), nearest(s, date.fromisoformat(r["s1_date"]))
        path = p["first"] / sf / r["file"]
        if r["kind"] == "clear" and sp is not None:
            x, _, cs = read_full(path)
            photos.append(L.photo_class(x, cs))
            s_photo.append(sp / cap)
            clear.append({**r, "path": path, "s_photo": sp / cap, "s_radar": (sr if sr is not None else sp) / cap})
        elif r["kind"] == "cloudy" and sr is not None:
            cloudy.append({**r, "path": path, "s": sr / cap})
    if photos:
        lo, hi, _ = L.fit_cover(np.stack(photos), np.array(s_photo))
    else:
        lo, hi = np.full(inside.shape, -np.inf), np.full(inside.shape, np.inf)
    passes = []
    with open(p["tables"] / "pass_manifests" / f"{sf}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["date"])
            st = nearest(s, d)
            e = {**r, "path": p["passes"] / r["file"], "s": None if st is None else st / cap, "transfer": None}
            gap = lambda c: abs((date.fromisoformat(c["s2_date"]) - d).days)  # noqa: E731
            near5 = [c for c in clear if gap(c) <= TRANSFER_DAYS]
            e["photo_near"] = min(near5, key=gap) if near5 else None
            if st is not None:
                near = [c for c in near5 if abs(c["s_photo"] - st / cap) <= TRANSFER_STORAGE]
                if near:
                    e["transfer"] = min(near, key=gap)
            passes.append(e)
    rr, cc = np.where(inside)
    known = float((np.isfinite(lo) | np.isfinite(hi))[inside].mean())
    return {"name": name, "safe": sf, "cap": cap, "inside": inside, "ctx": L.context(occ, inside),
            "lo": lo.astype(np.float32), "hi": hi.astype(np.float32), "box": (rr.min(), rr.max(), cc.min(), cc.max()),
            "clear": clear, "cloudy": cloudy, "passes": passes, "known": known,
            "bounds": (lon0, lat0, lon1, lat1), "centre": ((lat0 + lat1) / 2, (lon0 + lon1) / 2)}


def label_of(R, kind, e, mode="rule"):
    """The radar input (VV, VH in dB) and the label of one chip."""
    if mode == "photo":  # the fair control: the photo alone, on its own day or on a radar pass within 5 days of it
        c = e if kind == "clear" else e["photo_near"]
        xc, _, cs = read_full(c["path"])
        y = L.photo_label(L.photo_class(xc, cs), R["ctx"])
        x = xc[:2] if kind == "clear" else read_radar(e["path"])
    elif kind == "clear":
        x, _, cs = read_full(e["path"])
        y = L.clear_label(L.photo_class(x, cs), R["lo"], R["hi"], R["ctx"], e["s_photo"], e["s_radar"])
        x = x[:2]
    elif kind == "pass" and e.get("transfer"):
        t = e["transfer"]
        xc, _, csc = read_full(t["path"])
        y = L.clear_label(L.photo_class(xc, csc), R["lo"], R["hi"], R["ctx"], t["s_photo"], e["s"])
        x = read_radar(e["path"])
    else:
        x = read_radar(e["path"])
        y = L.radar_label(R["lo"], R["hi"], R["ctx"], e["s"])
    y[np.isnan(x).any(0)] = L.UNSURE
    return x, y


def s1f_split(root, split):
    """Sen1Floods11 hand-labelled chips of one split: (S1Hand path, LabelHand path)."""
    hand = Path(root) / "data/flood_events/HandLabeled"
    with open(Path(root) / f"splits/flood_handlabeled/flood_{split}_data.csv", encoding="utf-8") as f:
        ids = [ln.split(",")[0].strip().replace("_S1Hand.tif", "") for ln in f if ln.strip()]
    return [(hand / "S1Hand" / f"{c}_S1Hand.tif", hand / "LabelHand" / f"{c}_LabelHand.tif") for c in ids]


def flood_raw(s1_path, label_path):
    """VV, VH (dB, NaN = no data) and the expert label (1 water, 0 not water, -1 no data)."""
    with rasterio.open(s1_path) as src:
        x = src.read([1, 2]).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    with rasterio.open(label_path) as src:
        y = src.read(1)
    return x, y


def flood_chip(s1_path, label_path):
    """A Sen1Floods11 chip for training: expert water is water of either kind, the rest land; no data is unsure."""
    x, lab = flood_raw(s1_path, label_path)
    y = np.full(lab.shape, L.UNSURE, np.uint8)
    y[lab == 0], y[lab == 1] = L.LAND, L.ANY
    y[np.isnan(x).any(0)] = L.UNSURE
    return x, y


def coverage(res, names):
    """Label shares inside the outline, by source and storage band, on up to 3 chips per source and reservoir."""
    acc = {}
    for n in names:
        R = res[n]
        sources = {"clear": [("clear", e, "rule") for e in R["clear"]],
                   "pass, photo nearby": [("pass", e, "rule") for e in R["passes"] if e["transfer"]],
                   "pass, rule only": [("pass", e, "rule") for e in R["passes"] if e["s"] is not None and
                                       not e["transfer"]],
                   "cloudy": [("cloudy", e, "rule") for e in R["cloudy"]],
                   "control, pass within 5 days of a photo": [("pass", e, "photo") for e in R["passes"]
                                                              if e["photo_near"] and e["s"] is not None]}
        for src, items in sources.items():
            for kind, e, mode in items[:: max(1, len(items) // 3)][:3]:
                s = e["s_photo"] if kind == "clear" else e["s"]
                b = next(f"{int(100 * lo)}-{int(100 * hi) if hi < 9 else 'up'}%" for lo, hi in BANDS_STORAGE
                         if lo <= s < hi)
                _, y = label_of(R, kind, e, mode)
                v = y[R["inside"]]
                a = acc.setdefault((src, b), np.zeros(5))
                a += [(v == c).sum() for c in (L.LAND, L.OPEN, L.PLANTS, L.ANY, L.UNSURE)]
    return [{"source": src, "storage": b, **{k: round(float(a[i] / max(a.sum(), 1)), 3) for i, k in
                                             enumerate(("land", "open", "under_plants", "either", "unsure"))}}
            for (src, b), a in sorted(acc.items())]


# ---------- 2. training ----------

class Crops(Dataset):
    """Every FLOOD_EVERY-th crop is a Sen1Floods11 training chip. The others pick a kind by MIX, then a reservoir at
    random among those with chips of that kind, then one of its chips."""

    def __init__(self, res, names, flood, mean, std, crop, n, seed, mode):
        self.res, self.flood, self.mean, self.std, self.crop, self.n, self.seed, self.mode, self.epoch = \
            res, flood, mean, std, crop, n, seed, mode, 0
        mix = MIX[mode]
        self.by = {k: {} for k in mix}
        for name in names:
            R = res[name]
            pools = {"clear": R["clear"], "cloudy": R["cloudy"],
                     "pass": [e for e in R["passes"] if (e["photo_near"] if mode == "photo" else e["s"] is not None)]}
            for k in mix:
                if pools[k]:
                    self.by[k][name] = pools[k]
        self.kinds = [k for k in mix if self.by[k]]
        w = np.array([mix[k] for k in self.kinds])
        self.weights = w / w.sum()
        self.names = {k: sorted(self.by[k]) for k in self.kinds}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        box = None
        if self.flood and i % FLOOD_EVERY == 0:
            x, y = flood_chip(*self.flood[rng.integers(len(self.flood))])
        else:
            kind = self.kinds[rng.choice(len(self.kinds), p=self.weights)]
            name = self.names[kind][rng.integers(len(self.names[kind]))]
            pool = self.by[kind][name]
            e = pool[rng.integers(len(pool))]
            R = self.res[name]
            x, y = label_of(R, kind, e, self.mode)
            box = R["box"]
        h, w = y.shape
        c = self.crop
        if box is not None and rng.random() < 0.8:  # centred on the outline
            r0, r1, c0, c1 = box
            cy, cx = rng.integers(r0, r1 + 1), rng.integers(c0, c1 + 1)
        else:
            cy, cx = rng.integers(h), rng.integers(w)
        top, left = int(np.clip(cy - c // 2, 0, h - c)), int(np.clip(cx - c // 2, 0, w - c))
        x, y = x[:, top:top + c, left:left + c], y[top:top + c, left:left + c]
        x = np.nan_to_num((x - self.mean[:, None, None]) / self.std[:, None, None], nan=0.0)
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))


def loss_fn(logits, y):
    logp = F.log_softmax(logits.float(), dim=1)
    one = -logp.gather(1, y.clamp(max=2).unsqueeze(1)).squeeze(1)
    either = -torch.logsumexp(logp[:, 1:3], dim=1)
    loss = torch.where(y <= 2, one, torch.where(y == 3, either, torch.zeros_like(one)))
    return loss.sum() / (y <= 3).sum().clamp(min=1)


def load_model(path, classes, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=len(ck["bands"]), classes=classes)
    sd = dict(ck["state_dict"])
    if classes == 3:  # water under plants starts as a copy of water, less likely
        w, b = sd["segmentation_head.0.weight"], sd["segmentation_head.0.bias"]
        sd["segmentation_head.0.weight"] = torch.cat([w, w[1:2]], 0)
        sd["segmentation_head.0.bias"] = torch.cat([b, b[1:2] - 2.0], 0)
    model.load_state_dict(sd)
    return model.to(device), ck


def train(res, names, flood, start, args, device, seed, mode):
    torch.manual_seed(seed)
    model, ck = load_model(start, 3, device)
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
    data = Crops(res, names, flood, mean, std, args.crop, args.steps * args.batch, seed, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * args.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    for epoch in range(args.epochs):
        data.epoch = epoch
        loader = DataLoader(data, batch_size=args.batch, shuffle=False, drop_last=True, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
        model.train()
        t0, total, done = time.time(), 0.0, 0
        for xb, yb in loader:
            if (yb <= 3).sum() == 0:
                continue
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, done = total + loss.item(), done + 1
        log(f"   {mode} seed {seed} epoch {epoch + 1}/{args.epochs}: loss {total / max(done, 1):.4f}, "
            f"{time.time() - t0:.0f} s")
    model.eval()
    return model


# ---------- 3. testing ----------

def probs(models, zt, h, w):
    """Mean softmax over a list of models."""
    return torch.stack([torch.softmax(m(zt).float()[0, :, :h, :w], 0) for m in models]).mean(0).cpu().numpy()


def maps(models, x, mean, std, device):
    """Water maps of one chip from the two new ensembles: water = p_land < 0.5 (open or plants by the larger)."""
    z = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)
    h, w = z.shape[1:]
    ph, pw = (-h) % 32, (-w) % 32
    zt = F.pad(torch.from_numpy(z.astype(np.float32))[None], (0, pw, 0, ph), mode="reflect").to(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        p = probs(models["rule"], zt, h, w)
        pc = probs(models["photo"], zt, h, w) if models.get("photo") else None
    water = p[0] < 0.5
    out = {NEW: water, NEW_OPEN: water & (p[1] >= p[2]), NEW_PLANTS: water & (p[2] > p[1])}
    if pc is not None:
        out[CTRL] = pc[0] < 0.5
    return out


def consistency(acc, m, ref, ok):
    """Pixel counts against the storage reference (labels.rule: 1 under water, 0 dry), on decided pixels with radar."""
    d = ok & (ref >= 0)
    t = ref == 1
    for k in PIXEL:
        if k in m:
            a = acc.setdefault(k, [0, 0, 0, 0])
            a[0] += int((m[k] & t & d).sum())
            a[1] += int((m[k] & ~t & d).sum())
            a[2] += int((~m[k] & t & d).sum())
            a[3] += int(d.sum())
    acc.setdefault("_outline", [0])[0] += int(ok.sum())


def iou_f1(tp, fp, fn):
    return (round(100 * tp / (tp + fp + fn), 1) if tp + fp + fn else np.nan,
            round(100 * 2 * tp / (2 * tp + fp + fn), 1) if tp + fp + fn else np.nan)


def areas(m, inside, vv):
    row = {"outline_ha": round(float(inside.sum()) * 0.01, 1),
           "radar_gap_share": round(float(np.isnan(vv[inside]).mean()), 3)}
    row.update({k: round(float((v & inside).sum()) * 0.01, 1) for k, v in m.items()})
    return row


def test(R, models, mean, std, device, limit=0):
    """As run 2's test, for the new maps only: areas on every pass and first-check chip; pixel scores against the
    storage reference (all; under 20%; 85% and more; cloudy-photo passes); and, on clear days under 20% of capacity,
    each map's water share against the photo's on pixels where the photo is sure (open water or bare ground)."""
    inside = R["inside"]
    rows, acc = [], {g: {} for g in ("all", "low", "full", "cloudy")}
    for e in (R["passes"][:limit] if limit else R["passes"]):
        x = read_radar(e["path"])
        m = maps(models, x, mean, std, device)
        rows.append({**{k: e[k] for k in ("tank", "date", "orbit", "pass", "photo_date", "photo_clear")},
                     **areas(m, inside, x[0])})
        if e["s"] is not None:
            ref, ok = L.rule(R["lo"], R["hi"], inside, e["s"]), inside & np.isfinite(x[0])
            groups = ["all"] + (["low"] if e["s"] < 0.2 else []) + (["full"] if e["s"] >= 0.85 else []) + \
                (["cloudy"] if e["photo_clear"] not in ("", None) and float(e["photo_clear"]) < 0.7 else [])
            for g in groups:
                consistency(acc[g], m, ref, ok)
    cons = []
    for g, a in acc.items():
        outline = a.pop("_outline", [0])[0]
        for k, (tp, fp, fn, dec) in a.items():
            i, f1 = iou_f1(tp, fp, fn)
            cons.append({"reservoir": R["name"], "passes": g, "method": k, "iou": i, "f1": f1, "tp": tp, "fp": fp,
                         "fn": fn, "decided_share": round(dec / max(outline, 1), 3)})
    first = []
    for kind, items in (("clear", R["clear"]), ("cloudy", R["cloudy"])):
        for e in (items[:limit] if limit else items):
            x, _, cs = read_full(e["path"])
            m = maps(models, x[:2], mean, std, device)
            row = {"tank": R["name"], "kind": kind, "file": e["file"], "s2_date": e["s2_date"], "s1_date": e["s1_date"],
                   **areas(m, inside, x[0])}
            if kind == "clear" and e["s_photo"] < 0.2:
                photo = L.photo_class(x, cs)
                sure = inside & np.isfinite(x[0]) & ((photo == L.P_WATER) | (photo == L.P_BARE))
                if sure.sum() > 0:
                    share = float((photo == L.P_WATER)[sure].mean())
                    row.update({"storage": round(e["s_photo"], 3), "photo_share": round(share, 3),
                                **{f"lowdiff {k}": round(abs(float(m[k][sure].mean()) - share), 3)
                                   for k in PIXEL if k in m}})
            first.append(row)
    return rows, first, cons


def flood_scores(models, mean, std, device, root, limit=0):
    """Water IoU (radar only) of the new ensembles on Sen1Floods11's Bolivia chips (a flood kept out of all training)
    and on its test split, scored as run 2's bolivia(): pixels with an expert label (0 or 1); the retrained models'
    water is open water plus water under plants."""
    out = {}
    for split in ("bolivia", "test"):
        keys = (NEW, CTRL) if models.get("photo") else (NEW,)
        tot = {k: [0, 0, 0] for k in keys}
        chips = s1f_split(root, split)
        for s1, lab in (chips[:limit] if limit else chips):
            x, y = flood_raw(s1, lab)
            m = maps(models, x, mean, std, device)
            ok, t = y >= 0, y == 1
            for k in keys:
                tot[k][0] += int((m[k] & t & ok).sum())
                tot[k][1] += int((m[k] & ~t & ok).sum())
                tot[k][2] += int((~m[k] & t & ok).sum())
        out.update({f"{split}: {k}": iou_f1(*v)[0] for k, v in tot.items()})
    return out


# ---------- 4. scores (used by merge_run3.py on the laptop) ----------

def fit(a, s):
    keep = (s > 0) & (a > 0)
    if keep.sum() < 3:
        return None
    b, c = np.polyfit(np.log(a[keep]), np.log(s[keep]), 1)
    return lambda area: np.exp(c + b * np.log(np.maximum(area, 0.01)))


def rho(a, s):
    return float(stats.spearmanr(a, s)[0]) if len(a) >= 5 else np.nan


def med(v, nd=2):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), nd) if v else None


def score_passes(rows, series, cap):
    keep = []
    for r in rows:
        s = nearest(series, date.fromisoformat(r["date"]))
        if s is not None and float(r["radar_gap_share"]) <= 0.05:
            pc = float(r["photo_clear"]) if r["photo_clear"] not in ("", None) else None
            keep.append({**r, "storage": s, "pc": pc})
    keep.sort(key=lambda r: r["date"])
    if len(keep) < MIN_PASSES:
        return None
    s = np.array([r["storage"] for r in keep])
    clear = np.array([r["pc"] is not None and r["pc"] >= 0.98 for r in keep])
    cloudy = np.array([r["pc"] is not None and r["pc"] < 0.7 for r in keep])
    half = len(keep) // 2
    doy = np.array([date.fromisoformat(r["date"]).timetuple().tm_yday for r in keep])
    seasonal = []  # as the review's extra_checks.py: first-half passes within 15 days of the same day of the year
    for d in doy[half:]:
        near = s[:half][np.minimum(np.abs(doy[:half] - d), 365 - np.abs(doy[:half] - d)) <= 15]
        seasonal.append(np.median(near) if len(near) else np.median(s[:half]))
    out = {"passes": len(keep),
           "no model": {"later_error_pct": 100 * float(np.median(np.abs(np.median(s[:half]) - s[half:]))) / cap},
           "seasonal": {"later_error_pct": 100 * float(np.median(np.abs(np.array(seasonal) - s[half:]))) / cap}}
    for m in METHODS:
        if m not in keep[0]:
            continue
        a = np.array([float(r[m]) for r in keep])
        f = fit(a[:half], s[:half])
        full = (s >= 0.85 * cap) & (np.arange(len(s)) >= half)
        out[m] = {"spearman_all": rho(a, s), "spearman_clear": rho(a[clear], s[clear]),
                  "spearman_cloudy": rho(a[cloudy], s[cloudy]),
                  "later_error_pct": 100 * float(np.median(np.abs(f(a[half:]) - s[half:]))) / cap if f else np.nan,
                  "later_full_error_pct": 100 * float(np.median(f(a[full]) - s[full])) / cap if f and full.sum() >= 3 else np.nan}
    return out


def wilcoxon(a, b, lower_better):
    d = np.array([x - y for x, y in zip(a, b) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)])
    d = d * (-1 if lower_better else 1)
    if len(d) < 5:
        return None
    try:
        p = float(f"{stats.wilcoxon(d)[1]:.2g}")
    except ValueError:  # every difference is zero
        p = 1.0
    return {"reservoirs": len(d), "median_gain": round(float(np.median(d)), 3),
            "share_first_better": round(float((d > 0).mean()), 2), "wilcoxon_p": p}


def write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------- main ----------

def prep(args, p, out):
    """The CPU job: cleaned storage, run 2's folds, every reservoir's label inputs, saved as res.pkl."""
    info, series, cleaning = sheet(p["tables"])
    write(out / "cleaning.csv", cleaning)
    doubtful = {c["reservoir"] for c in cleaning if c["capacity_doubtful"]}
    with open(p["tables"] / "pass_flags.csv", encoding="utf-8") as f:
        used = [r["reservoir"] for r in csv.DictReader(f) if r["used"] == "True" and r["reservoir"] not in doubtful]
    if args.only:
        used = [n for n in used if n in args.only.split(",")]
    with open(p["folds2"], encoding="utf-8") as f:
        fold2 = {r["reservoir"]: int(r["fold"]) for r in csv.DictReader(f)}
    missing = [n for n in used if n not in fold2]
    if missing:
        raise RuntimeError(f"not in run 2's folds: {missing}")
    fold = {n: fold2[n] % args.folds for n in used}
    t0 = time.time()
    res = {n: prepare(n, info, series, p) for n in used}
    write(out / "folds.csv", [{"reservoir": n, "fold": fold[n]} for n in used])
    write(out / "reservoirs.csv", [{"reservoir": n, "cap_mcm": round(R["cap"], 2), "fold": fold[n],
                                    "clear": len(R["clear"]), "cloudy": len(R["cloudy"]), "passes": len(R["passes"]),
                                    "passes_with_photo_label": sum(bool(e["transfer"]) for e in R["passes"]),
                                    "passes_with_control_label": sum(bool(e["photo_near"]) for e in R["passes"]),
                                    "outline_known_share": round(R["known"], 3)} for n, R in res.items()])
    write(out / "label_coverage.csv", coverage(res, used))
    for R in res.values():  # paths relative to their dataset, so that the GPU jobs can re-root them
        for e in R["clear"] + R["cloudy"]:
            e["path"] = ("first", Path(e["path"]).relative_to(p["first"]).as_posix())
        for e in R["passes"]:
            e["path"] = ("passes", Path(e["path"]).relative_to(p["passes"]).as_posix())
    with open(out / "res.pkl", "wb") as f:
        pickle.dump({"res": res, "fold": fold, "used": used, "doubtful": sorted(doubtful),
                     "labels_version": L.VERSION}, f)
    log(f"prepared {len(res)} reservoirs in {time.time() - t0:.0f} s; folds of "
        f"{[sum(fold[n] == k for n in used) for k in range(args.folds)]} reservoirs (run 2's)")


def train_part(args, p, out, device):
    """A GPU job: train the two modes for its folds, then test them on the held-out reservoirs, fold by fold."""
    with open(p["prep"] / "res.pkl", "rb") as f:
        d = pickle.load(f)
    res, fold, used = d["res"], d["fold"], d["used"]
    for R in res.values():
        for e in R["clear"] + R["cloudy"] + R["passes"]:
            if isinstance(e["path"], tuple):
                root, rel = e["path"]
                e["path"] = p[root] / rel
    flood = s1f_split(p["s1f"], "train")
    log(f"{len(res)} reservoirs (labels {d['labels_version']}); {len(flood)} Sen1Floods11 training chips")
    seeds = [int(s) for s in args.seeds.split(",")]
    modes = args.modes.split(",")
    ck = torch.load(p["r6"][0], map_location="cpu", weights_only=False)
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
    folds = list(range(args.folds)) if args.dry_run else list(FOLDS_OF[args.part])
    t0 = time.time()
    for k in folds:
        train_names = [n for n in used if fold[n] != k]
        test_names = [n for n in used if fold[n] == k]
        log(f"fold {k + 1}/5: train {len(train_names)} reservoirs, test {len(test_names)}")
        trained = {}
        for mode in modes:
            trained[mode] = []
            for sd in seeds:
                model = train(res, train_names, flood, p["r6"][sd % 3], args, device, 100 * k + sd, mode)
                trained[mode].append(model)
                torch.save({"state_dict": model.state_dict(), "encoder": ck["encoder"], "bands": ck["bands"],
                            "mean": ck["mean"], "std": ck["std"], "classes": ["land", "open water", "water under plants"],
                            "fold": k, "mode": mode, "seed": sd, "start": f"run 6 seed {sd % 3}",
                            "labels_version": d["labels_version"], "test_reservoirs": test_names},
                           out / "weights" / f"fold{k}_{mode}_seed{sd}.pt")
        models = {"rule": trained["rule"], "photo": trained.get("photo")}
        fs = {"fold": k, **flood_scores(models, mean, std, device, p["s1f"], 3 if args.dry_run else 0)}
        write(out / f"flood_fold{k}.csv", [fs])
        log(f"   Sen1Floods11 water IoU: {fs}")
        cons_k = []
        for n in test_names:
            rows, first, cons = test(res[n], models, mean, std, device, args.limit)
            write(out / "areas_passes" / f"{res[n]['safe']}.csv", rows)
            write(out / "areas_first" / f"{res[n]['safe']}.csv", first)
            cons_k += cons
        write(out / f"consistency_fold{k}.csv", cons_k)
        log(f"fold {k + 1} done at {(time.time() - t0) / 60:.0f} min")
    with open(out / f"run_part_{args.part}.json", "w", encoding="utf-8") as f:
        json.dump({"part": args.part, "folds": folds, "labels_version": d["labels_version"], "seeds": seeds,
                   "modes": modes, "epochs": args.epochs, "steps": args.steps, "batch": args.batch, "lr": args.lr,
                   "flood_every": FLOOD_EVERY, "mix": MIX, "start": "run 6", "minutes": round((time.time() - t0) / 60)},
                  f, indent=2)


def main():
    global L
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--part", default=PART, choices=("prep", "A", "B", "B1", "B2"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--modes", default="rule,photo")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="", help="comma list of reservoirs (a test)")
    ap.add_argument("--limit", type=int, default=0, help="test at most this many chips of each kind (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="a few reservoirs and steps on a CPU, all parts in one go")
    args = ap.parse_args()
    if args.dry_run:
        args.folds, args.epochs, args.steps, args.batch, args.workers, args.limit = 2, 1, 2, 4, 0, 6
        args.seeds = "0"
        args.only = args.only or "Akathimuruppu,Radella,Nalanda,Huruluwewa"
    p = paths(args)
    log("part", args.part, "paths:", {k: str(v) for k, v in p.items()})
    sys.path.insert(0, str(p["code"]))
    import labels
    L = labels
    log("labels.py version", L.VERSION)
    out = Path(args.out)
    for sub in ("areas_passes", "areas_first", "weights"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if args.part == "prep" or args.dry_run:
        prep(args, p, out)
    if args.part != "prep" or args.dry_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log("device:", device)
        train_part(args, p, out, device)
    log("done")


if __name__ == "__main__":
    main()
