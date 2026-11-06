r"""Module A retraining on Kaggle, run 2 (3 Nov 2026): the fixes from the independent review and the outside reviews.

What changed from run 1 (../kernel/retrain.py), with the reason:
- labels.py version 2: more land to learn from (never-water ground outside the outline on every date; plants on the
  bed at low storage), a cloud-shadow guard, and bug fixes. Radar passes within 5 days of a clear photo, with
  storage within 1% of capacity, take that photo's label (run 1 left 75% of the low-water pass labels unsure).
- The storage sheet is cleaned: one-day spikes and flat runs below full are dropped. Reservoirs whose capacity looks
  wrong (over a fifth of days above 105%) are left out, and the cleaning is reported in cleaning.csv.
- Folds keep neighbours together (chips that overlap, or centres within 10 km). In run 1, five held-out reservoirs
  lay partly inside a training reservoir's chip.
- Two training modes, 3 seeds each, averaged:
  - "rule": the storage rule;
  - "photo": a control with photo labels only (clear days, no storage). It shows whether the rule itself helps or
    only the Sri Lankan chips.
- More comparisons:
  - run 6 (its 3 seeds averaged);
  - a per-scene Otsu radar threshold in a 300 m band around the outline;
  - the VV threshold tuned per reservoir, now on single-orbit pass chips of clear-photo days (run 1 tuned it on the
    first check's two-orbit mosaics).
- The low-water check per day: |map share - photo share| on pixels where the photo is sure (open water or bare
  ground), the median per reservoir, and paired tests.
- The win rules, with a minimum gain, are fixed in ../PROTOCOL-2.md before the run.
Everything else is as in run 1: run 2's radar U-Net is the starting point, with a 3-class head; random 256 x 256
crops; the last epoch is kept; scores on held-out reservoirs.
On Kaggle it runs with no arguments. Quick local check on a CPU (a few reservoirs, a few steps):
    python -I retrain.py --dry-run --input <project folder> --out <folder>
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
import urllib.request
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

ACFT_TO_MCM = 1233.48 / 1e6
OLD, R6, NEW, NEW_OPEN, NEW_PLANTS, CTRL = ("radar U-Net r2", "radar U-Net r6 (3 seeds)", "retrained, water",
                                            "retrained, open water", "retrained, water under plants",
                                            "photo-label control, water")
TUNED, OTSU = "VV threshold tuned to the photo", "Otsu radar threshold"
METHODS = (OLD, R6, NEW, NEW_OPEN, CTRL, TUNED, OTSU)
PIXEL = (OLD, R6, NEW, CTRL, TUNED, OTSU)  # methods scored pixel by pixel
BASELINES = ("no model", "seasonal")
MIX = {"rule": {"clear": 0.4, "cloudy": 0.1, "pass": 0.5}, "photo": {"clear": 1.0}}
THRESHOLDS = np.arange(-24, -10.01, 0.25)  # dB, as measure_areas.py
BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1/"
MIN_PASSES = 20
NEIGHBOUR_KM = 10
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
    if args.dry_run:  # the project folder on the laptop
        base = Path(args.input)
        w6 = base / "temp/module-a-tanks/weights/run-6"
        return {"tables": base / "temp/module-a-retrain/kaggle-data/tables",
                "first": base / "temp/module-a-retrain/kaggle-data/first-check",
                "passes": base / "temp/module-a-retrain/kaggle-data/passes",
                "code": base / "temp/module-a-retrain/scripts",
                "r2": base / "temp/module-a-tanks/weights/run-2-radar/unet_radar_best.pt",
                "r6": [w6 / "unet_radar_best.pt", w6 / "unet_radar_seed1_best.pt", w6 / "unet_radar_seed2_best.pt"]}
    i = args.input
    r6 = find(i, "unet_radar_seed1_best.pt", 1, "run-6")
    return {"tables": find(i, "storage_daily.csv", 1), "first": find(i, "Akathimuruppu_2023-07-04.tif", 2),
            "passes": find(i, "Akathimuruppu/2020-10-11.tif", 2), "code": find(i, "labels.py", 1),
            "r2": find(i, "unet_radar_best.pt", 0, "run-2"),
            "r6": [r6 / "unet_radar_best.pt", r6 / "unet_radar_seed1_best.pt", r6 / "unet_radar_seed2_best.pt"]}


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
    """The reservoir's outline, cover range, label context, chips with their storage, and its chip's footprint."""
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
            if st is not None:  # a clear photo within 5 days, with storage within 1% of capacity
                near = [c for c in clear if abs((date.fromisoformat(c["s2_date"]) - d).days) <= TRANSFER_DAYS
                        and abs(c["s_photo"] - st / cap) <= TRANSFER_STORAGE]
                if near:
                    e["transfer"] = min(near, key=lambda c: abs((date.fromisoformat(c["s2_date"]) - d).days))
            passes.append(e)
    rr, cc = np.where(inside)
    known = float((np.isfinite(lo) | np.isfinite(hi))[inside].mean())
    return {"name": name, "safe": sf, "cap": cap, "inside": inside, "ctx": L.context(occ, inside),
            "lo": lo.astype(np.float32), "hi": hi.astype(np.float32), "box": (rr.min(), rr.max(), cc.min(), cc.max()),
            "clear": clear, "cloudy": cloudy, "passes": passes, "known": known,
            "bounds": (lon0, lat0, lon1, lat1), "centre": ((lat0 + lat1) / 2, (lon0 + lon1) / 2)}


def label_of(R, kind, e, mode="rule"):
    """The radar input (VV, VH in dB) and the label of one chip."""
    if mode == "photo":  # the control: the clear photo alone
        x, _, cs = read_full(e["path"])
        y, x = L.photo_label(L.photo_class(x, cs), R["ctx"]), x[:2]
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


def neighbour_folds(res, used, k_folds):
    """Reservoirs whose chips overlap or whose chip centres are within 10 km go to the same fold. Groups are placed
    largest first, each into the fold with the fewest reservoirs."""
    parent = {n: n for n in used}

    def top(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for i, a in enumerate(used):
        for b in used[i + 1:]:
            (a0, a1, a2, a3), (b0, b1, b2, b3) = res[a]["bounds"], res[b]["bounds"]
            overlap = a0 < b2 and b0 < a2 and a1 < b3 and b1 < a3
            (la, oa), (lb, ob) = res[a]["centre"], res[b]["centre"]
            km = 111.0 * np.hypot(la - lb, (oa - ob) * np.cos(np.radians((la + lb) / 2)))
            if overlap or km < NEIGHBOUR_KM:
                parent[top(a)] = top(b)
    groups = {}
    for n in used:
        groups.setdefault(top(n), []).append(n)
    fold, count = {}, [0] * k_folds
    for g in sorted(groups.values(), key=lambda g: (-len(g), sorted(g)[0])):
        k = int(np.argmin(count))
        for n in g:
            fold[n] = k
        count[k] += len(g)
    return fold, groups


def coverage(res, names):
    """Label shares inside the outline, by source and storage band, on up to 3 chips per source and reservoir."""
    acc = {}
    for n in names:
        R = res[n]
        sources = {"clear": [("clear", e) for e in R["clear"]],
                   "pass, photo nearby": [("pass", e) for e in R["passes"] if e["transfer"]],
                   "pass, rule only": [("pass", e) for e in R["passes"] if e["s"] is not None and not e["transfer"]],
                   "cloudy": [("cloudy", e) for e in R["cloudy"]]}
        for src, items in sources.items():
            for kind, e in items[:: max(1, len(items) // 3)][:3]:
                s = e["s_photo"] if kind == "clear" else e["s"]
                b = next(f"{int(100 * lo)}-{int(100 * hi) if hi < 9 else 'up'}%" for lo, hi in BANDS_STORAGE
                         if lo <= s < hi)
                _, y = label_of(R, kind, e)
                v = y[R["inside"]]
                a = acc.setdefault((src, b), np.zeros(5))
                a += [(v == c).sum() for c in (L.LAND, L.OPEN, L.PLANTS, L.ANY, L.UNSURE)]
    return [{"source": src, "storage": b, **{k: round(float(a[i] / max(a.sum(), 1)), 3) for i, k in
                                             enumerate(("land", "open", "under_plants", "either", "unsure"))}}
            for (src, b), a in sorted(acc.items())]


# ---------- 2. training ----------

class Crops(Dataset):
    def __init__(self, res, names, mean, std, crop, n, seed, mode):
        self.res, self.mean, self.std, self.crop, self.n, self.seed, self.mode, self.epoch = \
            res, mean, std, crop, n, seed, mode, 0
        mix = MIX[mode]
        self.items = {k: [] for k in mix}
        for name in names:
            R = res[name]
            if "clear" in mix:
                self.items["clear"] += [(name, e) for e in R["clear"]]
            if "cloudy" in mix:
                self.items["cloudy"] += [(name, e) for e in R["cloudy"]]
            if "pass" in mix:
                self.items["pass"] += [(name, e) for e in R["passes"] if e["s"] is not None]
        self.kinds = [k for k in mix if self.items[k]]
        w = np.array([mix[k] for k in self.kinds])
        self.weights = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        kind = self.kinds[rng.choice(len(self.kinds), p=self.weights)]
        name, e = self.items[kind][rng.integers(len(self.items[kind]))]
        R = self.res[name]
        x, y = label_of(R, kind, e, self.mode)
        h, w = y.shape
        c = self.crop
        if rng.random() < 0.8:  # centred on the outline
            r0, r1, c0, c1 = R["box"]
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


def train(res, names, ck_path, args, device, seed, mode):
    torch.manual_seed(seed)
    model, ck = load_model(ck_path, 3, device)
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
    data = Crops(res, names, mean, std, args.crop, args.steps * args.batch, seed, mode)
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


def otsu(v):
    v = v[np.isfinite(v)]
    if v.size < 100:
        return -16.75
    hist, edges = np.histogram(np.clip(v, -30, 5), bins=256)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return float(edges[int(np.argmax(between)) + 1])


def maps(models, x, mean, std, device, vv_t=None, band=None):
    """Water maps of one chip: run 2, run 6 (3 seeds averaged), the retrained ensemble (water = p_land < 0.5; open or
    plants by the larger), the photo-label control ensemble, and with vv_t / band the tuned and the Otsu thresholds."""
    z = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)
    h, w = z.shape[1:]
    ph, pw = (-h) % 32, (-w) % 32
    zt = F.pad(torch.from_numpy(z.astype(np.float32))[None], (0, pw, 0, ph), mode="reflect").to(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        old = probs([models["old"]], zt, h, w)[1] > 0.5
        r6 = probs(models["r6"], zt, h, w)[1] > 0.5
        p = probs(models["rule"], zt, h, w)
        pc = probs(models["photo"], zt, h, w) if models.get("photo") else None
    water = p[0] < 0.5
    out = {OLD: old, R6: r6, NEW: water, NEW_OPEN: water & (p[1] >= p[2]), NEW_PLANTS: water & (p[2] > p[1])}
    if pc is not None:
        out[CTRL] = pc[0] < 0.5
    vv = ndimage.median_filter(np.nan_to_num(x[0], nan=0.0), size=3)
    if vv_t is not None:
        out[TUNED] = vv < vv_t
    if band is not None:
        vb = np.where(np.isfinite(x[0]), vv, np.nan)[band]
        out[OTSU] = vv < otsu(vb)
    return out


def tune_vv(R):
    """The VV threshold (after a 3 x 3 median) whose water inside the outline, summed over the reservoir's clear days,
    matches the photo's (NDWI >= -0.06 on clear pixels). The radar is the single-orbit pass chip of the photo's
    radar date (or within a day of the photo) where there is one, else the first-check chip. Storage is not used."""
    by_date = {e["date"]: e for e in R["passes"]}
    radar_px, photo_px = np.zeros(len(THRESHOLDS)), 0
    for e in R["clear"]:
        x, _, cs = read_full(e["path"])
        d = date.fromisoformat(e["s2_date"])
        pas = by_date.get(e["s1_date"]) or next((by_date[(d + timedelta(days=k)).isoformat()] for k in (0, -1, 1)
                                                 if (d + timedelta(days=k)).isoformat() in by_date), None)
        vv_raw = read_radar(pas["path"])[0] if pas else x[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            ndwi = (x[3] - x[8]) / (x[3] + x[8])
        ok = R["inside"] & (cs >= 0.6) & np.isfinite(vv_raw) & np.isfinite(ndwi)
        vv = ndimage.median_filter(np.nan_to_num(vv_raw, nan=0.0), size=3)
        photo_px += int((ok & (ndwi >= -0.06)).sum())
        radar_px += np.searchsorted(np.sort(vv[ok]), THRESHOLDS)
    return float(THRESHOLDS[np.argmin(np.abs(radar_px - photo_px))]) if R["clear"] else -16.75


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
    """Areas on every pass and first-check chip; pixel scores against the storage reference on every pass with storage
    (all; under 20%; 85% and more; cloudy-photo passes); and, on clear days under 20% of capacity, each map's water
    share against the photo's on pixels where the photo is sure (open water or bare ground)."""
    inside, vv_t = R["inside"], tune_vv(R)
    band = ndimage.binary_dilation(inside, iterations=30)
    rows, acc = [], {g: {} for g in ("all", "low", "full", "cloudy")}
    for e in (R["passes"][:limit] if limit else R["passes"]):
        x = read_radar(e["path"])
        m = maps(models, x, mean, std, device, vv_t, band)
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
            m = maps(models, x[:2], mean, std, device, vv_t, band)
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
    return rows, first, cons, vv_t


# ---------- 4. scores ----------

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


def summarise(per, cons, first, info, out):
    rows = []
    for name, sc in per.items():
        for m in [k for k in METHODS + BASELINES if k in sc]:
            rows.append({"reservoir": name, "cap_mcm": round(info[name]["cap"], 2), "method": m, "passes": sc["passes"],
                         **{k: round(v, 3) for k, v in sc[m].items() if np.isfinite(v)}})
    write(out / "pass_scores.csv", rows)
    names = sorted(per)
    summary = []
    for group, lo, hi in SIZES:
        sub = [n for n in names if lo <= info[n]["cap"] < hi]
        for m in METHODS + BASELINES:
            if not any(m in per[n] for n in sub):
                continue
            summary.append({"group": group, "reservoirs": len(sub), "method": m,
                            **{k: med([per[n].get(m, {}).get(k, np.nan) for n in sub]) for k in MEASURES},
                            "reading_15_low_near_full": sum(per[n].get(m, {}).get("later_full_error_pct", np.nan) <= -15
                                                            for n in sub)})
    write(out / "pass_summary.csv", summary)
    get = lambda n, m, k: per[n].get(m, {}).get(k, np.nan)  # noqa: E731
    tests = []
    pairs = [(NEW, x, "later_error_pct") for x in (OLD, R6, TUNED, OTSU, CTRL, "seasonal", "no model")]
    pairs += [(NEW, OLD, "spearman_all"), (NEW, OLD, "spearman_cloudy"), (NEW, OLD, "later_full_error_pct"),
              (CTRL, OLD, "later_error_pct"), (OLD, "seasonal", "later_error_pct"), (R6, OLD, "later_error_pct"),
              (TUNED, "seasonal", "later_error_pct"), (NEW_OPEN, NEW, "later_error_pct")]
    for first_m, second, k in pairs:
        size = abs if k == "later_full_error_pct" else (lambda v: v)  # near full: how far off, either way
        t = wilcoxon([size(get(n, first_m, k)) for n in names], [size(get(n, second, k)) for n in names],
                     k.endswith("error_pct"))
        if t:
            tests.append({"first": first_m, "second": second, "measure": k, **t})
    low = {}  # the low-water check: per reservoir, the median |map share - photo share| over its low days
    for r in first:
        for k in PIXEL:
            v = r.get(f"lowdiff {k}")
            if v is not None:
                low.setdefault((r["tank"], k), []).append(v)
    lowmed = {key: float(np.median(v)) for key, v in low.items()}
    low_names = sorted({n for n, _ in lowmed})
    for first_m, second in ((NEW, OLD), (NEW, CTRL), (NEW, TUNED), (NEW, OTSU), (NEW, R6)):
        t = wilcoxon([lowmed.get((n, first_m), np.nan) for n in low_names],
                     [lowmed.get((n, second), np.nan) for n in low_names], True)
        if t:
            tests.append({"first": first_m, "second": second, "measure": "low_water_share_gap", **t})
    for g in ("all", "low", "full", "cloudy"):  # pixel IoU against the storage reference, per reservoir
        by = {(c["reservoir"], c["method"]): c["iou"] for c in cons if c["passes"] == g}
        for first_m, second in ((NEW, OLD), (NEW, TUNED), (NEW, CTRL)):
            t = wilcoxon([by.get((n, first_m), np.nan) for n in names], [by.get((n, second), np.nan) for n in names],
                         False)
            if t:
                tests.append({"first": first_m, "second": second, "measure": f"consistency_iou_{g}", **t})
    write(out / "pass_tests.csv", tests)
    write(out / "consistency.csv", cons)
    cons_summary = []
    for g in ("all", "low", "full", "cloudy"):
        for m in PIXEL:
            x = [c for c in cons if c["passes"] == g and c["method"] == m]
            if x:
                cons_summary.append({"passes": g, "method": m, "reservoirs": len(x),
                                     "iou_median": med([c["iou"] for c in x], 1),
                                     "f1_median": med([c["f1"] for c in x], 1),
                                     "decided_share_median": med([c["decided_share"] for c in x], 3)})
    write(out / "consistency_summary.csv", cons_summary)
    write(out / "low_water.csv", [{"method": k, "reservoirs": len([1 for (n, kk) in lowmed if kk == k]),
                                   "median_share_gap": med([v for (n, kk), v in lowmed.items() if kk == k], 3)}
                                  for k in PIXEL])
    for s in summary:
        log(f"   {s['group']:24s} {s['method']:32s} " + "  ".join(f"{k} {s[k]}" for k in MEASURES))
    for t in tests:
        log(f"   {t['first']} vs {t['second']}, {t['measure']}: gain {t['median_gain']}, better at "
            f"{t['share_first_better']:.0%} of {t['reservoirs']}, p = {t['wilcoxon_p']}")
    for c in cons_summary:
        log(f"   consistency {c['passes']}, {c['method']}: IoU {c['iou_median']}, F1 {c['f1_median']}, "
            f"decided {c['decided_share_median']}")


def fetch(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, dest.with_suffix(".part"))
            dest.with_suffix(".part").replace(dest)
            return
        except OSError:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def bolivia(models, mean, std, device, root=Path("/tmp/sen1floods11")):
    """Water IoU (radar only) on Sen1Floods11's 15 Bolivia chips: expert labels from a flood kept out of training.
    The retrained models' water is open water plus water under plants."""
    rel = "splits/flood_handlabeled/flood_bolivia_data.csv"
    fetch(BUCKET + rel, root / rel)
    ids = [ln.split(",")[0].replace("_S1Hand.tif", "") for ln in (root / rel).read_text().split() if ln.strip()]
    keys = (OLD, R6, NEW) + ((CTRL,) if models.get("photo") else ())
    tot = {k: [0, 0, 0] for k in keys}
    for c in ids:
        for layer in ("S1Hand", "LabelHand"):
            fetch(BUCKET + f"data/flood_events/HandLabeled/{layer}/{c}_{layer}.tif", root / layer / f"{c}.tif")
        with rasterio.open(root / "S1Hand" / f"{c}.tif") as src:
            x = src.read().astype(np.float32)
        x[~np.isfinite(x)] = np.nan
        with rasterio.open(root / "LabelHand" / f"{c}.tif") as src:
            y = src.read(1)
        m = maps(models, x, mean, std, device)
        ok, t = y >= 0, y == 1
        for k in keys:
            tot[k][0] += int((m[k] & t & ok).sum())
            tot[k][1] += int((m[k] & ~t & ok).sum())
            tot[k][2] += int((~m[k] & t & ok).sum())
    return {k: iou_f1(*v)[0] for k, v in tot.items()}


# ---------- main ----------

def main():
    global L
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
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
    ap.add_argument("--dry-run", action="store_true", help="a few reservoirs and steps on a CPU")
    args = ap.parse_args()
    if args.dry_run:
        args.folds, args.epochs, args.steps, args.batch, args.workers, args.limit = 2, 1, 2, 2, 0, 6
        args.seeds = "0"
        args.only = args.only or "Akathimuruppu,Radella,Nalanda,Huruluwewa"
    seeds = [int(s) for s in args.seeds.split(",")]
    modes = args.modes.split(",")
    p = paths(args)
    log("paths:", {k: str(v) for k, v in p.items()})
    sys.path.insert(0, str(p["code"]))
    import labels
    L = labels
    log("labels.py version", L.VERSION)
    out = Path(args.out)
    for sub in ("areas_passes", "areas_first", "weights"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device:", device)
    info, series, cleaning = sheet(p["tables"])
    write(out / "cleaning.csv", cleaning)
    doubtful = {c["reservoir"] for c in cleaning if c["capacity_doubtful"]}
    with open(p["tables"] / "pass_flags.csv", encoding="utf-8") as f:
        used = [r["reservoir"] for r in csv.DictReader(f) if r["used"] == "True" and r["reservoir"] not in doubtful]
    if args.only:
        used = [n for n in used if n in args.only.split(",")]
    log(f"storage cleaning: {sum(c['spikes'] for c in cleaning)} spikes and {sum(c['flat_days'] for c in cleaning)} "
        f"flat days dropped; capacity doubtful, left out: {sorted(doubtful)}")
    t0 = time.time()
    res = {n: prepare(n, info, series, p) for n in used}
    fold, groups = neighbour_folds(res, used, args.folds)
    write(out / "folds.csv", [{"reservoir": n, "fold": fold[n], "group_size": len(g)}
                              for g in groups.values() for n in g])
    write(out / "reservoirs.csv", [{"reservoir": n, "cap_mcm": round(R["cap"], 2), "fold": fold[n],
                                    "clear": len(R["clear"]), "cloudy": len(R["cloudy"]), "passes": len(R["passes"]),
                                    "passes_with_photo_label": sum(bool(e["transfer"]) for e in R["passes"]),
                                    "outline_known_share": round(R["known"], 3)} for n, R in res.items()])
    write(out / "label_coverage.csv", coverage(res, used))
    log(f"prepared {len(res)} reservoirs in {time.time() - t0:.0f} s; folds of "
        f"{[sum(fold[n] == k for n in used) for k in range(args.folds)]} reservoirs; largest neighbour group "
        f"{max(len(g) for g in groups.values())}")
    old, ck = load_model(p["r2"], 2, device)
    r6 = [load_model(w, 2, device)[0] for w in p["r6"]]
    for m in [old] + r6:
        m.eval()
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
    per, first_all, cons_all, tuned, bol = {}, [], [], [], []
    for k in range(args.folds):
        train_names = [n for n in used if fold[n] != k]
        test_names = [n for n in used if fold[n] == k]
        log(f"fold {k + 1}/{args.folds}: train {len(train_names)} reservoirs, test {len(test_names)}")
        trained = {}
        for mode in modes:
            trained[mode] = []
            for sd in seeds:
                model = train(res, train_names, p["r2"], args, device, 100 * k + sd, mode)
                trained[mode].append(model)
                torch.save({"state_dict": model.state_dict(), "encoder": ck["encoder"], "bands": ck["bands"],
                            "mean": ck["mean"], "std": ck["std"], "classes": ["land", "open water", "water under plants"],
                            "fold": k, "mode": mode, "seed": sd, "labels_version": L.VERSION,
                            "test_reservoirs": test_names}, out / "weights" / f"fold{k}_{mode}_seed{sd}.pt")
        models = {"old": old, "r6": r6, "rule": trained["rule"], "photo": trained.get("photo")}
        if not args.dry_run:
            bol.append({"fold": k, **bolivia(models, mean, std, device)})
            log(f"   Bolivia water IoU: {bol[-1]}")
        for n in test_names:
            rows, first, cons, vv_t = test(res[n], models, mean, std, device, args.limit)
            write(out / "areas_passes" / f"{res[n]['safe']}.csv", rows)
            write(out / "areas_first" / f"{res[n]['safe']}.csv", first)
            first_all += first
            cons_all += cons
            tuned.append({"reservoir": n, "fold": k, "vv_threshold_db": vv_t})
            sc = score_passes(rows, series.get(n, {}), info[n]["cap"])
            if sc:
                per[n] = sc
        log(f"fold {k + 1} done at {(time.time() - t0) / 60:.0f} min")
    write(out / "bolivia.csv", bol)
    write(out / "tuned_thresholds.csv", tuned)
    summarise(per, cons_all, first_all, info, out)
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump({"labels_version": L.VERSION, "seeds": seeds, "modes": modes, "folds": args.folds,
                   "epochs": args.epochs, "steps": args.steps, "batch": args.batch, "lr": args.lr,
                   "left_out_capacity": sorted(doubtful), "minutes": round((time.time() - t0) / 60)}, f, indent=2)
    log(f"all done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
