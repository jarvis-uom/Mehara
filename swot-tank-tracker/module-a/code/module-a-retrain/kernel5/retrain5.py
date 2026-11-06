r"""Module A retraining run 5, one fold per process (5 Nov 2026), under ../PROTOCOL-5.md.

Run 4's retrain4.py with one change, the batch mix (PROTOCOL-5): only the batch positions, the run's name ("run 5")
and the weights' name (photo5) differ from retrain4.py. The same seeds, starts and inputs as run 4.

What it does for fold k (--fold 0 to 4):
1. trains 3 seeds of the radar U-Net from run 6 (seed s from run 6's seed s), photo labels only. Every batch of 16
   has 10 crops from our reservoirs (training folds), 4 Sen1Floods11 training chips and 2 S1S2-Water training
   patches; dry-bed chips count twice within each reservoir's pool. 10 epochs of 300 batches, lr 3e-4, the last
   epoch kept;
2. scores the 3-seed ensemble (water = p(land) < 0.5) on:
   - Sen1Floods11's Bolivia chips and test split (rule 4);
   - every radar pass of the fold's held-out reservoirs: water area inside the outline (rule 3, scored on the
     laptop with the storage records, which never reach this job);
   - the held-out reservoirs' clear chips against Dynamic World (rule 2), as photo_check.py.
Everything is written as it is made: weights/, flood_fold<k>.csv, areas_passes/, dw_counts_fold<k>.csv,
run_fold<k>.json. A seed whose weights already exist is loaded, not trained again, so a session that was cut off
continues where it stopped when the same fold is run again. The same file runs on Kaggle, Colab or Lightning (inputs found by name).
On Kaggle it is started by run5_part.py, one process per fold.
Quick local check on a CPU (2 reservoirs, a few steps):
    python -I retrain5.py --dry-run --fold 0 --input <project folder> --out <folder>
"""
import argparse
import csv
import json
import pickle
import subprocess
import sys
import time
from datetime import date
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

FOLD = 0  # set by the notebook, or --fold
LAND, OPEN, PLANTS, ANY, UNSURE = 0, 1, 2, 3, 255  # as labels.py
MIX = {"clear": 0.4, "pass": 0.6}  # run 3's photo mode
DW_WATER, DW_LAND = (0,), (1, 2, 4, 5, 6, 7)
NEW = "run 5, water"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(base, pattern, must=""):
    hits = sorted(p for p in Path(base).rglob(pattern) if must in p.as_posix())
    if not hits:
        raise FileNotFoundError(f"{pattern} ({must}) under {base}")
    return hits


def paths(args):
    if args.dry_run:
        b = Path(args.input)
        r = b / "temp/module-a-retrain"
        return {"inputs": r / "run4-inputs", "first": r / "kaggle-data/first-check", "passes": r / "kaggle-data/passes",
                "s1f": b / "temp/module-a-kaggle/downloads/sen1floods11/v1.1", "s1s2": b / "temp/s1s2water-check/train",
                "dw": b / "temp/module-a-photo-check/dw"}
    i = Path(args.input)
    return {"inputs": find(i, "res4.pkl")[0].parent, "first": find(i, "Akathimuruppu_2023-07-04.tif")[0].parents[1],
            "passes": find(i, "2020-10-11.tif", "Akathimuruppu")[0].parents[1],
            "s1f": find(i, "flood_train_data.csv")[0].parents[2], "s1s2": find(i, "scenes.csv", "s1s2water-train")[0].parent,
            "dw": find(i, "chips.csv", "dynamic-world")[0].parent}


def read_radar(path):
    """VV, VH in dB (NaN = no data) from a chip whose first two bands are the radar as dB x 100."""
    with rasterio.open(path) as src:
        a = src.read([1, 2]).astype(np.float32)
    a[a == -32768] = np.nan
    return a / 100


def tank_free_cs(path):
    with rasterio.open(path) as src:
        return src.read(14).astype(np.float32) / 100


def flood_raw(s1_path, label_path):
    with rasterio.open(s1_path) as src:
        x = src.read([1, 2]).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    with rasterio.open(label_path) as src:
        y = src.read(1)
    return x, y


def s1f_split(root, split):
    hand = Path(root) / "data/flood_events/HandLabeled"
    with open(Path(root) / f"splits/flood_handlabeled/flood_{split}_data.csv", encoding="utf-8") as f:
        ids = [ln.split(",")[0].strip().replace("_S1Hand.tif", "") for ln in f if ln.strip()]
    return [(hand / "S1Hand" / f"{c}_S1Hand.tif", hand / "LabelHand" / f"{c}_LabelHand.tif") for c in ids]


def load_s1s2(folder):
    """All S1S2-Water training patches: radar kept as stored (int16 dB x 100, -32768 = no data; about 2 GB) and labels
    (water of either kind, land, unsure). Each patch becomes dB when it is used."""
    xs, ys = [], []
    for f in sorted(Path(folder).glob("*.npz")):
        d = np.load(f)
        xs.append(d["radar"])
        t = d["truth"]
        y = np.full(t.shape, UNSURE, np.uint8)
        y[t == 1], y[t == 0] = ANY, LAND
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


class Crops(Dataset):
    """Batch position i % 8 in (0, 4): a Sen1Floods11 training chip; i % 8 == 2: an S1S2-Water patch; else our
    reservoirs, 10 of every 16 (PROTOCOL-5; a kind
    by MIX, a reservoir at random, then a chip, dry beds counting twice). Crops are 256 x 256: on our reservoirs
    centred on the outline 80% of the time, otherwise at a uniform position over every window that fits."""

    def __init__(self, res, labels, names, p, flood, s1s2, mean, std, crop, n, seed):
        self.res, self.labels, self.p, self.flood, self.s1s2 = res, labels, p, flood, s1s2
        self.mean, self.std, self.crop, self.n, self.seed, self.epoch = mean, std, crop, n, seed, 0
        self.by = {k: {} for k in MIX}
        for name in names:
            R = res[name]
            pools = {"clear": R["clear"], "pass": [e for e in R["passes"] if e["photo_near"]]}
            for k in MIX:
                if pools[k]:
                    w = np.array([2.0 if e["dry"] else 1.0 for e in pools[k]])
                    self.by[k][name] = (pools[k], w / w.sum())
        self.kinds = [k for k in MIX if self.by[k]]
        w = np.array([MIX[k] for k in self.kinds])
        self.weights = w / w.sum()
        self.names = {k: sorted(self.by[k]) for k in self.kinds}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        box = None
        if i % 8 in (0, 4):
            x, lab = flood_raw(*self.flood[rng.integers(len(self.flood))])
            y = np.full(lab.shape, UNSURE, np.uint8)
            y[lab == 0], y[lab == 1] = LAND, ANY
        elif i % 8 == 2:
            j = rng.integers(len(self.s1s2[0]))
            r = self.s1s2[0][j]
            x = r.astype(np.float32) / 100
            x[r == -32768] = np.nan
            y = self.s1s2[1][j].copy()
        else:
            kind = self.kinds[rng.choice(len(self.kinds), p=self.weights)]
            name = self.names[kind][rng.integers(len(self.names[kind]))]
            pool, pw = self.by[kind][name]
            e = pool[rng.choice(len(pool), p=pw)]
            R = self.res[name]
            if kind == "clear":
                x, y = read_radar(self.p["first"] / e["path"][1]), self.labels[name][e["file"]].copy()
            else:
                x, y = read_radar(self.p["passes"] / e["path"][1]), self.labels[name][e["photo_near"]].copy()
            box = R["box"]
        y[np.isnan(x).any(0)] = UNSURE
        h, w = y.shape
        c = self.crop
        if box is not None and rng.random() < 0.8:  # centred on the outline, as run 3
            r0, r1, c0, c1 = box
            cy, cx = rng.integers(r0, r1 + 1), rng.integers(c0, c1 + 1)
            top, left = int(np.clip(cy - c // 2, 0, h - c)), int(np.clip(cx - c // 2, 0, w - c))
        else:  # uniform over every window that fits (PROTOCOL-4 item 5)
            top, left = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))
        x, y = x[:, top:top + c, left:left + c], y[top:top + c, left:left + c]
        x = np.nan_to_num((x - self.mean[:, None, None]) / self.std[:, None, None], nan=0.0)
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(x, np.float32)),
                torch.from_numpy(np.ascontiguousarray(y).astype(np.int64)))


def loss_fn(logits, y):
    """Run 3's loss: one class for labels 0-2, either water class for 3, nothing for unsure."""
    logp = F.log_softmax(logits.float(), dim=1)
    one = -logp.gather(1, y.clamp(max=2).unsqueeze(1)).squeeze(1)
    either = -torch.logsumexp(logp[:, 1:3], dim=1)
    loss = torch.where(y <= 2, one, torch.where(y == 3, either, torch.zeros_like(one)))
    return loss.sum() / (y <= 3).sum().clamp(min=1)


def load_start(path, device):
    """Run 6's two-class model with a third class (water under plants) started as a copy of water, less likely."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=len(ck["bands"]), classes=3)
    sd = dict(ck["state_dict"])
    w, b = sd["segmentation_head.0.weight"], sd["segmentation_head.0.bias"]
    sd["segmentation_head.0.weight"] = torch.cat([w, w[1:2]], 0)
    sd["segmentation_head.0.bias"] = torch.cat([b, b[1:2] - 2.0], 0)
    model.load_state_dict(sd)
    return model.to(device), ck


def train(data, start, args, device, seed):
    torch.manual_seed(seed)
    model, ck = load_start(start, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * args.steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    data.seed = seed
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
        log(f"   seed {seed} epoch {epoch + 1}/{args.epochs}: loss {total / max(done, 1):.4f}, {time.time() - t0:.0f} s")
    model.eval()
    return model, ck


def water(models, x, mean, std, device):
    """The ensemble's water (p(land) < 0.5) on one chip."""
    z = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)
    h, w = z.shape[1:]
    zt = F.pad(torch.from_numpy(z.astype(np.float32))[None], (0, (-w) % 32, 0, (-h) % 32), mode="reflect").to(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        p = torch.stack([torch.softmax(m(zt).float()[0, :, :h, :w], 0) for m in models]).mean(0)
    return (p[0] < 0.5).cpu().numpy()


def iou(tp, fp, fn):
    return round(100 * tp / (tp + fp + fn), 2) if tp + fp + fn else float("nan")


def write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--fold", type=int, default=FOLD)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        args.epochs, args.steps, args.batch, args.workers, args.seeds = 1, 2, 4, 0, "0"
    elif args.batch % 8:
        raise SystemExit(f"PROTOCOL-5's mix (10 + 4 + 2 of 16) needs a batch that is a multiple of 8, not {args.batch}")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    p = paths(args)
    out = Path(args.out)
    for sub in ("weights", "areas_passes"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with open(p["inputs"] / "res4.pkl", "rb") as f:
        d = pickle.load(f)
    res, fold, used = d["res"], d["fold"], d["used"]
    k = args.fold
    train_names = [n for n in used if fold[n] != k]
    test_names = [n for n in used if fold[n] == k]
    if args.dry_run:
        train_names, test_names = train_names[:2], test_names[:2]
    labels = {}
    for n in train_names + test_names:
        z = np.load(p["inputs"] / "labels4" / f"{res[n]['safe']}.npz")
        labels[n] = {f + ".tif": z[f] for f in z.files}
    flood = s1f_split(p["s1f"], "train")
    s1s2 = load_s1s2(p["s1s2"])
    if args.dry_run:
        s1s2 = (s1s2[0][:8], s1s2[1][:8])
    r6 = p["inputs"] / "run-6"
    starts = [r6 / "unet_radar_best.pt", r6 / "unet_radar_seed1_best.pt", r6 / "unet_radar_seed2_best.pt"]
    ck = torch.load(starts[0], map_location="cpu", weights_only=False)
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
    log(f"PROTOCOL-5, fold {k + 1}/5 on {gpu}: train {len(train_names)} reservoirs, test {len(test_names)}; "
        f"{len(flood)} Sen1Floods11 chips; {len(s1s2[0])} S1S2-Water patches; labels {d['labels_version']}")
    seeds = [int(s) for s in args.seeds.split(",")]
    models = []
    for sd in seeds:
        wpath = out / "weights" / f"fold{k}_photo5_seed{sd}.pt"
        if wpath.exists():  # a session that was cut off: this seed is already trained, so continue from here
            ckw = torch.load(wpath, map_location="cpu", weights_only=False)
            model = smp.Unet(ckw["encoder"], encoder_weights=None, in_channels=len(ckw["bands"]), classes=3)
            model.load_state_dict(ckw["state_dict"])
            models.append(model.to(device).eval())
            log(f"   seed {sd}: already trained, loaded {wpath.name}")
            continue
        data = Crops(res, labels, train_names, p, flood, s1s2, mean, std, args.crop, args.steps * args.batch, sd)
        model, _ = train(data, starts[sd % 3], args, device, 100 * k + sd)
        torch.save({"state_dict": model.state_dict(), "encoder": ck["encoder"], "bands": ck["bands"], "mean": ck["mean"],
                    "std": ck["std"], "classes": ["land", "open water", "water under plants"], "fold": k,
                    "mode": "photo5", "seed": sd, "start": f"run 6 seed {sd % 3}", "test_reservoirs": test_names},
                   out / "weights" / f"fold{k}_photo5_seed{sd}.pt")
        models.append(model)
    log(f"trained in {(time.time() - t0) / 60:.0f} min")
    fs = {"fold": k}
    for split in ("bolivia", "test"):
        tp = fp = fn = 0
        chips = s1f_split(p["s1f"], split)
        for s1, lab in (chips[:3] if args.dry_run else chips):
            x, y = flood_raw(s1, lab)
            m = water(models, x, mean, std, device)
            ok, t = y >= 0, y == 1
            tp, fp, fn = tp + int((m & t & ok).sum()), fp + int((m & ~t & ok).sum()), fn + int((~m & t & ok).sum())
        fs[f"{split}: {NEW}"] = iou(tp, fp, fn)
    write(out / f"flood_fold{k}.csv", [fs])
    log("Sen1Floods11 water IoU:", fs)
    dw_index = {f.name: f for f in Path(p["dw"]).rglob("*.tif")}
    dw_rows = []
    for n in test_names:
        R = res[n]
        inside = R["inside"]
        rows = []
        for e in (R["passes"][:4] if args.dry_run else R["passes"]):
            x = read_radar(p["passes"] / e["path"][1])
            m = water(models, x, mean, std, device)
            rows.append({**{c: e[c] for c in ("tank", "date", "orbit", "pass", "photo_date", "photo_clear")},
                         "outline_ha": round(float(inside.sum()) * 0.01, 1),
                         "radar_gap_share": round(float(np.isnan(x[0][inside]).mean()), 3),
                         NEW: round(float((m & inside).sum()) * 0.01, 1)})
        write(out / "areas_passes" / f"{R['safe']}.csv", rows)
        for e in (R["clear"][:2] if args.dry_run else R["clear"]):
            if e["file"] not in dw_index:
                continue
            x = read_radar(p["first"] / e["path"][1])
            cs = tank_free_cs(p["first"] / e["path"][1])
            with rasterio.open(dw_index[e["file"]]) as src:
                dw = src.read(1)
            m = water(models, x, mean, std, device)
            ok = np.isin(dw, DW_WATER + DW_LAND) & (cs >= 0.6) & np.isfinite(x[0]) & np.isfinite(x[1])
            t = np.isin(dw, DW_WATER)
            dw_rows.append({"reservoir": n, "file": e["file"], "reference": "B Dynamic World", "method": "run 5",
                            "tp": int((m & t & ok).sum()), "fp": int((m & ~t & ok).sum()),
                            "fn": int((~m & t & ok).sum()), "tn": int((~m & ~t & ok).sum())})
        log(f"   tested {n}: {len(rows)} passes")
    write(out / f"dw_counts_fold{k}.csv", dw_rows)
    with open(out / f"run_fold{k}.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "PROTOCOL-5.md", "fold": k, "gpu": gpu, "seeds": seeds, "epochs": args.epochs,
                   "steps": args.steps, "batch": args.batch, "lr": args.lr, "train": len(train_names),
                   "test": test_names, "s1s2_patches": int(len(s1s2[0])), "minutes": round((time.time() - t0) / 60),
                   "dry_run": args.dry_run}, f, indent=2)
    log(f"fold {k + 1} done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
