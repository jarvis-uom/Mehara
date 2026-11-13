r"""Module A series 2, development runs, one fold per process (5 Nov 2026), under ../PROTOCOL-6.md (part 1, section B)
and ../DEV-6.md (the development plan and its keep rules).

Run 5's retrain5.py with three kinds of change:
1. S1S2-Water training scenes 39, 49, 54, 76 and 83 are held out of training (PROTOCOL-6 B), in every variant;
2. the variant (--variant):
   - control: run 5's recipe (10 epochs of 300 batches, cross-entropy, the last epoch);
   - long: 30 epochs of 300 batches (9,000 steps), the same OneCycle shape, and the weights' moving average (EMA,
     decay 0.999) is the model that is scored;
   - lovasz: long, plus 0.5 x a Lovasz hinge (Berman et al., CVPR 2018) on water (either class) against land, per
     crop, beside the cross-entropy;
   the scheduler also steps on a batch with no label (never happens in practice);
3. development scoring only (PROTOCOL-6 B); nothing from the old tests or the fresh tests is scored:
   - Sen1Floods11's validation split (89 chips);
   - the 5 held-out S1S2-Water training scenes (their 64 patches each) and the 5 Asian scenes (6, 7, 11, 12, 52);
   - the held-out reservoirs' clear chips against Dynamic World, only those with radar dated before the reservoir's
     later half (split_dates.csv);
   - the held-out reservoirs' passes dated before the later half: water area inside the outline (the storage step
     is scored on the laptop, on the first and second quarters).
Writes weights/, dev_flood_fold<k>.csv, dev_s1s2_fold<k>.csv, dw_counts_fold<k>.csv, areas_passes/,
run_fold<k>.json. A seed whose weights exist is loaded, not trained again.

Development batch 3 (DEV-6's addendum of 11 Nov 2026) adds four variants and one inference rule:
- edge: the reservoirs' labels are cleaned at the water's edge. Inside the outline, plants within 30 m of the photo's
  open water become unsure (they were land). When the label's photo is 1 day or more from the radar, a band at the
  line between open water and land becomes unsure on both sides (1 pixel for 1 or 2 days, 2 pixels for 3 to 5);
- aug: every training crop also gets a brightness shift, a contrast factor and, half of the time, extra speckle;
- edgeaug: both; longedgeaug: both, with long's steps and moving average;
- grown maps (scored beside the plain maps in every variant): the plain water plus every touching pixel whose
  p(land) is under 0.7.
The plain scores keep their files, so the control's can be compared with batch 1's byte for byte. Batch 3's extra
counts go to dev3_flood_fold<k>.csv (per chip), dev3_s1s2_fold<k>.csv, dev3_dw_fold<k>.csv (the whole chip and
inside the outline) and one more column in areas_passes/.
Quick local check on a CPU (2 reservoirs, a few steps):
    python -I retrain6.py --dry-run --variant lovasz --fold 0 --input <project folder> --out <folder>
The same tiny check on Kaggle's own folders (no GPU needed): --quick in place of --dry-run.
"""
import argparse
import copy
import csv
import importlib.util
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


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
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
FOLD = 0  # set by the notebook, or --fold
LAND, OPEN, PLANTS, ANY, UNSURE = 0, 1, 2, 3, 255  # as labels.py
MIX = {"clear": 0.4, "pass": 0.6}  # run 3's photo mode
DW_WATER, DW_LAND = (0,), (1, 2, 4, 5, 6, 7)
DEV_S1S2 = ("39", "49", "54", "76", "83")  # PROTOCOL-6 B: held out of training, one per land-cover group (seed 6)
ASIAN = ("6", "7", "11", "12", "52")  # scored before as extras, never trained on
VARIANTS = {"control": {"epochs": 10, "ema": 0.0, "lovasz": 0.0},
            "long": {"epochs": 30, "ema": 0.999, "lovasz": 0.0},
            "lovasz": {"epochs": 30, "ema": 0.999, "lovasz": 0.5},
            # development batch 3 (DEV-6's addendum of 11 Nov 2026)
            "edge": {"epochs": 10, "ema": 0.0, "lovasz": 0.0, "edge": True},
            "aug": {"epochs": 10, "ema": 0.0, "lovasz": 0.0, "aug": True},
            "edgeaug": {"epochs": 10, "ema": 0.0, "lovasz": 0.0, "edge": True, "aug": True},
            "longedgeaug": {"epochs": 30, "ema": 0.999, "lovasz": 0.0, "edge": True, "aug": True}}
NEAR_PX = 3  # 30 m, as labels.py
JRC_OUTSIDE_MIN = 10  # as labels.py
GROW_LAND = 0.7  # grown maps: water spreads from p(land) < 0.5 into touching pixels with p(land) under this
AUG = {"shift": 2.0, "band_shift": 0.5, "contrast": 0.1, "speckle_p": 0.5, "looks": (8.0, 32.0)}


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
                "gee": b / "temp/s1s2water-check/gee", "dw": b / "temp/module-a-photo-check/dw",
                "split": HERE / "split_dates.csv"}
    i = Path(args.input)
    return {"inputs": find(i, "res4.pkl")[0].parent, "first": find(i, "Akathimuruppu_2023-07-04.tif")[0].parents[1],
            "passes": find(i, "2020-10-11.tif", "Akathimuruppu")[0].parents[1],
            "s1f": find(i, "flood_train_data.csv")[0].parents[2], "s1s2": find(i, "scenes.csv", "s1s2water-train")[0].parent,
            "gee": find(i, "scenes.csv", "s1s2water-gee")[0].parent,
            "dw": find(i, "chips.csv", "dynamic-world")[0].parent, "split": find(i, "split_dates.csv")[0]}


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


def load_s1s2(folder, scenes=None, skip=()):
    """S1S2-Water training patches: radar as stored (int16 dB x 100, -32768 = no data) and labels (water of either
    kind, land, unsure). scenes: only these; skip: all but these. Returns (radar, labels, scene of each patch)."""
    xs, ys, ids = [], [], []
    for f in sorted(Path(folder).glob("*.npz")):
        if (scenes is not None and f.stem not in scenes) or f.stem in skip:
            continue
        d = np.load(f)
        xs.append(d["radar"])
        t = d["truth"]
        y = np.full(t.shape, UNSURE, np.uint8)
        y[t == 1], y[t == 0] = ANY, LAND
        ys.append(y)
        ids += [f.stem] * len(t)
    return np.concatenate(xs), np.concatenate(ys), ids


def photo_parts(path):
    """What a first-check chip's photo shows, exactly as labels.py's photo_class: open water, bare ground, plants
    (anything else is cloud, shadow or unclear). Reads B3, B4, B8, B11 (reflectance x 10000, 0 = no data) and Cloud
    Score+ (x 100)."""
    with rasterio.open(path) as src:
        b3, b4, b8, b11, cs = src.read([4, 5, 9, 11, 14]).astype(np.float32)
    for b in (b3, b4, b8, b11):
        b[b == 0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi, mndwi, ndvi = (b3 - b8) / (b3 + b8), (b3 - b11) / (b3 + b11), (b8 - b4) / (b8 + b4)
        wet = (ndwi >= -0.06) & (mndwi >= 0.0) & (b11 < 1000)
        dry = (ndwi < -0.06) & (mndwi < 0.0)
        bare, plants = dry & (ndvi < 0.2), dry & (ndvi >= 0.2)
    seen = ~(cs / 100 < 0.6)
    return wet & seen, bare & seen, plants & seen


def plants_by_water(path, y, ctx):
    """Batch 3's edge clean-up, part 1: the plants inside the outline within 30 m of the photo's open water (a mask).
    Also checks that the photo read here gives the stored label y again (prep4.py's label4)."""
    water, bare, plants = photo_parts(path)
    again = np.full(y.shape, UNSURE, np.uint8)
    again[water & (ctx["inside"] | (ctx["occ"] > JRC_OUTSIDE_MIN))] = OPEN
    again[bare | plants] = LAND
    again[~(water | bare | plants) & ctx["land_out"]] = LAND
    near = ndimage.distance_transform_edt(~water) <= NEAR_PX if water.any() else np.zeros(y.shape, bool)
    return plants & ctx["inside"] & near, int((again != y).sum())


def shore_band(y, r):
    """Batch 3's edge clean-up, part 2: the pixels within r of the line between the label's open water and its land,
    on both sides (a mask)."""
    water, land, st = y == OPEN, y == LAND, np.ones((3, 3), bool)
    return (land & ndimage.binary_dilation(water, st, iterations=r)) | \
        (water & ndimage.binary_dilation(land, st, iterations=r))


def band_px(days):
    """The band's width for a label whose photo is this many days from the radar."""
    return 0 if days == 0 else 1 if days <= 2 else 2


def radar_aug(x, mean, rng):
    """Batch 3's radar augmentation of one crop in dB (NaN = no data): contrast about the training mean, a brightness
    shift (one for both bands, plus a small one per band) and, half of the time, extra speckle (gamma noise on the
    power, pixel by pixel)."""
    g = rng.uniform(1 - AUG["contrast"], 1 + AUG["contrast"])
    shift = rng.uniform(-AUG["shift"], AUG["shift"]) + rng.uniform(-AUG["band_shift"], AUG["band_shift"], (2, 1, 1))
    x = mean[:, None, None] + g * (x - mean[:, None, None]) + shift
    if rng.random() < AUG["speckle_p"]:
        looks = rng.uniform(*AUG["looks"])
        x = x + 10 * np.log10(rng.gamma(looks, 1 / looks, x.shape))
    return x.astype(np.float32)


class Crops(Dataset):
    """As run 5: batch position i % 8 in (0, 4): a Sen1Floods11 training chip; i % 8 == 2: an S1S2-Water patch; else
    our reservoirs, 10 of every 16 (a kind by MIX, a reservoir at random, then a chip, dry beds counting twice).
    Crops are 256 x 256: on our reservoirs centred on the outline 80% of the time, otherwise at a uniform position.
    Batch 3: near (the plants beside the water of each photo; None = no edge clean-up) and aug (radar augmentation).
    The augmentation draws from its own generator, so every variant sees the same crops in the same order."""

    def __init__(self, res, labels, names, p, flood, s1s2, mean, std, crop, n, seed, near=None, aug=False):
        self.res, self.labels, self.p, self.flood, self.s1s2 = res, labels, p, flood, s1s2
        self.near, self.aug = near, aug
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
            photo, days = (e["file"], int(e["days_apart"])) if kind == "clear" else (e["photo_near"], e["near_gap"])
            x, y = read_radar(self.p["first" if kind == "clear" else "passes"] / e["path"][1]), \
                self.labels[name][photo].copy()
            if self.near is not None:  # the edge clean-up; both parts are taken from the label as stored
                band = shore_band(y, band_px(days)) if days else False
                y[self.near[name][photo] | band] = UNSURE
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
        if self.aug:
            x = radar_aug(x, self.mean, np.random.default_rng([self.seed, self.epoch, i, 3]))
        x = np.nan_to_num((x - self.mean[:, None, None]) / self.std[:, None, None], nan=0.0)
        k = int(rng.integers(4))
        x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(x, np.float32)),
                torch.from_numpy(np.ascontiguousarray(y).astype(np.int64)))


def ce_loss(logp, y):
    """Run 3's loss: one class for labels 0-2, either water class for 3, nothing for unsure."""
    one = -logp.gather(1, y.clamp(max=2).unsqueeze(1)).squeeze(1)
    either = -torch.logsumexp(logp[:, 1:3], dim=1)
    loss = torch.where(y <= 2, one, torch.where(y == 3, either, torch.zeros_like(one)))
    return loss.sum() / (y <= 3).sum().clamp(min=1)


def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension of the Jaccard loss (Berman et al., 2018, lovasz_losses.py)."""
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jac = 1.0 - inter / union
    if len(gt_sorted) > 1:
        jac[1:] = jac[1:] - jac[:-1]
    return jac


def lovasz_hinge(logp, y):
    """Binary Lovasz hinge per crop, averaged over crops: water (labels 1 to 3) against land (0); unsure left out.
    The logit is log p(water) - log p(land)."""
    logit = torch.logsumexp(logp[:, 1:3], dim=1) - logp[:, 0]
    losses = []
    for lg, yy in zip(logit, y):
        ok = yy <= 3
        if not ok.any():
            continue
        lab = (yy[ok] >= 1).float()
        err = 1.0 - lg[ok] * (2.0 * lab - 1.0)
        err_sorted, perm = torch.sort(err, descending=True)
        losses.append(torch.dot(F.relu(err_sorted), lovasz_grad(lab[perm])))
    return torch.stack(losses).mean() if losses else logit.sum() * 0.0


def loss_fn(logits, y, w_lovasz):
    logp = F.log_softmax(logits.float(), dim=1)
    loss = ce_loss(logp, y)
    return loss + w_lovasz * lovasz_hinge(logp, y) if w_lovasz else loss


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


def train(data, start, args, device, seed, v):
    """Returns (last model, EMA model or None)."""
    torch.manual_seed(seed)
    model, ck = load_start(start, device)
    ema = copy.deepcopy(model).eval() if v["ema"] else None
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
                sched.step()  # keep the schedule on time
                continue
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(xb)
            loss = loss_fn(logits, yb, v["lovasz"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
                        if e.dtype.is_floating_point:
                            e.mul_(v["ema"]).add_(m.detach(), alpha=1 - v["ema"])
                        else:
                            e.copy_(m)
            total, done = total + loss.item(), done + 1
        log(f"   seed {seed} epoch {epoch + 1}/{args.epochs}: loss {total / max(done, 1):.4f}, {time.time() - t0:.0f} s")
    model.eval()
    return model, ema


def probs(models, zt):
    with torch.no_grad(), torch.autocast(zt.device.type, dtype=torch.float16, enabled=zt.device.type == "cuda"):
        return torch.stack([torch.softmax(m(zt).float(), 1) for m in models]).mean(0)


def land_prob(models, x, mean, std, device):
    """The ensemble's p(land) on one chip (normalised, no data = 0, reflect-padded to a multiple of 32), as run 5."""
    z = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)
    h, w = z.shape[1:]
    zt = F.pad(torch.from_numpy(z.astype(np.float32))[None], (0, (-w) % 32, 0, (-h) % 32), mode="reflect").to(device)
    return probs(models, zt)[0, 0, :h, :w].cpu().numpy()


def water(models, x, mean, std, device):
    """The ensemble's water (p(land) < 0.5) on one chip, as run 5."""
    return land_prob(models, x, mean, std, device) < 0.5


def grow(pl):
    """Batch 3's grown map from p(land): the plain water (under 0.5) plus every pixel under GROW_LAND that is joined
    to it (8 neighbours)."""
    lab, n = ndimage.label(pl < GROW_LAND, structure=np.ones((3, 3), bool))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[pl < 0.5])] = True
    keep[0] = False
    return keep[lab]


def both_maps(models, x, mean, std, device):
    """The ensemble's plain water (p(land) < 0.5, as run 5) and batch 3's grown water on one chip."""
    pl = land_prob(models, x, mean, std, device)
    return pl < 0.5, grow(pl)


def counts(m, t, ok):
    return {"tp": int((m & t & ok).sum()), "fp": int((m & ~t & ok).sum()), "fn": int((~m & t & ok).sum()),
            "tn": int((~m & ~t & ok).sum())}


def add_counts(acc, m, t, ok):
    for key, v in counts(m, t, ok).items():
        acc[key] += v


def score_scene(models, radar_path, truth_path, mean, std, device, S, crop=None):
    """Pixel counts on one prepared S1S2-Water scene, tiled as score_s1s2.py (each tile scored on its centre): the
    plain map's, and the grown map's (grown inside each tile with its margin)."""
    with rasterio.open(radar_path) as src:
        img = src.read()
    with rasterio.open(truth_path) as src:
        tr = src.read(1)
    if crop:
        r, c, h, w = crop
        img, tr = img[:, r:r + h, c:c + w], tr[r:r + h, c:c + w]
    ok = (tr != 255) & (img != -32768).all(0)
    truth = tr == 1
    H, W = ok.shape
    acc, acc_g = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r0 in range(0, H, S.T):
        for c0 in range(0, W, S.T):
            if not ok[r0:r0 + S.T, c0:c0 + S.T].any():
                continue
            h, w = min(S.T, H - r0), min(S.T, W - c0)
            z, _ = S.tile_input(img, r0, c0, mean, std)
            pl = probs(models, torch.from_numpy(z[None]).to(device))[0, 0].cpu().numpy()
            t, o = truth[r0:r0 + h, c0:c0 + w], ok[r0:r0 + h, c0:c0 + w]
            add_counts(acc, (pl < 0.5)[S.M:S.M + h, S.M:S.M + w], t, o)
            add_counts(acc_g, grow(pl)[S.M:S.M + h, S.M:S.M + w], t, o)
    return acc, acc_g


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
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="the dry run's tiny sizes on Kaggle's own folders")
    args = ap.parse_args()
    v = VARIANTS[args.variant]
    args.epochs = v["epochs"]
    tiny = args.dry_run or args.quick
    if tiny:
        args.epochs, args.steps, args.batch, args.workers, args.seeds = 1, 2, 8, 0, "0"
    elif args.batch % 8:
        raise SystemExit(f"run 5's mix (10 + 4 + 2 of 16) needs a batch that is a multiple of 8, not {args.batch}")
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
    later = {r["reservoir"]: r["later_from"] for r in csv.DictReader(open(p["split"], encoding="utf-8"))}
    k = args.fold
    train_names = [n for n in used if fold[n] != k]
    test_names = [n for n in used if fold[n] == k]
    if tiny:
        test_names = test_names[:2]
    if args.dry_run:
        train_names = train_names[:2]
    labels = {}
    for n in train_names + test_names:
        z = np.load(p["inputs"] / "labels4" / f"{res[n]['safe']}.npz")
        labels[n] = {f + ".tif": z[f] for f in z.files}
    seeds = [int(s) for s in args.seeds.split(",")]
    to_train = [sd for sd in seeds if not (out / "weights" / f"fold{k}_{args.variant}6_seed{sd}.pt").exists()]
    near, edge_stats = None, {}
    if v.get("edge") and to_train:  # the edge clean-up, part 1: every training photo is read once (also with --quick)
        near, n_near, n_in, n_diff = {}, 0, 0, 0
        for n in train_names:
            R = res[n]
            near[n] = {}
            for e in R["clear"]:
                near[n][e["file"]], diff = plants_by_water(p["first"] / e["path"][1], labels[n][e["file"]], R["ctx"])
                n_near, n_in, n_diff = n_near + int(near[n][e["file"]].sum()), n_in + int(R["inside"].sum()), n_diff + diff
        if n_diff:
            raise SystemExit(f"the photos read here do not give the stored labels again ({n_diff} pixels differ)")
        edge_stats = {"plants_by_water_px": n_near, "share_of_outline": round(n_near / max(n_in, 1), 4)}
        log(f"edge clean-up: the stored labels were rebuilt exactly; plants beside the water: {n_near} pixels, "
            f"{100 * n_near / max(n_in, 1):.1f}% of the outlines' pixels over {sum(len(x) for x in near.values())} photos")
    flood = s1f_split(p["s1f"], "train")
    s1s2 = load_s1s2(p["s1s2"], skip=DEV_S1S2)
    if tiny:
        s1s2 = (s1s2[0][:8], s1s2[1][:8], s1s2[2][:8])
    r6 = p["inputs"] / "run-6"
    starts = [r6 / "unet_radar_best.pt", r6 / "unet_radar_seed1_best.pt", r6 / "unet_radar_seed2_best.pt"]
    ck = torch.load(starts[0], map_location="cpu", weights_only=False)
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
    log(f"PROTOCOL-6 development, variant {args.variant} {v}, fold {k + 1}/5 on {gpu}: train {len(train_names)} "
        f"reservoirs, test {len(test_names)}; {len(flood)} Sen1Floods11 chips; {len(s1s2[0])} S1S2-Water patches "
        f"from {len(set(s1s2[2]))} scenes (held out: {', '.join(DEV_S1S2)})")
    models = []
    for sd in seeds:
        tag = f"fold{k}_{args.variant}6_seed{sd}"
        wpath = out / "weights" / f"{tag}.pt"
        if wpath.exists():  # a session that was cut off, or weights copied in: this seed is already trained
            ckw = torch.load(wpath, map_location="cpu", weights_only=False)
            model = smp.Unet(ckw["encoder"], encoder_weights=None, in_channels=len(ckw["bands"]), classes=3)
            model.load_state_dict(ckw["state_dict"])
            models.append(model.to(device).eval())
            log(f"   seed {sd}: already trained, loaded {wpath.name}")
            continue
        data = Crops(res, labels, train_names[:2] if tiny else train_names, p, flood, s1s2[:2], mean, std, args.crop,
                     args.steps * args.batch, sd, near, bool(v.get("aug")))
        last, ema = train(data, starts[sd % 3], args, device, 100 * k + sd, v)
        meta = {"encoder": ck["encoder"], "bands": ck["bands"], "mean": ck["mean"], "std": ck["std"],
                "classes": ["land", "open water", "water under plants"], "fold": k, "variant": args.variant,
                "recipe": v, "seed": sd, "start": f"run 6 seed {sd % 3}", "test_reservoirs": test_names,
                "held_out_s1s2": list(DEV_S1S2)}
        if ema is not None:  # the scored model is the moving average; the last weights are kept beside it
            torch.save({"state_dict": last.state_dict(), **meta, "kind": "last"}, out / "weights" / f"{tag}_last.pt")
        scored = ema if ema is not None else last
        torch.save({"state_dict": scored.state_dict(), **meta, "kind": "ema" if ema is not None else "last"}, wpath)
        models.append(scored)
    log(f"trained in {(time.time() - t0) / 60:.0f} min")
    # Development scores only (PROTOCOL-6 B). The plain map's files are as in batches 1 and 2; the dev3 files hold
    # batch 3's extra counts (the plain and the grown map)
    zero = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    fl, fl3 = dict(zero), []
    chips = s1f_split(p["s1f"], "valid")
    for s1, lab in (chips[:3] if tiny else chips):
        x, y = flood_raw(s1, lab)
        for kind, m in zip(("plain", "grown"), both_maps(models, x, mean, std, device)):
            c = counts(m, y == 1, y >= 0)
            fl3.append({"fold": k, "variant": args.variant, "chip": Path(s1).name.replace("_S1Hand.tif", ""),
                        "map": kind, **c})
            if kind == "plain":
                for key, val in c.items():
                    fl[key] += val
    write(out / f"dev_flood_fold{k}.csv", [{"fold": k, "variant": args.variant, "split": "valid",
                                            "chips": len(chips), **fl}])
    write(out / f"dev3_flood_fold{k}.csv", fl3)
    log("Sen1Floods11 validation split:", fl)
    s_rows, s3 = [], []
    dx, dy, dids = load_s1s2(p["s1s2"], scenes=DEV_S1S2)
    for sid in DEV_S1S2:
        acc, acc_g = dict(zero), dict(zero)
        idx = [j for j, s in enumerate(dids) if s == sid][:2 if tiny else None]
        for j in idx:
            x = dx[j].astype(np.float32) / 100
            x[dx[j] == -32768] = np.nan
            yj = dy[j]
            ok = (yj <= 3) & np.isfinite(x).all(0)
            plain, grown = both_maps(models, x, mean, std, device)
            add_counts(acc, plain, yj == ANY, ok)
            add_counts(acc_g, grown, yj == ANY, ok)
        row = {"fold": k, "variant": args.variant, "set": "held-out training", "scene": sid, "patches": len(idx)}
        s_rows.append({**row, **acc})
        s3 += [{**row, "map": "plain", **acc}, {**row, "map": "grown", **acc_g}]
    spec = importlib.util.spec_from_file_location("score_s1s2", HERE / "score_s1s2.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    for sid in (ASIAN[:1] if tiny else ASIAN):
        acc, acc_g = score_scene(models, p["gee"] / f"{sid}_radar.tif", p["gee"] / f"{sid}_truth.tif", mean, std,
                                 device, S, (5000, 5000, 1100, 1300) if tiny else None)
        row = {"fold": k, "variant": args.variant, "set": "asian", "scene": sid, "patches": ""}
        s_rows.append({**row, **acc})
        s3 += [{**row, "map": "plain", **acc}, {**row, "map": "grown", **acc_g}]
    write(out / f"dev_s1s2_fold{k}.csv", s_rows)
    write(out / f"dev3_s1s2_fold{k}.csv", s3)
    log("S1S2-Water development scenes:", {r["scene"]: round(100 * r["tp"] / max(r["tp"] + r["fp"] + r["fn"], 1), 1)
                                           for r in s_rows})
    dw_index = {f.name: f for f in Path(p["dw"]).rglob("*.tif")}
    dw_rows, dw3, n_pass = [], [], 0
    (out / "areas_grown").mkdir(exist_ok=True)
    for n in test_names:
        R = res[n]
        inside, cut = R["inside"], later[n]
        rows, rows_g = [], []
        first = [e for e in R["passes"] if e["date"] < cut]
        for e in (first[:4] if tiny else first):
            x = read_radar(p["passes"] / e["path"][1])
            m, g = both_maps(models, x, mean, std, device)
            gap = ~np.isfinite(x).all(0)
            rows.append({**{c: e[c] for c in ("tank", "date", "orbit", "pass", "photo_date", "photo_clear")},
                         "outline_ha": round(float(inside.sum()) * 0.01, 1),
                         "radar_gap_share": round(float(np.isnan(x[0][inside]).mean()), 3),
                         "radar_gap_share_both": round(float(gap[inside].mean()), 3),
                         f"{args.variant}6, water": round(float((m & inside).sum()) * 0.01, 1)})
            rows_g.append({"tank": e["tank"], "date": e["date"],
                           f"{args.variant}6, grown": round(float((g & inside).sum()) * 0.01, 1)})
        write(out / "areas_passes" / f"{R['safe']}.csv", rows)
        write(out / "areas_grown" / f"{R['safe']}.csv", rows_g)
        n_pass += len(rows)
        clear = [e for e in R["clear"] if e["s1_date"] < cut]
        for e in (clear[:2] if tiny else clear):
            if e["file"] not in dw_index:
                continue
            x = read_radar(p["first"] / e["path"][1])
            cs = tank_free_cs(p["first"] / e["path"][1])
            with rasterio.open(dw_index[e["file"]]) as src:
                dw = src.read(1)
            m, g = both_maps(models, x, mean, std, device)
            ok = np.isin(dw, DW_WATER + DW_LAND) & (cs >= 0.6) & np.isfinite(x[0]) & np.isfinite(x[1])
            row = {"reservoir": n, "file": e["file"], "s1_date": e["s1_date"]}
            dw_rows.append({**row, "method": args.variant, **counts(m, np.isin(dw, DW_WATER), ok)})
            dw3 += [{**row, "variant": args.variant, "map": kind, "region": region,
                     **counts(mm, np.isin(dw, DW_WATER), ok & where)}
                    for kind, mm in (("plain", m), ("grown", g)) for region, where in (("chip", True), ("outline", inside))]
        log(f"   tested {n}: {len(rows)} first-half passes")
    write(out / f"dw_counts_fold{k}.csv", dw_rows)
    write(out / f"dev3_dw_fold{k}.csv", dw3)
    with open(out / f"run_fold{k}.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "PROTOCOL-6.md part 1, section B; DEV-6.md", "variant": args.variant, "recipe": v,
                   "fold": k, "gpu": gpu, "seeds": seeds, "epochs": args.epochs, "steps": args.steps,
                   "batch": args.batch, "lr": args.lr, "train": len(train_names), "test": test_names,
                   "s1s2_patches": int(len(s1s2[0])), "held_out_s1s2": list(DEV_S1S2), "first_half_passes": n_pass,
                   "dw_chips": len(dw_rows), "minutes": round((time.time() - t0) / 60), "dry_run": args.dry_run,
                   **({"quick": True} if args.quick else {}), **({"edge": edge_stats} if edge_stats else {}),
                   **({"trained_here": to_train} if len(to_train) < len(seeds) else {})},
                  f, indent=2)
    log(f"fold {k + 1} done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
