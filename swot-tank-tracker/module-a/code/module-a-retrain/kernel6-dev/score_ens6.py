r"""Series 2, development batch 2 (10 Nov 2026), under ../DEV-6.md's addendum (fixed before any run, commit f0858af).

Inference only. Combines the control's fold models (retrain6.py --variant control, folds 1 and 2) on the development
data abroad and writes pixel counts for:
- A0 fold<k>:        the fold's 3 seeds averaged (today's product; as retrain6.py scores a fold);
- A0+flips fold<k>:  the same, each model also averaged over the 8 turns and flips (reported beside the rules);
- A1:                all folds' models averaged into one map;
- A2:                A1 with the 8 turns and flips.
Data: the 5 held-out S1S2-Water training scenes (64 patches each), the 5 Asian scenes (tiled as score_s1s2.py) and
Sen1Floods11's validation split (one row per chip). Nothing else is read.
The work is split by --part i/n so that two GPUs can share it; each part writes ens_part<i>.csv.
    python -u score_ens6.py --run <folder with fold1/, fold2/> --folds 1,2 --part 0/2 [--input /kaggle/input]
Local check on a CPU (tiny):
    python -B score_ens6.py --dry-run --run <folder> --folds 1,2 --input <project folder>
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = load("retrain6")  # paths, the S1S2-Water and Sen1Floods11 readers, counts
smp = R.smp


def fold_models(run, variant, k, device, seeds):
    out = []
    for sd in seeds:
        ck = torch.load(Path(run) / f"fold{k}" / "weights" / f"fold{k - 1}_{variant}6_seed{sd}.pt", map_location="cpu",
                        weights_only=False)
        m = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=len(ck["bands"]), classes=3)
        m.load_state_dict(ck["state_dict"])
        out.append(m.to(device).eval())
    return out, np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)


def land_probs(models, zt):
    """p(land) of each fold, plain and averaged over the 8 turns and flips. zt: (1, C, H, W), H and W multiples of 32.
    Returns {fold: (plain, flips)}, each (H, W)."""
    out = {}
    with torch.no_grad(), torch.autocast(zt.device.type, dtype=torch.float16, enabled=zt.device.type == "cuda"):
        for k, ms in models.items():
            plain = flips = 0
            for m in ms:
                for turn in range(4):
                    for mirror in (False, True):
                        x = torch.rot90(zt, turn, dims=(2, 3))
                        x = torch.flip(x, dims=(3,)) if mirror else x
                        p = torch.softmax(m(x).float(), 1)[:, 0]
                        p = torch.flip(p, dims=(2,)) if mirror else p
                        p = torch.rot90(p, -turn, dims=(1, 2))[0]
                        flips = flips + p
                        if turn == 0 and not mirror:
                            plain = plain + p
            out[k] = (plain / len(ms), flips / (8 * len(ms)))
    return out


def maps(models, zt, h, w, r0=0, c0=0):
    """The candidates' water maps (p(land) < 0.5) on rows r0:r0+h, columns c0:c0+w of the padded input."""
    p = land_probs(models, zt)
    cut = lambda a: (a[r0:r0 + h, c0:c0 + w] < 0.5).cpu().numpy()  # noqa: E731
    out = {}
    for k, (plain, flips) in p.items():
        out[f"A0 fold{k}"], out[f"A0+flips fold{k}"] = cut(plain), cut(flips)
    out["A1"] = cut(sum(v[0] for v in p.values()) / len(p))
    out["A2"] = cut(sum(v[1] for v in p.values()) / len(p))
    return out


def chip_maps(models, x, mean, std, device):
    """As retrain6.water(): normalised, no data = 0, reflect-padded at the right and bottom to a multiple of 32."""
    z = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)
    h, w = z.shape[1:]
    zt = F.pad(torch.from_numpy(z.astype(np.float32))[None], (0, (-w) % 32, 0, (-h) % 32), mode="reflect").to(device)
    return maps(models, zt, h, w)


def add(acc, m, t, ok):
    for cand, mm in m.items():
        a = acc.setdefault(cand, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        for key, v in R.counts(mm, t, ok).items():
            a[key] += v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--run", default="/kaggle/working")
    ap.add_argument("--out", default="")
    ap.add_argument("--variant", default="control")
    ap.add_argument("--folds", default="1,2")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    i, n = (int(v) for v in args.part.split("/"))
    out = Path(args.out or args.run)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    p = R.paths(args)
    folds = [int(f) for f in args.folds.split(",")]
    models = {}
    for k in folds:
        models[k], mean, std = fold_models(args.run, args.variant, k, device, [0] if args.dry_run else [0, 1, 2])
    S = load("score_s1s2")
    work = [("asian", s) for s in R.ASIAN] + [("held-out training", s) for s in R.DEV_S1S2] + [("flood", "valid")]
    if args.dry_run:
        work = [work[0], work[5], work[-1]]
    work = work[i::n]
    R.log(f"batch 2, part {i + 1} of {n} on {device}: folds {folds}, {sum(len(m) for m in models.values())} models; "
          f"work {work}")
    rows = []

    def keep(kind, scene, chip, acc):
        rows.extend({"set": kind, "scene": scene, "chip": chip, "candidate": c, **a} for c, a in acc.items())
        R.write(out / f"ens_part{i}.csv", rows)

    for kind, sid in work:
        t1 = time.time()
        if kind == "asian":
            with rasterio.open(p["gee"] / f"{sid}_radar.tif") as src:
                img = src.read()
            with rasterio.open(p["gee"] / f"{sid}_truth.tif") as src:
                tr = src.read(1)
            if args.dry_run:
                img, tr = img[:, 5000:6100, 5000:6300], tr[5000:6100, 5000:6300]
            ok, truth = (tr != 255) & (img != -32768).all(0), tr == 1
            H, W = ok.shape
            acc = {}
            for r0 in range(0, H, S.T):  # as retrain6.score_scene: each tile scored on its centre
                for c0 in range(0, W, S.T):
                    if not ok[r0:r0 + S.T, c0:c0 + S.T].any():
                        continue
                    h, w = min(S.T, H - r0), min(S.T, W - c0)
                    z, _ = S.tile_input(img, r0, c0, mean, std)
                    m = maps(models, torch.from_numpy(z[None]).to(device), h, w, S.M, S.M)
                    add(acc, m, truth[r0:r0 + h, c0:c0 + w], ok[r0:r0 + h, c0:c0 + w])
            keep(kind, sid, "", acc)
        elif kind == "held-out training":
            dx, dy, _ = R.load_s1s2(p["s1s2"], scenes=(sid,))
            acc = {}
            for j in range(2 if args.dry_run else len(dx)):
                x = dx[j].astype(np.float32) / 100
                x[dx[j] == -32768] = np.nan
                add(acc, chip_maps(models, x, mean, std, device), dy[j] == R.ANY, (dy[j] <= 3) & np.isfinite(x).all(0))
            keep(kind, sid, "", acc)
        else:
            chips = R.s1f_split(p["s1f"], "valid")
            for s1, lab in (chips[:3] if args.dry_run else chips):
                x, y = R.flood_raw(s1, lab)
                acc = {}
                add(acc, chip_maps(models, x, mean, std, device), y == 1, y >= 0)
                keep(kind, "valid", Path(s1).name.replace("_S1Hand.tif", ""), acc)
        R.log(f"   {kind} {sid}: {(time.time() - t1) / 60:.1f} min")
    R.log(f"part {i + 1} of {n} done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
