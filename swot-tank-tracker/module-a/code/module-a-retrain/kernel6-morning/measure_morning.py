r"""Run 5's fold models measure the morning passes (PROTOCOL-6 section D; PROTOCOL-CURVE step 3). 6 Nov 2026.

Each of run 5's 103 reservoirs is measured by the three models of the fold that never trained on it (out of fold,
as the main passes), with run 5's own water rule (kernel5/retrain5.py: water, p(land) < 0.5 over the three seeds)
inside the same fixed outline (res4.pkl). The morning chips are on the main chips' grid; a chip of another shape is
skipped and reported. Writes areas_morning/<reservoir>.csv (the main passes' columns) and run.json.
On Kaggle (CPU) it reads the private datasets run4-inputs, run5-weights and morning-pass-chips. The same code checks
itself on the laptop against run 5's own main-pass areas (it must give the same numbers):
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\kernel6-morning\measure_morning.py
        --input temp --chips temp\module-a-reservoirs\passes --out temp\module-a-retrain\tmp\morning-check --limit 2
"""
import argparse
import csv
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

NEW = "run 5, water"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(base, pattern, must=""):
    hits = sorted(p for p in Path(base).rglob(pattern) if must in p.as_posix())
    if not hits:
        raise FileNotFoundError(f"{pattern} ({must}) under {base}")
    return hits


def read_radar(path):
    """VV, VH in dB (NaN = no data) from a chip whose first two bands are the radar as dB x 100."""
    with rasterio.open(path) as src:
        a = src.read([1, 2]).astype(np.float32)
    a[a == -32768] = np.nan
    return a / 100


def water(models, x, mean, std, device):
    """The ensemble's water (p(land) < 0.5) on one chip, as kernel5/retrain5.py."""
    z = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)
    h, w = z.shape[1:]
    zt = F.pad(torch.from_numpy(z.astype(np.float32))[None], (0, (-w) % 32, 0, (-h) % 32), mode="reflect").to(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        p = torch.stack([torch.softmax(m(zt).float()[0, :, :h, :w], 0) for m in models]).mean(0)
    return (p[0] < 0.5).cpu().numpy()


def load_fold(wdir, k, device):
    models, ck = [], None
    for sd in (0, 1, 2):
        ck = torch.load(wdir / f"fold{k}_photo5_seed{sd}.pt", map_location="cpu", weights_only=False)
        m = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=len(ck["bands"]), classes=3)
        m.load_state_dict(ck["state_dict"])
        models.append(m.to(device).eval())
    return models, np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--chips", default="", help="the chips' folder (default: the morning dataset under --input)")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--limit", type=int, default=0, help="laptop check: only this many reservoirs, 5 passes each")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    i = Path(args.input)
    res_path = find(i, "res4.pkl")[0]
    wdir = find(i, "fold0_photo5_seed0.pt")[0].parent
    chips = Path(args.chips) if args.chips else find(i, "Akathimuruppu.csv", "morning")[0].parents[1]
    tifs = chips / "chips" if (chips / "chips").is_dir() else chips  # the laptop keeps chips/ beside manifests/
    with open(res_path, "rb") as f:
        d = pickle.load(f)
    res, fold, used = d["res"], d["fold"], d["used"]
    out = Path(args.out)
    (out / "areas_morning").mkdir(parents=True, exist_ok=True)
    log(f"device {device}, {torch.get_num_threads()} threads; chips {chips}; weights {wdir}")
    names = sorted(used, key=lambda n: (fold[n], n))
    if args.limit:
        names = names[:args.limit]
    report = {"reservoirs": 0, "passes": 0, "no_manifest": [], "wrong_shape": [], "unreadable": []}
    loaded = None
    for n in names:
        R, k = res[n], fold[n]
        man = chips / "manifests" / f"{R['safe']}.csv"
        if not man.exists():
            report["no_manifest"].append(n)
            continue
        if loaded is None or loaded[0] != k:
            loaded = (k, *load_fold(wdir, k, device))
            log(f"fold {k + 1}: models loaded")
        _, models, mean, std = loaded
        inside = R["inside"]
        with open(man, encoding="utf-8") as f:
            entries = list(csv.DictReader(f))
        rows = []
        for e in (entries[:5] if args.limit else entries):
            try:
                x = read_radar(tifs / e["file"])
            except Exception as ex:  # noqa: BLE001  (reported, not fatal)
                report["unreadable"].append(f"{e['file']}: {ex}")
                continue
            if x.shape[1:] != inside.shape:
                report["wrong_shape"].append(f"{e['file']}: {x.shape[1:]} vs {inside.shape}")
                continue
            m = water(models, x, mean, std, device)
            rows.append({**{c: e[c] for c in ("tank", "date", "orbit", "pass", "photo_date", "photo_clear")},
                         "outline_ha": round(float(inside.sum()) * 0.01, 1),
                         "radar_gap_share": round(float(np.isnan(x[0][inside]).mean()), 3),
                         NEW: round(float((m & inside).sum()) * 0.01, 1)})
        if rows:
            with open(out / "areas_morning" / f"{R['safe']}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        report["reservoirs"] += 1
        report["passes"] += len(rows)
        log(f"   {n} (fold {k + 1}): {len(rows)} passes; {report['passes']} so far, "
            f"{(time.time() - t0) / 60:.0f} min")
    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["device"] = str(device)
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"done: {report['reservoirs']} reservoirs, {report['passes']} passes, {report['minutes']} min; "
        f"no manifest {len(report['no_manifest'])}, wrong shape {len(report['wrong_shape'])}, "
        f"unreadable {len(report['unreadable'])}")


if __name__ == "__main__":
    main()
