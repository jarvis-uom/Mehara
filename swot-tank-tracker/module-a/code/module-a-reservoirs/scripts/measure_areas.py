r"""Water area in every reservoir chip, by each method (2 Nov 2026). The label-free check at scale, step 1 of 2.

For each chip listed in manifests/ it measures the water inside the reservoir's outline (the JRC water body that holds
the chip centre, widened by 30 m), with:
- the radar-only U-Nets: run 2, and run 6 seeds 0, 1 and 2 (the best method on our 7 tanks);
- the radar + optical U-Net of run 3 (fake clouds), and run 3 with the radar-only U-Net of run 2 where Cloud Score+
  calls the pixel cloudy;
- NDWI >= -0.06 and VV < -16.75 dB (thresholds tuned on the Sen1Floods11 training chips).
It also records the share of the outline under cloud and without radar, and how many outline pixels fall below each
VV threshold from -24 to -10 dB (3 x 3 median), so reservoir_stats.py can tune a radar rule for each reservoir.
Writes areas/<reservoir>.csv and areas/<reservoir>_vv.npy; reservoirs already measured are skipped, so it can be
stopped and started again.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\measure_areas.py
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
TANKS = REPO / "temp/module-a-tanks"
sys.path.insert(0, str(TANKS / "scripts"))
import run_model as M  # noqa: E402  (load, predict, tank_mask)

sys.stdout.reconfigure(encoding="utf-8")
MODELS = {"radar U-Net r2": "weights/run-2-radar/unet_radar_best.pt",
          "radar U-Net r6 s0": "weights/run-6/unet_radar_best.pt",
          "radar U-Net r6 s1": "weights/run-6/unet_radar_seed1_best.pt",
          "radar U-Net r6 s2": "weights/run-6/unet_radar_seed2_best.pt",
          "fusion U-Net r3": "weights/run-3/unet_radar_optical_clouds_best.pt"}
SWITCH = "r3 where clear, radar U-Net r2 under cloud"
THRESHOLDS = np.arange(-24, -10.01, 0.25)  # dB
AREAS = ROOT / "areas"


def load_model(path):
    ck = torch.load(TANKS / path, map_location="cpu", weights_only=False)
    model = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=len(ck["bands"]), classes=2)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def measure(models, path):
    x, occ = M.load(path)
    mask = M.tank_mask(occ)
    with rasterio.open(path) as src:
        cs = src.read(14).astype(np.float32) / 100
    cloudy = (cs > 0) & (cs < 0.6)
    vv_raw = x[M.BANDS.index("VV")]
    maps = {name: M.predict(*models[name], x).astype(bool) for name in MODELS}
    maps[SWITCH] = np.where(cloudy, maps["radar U-Net r2"], maps["fusion U-Net r3"])
    b3, b8 = x[M.BANDS.index("B3")], x[M.BANDS.index("B8")]
    with np.errstate(invalid="ignore", divide="ignore"):
        maps["NDWI >= -0.06"] = (b3 - b8) / (b3 + b8) >= -0.06
        maps["VV < -16.75 dB"] = vv_raw < -16.75
    vv = ndimage.median_filter(np.nan_to_num(vv_raw, nan=0.0), size=3)
    counts = np.searchsorted(np.sort(vv[mask]), THRESHOLDS).astype(np.int32)  # outline pixels below each threshold
    row = {"outline_ha": round(float(mask.sum()) * 0.01, 1),
           "cloud_share": round(float(cloudy[mask].mean()), 3),
           "radar_gap_share": round(float(np.isnan(vv_raw[mask]).mean()), 3),
           **{m: round(float((w & mask).sum()) * 0.01, 1) for m, w in maps.items()}}
    return row, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of reservoir file names (for a test)")
    args = ap.parse_args()
    AREAS.mkdir(exist_ok=True)
    models = {name: load_model(p) for name, p in MODELS.items()}
    manifests = sorted((ROOT / "manifests").glob("*.csv"))
    if args.only:
        manifests = [m for m in manifests if m.stem in args.only.split(",")]
    started = time.time()
    for k, man in enumerate(manifests, 1):
        rows_in = list(csv.DictReader(open(man, encoding="utf-8")))
        out = AREAS / f"{man.stem}.csv"
        if out.exists() and sum(1 for _ in open(out, encoding="utf-8")) - 1 == len(rows_in):
            continue
        rows, counts = [], []
        for r in rows_in:
            res, c = measure(models, ROOT / "chips" / r["file"])
            rows.append({k2: r[k2] for k2 in ("tank", "kind", "file", "s2_date", "s1_date", "days_apart", "clear")} | res)
            counts.append(c)
        np.save(AREAS / f"{man.stem}_vv.npy", np.array(counts))
        tmp = out.with_suffix(".part")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        tmp.replace(out)
        print(f"[{(time.time() - started) / 60:5.1f} min] {k}/{len(manifests)} {man.stem}: {len(rows)} chips", flush=True)
    print("all reservoirs measured", flush=True)


if __name__ == "__main__":
    main()
