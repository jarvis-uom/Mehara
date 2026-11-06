r"""Is the radar's extra water at low storage wet ground? A check by eye and by the photo's own colours (2 Nov 2026).

On the first check's clear days at low storage (under 20% of capacity), the radar U-Net (run 2) calls more of the
outline water than the photo does (51% against 35%; high_water.py). For each such day, at the reservoirs that
pass_stats.py used, it splits the outline's clear pixels with radar into four parts: water in both, radar only (the
extra), photo only and neither. The photo's water is NDWI >= -0.06, as in measure_areas.py. For each part it records
the photo's NDVI (green plants about 0.4 and more; bare ground, mud or water under 0.2), the radar's VV (dB) and how
often JRC saw water there (1984 to 2021). The extra is split by the photo's own colours: water by the SWIR test
(MNDWI = (B3 - B11) / (B3 + B11) >= 0; SWIR stays dark over water, so it finds water that NDWI misses under haze or
thin cloud), and the rest by NDVI into green plants, sparse plants and bare ground. It also records the share of the
extra within 30 m of the photo's water. Shares are means over days. Nothing is marked by hand.
It draws the days with the largest extra and days near the median: the photo, the radar and the two water maps.
Writes out/low_water_days.csv (one line per day) and previews/low_water_days.png.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\look_low_water.py
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_areas as A  # noqa: E402  (load_model, MODELS)
import reservoir_stats as S  # noqa: E402  (sheet, nearest, write)
import run_model as M  # noqa: E402  (load, predict, tank_mask, photo, BANDS)

sys.stdout.reconfigure(encoding="utf-8")
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
COLOURS = {"both": (0.16, 0.47, 0.84), "radar_only": (0.92, 0.41, 0.20), "photo_only": (0.55, 0.36, 0.84)}
LOW = 0.2  # storage under 20% of capacity
GREEN, BARE = 0.4, 0.2  # NDVI
NEAR_PX = 3  # 30 m
PARTS = ("both", "radar_only", "photo_only", "neither")


def med(v, nd=2):
    return round(float(np.median(v)), nd) if len(v) else None


def mean(rows, k):
    v = [r[k] for r in rows if r[k] is not None]
    return round(float(np.mean(v)), 2) if v else None


def low_days(used):
    info, series = S.sheet()
    days = []
    for path in sorted((ROOT / "areas").glob("*.csv")):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["kind"] != "clear" or r["tank"] not in used:
                    continue
                s = S.nearest(series.get(r["tank"], {}), date.fromisoformat(r["s2_date"]))
                if s is not None and s < LOW * info[r["tank"]]["cap"]:
                    days.append({**r, "cap": info[r["tank"]]["cap"], "storage": s})
    return days


def split(model, r):
    """The chip, the photo's NDVI and MNDWI, the radar's VV and the four parts of the outline on one day."""
    path = ROOT / "chips" / r["file"]
    x, occ = M.load(path)
    with rasterio.open(path) as src:
        cs = src.read(14).astype(np.float32) / 100
    b3, b4, b8, b11, vv = (x[M.BANDS.index(b)] for b in ("B3", "B4", "B8", "B11", "VV"))
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi, ndvi, mndwi = (b3 - b8) / (b3 + b8), (b8 - b4) / (b8 + b4), (b3 - b11) / (b3 + b11)
    mask = M.tank_mask(occ)
    ok = mask & ~((cs > 0) & (cs < 0.6)) & np.isfinite(vv) & np.isfinite(ndwi)  # clear, with radar
    photo = ok & (ndwi >= -0.06)
    radar = ok & M.predict(*model, x).astype(bool)
    parts = {"both": photo & radar, "radar_only": radar & ~photo, "photo_only": photo & ~radar,
             "neither": ok & ~photo & ~radar}
    return x, occ, mask, {"ndvi": ndvi, "mndwi": mndwi, "vv": vv}, parts


def describe(r, occ, ix, parts):
    ndvi, vv = ix["ndvi"], ix["vv"]
    n = sum(int(p.sum()) for p in parts.values())
    extra, water = parts["radar_only"], parts["both"] | parts["photo_only"]
    dist = ndimage.distance_transform_edt(~water) if water.any() else np.full(extra.shape, np.inf)
    e, wet = ndvi[extra], ix["mndwi"][extra] >= 0
    kinds = {"swir_water": wet, "green": ~wet & (e >= GREEN), "sparse": ~wet & (e >= BARE) & (e < GREEN),
             "bare": ~wet & ~(e >= BARE)}
    row = {"reservoir": r["tank"], "cap_mcm": round(r["cap"], 2), "s2_date": r["s2_date"], "s1_date": r["s1_date"],
           "days_apart": int(r["days_apart"]), "storage_pct": round(100 * r["storage"] / r["cap"], 1), "pixels": n,
           "photo_share": round(float(water.sum()) / n, 3), "radar_share": round(float((parts["both"] | extra).sum()) / n, 3),
           "extra_share": round(float(extra.sum()) / n, 3),
           **{f"extra_{k}_share": round(float(v.mean()), 3) if len(e) else None for k, v in kinds.items()},
           "extra_within_30m_share": round(float((dist[extra] <= NEAR_PX).mean()), 3) if len(e) else None}
    for k, p in parts.items():
        row.update({f"{k}_ndvi": med(ndvi[p]), f"{k}_vv_db": med(vv[p], 1), f"{k}_jrc_pct": med(occ[p], 0)})
    return row


def pick(rows, k):
    """k days with the largest extra, then k near the median, one day per reservoir."""
    chosen, seen = [], set()
    mid = float(np.median([r["extra_share"] for r in rows]))
    for order in (sorted(rows, key=lambda r: -r["extra_share"]), sorted(rows, key=lambda r: abs(r["extra_share"] - mid))):
        got = 0
        for r in order:
            if got < k and r["reservoir"] not in seen:
                chosen.append(r)
                seen.add(r["reservoir"])
                got += 1
    return chosen


def draw(model, days, chosen):
    by = {(d["tank"], d["s2_date"]): d for d in days}
    fig, axes = plt.subplots(len(chosen), 3, figsize=(14, 4.1 * len(chosen)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, row in enumerate(chosen):
        x, occ, mask, ix, parts = split(model, by[(row["reservoir"], row["s2_date"])])
        vv = ix["vv"]
        rr, cc = np.where(mask)
        win = (slice(max(rr.min() - 20, 0), rr.max() + 21), slice(max(cc.min() - 20, 0), cc.max() + 21))
        rgb = M.photo(x)[win]
        over = 0.45 * rgb + 0.55
        for k, colour in COLOURS.items():
            over[parts[k][win]] = colour
        for ax, img, kw in ((axes[i, 0], rgb, {}), (axes[i, 1], vv[win], {"cmap": "gray", "vmin": -25, "vmax": -5}),
                            (axes[i, 2], over, {})):
            ax.imshow(img, **kw)
            ax.contour(mask[win], levels=[0.5], colors="yellow", linewidths=0.6)
            ax.set_xticks([]), ax.set_yticks([])
        axes[i, 0].set_title(f"{row['reservoir']} ({row['cap_mcm']} million m3), photo {row['s2_date']}\n"
                             f"storage {row['storage_pct']:.0f}% of capacity", fontsize=9, color=INK)
        axes[i, 1].set_title(f"Radar VV, {row['s1_date']}\n(dark = smooth: water, or wet flat ground)",
                             fontsize=9, color=INK)
        w, b, s, g = (row[f"extra_{k}_share"] or 0 for k in ("swir_water", "bare", "sparse", "green"))
        axes[i, 2].set_title(f"Radar only (orange): {row['extra_share']:.0%} of the outline. Of it:\n"
                             f"water by SWIR {w:.0%}; bare {b:.0%}, sparse plants {s:.0%}, green {g:.0%}",
                             fontsize=9, color=INK)
    fig.suptitle("Low water (under 20% of capacity): water in the photo and on the radar. Blue: both; orange: radar "
                 "only; purple: photo only; yellow: outline.\nTop rows: the largest radar-only share; bottom rows: "
                 "near the median.", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (4.1 * len(chosen))))
    out = ROOT / "previews/low_water_days.png"
    fig.savefig(out, dpi=90, facecolor=SURFACE)
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3, help="days drawn in each group")
    args = ap.parse_args()
    with open(ROOT / "passes/out/pass_flags.csv", encoding="utf-8") as f:
        used = {r["reservoir"] for r in csv.DictReader(f) if r["used"] == "True"}
    model = A.load_model(A.MODELS["radar U-Net r2"])
    days = low_days(used)
    rows = []
    for d in days:
        _, occ, _, ix, parts = split(model, d)
        rows.append(describe(d, occ, ix, parts))
    S.write(ROOT / "out/low_water_days.csv", rows)
    print(f"{len(rows)} clear days under {LOW:.0%} of capacity at {len({r['reservoir'] for r in rows})} reservoirs")
    print(f"Share of the outline (clear, with radar), mean over days: photo {mean(rows, 'photo_share')}, radar "
          f"{mean(rows, 'radar_share')}, radar only {mean(rows, 'extra_share')}")
    print(f"The radar-only part, mean over days: water by the photo's SWIR test (MNDWI >= 0) "
          f"{mean(rows, 'extra_swir_water_share')}; the rest: bare ground (NDVI < {BARE}) {mean(rows, 'extra_bare_share')},"
          f" sparse plants {mean(rows, 'extra_sparse_share')}, green plants (NDVI >= {GREEN}) "
          f"{mean(rows, 'extra_green_share')}. Within 30 m of the photo's water: {mean(rows, 'extra_within_30m_share')}")
    x = [r for r in rows if r["extra_swir_water_share"] is not None]
    print(f"   days where the SWIR test calls over half the radar-only part water: "
          f"{sum(r['extra_swir_water_share'] > 0.5 for r in x)} of {len(x)}")
    print("Median over days, each part (NDVI, VV dB, JRC % of years seen as water):")
    for k in PARTS:
        v = [med([r[f"{k}_{m}"] for r in rows if r[f"{k}_{m}"] is not None]) for m in ("ndvi", "vv_db", "jrc_pct")]
        print(f"   {k:10s} NDVI {v[0]!s:>5}   VV {v[1]!s:>6}   JRC {v[2]!s:>5}")
    for d in (0, 1, 2):
        x = [r for r in rows if r["days_apart"] == d]
        if x:
            print(f"   radar {d} days from the photo: {len(x)} days, radar only {mean(x, 'extra_share')}")
    draw(model, days, pick(rows, args.rows))


if __name__ == "__main__":
    main()
