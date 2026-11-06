r"""Were the chips cut in the right place? A check of all 117 reservoirs, by eye and by numbers (2 Nov 2026).

For each reservoir it takes its first-check chip with the highest storage on a clear day (or the clearest chip):
- by eye (previews/crops_page<k>.png): the photo, the outline (the JRC water body the chip is centred on, widened by
  30 m, as the fixed outline in temp/module-a-retrain/kernel/retrain.py) and the department's point (red cross);
- by numbers (out/crop_checks.csv):
  - whether the water body touches the chip's edge (the lake would be cut off);
  - the distance from the department's point to that water body;
  - the chip centre with a Google Maps link, to compare with an independent map;
  - the radar U-Net's Spearman with storage on every pass, where the reservoir was scored. A wrong lake would not
    follow the named reservoir's storage.
Light: one chip per reservoir. Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\check_crops.py
"""
import csv
import sys
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from rasterio.warp import transform as warp
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
sys.path.insert(0, str(REPO / "temp/module-a-retrain/kernel"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir_stats as S  # noqa: E402  (sheet, nearest, write)
import retrain as K  # noqa: E402  (tank_mask: the fixed outline)
import run_model as M  # noqa: E402  (load, photo)

sys.stdout.reconfigure(encoding="utf-8")
INK, SURFACE = "#0b0b0b", "#fcfcfb"
PER_PAGE, COLS = 30, 6


def points():
    out = {}
    with open(S.SHEET, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["reservoir"] and r["reservoir"] not in out:
                out[r["reservoir"]] = (float(r["lat"]), float(r["lon"]))
    return out


def best_chip(name, info, series):
    with open(ROOT / "manifests" / f"{K.safe(name)}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clear = [r for r in rows if r["kind"] == "clear"]
    for r in clear:
        s = S.nearest(series.get(name, {}), date.fromisoformat(r["s2_date"]))
        r["pct"] = None if s is None else 100 * s / info[name]["cap"]
    with_s = [r for r in clear if r["pct"] is not None]
    if with_s:
        return max(with_s, key=lambda r: r["pct"])
    return max(rows, key=lambda r: float(r["clear"] or 0)) | {"pct": None}


def check(name, info, series, pts, spearman):
    r = best_chip(name, info, series)
    path = ROOT / "chips" / r["file"]
    x, occ = M.load(path)
    with rasterio.open(path) as src:
        crs, tr = src.crs, src.transform
    outline = K.tank_mask(occ)
    body = ndimage.binary_erosion(outline, iterations=3) & (occ > 0)  # the water body itself, without the 30 m
    h, w = occ.shape
    lat, lon = pts[name]
    (px,), (py,) = warp("EPSG:4326", crs, [lon], [lat])
    col, row = ~tr * (px, py)
    rr, cc = np.where(body if body.any() else outline)
    dist = float(np.min(np.hypot(rr - row, cc - col))) * 10 if len(rr) else np.nan
    edge = bool(body[0].any() or body[-1].any() or body[:, 0].any() or body[:, -1].any())
    (clon,), (clat,) = warp(crs, "EPSG:4326", [tr.c + tr.a * w / 2], [tr.f + tr.e * h / 2])
    return {"reservoir": name, "cap_mcm": round(info[name]["cap"], 2), "chip": r["file"],
            "storage_pct": None if r["pct"] is None else round(r["pct"]),
            "chip_km": f"{w * 0.01:.1f} x {h * 0.01:.1f}", "outline_ha": round(float(outline.sum()) * 0.01, 1),
            "lake_touches_edge": edge, "point_to_lake_m": round(dist), "point_in_chip": 0 <= col < w and 0 <= row < h,
            "radar_spearman": spearman.get(name), "map": f"https://www.google.com/maps?q={clat:.5f},{clon:.5f}"}, \
        (M.photo(x), outline, (col, row))


def main():
    info, series = S.sheet()
    pts = points()
    with open(ROOT / "passes/out/pass_scores.csv", encoding="utf-8") as f:
        spearman = {r["reservoir"]: float(r["spearman_all"]) for r in csv.DictReader(f)
                    if r["method"] == "radar U-Net r2" and r["spearman_all"]}
    names = sorted(info)
    rows, pics = [], []
    for n in names:
        row, pic = check(n, info, series, pts, spearman)
        rows.append(row)
        pics.append(pic)
    S.write(ROOT / "out/crop_checks.csv", rows)
    for k in range(0, len(names), PER_PAGE):
        chunk = list(zip(rows[k:k + PER_PAGE], pics[k:k + PER_PAGE]))
        nrow = int(np.ceil(len(chunk) / COLS))
        fig, axes = plt.subplots(nrow, COLS, figsize=(2.6 * COLS, 2.9 * nrow), squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (r, (rgb, outline, (col, row))) in zip(axes.ravel(), chunk):
            ax.imshow(rgb)
            ax.contour(outline, levels=[0.5], colors="yellow", linewidths=0.6)
            ax.plot(col, row, "r+", ms=9, mew=1.5)
            flag = " (EDGE)" if r["lake_touches_edge"] else ""
            pct = "" if r["storage_pct"] is None else f", {r['storage_pct']}% full"
            ax.set_title(f"{r['reservoir']}{flag}\n{r['cap_mcm']} MCM{pct}", fontsize=7.5, color=INK)
        fig.suptitle("Were the chips cut in the right place? Photo, outline (yellow) and the department's point "
                     f"(red cross). Page {k // PER_PAGE + 1}", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = ROOT / "previews" / f"crops_page{k // PER_PAGE + 1}.png"
        fig.savefig(out, dpi=80, facecolor=SURFACE)
        plt.close(fig)
        print("saved", out)
    edge = [r["reservoir"] for r in rows if r["lake_touches_edge"]]
    far = sorted(((r["point_to_lake_m"], r["reservoir"]) for r in rows if r["point_to_lake_m"] > 500), reverse=True)
    out_pt = [r["reservoir"] for r in rows if not r["point_in_chip"]]
    print(f"{len(rows)} reservoirs. Lake touches the chip edge: {len(edge)} {edge}")
    print(f"Department's point outside the chip: {len(out_pt)} {out_pt}")
    print(f"Department's point more than 500 m from the lake: {len(far)} {far}")
    low = sorted((v, n) for n, v in spearman.items() if v < 0.3)
    print(f"Radar Spearman under 0.3 (look at these): {len(low)} {[(n, round(v, 2)) for v, n in low]}")


if __name__ == "__main__":
    main()
