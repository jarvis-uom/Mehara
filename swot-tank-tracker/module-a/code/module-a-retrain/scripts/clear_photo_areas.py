r"""The photo's water area on every clear Sentinel-2 date, for the "last clear photo carried forward" baseline
(3 Nov 2026). The independent review and the outside reviews asked for it: without radar, a user would read storage
from the water area of the most recent clear photo.

For each reservoir of the every-pass check, Earth Engine (on Google's servers) takes every Sentinel-2 L1C date from
Aug 2020 to Oct 2026 (one mosaic per date, as the chip export), and inside the reservoir's outline (the fixed outline
of kernel2/retrain.py, on the chip's own 10 m grid) counts the pixels with data, the clear pixels (Cloud Score+ cs_cdf
>= 0.6) and the clear pixels with NDWI >= -0.06. Writes out/clear_photos/<reservoir>.csv (one line per date) and skips
reservoirs already done. score_last_clear.py reads them.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\clear_photo_areas.py
The largest outline (Senanayaka Sam) timed out with a year per request; it was fetched alone with --months 1.
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RES = REPO / "temp/module-a-reservoirs"
sys.path.insert(0, str(ROOT / "kernel2"))
import retrain as K  # noqa: E402  (tank_mask: the fixed outline; safe; read_full)

sys.stdout.reconfigure(encoding="utf-8")
PROJECT = "project-c14134e4-dd6a-4968-b12"
START, END = date(2020, 8, 1), date(2026, 11, 1)
OUT = ROOT / "out/clear_photos"
FIELDS = ("date", "pixels", "valid", "clear", "water")


def windows(months):
    """START to END in steps of this many months: one Earth Engine request each (shorter for the largest outlines)."""
    out, d = [], START
    while d < END:
        m = d.month - 1 + months
        nxt = min(date(d.year + m // 12, m % 12 + 1, 1), END)
        out.append((d.isoformat(), nxt.isoformat()))
        d = nxt
    return out


def outline(name):
    """The fixed outline as a polygon on the chip's grid, and the grid (CRS and transform)."""
    with open(RES / "manifests" / f"{K.safe(name)}.csv", encoding="utf-8") as f:
        first = RES / "chips" / next(csv.DictReader(f))["file"]
    _, occ, _ = K.read_full(first)
    with rasterio.open(first) as src:
        crs, tr = src.crs.to_string(), src.transform
    mask = K.tank_mask(occ)
    geoms = [shape(g) for g, v in features.shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
    return unary_union(geoms).simplify(1), crs, [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f], int(mask.sum())


def year_rows(geom, crs, transform, start, end):
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geom).filterDate(start, end)
          .linkCollection(cs, ["cs_cdf"]))
    days = s2.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY-MM-dd")).distinct()

    def one(d):
        d0 = ee.Date.parse("YYYY-MM-dd", d)
        img = s2.filterDate(d0, d0.advance(1, "day")).mosaic()
        b3, b8 = img.select("B3"), img.select("B8")
        valid = b3.mask().And(b8.mask())
        clear = img.select("cs_cdf").gte(0.6).And(valid)
        water = b3.subtract(b8).divide(b3.add(b8)).gte(-0.06).And(clear)
        stack = ee.Image.cat([ee.Image(1).rename("pixels"), valid.rename("valid"), clear.rename("clear"),
                              water.rename("water")]).unmask(0)
        r = stack.reduceRegion(ee.Reducer.sum().unweighted(), geom, crs=crs, crsTransform=transform, maxPixels=1e9)
        return ee.Feature(None, r).set("date", d)
    feats = ee.FeatureCollection(days.map(one)).getInfo()["features"]
    return [{k: f["properties"].get(k) for k in FIELDS} for f in feats]


def do(name, months=12):
    out = OUT / f"{K.safe(name)}.csv"
    if out.exists():
        return name, "skipped", 0
    t0 = time.time()
    poly, crs, transform, px = outline(name)
    geom = ee.Geometry(mapping(poly), crs, False)
    rows = []
    for start, end in windows(months):
        for attempt in range(4):
            try:
                rows += year_rows(geom, crs, transform, start, end)
                break
            except ee.EEException as e:
                print(f"   {name} {start}: {e} (try {attempt + 1})", flush=True)
                if attempt == 3:
                    return name, f"failed: {e}", 0
                time.sleep(20 * (attempt + 1))
    rows.sort(key=lambda r: r["date"])
    part = out.with_suffix(".part")
    with open(part, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ("mask_pixels",))
        w.writeheader()
        w.writerows({**r, "mask_pixels": px} for r in rows)
    part.replace(out)
    return name, f"{len(rows)} dates, {time.time() - t0:.0f} s", len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of reservoirs (a test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--months", type=int, default=12, help="months per Earth Engine request")
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(RES / "passes/out/pass_flags.csv", encoding="utf-8") as f:
        names = [r["reservoir"] for r in csv.DictReader(f) if r["used"] == "True"]
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (name, status, _) in enumerate(pool.map(lambda n: do(n, args.months), names), 1):
            print(f"[{(time.time() - started) / 60:5.1f} min] {i}/{len(names)} {name}: {status}", flush=True)
    print("clear photo areas finished", flush=True)


if __name__ == "__main__":
    main()
