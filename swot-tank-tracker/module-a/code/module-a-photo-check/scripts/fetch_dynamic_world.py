r"""Fetches Dynamic World for every clear first-check chip of run 3's 103 reservoirs (4 Nov 2026), under
../PROTOCOL-PHOTO.md: reference B.

For each chip it takes the Dynamic World image(s) of the chip's own Sentinel-2 product(s) (the manifest's
s2_images; GOOGLE/DYNAMICWORLD/V1 shares Sentinel-2's system:index), mosaics their "label" band (0 water, 1 trees,
2 grass, 3 flooded vegetation, 4 crops, 5 shrub and scrub, 6 built, 7 bare, 8 snow and ice; 255 = none) and saves it
on the chip's exact grid as dw/<reservoir>/<chip file>. dw/chips.csv lists every chip with the number of Dynamic
World images found. Light: downloads only. It resumes: a chip whose file exists is skipped.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-photo-check\scripts\fetch_dynamic_world.py
"""
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.io import MemoryFile

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[1]
DATA = REPO / "temp/module-a-retrain/kaggle-data"
FOLDS = REPO / "temp/module-a-retrain/runs/2026-11-03-retrain-3/prep/folds.csv"
OUT = HERE / "dw"
PROJECT = "project-c14134e4-dd6a-4968-b12"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")  # as retrain.safe


def one(job):
    name, r = job
    chip = DATA / "first-check" / safe(name) / r["file"]
    dest = OUT / safe(name) / r["file"]
    ids = [i for i in re.split(r"[;+ ]+", r["s2_images"]) if i]
    row = {"reservoir": name, "file": r["file"], "s2_date": r["s2_date"], "s2_images": r["s2_images"]}
    if dest.exists():
        return {**row, "status": "done before"}
    with rasterio.open(chip) as src:
        crs, t, w, h = src.crs.to_string(), src.transform, src.width, src.height
    col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filter(ee.Filter.inList("system:index", ids))
    for attempt in range(5):
        try:
            found = col.size().getInfo()
            if found == 0:
                return {**row, "dw_images": 0, "status": "no Dynamic World image"}
            img = col.select("label").mosaic().unmask(255).toUint8()
            g = {"dimensions": {"width": w, "height": h}, "crsCode": crs,
                 "affineTransform": {"scaleX": t.a, "shearX": t.b, "translateX": t.c, "shearY": t.d, "scaleY": t.e,
                                     "translateY": t.f}}
            raw = ee.data.computePixels({"expression": img, "fileFormat": "GEO_TIFF", "grid": g})
            with MemoryFile(raw) as mem, mem.open() as s:
                lab = s.read(1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dest, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs=crs,
                               transform=t, nodata=255, compress="deflate") as d:
                d.write(lab, 1)
            return {**row, "dw_images": found, "status": "ok",
                    "dw_water_share": round(float((lab == 0).mean()), 4),
                    "dw_flooded_veg_share": round(float((lab == 3).mean()), 4),
                    "dw_none_share": round(float((lab == 255).mean()), 4)}
        except Exception as e:  # Earth Engine's occasional time-outs and rate limits
            log(f"   {name} {r['file']}: {e}; retry {attempt + 1}")
            time.sleep(10 * (attempt + 1))
    return {**row, "status": "failed"}


def main():
    ee.Initialize(project=PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(FOLDS, encoding="utf-8") as f:
        names = [r["reservoir"] for r in csv.DictReader(f)]
    jobs = []
    for name in names:
        with open(DATA / "tables/manifests" / f"{safe(name)}.csv", encoding="utf-8") as f:
            jobs += [(name, r) for r in csv.DictReader(f) if r["kind"] == "clear"]
    log(f"{len(names)} reservoirs, {len(jobs)} clear chips")
    rows, t0 = [], time.time()
    with ThreadPoolExecutor(8) as pool:
        for i, row in enumerate(pool.map(one, jobs), 1):
            rows.append(row)
            if i % 100 == 0:
                log(f"{i} of {len(jobs)} in {time.time() - t0:.0f} s")
    old = {}
    if (OUT / "chips.csv").exists():
        with open(OUT / "chips.csv", encoding="utf-8") as f:
            old = {(r["reservoir"], r["file"]): r for r in csv.DictReader(f)}
    rows = [old.get((r["reservoir"], r["file"]), r) if r["status"] == "done before" else r for r in rows]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "chips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    log("done:", Counter(r["status"] for r in rows))


if __name__ == "__main__":
    main()
