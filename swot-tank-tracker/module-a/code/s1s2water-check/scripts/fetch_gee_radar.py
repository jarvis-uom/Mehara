r"""Makes S1S2-Water's scenes ready for our models (3 Nov 2026), under ../PROTOCOL-S1S2.md, amendment 1.

For each scene of the main and extra sets:
1. downloads S1S2-Water's Sentinel-1 water mask and valid mask from Zenodo (one range request each, a few MB);
2. fetches Earth Engine's COPERNICUS/S1_GRD for the scene's own products (VV, VH, dB x 100, int16, -32768 = no
   data, as export_passes.py made our training radar) at 10 m on the scene's UTM grid, from the same corner;
3. resamples the mask to that grid by area: valid if at least half valid, water if at least half of the valid area;
4. registers it: the whole-pixel shift (-3 to 3 each way) with the largest VV contrast (mean outside the water minus
   inside), on every 2nd row and column; shift (dy, dx) puts the mask's cell (r + dy, c + dx) on the radar's (r, c);
5. writes gee/<scene>_radar.tif (2 bands), gee/<scene>_truth.tif (0 not water, 1 water, 255 not scored) and
   gee/scenes.csv, ready for the private Kaggle dataset is4990-s1s2water-gee.
Light: downloads only. It resumes: a scene whose two files exist is skipped.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\s1s2water-check\scripts\fetch_gee_radar.py
"""
import csv
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s1s2_zip as S  # noqa: E402  (the Zenodo reader and the scene lists)

PROJECT = "project-c14134e4-dd6a-4968-b12"
OUT = HERE / "gee"
TMP = HERE / "tmp" / "masks"
TILE, SCALE, SHIFT = 2048, 10, 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gee_radar(ids, crs, x0, y1, w, h):
    """The scene's products mosaicked, VV and VH as dB x 100, fetched in tiles on the 10 m grid."""
    img = (ee.ImageCollection([ee.Image(f"COPERNICUS/S1_GRD/{i}") for i in ids]).select(["VV", "VH"]).mosaic()
           .multiply(100).round().toInt16().unmask(-32768))
    out = np.full((2, h, w), -32768, np.int16)

    def one(rc):
        r0, c0 = rc
        th, tw = min(TILE, h - r0), min(TILE, w - c0)
        g = {"dimensions": {"width": tw, "height": th}, "crsCode": crs,
             "affineTransform": {"scaleX": SCALE, "shearX": 0, "translateX": x0 + c0 * SCALE, "shearY": 0,
                                 "scaleY": -SCALE, "translateY": y1 - r0 * SCALE}}
        for attempt in range(5):
            try:
                raw = ee.data.computePixels({"expression": img, "fileFormat": "GEO_TIFF", "grid": g})
                with MemoryFile(raw) as mem, mem.open() as src:
                    out[:, r0:r0 + th, c0:c0 + tw] = src.read()
                return
            except Exception as e:  # Earth Engine's occasional "too many requests" or time-outs
                log(f"   tile {r0},{c0}: {e}; retry {attempt + 1}")
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"tile {r0},{c0} failed 5 times")

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(one, [(r0, c0) for r0 in range(0, h, TILE) for c0 in range(0, w, TILE)]))
    return out


def resample_mask(msk_path, valid_path, crs, transform10, w, h):
    """Water share of the valid area and valid share of each 10 m cell (by area)."""
    with rasterio.open(msk_path) as src:
        water = src.read(1) == 1
        src_t, src_crs = src.transform, src.crs
    with rasterio.open(valid_path) as src:
        valid = src.read(1) == 1
    src = water.astype(np.float32)
    src[~valid] = np.nan
    share = np.full((h, w), np.nan, np.float32)
    reproject(src, share, src_transform=src_t, src_crs=src_crs,
              src_nodata=np.nan, dst_transform=transform10, dst_crs=crs, dst_nodata=np.nan,
              resampling=Resampling.average)
    vshare = np.zeros((h, w), np.float32)
    reproject(valid.astype(np.float32), vshare, src_transform=src_t, src_crs=src_crs, dst_transform=transform10,
              dst_crs=crs, resampling=Resampling.average)
    truth = np.full((h, w), 255, np.uint8)
    ok = (vshare >= 0.5) & np.isfinite(share)
    truth[ok] = (share[ok] >= 0.5).astype(np.uint8)
    return truth, float((water & valid).sum() / max(valid.sum(), 1)), abs(src_t.a)


def register(truth, vv):
    """The shift with the largest VV contrast (outside the water minus inside), on every 2nd row and column."""
    H, W = truth.shape
    v = vv[SHIFT:H - SHIFT:2, SHIFT:W - SHIFT:2]
    good = np.isfinite(v)
    best, table = None, {}
    for dy in range(-SHIFT, SHIFT + 1):
        for dx in range(-SHIFT, SHIFT + 1):
            t = truth[SHIFT + dy:H - SHIFT + dy:2, SHIFT + dx:W - SHIFT + dx:2]
            wet, dry = good & (t == 1), good & (t == 0)
            if wet.sum() < 100 or dry.sum() < 100:
                continue
            c = float(v[dry].mean() - v[wet].mean())
            table[(dy, dx)] = c
            if best is None or c > table[best]:
                best = (dy, dx)
    return best, table


def shifted(truth, dy, dx):
    """The mask moved so that its cell (r + dy, c + dx) sits on (r, c); cells moved in from outside are not scored."""
    out = np.full_like(truth, 255)
    H, W = truth.shape
    out[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] = \
        truth[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
    return out


def main():
    ee.Initialize(project=PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((HERE / "data/scenes.json").read_text(encoding="utf-8"))
    meta = {int(k.split("/")[0]): v for k, v in scenes.items()}
    table_path = OUT / "scenes.csv"
    rows = {}
    if table_path.exists():
        with open(table_path, encoding="utf-8") as f:
            rows = {int(r["scene"]): r for r in csv.DictReader(f)}
    cache = {}
    todo = [("main", s, p) for s, p in S.MAIN.items()] + [("extra", s, p) for s, p in S.EXTRA.items()]
    for group, sid, part in todo:
        if (OUT / f"{sid}_radar.tif").exists() and (OUT / f"{sid}_truth.tif").exists() and sid in rows:
            log(f"scene {sid}: done before")
            continue
        t0 = time.time()
        m = meta[sid]
        p = m["properties"]
        lat, lon = S.latlon(m)
        row = {"set": group, "scene": sid, "lat": round(lat, 2), "lon": round(lon, 2), "landcover": p["landcover"],
               "date_s1": p["date_s1"], "products": " ".join(p["s1_srcids"])}
        found = (ee.ImageCollection("COPERNICUS/S1_GRD").filter(ee.Filter.inList("system:index", p["s1_srcids"]))
                 .aggregate_array("system:index").getInfo())
        row["products_in_gee"] = len(found)
        if not found:
            rows[sid] = {**row, "status": "left out: products not in Earth Engine"}
            log(f"scene {sid}: products not in Earth Engine")
            continue
        files = {k: S.fetch_member(part, f"{sid}/sentinel12_s1_{sid}_{k}.tif", TMP / f"{sid}_{k}.tif", cache)
                 for k in ("msk", "valid")}
        with rasterio.open(files["msk"]) as src:
            crs, t, w0, h0 = src.crs, src.transform, src.width, src.height
        w, h = round(w0 * t.a / SCALE), round(h0 * -t.e / SCALE)
        t10 = Affine(SCALE, 0, t.c, 0, -SCALE, t.f)
        radar = gee_radar(found, crs.to_string(), t.c, t.f, w, h)
        truth, water_native, res = resample_mask(files["msk"], files["valid"], crs, t10, w, h)
        vv = np.where(radar[0] == -32768, np.nan, radar[0] / 100).astype(np.float32)
        best, table = register(truth, vv)
        dy, dx = best if best else (0, 0)
        if (dy, dx) != (0, 0):
            truth = shifted(truth, dy, dx)
        prof = {"driver": "GTiff", "crs": crs, "transform": t10, "width": w, "height": h, "tiled": True,
                "blockxsize": 512, "blockysize": 512, "compress": "deflate"}
        with rasterio.open(OUT / f"{sid}_radar.tif", "w", **prof, count=2, dtype="int16", nodata=-32768,
                           predictor=2) as dst:
            dst.write(radar)
        with rasterio.open(OUT / f"{sid}_truth.tif", "w", **prof, count=1, dtype="uint8", nodata=255) as dst:
            dst.write(truth, 1)
        scored = truth != 255
        rows[sid] = {**row, "status": "ok", "mask_m": res, "width": w, "height": h, "shift_dy": dy, "shift_dx": dx,
                     "contrast_db_best": round(table.get((dy, dx), math.nan), 3),
                     "contrast_db_unshifted": round(table.get((0, 0), math.nan), 3),
                     "water_share": round(float((truth == 1).sum() / max(scored.sum(), 1)), 4),
                     "water_share_native": round(water_native, 4),
                     "scored_share": round(float(scored.mean()), 4),
                     "radar_nodata_share": round(float((radar[0] == -32768).mean()), 4),
                     "seconds": round(time.time() - t0)}
        for f in files.values():
            f.unlink()
        log(f"scene {sid} ({group}): {len(found)} products, mask {res:g} m, shift ({dy}, {dx}), water "
            f"{rows[sid]['water_share']:.3f} (native {water_native:.3f}), radar no-data "
            f"{rows[sid]['radar_nodata_share']:.3f}, {time.time() - t0:.0f} s")
        keys = list(dict.fromkeys(k for r in rows.values() for k in r))
        with open(table_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows[k] for k in sorted(rows))
    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows[k] for k in sorted(rows))
    log("done:", sum(r.get("status") == "ok" for r in rows.values()), "scenes ready")


if __name__ == "__main__":
    main()
