r"""S1S2-Water training patches for retraining run 4 (4 Nov 2026), under temp/module-a-retrain/PROTOCOL-4.md (fixed).

For each of the 30 training scenes (the training split without the Asian extra scenes 6, 12 and 52), it:
1. downloads the expert water mask and the valid mask from Zenodo (one range request each);
2. resamples them to 10 m on the scene's UTM grid: water if at least 0.8 of a cell, land if at most 0.2, left out
   (255) in between or where under half the cell is valid;
3. checks the scene: mask 10 m or finer, and Earth Engine radar on at least 70% of its valid area (from a 160 m
   look); scenes that fail are listed and left out;
4. picks 64 windows of 512 x 512 (at least 90% scored, seed [4, scene]): 32 with shoreline (5 to 95% water) and
   32 with dark land (at least 20% of the window is land with VV under -14 dB at 160 m); short kinds are filled
   with other valid windows;
5. fetches those windows' radar at 10 m from Earth Engine (the scene's own products; VV, VH as dB x 100, int16,
   -32768 = no data);
6. registers the mask on all 64 patches together: the whole-pixel shift (-3 to 3) with the largest VV contrast
   (mean outside the water minus inside), as PROTOCOL-S1S2;
7. writes train/<scene>.npz (radar (64, 2, 512, 512) int16, truth (64, 512, 512) uint8, windows, kinds) and
   train/scenes.csv.
Light: downloads only. It resumes: a scene whose npz exists is skipped.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\s1s2water-check\scripts\fetch_train_patches.py
"""
import csv
import json
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
import s1s2_zip as S  # noqa: E402

PROJECT = "project-c14134e4-dd6a-4968-b12"
OUT = HERE / "train"
TMP = HERE / "tmp" / "train-masks"
KEEP_OUT = {6, 12, 52}
P, N_EACH, SHIFT, COARSE = 512, 32, 3, 16  # patch, patches per kind, registration range, coarse factor (160 m)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def truth_10m(msk, valid, crs, t10, w, h):
    with rasterio.open(msk) as src:
        water = src.read(1) == 1
        st, scrs, res = src.transform, src.crs, abs(src.transform.a)
    with rasterio.open(valid) as src:
        ok = src.read(1) == 1
    a = water.astype(np.float32)
    a[~ok] = np.nan
    share = np.full((h, w), np.nan, np.float32)
    reproject(a, share, src_transform=st, src_crs=scrs, src_nodata=np.nan, dst_transform=t10, dst_crs=crs,
              dst_nodata=np.nan, resampling=Resampling.average)
    vshare = np.zeros((h, w), np.float32)
    reproject(ok.astype(np.float32), vshare, src_transform=st, src_crs=scrs, dst_transform=t10, dst_crs=crs,
              resampling=Resampling.average)
    t = np.full((h, w), 255, np.uint8)
    good = (vshare >= 0.5) & np.isfinite(share)
    t[good & (share >= 0.8)] = 1
    t[good & (share <= 0.2)] = 0
    return t, res


def image(ids):
    return (ee.ImageCollection([ee.Image(f"COPERNICUS/S1_GRD/{i}") for i in ids]).select(["VV", "VH"]).mosaic()
            .multiply(100).round().toInt16().unmask(-32768))


def pixels(img, crs, x0, y1, scale, w, h):
    g = {"dimensions": {"width": w, "height": h}, "crsCode": crs,
         "affineTransform": {"scaleX": scale, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -scale,
                             "translateY": y1}}
    for attempt in range(5):
        try:
            raw = ee.data.computePixels({"expression": img, "fileFormat": "GEO_TIFF", "grid": g})
            with MemoryFile(raw) as mem, mem.open() as s:
                return s.read()
        except Exception as e:  # Earth Engine's occasional time-outs and rate limits
            log(f"   pixels: {e}; retry {attempt + 1}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Earth Engine failed 5 times")


def choose(truth, vv_c, sid):
    """64 windows: 32 with shoreline, 32 with dark land, filled from other valid windows if short."""
    H, W = truth.shape
    rows = []
    for r0 in range(0, H - P + 1, P):
        for c0 in range(0, W - P + 1, P):
            t = truth[r0:r0 + P, c0:c0 + P]
            scored = t != 255
            if scored.mean() < 0.9:
                continue
            water = float((t == 1)[scored].mean())
            land_c = (t == 0).reshape(P // COARSE, COARSE, P // COARSE, COARSE).mean((1, 3)) >= 0.5
            v = vv_c[r0 // COARSE:(r0 + P) // COARSE, c0 // COARSE:(c0 + P) // COARSE]
            dark = float((land_c & (v != -32768) & (v < -1400)).mean())
            rows.append((r0, c0, water, dark))
    rng = np.random.default_rng([4, sid])
    shore = [w for w in rows if 0.05 <= w[2] <= 0.95]
    pick = [shore[i] for i in rng.permutation(len(shore))[:N_EACH]]
    kinds = ["shoreline"] * len(pick)
    rest = [w for w in rows if w not in pick]
    dark = [w for w in rest if w[3] >= 0.2]
    d = [dark[i] for i in rng.permutation(len(dark))[:N_EACH]]
    pick, kinds = pick + d, kinds + ["dark land"] * len(d)
    rest = [w for w in rows if w not in pick]
    need = 2 * N_EACH - len(pick)
    f = [rest[i] for i in rng.permutation(len(rest))[:need]]
    return pick + f, kinds + ["filled"] * len(f), len(rows)


def main():
    rev = "--reverse" in sys.argv  # a second copy working from the last scene back; it writes scenes_rev.csv
    ee.Initialize(project=PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((HERE / "data/scenes.json").read_text(encoding="utf-8"))
    todo = sorted((int(k.split("/")[0]), v) for k, v in scenes.items()
                  if v["properties"]["split"] == "train" and int(k.split("/")[0]) not in KEEP_OUT)
    if rev:
        todo = todo[::-1]
    log(f"{len(todo)} training scenes" + (" (from the last)" if rev else ""))
    table = OUT / ("scenes_rev.csv" if rev else "scenes.csv")
    rows = {}
    if table.exists():
        with open(table, encoding="utf-8") as f:
            rows = {int(r["scene"]): r for r in csv.DictReader(f)}
    cache = {}
    for sid, m in todo:
        if (OUT / f"{sid}.npz").exists() and (sid in rows or rev):
            continue
        if (OUT / f"{sid}.started").exists():  # the other copy has this scene: the two have met
            break
        (OUT / f"{sid}.started").touch()
        t0 = time.time()
        p = m["properties"]
        lat, lon = S.latlon(m)
        row = {"scene": sid, "lat": round(lat, 2), "lon": round(lon, 2), "landcover": p["landcover"],
               "date_s1": p["date_s1"]}
        found = (ee.ImageCollection("COPERNICUS/S1_GRD").filter(ee.Filter.inList("system:index", p["s1_srcids"]))
                 .aggregate_array("system:index").getInfo())
        if not found:
            rows[sid] = {**row, "status": "left out: products not in Earth Engine"}
            log(f"scene {sid}: products not in Earth Engine")
            continue
        files = {k: S.fetch_member(m["part"], f"{sid}/sentinel12_s1_{sid}_{k}.tif", TMP / f"{sid}_{k}.tif", cache)
                 for k in ("msk", "valid")}
        with rasterio.open(files["msk"]) as src:
            crs, t, w0, h0 = src.crs.to_string(), src.transform, src.width, src.height
        w, h = round(w0 * t.a / 10), round(h0 * -t.e / 10)
        t10 = Affine(10, 0, t.c, 0, -10, t.f)
        truth, res = truth_10m(files["msk"], files["valid"], crs, t10, w, h)
        img = image(found)
        vv_c = pixels(img, crs, t.c, t.f, 10 * COARSE, -(-w // COARSE), -(-h // COARSE))[0]
        tc = truth[:h - h % COARSE, :w - w % COARSE]
        scored_c = (tc != 255).reshape(tc.shape[0] // COARSE, COARSE, tc.shape[1] // COARSE, COARSE).mean((1, 3)) >= 0.5
        cover = float((vv_c[:scored_c.shape[0], :scored_c.shape[1]][scored_c] != -32768).mean()) if scored_c.any() else 0.0
        row.update({"mask_m": round(res, 2), "radar_cover": round(cover, 3), "products": len(found)})
        if res > 10.5 or cover < 0.7:
            rows[sid] = {**row, "status": f"left out: mask {res:.1f} m, radar cover {cover:.2f}"}
            log(f"scene {sid}: {rows[sid]['status']}")
            for f in files.values():
                f.unlink()
            continue
        wins, kinds, n_ok = choose(truth, vv_c, sid)
        radar = np.full((len(wins), 2, P, P), -32768, np.int16)

        def one(k):
            r0, c0 = wins[k][:2]
            radar[k] = pixels(img, crs, t.c + c0 * 10, t.f - r0 * 10, 10, P, P)

        with ThreadPoolExecutor(6) as pool:
            list(pool.map(one, range(len(wins))))
        pad = np.pad(truth, SHIFT, constant_values=255)
        vv = np.where(radar[:, 0] == -32768, np.nan, radar[:, 0] / 100.0)
        best, score = (0, 0), None
        for dy in range(-SHIFT, SHIFT + 1):
            for dx in range(-SHIFT, SHIFT + 1):
                ts = np.stack([pad[r0 + SHIFT + dy:r0 + SHIFT + dy + P, c0 + SHIFT + dx:c0 + SHIFT + dx + P]
                               for r0, c0, *_ in wins])
                good = np.isfinite(vv)
                wet, dry = good & (ts == 1), good & (ts == 0)
                if wet.sum() < 100 or dry.sum() < 100:
                    continue
                c = float(np.nanmean(vv[dry]) - np.nanmean(vv[wet]))
                if score is None or c > score:
                    best, score = (dy, dx), c
        dy, dx = best
        tr = np.stack([pad[r0 + SHIFT + dy:r0 + SHIFT + dy + P, c0 + SHIFT + dx:c0 + SHIFT + dx + P]
                       for r0, c0, *_ in wins]).astype(np.uint8)
        np.savez_compressed(OUT / f"{sid}.npz", radar=radar, truth=tr,
                            windows=np.array([w_[:2] for w_ in wins], np.int32), kinds=np.array(kinds))
        for f in files.values():
            f.unlink()
        scored = tr != 255
        rows[sid] = {**row, "status": "ok", "windows_ok": n_ok, "patches": len(wins),
                     "shoreline": kinds.count("shoreline"), "dark_land": kinds.count("dark land"),
                     "filled": kinds.count("filled"), "shift_dy": dy, "shift_dx": dx,
                     "contrast_db": None if score is None else round(score, 2),
                     "water_share": round(float((tr == 1)[scored].mean()), 3),
                     "radar_nodata_share": round(float((radar[:, 0] == -32768).mean()), 3),
                     "seconds": round(time.time() - t0)}
        log(f"scene {sid}: {len(wins)} patches ({kinds.count('shoreline')} shoreline, {kinds.count('dark land')} "
            f"dark land), shift ({dy}, {dx}), water {rows[sid]['water_share']}, {time.time() - t0:.0f} s")
        keys = list(dict.fromkeys(k for r in rows.values() for k in r))
        with open(table, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows[k] for k in sorted(rows))
    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    with open(table, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows[k] for k in sorted(rows))
    log("done:", sum(r.get("status") == "ok" for r in rows.values()), "of", len(todo), "scenes ready")


if __name__ == "__main__":
    main()
