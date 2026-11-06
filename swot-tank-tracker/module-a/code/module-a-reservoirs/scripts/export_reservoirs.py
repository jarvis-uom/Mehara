r"""Satellite chips for every reservoir in the department's daily storage sheet (117), for the label-free check of
Module A at scale: does the water area follow official storage, on clear and on cloudy days? (2 Nov 2026)

For each reservoir (position and capacity from temp/model-check/storage_daily.csv) it:
1. finds its water body in the JRC map (water seen at least once, 1984 to 2021) at or nearest to the sheet's position,
   and sizes a 10 m chip around it. The search window grows when the water body runs off its edge;
2. lists the Sentinel-2 L1C dates from Jul 2023 to Dec 2025 that fully cover the chip, with the share of the
   reservoir's JRC pixels that Cloud Score+ (cs_cdf) calls clear: clear days have at least 98%, cloudy days less than
   70%. Each needs Sentinel-1 within 2 days;
3. saves up to 20 clear and 20 cloudy chips, spread over the period, as 14-band GeoTIFFs in the same form as
   temp/module-a-tanks/scripts/export_chips.py: VV and VH (dB x 100, -32768 = no data), B2 B3 B4 B5 B6 B7 B8 B8A B11
   B12 (0 = no data), JRC occurrence (%), cs_cdf (x 100). Big chips are fetched in band groups.
It skips chips that already exist, so it can be stopped and started again. Each reservoir's chips are listed in
manifests/<reservoir>.csv, with the satellites and images that made each chip (sources()), and each part writes one
line per reservoir to logs/progress_part<k>.csv.
Run from the project folder; parts can run at the same time, each in its own window or detached:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\export_reservoirs.py --part 1/4
"""
import argparse
import csv
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import ee
import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
import export_chips as X  # noqa: E402  (grid, pixels, candidates, s1_near)

sys.stdout.reconfigure(encoding="utf-8")
SHEET = REPO / "temp/model-check/storage_daily.csv"
CHIPS, MANIFESTS, LOGS = ROOT / "chips", ROOT / "manifests", ROOT / "logs"
WANTED = 20  # chips per kind (clear, cloudy)
CLEAR_MIN, CLOUDY_MAX, S1_DAYS = 0.98, 0.7, 2
MAX_REQUEST = 40e6  # bytes per computePixels request (Earth Engine allows about 48 MB)
BANDS = ["VV", "VH"] + X.S2_USE + ["JRC_occurrence", "cloud_score_plus_cs_cdf"]


def reservoirs():
    """name -> kind, lat, lon, capacity (million m3) from the department's daily sheet."""
    out = {}
    for r in csv.DictReader(open(SHEET, encoding="utf-8")):
        if r["reservoir"] and r["reservoir"] not in out:
            out[r["reservoir"]] = {"name": r["reservoir"], "kind": r["kind"], "lat": float(r["lat"]),
                                   "lon": float(r["lon"]), "cap_mcm": float(r["capacity_acft"]) * 1233.48 / 1e6}
    return [out[k] for k in sorted(out)]


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def retry(fn, *a, tries=4, **kw):
    for k in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:  # Earth Engine and network hiccups
            if k == tries - 1 or "quota" in str(e).lower():
                raise
            time.sleep(10 * (k + 1))


def size_chip(t):
    """Chip around the JRC water body at (or nearest to) the sheet's position. The first search window is about three
    times the side of a tank of this capacity; it doubles while the water body touches its edge."""
    area_ha = 50 * t["cap_mcm"] ** 0.9  # rough area of a Sri Lankan tank of this capacity
    px = int(min(2048, max(256, np.ceil(3 * np.sqrt(area_ha * 1e4) / 10 / 32) * 32)))
    occ_img = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).toUint8()
    while True:
        g = X.grid(t["lon"], t["lat"], px, px)
        occ = retry(X.pixels, occ_img, g)[0][0]
        lab, n = ndimage.label(occ > 0)
        if n == 0:
            return None, px, 0.0
        c = px // 2
        k = lab[c, c]
        if k == 0:  # the sheet's point is on land (often the dam): take the nearest water body
            _, (ii, jj) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
            k = lab[ii[c, c], jj[c, c]]
        rows, cols = np.where(lab == k)
        touches = rows.min() == 0 or cols.min() == 0 or rows.max() == px - 1 or cols.max() == px - 1
        if touches and px < 4096:
            px *= 2
            continue
        a = g["affineTransform"]
        cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
        h = max(256, int(np.ceil((rows.max() - rows.min() + 60) / 32)) * 32)
        w = max(256, int(np.ceil((cols.max() - cols.min() + 60) / 32)) * 32)
        lons, lats = X.warp_transform(X.CRS, "EPSG:4326", [a["translateX"] + (cx + 0.5) * X.SCALE],
                                      [a["translateY"] - (cy + 0.5) * X.SCALE])
        return X.grid(lons[0], lats[0], w, h), px, len(rows) * 0.01


def candidates_mosaic(g):
    """Like export_chips.candidates, but scored on each date's mosaic of all tiles: for a reservoir that crosses a
    Sentinel-2 tile edge, no single tile covers the chip (Senanayake Samudra)."""
    roi = X.region(g)
    s2 = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(roi).filterDate(X.START, X.END)
          .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs_cdf"]))
    tank = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(0)
    days = s2.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY-MM-dd")).distinct()

    def stats(day):
        d = ee.Date.parse("YYYY-MM-dd", day)
        img = s2.filterDate(d, d.advance(1, "day")).mosaic()
        clear = img.select("cs_cdf").gte(0.6).updateMask(tank).rename("clear")
        cover = ee.Image.constant(1).updateMask(img.select("B2").mask()).unmask(0).rename("cover")
        s = clear.addBands(cover).reduceRegion(ee.Reducer.mean(), roi, 60, maxPixels=1e9)
        return ee.Feature(None, {"date": day, "clear": s.get("clear"), "cover": s.get("cover")})

    fc = ee.FeatureCollection(days.map(stats)).filter(ee.Filter.gte("cover", 0.999))
    return sorted((f["properties"] for f in fc.getInfo()["features"]), key=lambda p: p["date"])


def fetch(img, g):
    """computePixels in band groups small enough for one request each; returns (bands, h, w) and the profile."""
    per_band = g["dimensions"]["width"] * g["dimensions"]["height"] * 3  # int16 plus Earth Engine's 1-byte mask
    step = max(1, int(MAX_REQUEST // per_band))
    parts, profile = [], None
    for i in range(0, len(BANDS), step):
        arr, profile = retry(X.pixels, img.select(list(range(i, min(i + step, len(BANDS))))), g)
        parts.append(arr)
    return np.concatenate(parts), profile


def save_chip(g, day, s1, path):
    d = date.fromisoformat(day)
    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(X.region(g))
           .filterDate(str(d), str(d + timedelta(days=1)))
           .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs_cdf"]))
    img = (s1.multiply(100).round().toInt16().unmask(-32768)
           .addBands(col.select(X.S2_USE).mosaic().toInt16().unmask(0))
           .addBands(ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).toInt16())
           .addBands(col.select("cs_cdf").mosaic().multiply(100).round().toInt16().unmask(0)))
    arr, profile = fetch(img, g)
    profile.update(driver="GTiff", compress="deflate", count=arr.shape[0], dtype="int16")
    tmp = path.with_suffix(".part")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr.astype(np.int16))
        dst.descriptions = tuple(BANDS)
    tmp.replace(path)  # a stopped run never leaves a half-written chip behind


def spread(days, n):
    """Up to 2n dates spread evenly over the period (spares for dates without radar)."""
    return [days[round(i)] for i in np.linspace(0, len(days) - 1, min(2 * n, len(days)))] if days else []


def sources(roi, rows):
    """Adds to each manifest row which satellites and images made the chip (a chip is the mosaic of all the images on
    its dates): Sentinel-2 satellite and image ids; Sentinel-1 satellite, pass, relative orbit and image ids."""
    days = [r[k] for r in rows for k in ("s2_date", "s1_date") if r[k]]
    a, b = min(days), str(date.fromisoformat(max(days)) + timedelta(days=1))
    s2 = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(roi).filterDate(a, b)
          .reduceColumns(ee.Reducer.toList(3), ["system:index", "system:time_start", "SPACECRAFT_NAME"]).get("list"))
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi).filterDate(a, b)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .reduceColumns(ee.Reducer.toList(5), ["system:index", "system:time_start", "platform_number",
                                                 "orbitProperties_pass", "relativeOrbitNumber_start"]).get("list"))
    s2, s1 = retry(ee.List([s2, s1]).getInfo)
    on = {}
    for kind, items in (("s2", s2), ("s1", s1)):
        for it in items:
            on.setdefault((kind, str(date(1970, 1, 1) + timedelta(milliseconds=it[1]))), []).append(it)

    def join(values):
        return "+".join(sorted({str(v) for v in values}))

    for r in rows:
        a2, a1 = on.get(("s2", r["s2_date"]), []), on.get(("s1", r["s1_date"]), [])
        r.update({"s2_satellite": join(it[2] for it in a2), "s2_images": ";".join(it[0] for it in a2),
                  "s1_satellite": join(f"Sentinel-1{it[2]}" for it in a1), "s1_pass": join(it[3] for it in a1),
                  "s1_orbit": join(it[4] for it in a1), "s1_images": ";".join(it[0] for it in a1)})
    return rows


def one(t):
    started = time.time()
    g, search_px, area_ha = size_chip(t)
    if g is None:
        return {"status": "no JRC water near the sheet's position", "search_px": search_px}
    dims = g["dimensions"]
    days = retry(X.candidates, g, "tank", 0.0, "cs_cdf", 1.01) or retry(candidates_mosaic, g)
    by_kind = {"clear": [p for p in days if p["clear"] >= CLEAR_MIN], "cloudy": [p for p in days if p["clear"] < CLOUDY_MAX]}
    rows = []
    for kind, pool in by_kind.items():
        saved = 0
        for p in spread(pool, WANTED):
            if saved >= WANTED:
                break
            s1, s1_day = retry(X.s1_near, g, p["date"], S1_DAYS)
            if s1 is None:
                continue
            path = CHIPS / f"{safe(t['name'])}_{p['date']}.tif"
            if not path.exists():
                save_chip(g, p["date"], s1, path)
            rows.append({"tank": t["name"], "kind": kind, "file": path.name, "s2_date": p["date"], "s1_date": s1_day,
                         "days_apart": abs((date.fromisoformat(s1_day) - date.fromisoformat(p["date"])).days),
                         "clear": round(p["clear"], 3), "width": dims["width"], "height": dims["height"]})
            saved += 1
    if rows:
        sources(X.region(g), rows)
        with open(MANIFESTS / f"{safe(t['name'])}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return {"status": "done", "search_px": search_px, "chip": f"{dims['width']}x{dims['height']}",
            "jrc_area_ha": round(area_ha), "clear_dates": len(by_kind["clear"]), "cloudy_dates": len(by_kind["cloudy"]),
            "clear_saved": sum(r["kind"] == "clear" for r in rows), "cloudy_saved": sum(r["kind"] == "cloudy" for r in rows),
            "minutes": round((time.time() - started) / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="1/1", help="k/n: do every n-th reservoir, starting with the k-th")
    ap.add_argument("--only", default="", help="comma list of reservoir names (for a test)")
    args = ap.parse_args()
    k, n = (int(v) for v in args.part.split("/"))
    for d in (CHIPS, MANIFESTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=X.PROJECT)
    todo = [t for i, t in enumerate(reservoirs()) if i % n == k - 1]
    if args.only:
        todo = [t for t in reservoirs() if t["name"] in args.only.split(",")]
    log = LOGS / f"progress_part{k}.csv"
    done = {r["reservoir"] for r in csv.DictReader(open(log, encoding="utf-8"))
            if r["status"] == "done"} if log.exists() and not args.only else set()
    print(f"part {k}/{n}: {len(todo)} reservoirs, {len(done)} already done", flush=True)
    for t in todo:
        if t["name"] in done:
            continue
        try:
            res = one(t)
        except Exception as e:
            res = {"status": f"error: {str(e)[:200]}"}
        line = {"reservoir": t["name"], "kind": t["kind"], "cap_mcm": round(t["cap_mcm"], 2), "status": res["status"],
                **{c: res.get(c, "") for c in ("search_px", "chip", "jrc_area_ha", "clear_dates", "cloudy_dates",
                                                "clear_saved", "cloudy_saved", "minutes")},
                "finished": time.strftime("%Y-%m-%d %H:%M")}
        new = not log.exists()
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(line))
            if new:
                w.writeheader()
            w.writerow(line)
        print(" ".join(f"{v}" for v in line.values()), flush=True)
    print(f"part {k}/{n} finished", flush=True)


if __name__ == "__main__":
    main()
