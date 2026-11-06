"""Cut Sentinel-1 + Sentinel-2 chips of our tanks with Earth Engine, in the same form as Sen1Floods11.

For each tank it:
1. finds the tank's water area in the JRC Global Surface Water map (water seen at least once, 1984 to 2021)
   and sizes a 10 m chip around it (a multiple of 32 pixels, at least 256);
2. lists the Sentinel-2 L1C photos (COPERNICUS/S2_HARMONIZED, as in Sen1Floods11) whose chip is fully covered
   and at least 95% clear by Cloud Score+, and keeps those with a Sentinel-1 radar image within 3 days;
3. saves up to N dates per tank, spread over the period, as GeoTIFFs with 14 bands:
   VV and VH (dB x 100, -32768 = no data), B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 (0 = no data), JRC occurrence (%),
   Cloud Score+ cs (x 100; 100 = clear).

First look (1 Nov): every tank, at least 95% of the whole chip clear:
    temp\\feasibility-check\\.venv\\Scripts\\python.exe -I temp\\module-a-tanks\\scripts\\export_chips.py
More Mahakanadarawa dates: only the tank (JRC water pixels) must be clear:
    ... export_chips.py --tanks Mahakanadarawa --dates 25 --clear-over tank --min-clear 0.98 --manifest manifest_more.csv
Cloudy days (less than 70% of the tank clear, radar within 2 days), for the cloudy-day check:
    ... export_chips.py --tanks Mahakanadarawa --dates 20 --clear-over tank --score cs_cdf --min-clear 0 --max-clear 0.7
        --s1-days 2 --manifest manifest_cloudy_mahakanadarawa.csv
"""
import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import transform as warp_transform
from scipy import ndimage

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "chips"
PROJECT = "project-c14134e4-dd6a-4968-b12"
CRS, SCALE = "EPSG:32644", 10
S2_USE = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
START, END = "2023-07-01", "2026-01-01"  # the SWOT years with department storage
VILLAGES = {"4580006282": "V3", "4580010542": "V4", "4580011422": "V6", "4580007392": "V7"}
DATES_PER_TANK = {"Mahakanadarawa": 12, "major": 12, "village": 6, "extra": 12}
MAJOR = ("Nachchaduwa", "Hurulu wewa", "Nuwara Wewa")  # the other big tanks (positions from A1_tanks.csv)


def tanks():
    out = [{"name": "Mahakanadarawa", "lon": 80.5521, "lat": 8.3829, "search_px": 1024, "kind": "Mahakanadarawa"}]
    for row in csv.DictReader(open(REPO / "temp/feasibility-check/results/A1_tanks.csv", encoding="utf-8")):
        if row["name"] in MAJOR:
            out.append({"name": row["name"], "lon": float(row["lon"]), "lat": float(row["lat"]), "search_px": 1280,
                        "kind": "major"})
    g = json.loads((REPO / "temp/cascade-check/out/candidates.geojson").read_text(encoding="utf-8"))
    for f in g["features"]:
        p = f["properties"]
        if p["lake_id"] in VILLAGES:
            out.append({"name": f"{VILLAGES[p['lake_id']]} ({p['lake_id']})", "lon": p["lon"], "lat": p["lat"],
                        "search_px": 384, "kind": "village"})
    return out


def grid(lon, lat, w, h):
    xs, ys = warp_transform("EPSG:4326", CRS, [lon], [lat])
    x0 = round((xs[0] - w * SCALE / 2) / SCALE) * SCALE
    y1 = round((ys[0] + h * SCALE / 2) / SCALE) * SCALE
    return {"dimensions": {"width": w, "height": h},
            "affineTransform": {"scaleX": SCALE, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -SCALE,
                                "translateY": y1}, "crsCode": CRS}


def region(g):
    a, d = g["affineTransform"], g["dimensions"]
    x0, y1 = a["translateX"], a["translateY"]
    return ee.Geometry.Rectangle([x0, y1 - d["height"] * SCALE, x0 + d["width"] * SCALE, y1], CRS, False)


def pixels(image, g):
    with MemoryFile(ee.data.computePixels({"expression": image, "fileFormat": "GEO_TIFF", "grid": g})) as mem:
        with mem.open() as src:
            return src.read(), src.profile


def size_chip(t):
    """Chip around the JRC water body that contains (or is nearest to) the tank's point."""
    g = grid(t["lon"], t["lat"], t["search_px"], t["search_px"])
    occ = pixels(ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).toUint8(), g)[0][0]
    lab, n = ndimage.label(occ > 0)
    c = t["search_px"] // 2
    if n == 0:
        w = h = 256
    else:
        k = lab[c, c]
        if k == 0:  # the point is on land: take the nearest water body
            dist, (ii, jj) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
            k = lab[ii[c, c], jj[c, c]]
        rows, cols = np.where(lab == k)
        cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
        h = max(256, int(np.ceil((rows.max() - rows.min() + 60) / 32)) * 32)
        w = max(256, int(np.ceil((cols.max() - cols.min() + 60) / 32)) * 32)
        a = g["affineTransform"]
        xs, ys = a["translateX"] + (cx + 0.5) * SCALE, a["translateY"] - (cy + 0.5) * SCALE
        lons, lats = warp_transform(CRS, "EPSG:4326", [xs], [ys])
        t["lon"], t["lat"] = lons[0], lats[0]
    return grid(t["lon"], t["lat"], w, h)


def candidates(g, clear_over="chip", min_clear=0.95, score="cs", max_clear=1.01):
    roi = region(g)
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(roi).filterDate(START, END).linkCollection(cs, [score])
    tank = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(0)

    def stats(img):
        clear = img.select(score).gte(0.6).rename("clear")
        if clear_over == "tank":  # only pixels where JRC has ever seen water
            clear = clear.updateMask(tank)
        cover = ee.Image.constant(1).updateMask(img.select("B2").mask()).unmask(0).rename("cover")
        s = clear.addBands(cover).reduceRegion(ee.Reducer.mean(), roi, 20, maxPixels=1e9)
        return ee.Feature(None, {"id": img.get("system:index"), "date": img.date().format("YYYY-MM-dd"),
                                 "clear": s.get("clear"), "cover": s.get("cover")})

    rows = (ee.FeatureCollection(s2.map(stats)).filter(ee.Filter.gte("cover", 0.999))
            .filter(ee.Filter.gte("clear", min_clear)).filter(ee.Filter.lt("clear", max_clear)))
    return sorted({f["properties"]["date"]: f["properties"] for f in rows.getInfo()["features"]}.values(),
                  key=lambda p: p["date"])


def s1_near(g, day, days=3):
    d = date.fromisoformat(day)
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region(g))
          .filterDate(str(d - timedelta(days=days)), str(d + timedelta(days=days + 1)))
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    info = s1.aggregate_array("system:time_start").getInfo()
    if not info:
        return None, None
    best = min(info, key=lambda ms: abs(ms / 86400000 - (d - date(1970, 1, 1)).days - 0.5))
    day1 = date(1970, 1, 1) + timedelta(milliseconds=best)
    same_day = s1.filterDate(str(day1), str(day1 + timedelta(days=1)))
    return same_day.select(["VV", "VH"]).mosaic(), str(day1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tanks", default="all", help="all, or one name such as Mahakanadarawa")
    ap.add_argument("--dates", type=int, default=0, help="dates per tank (0 = 12 for Mahakanadarawa, 6 per village)")
    ap.add_argument("--clear-over", choices=["chip", "tank"], default="chip")
    ap.add_argument("--min-clear", type=float, default=0.95)
    ap.add_argument("--max-clear", type=float, default=1.01, help="keep only dates less clear than this (cloudy days)")
    ap.add_argument("--s1-days", type=int, default=3, help="largest gap in days between the radar and the photo")
    ap.add_argument("--score", choices=["cs", "cs_cdf"], default="cs",
                    help="Cloud Score+ band; cs_cdf marks water as cloudy less often")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--extra", action="append", default=[],
                    help="another tank as Name:lon:lat:search_px (repeatable), e.g. Mahawewa:81.011:6.459:384")
    args = ap.parse_args()
    ee.Initialize(project=PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    extras = [{"name": n, "lon": float(lo), "lat": float(la), "search_px": int(px), "kind": "extra"}
              for n, lo, la, px in (e.split(":") for e in args.extra)]
    for t in tanks() + extras:
        if args.tanks != "all" and not t["name"].startswith(args.tanks):
            continue
        g = size_chip(t)
        dims = g["dimensions"]
        days = candidates(g, args.clear_over, args.min_clear, args.score, args.max_clear)
        print(f"{t['name']}: chip {dims['width']} x {dims['height']} px, {len(days)} dates that pass the cloud test",
              flush=True)
        wanted = args.dates or DATES_PER_TANK[t["kind"]]
        picks = [days[round(i)] for i in np.linspace(0, len(days) - 1, min(wanted * 2, len(days)))] if days else []
        saved = 0
        for p in picks:
            if saved >= wanted:
                break
            s1, s1_day = s1_near(g, p["date"], args.s1_days)
            if s1 is None:
                continue
            d = date.fromisoformat(p["date"])
            name = f"{t['name'].split(' ')[0]}_{p['date']}.tif"
            if (OUT / name).exists():
                with rasterio.open(OUT / name) as src:
                    have = src.count == 14
                if have:  # already saved with all 14 bands: reuse it
                    manifest.append({"tank": t["name"], "file": name, "s2_date": p["date"], "s1_date": s1_day,
                                     "days_apart": abs((date.fromisoformat(s1_day) - d).days),
                                     "clear": round(p["clear"], 3), "width": dims["width"], "height": dims["height"],
                                     "lon": round(t["lon"], 5), "lat": round(t["lat"], 5)})
                    saved += 1
                    print(f"  have {name}", flush=True)
                    continue
            day = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(region(g))
                   .filterDate(str(d), str(d + timedelta(days=1)))
                   .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), [args.score]))
            s2, cs = day.select(S2_USE).mosaic(), day.select(args.score).mosaic()
            img = (s1.multiply(100).round().toInt16().unmask(-32768)
                   .addBands(s2.toInt16().unmask(0))
                   .addBands(ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).toInt16())
                   .addBands(cs.multiply(100).round().toInt16().unmask(0)))
            arr, profile = pixels(img, g)
            profile.update(driver="GTiff", compress="deflate", count=arr.shape[0], dtype="int16")
            with rasterio.open(OUT / name, "w", **profile) as dst:
                dst.write(arr.astype(np.int16))
                dst.descriptions = tuple(["VV", "VH"] + S2_USE + ["JRC_occurrence", "cloud_score_plus_cs"])
            manifest.append({"tank": t["name"], "file": name, "s2_date": p["date"], "s1_date": s1_day,
                             "days_apart": abs((date.fromisoformat(s1_day) - d).days), "clear": round(p["clear"], 3),
                             "width": dims["width"], "height": dims["height"],
                             "lon": round(t["lon"], 5), "lat": round(t["lat"], 5)})
            saved += 1
            print(f"  saved {name} (radar {s1_day})", flush=True)
    with open(OUT / args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"{len(manifest)} chips saved in {OUT}")


if __name__ == "__main__":
    main()
