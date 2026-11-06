r"""Counts only, saves no chips (2 Nov 2026). What would other satellites in Earth Engine add for the 7 reservoirs left
out for too few clear days? Run count_more_days.py first: this reads its list of Sentinel-2 photos.

For the first reservoir it prints the dates per year of Sentinel-1 and Sentinel-2 by satellite, Landsat 8 and 9, and
ALOS-2 PALSAR-2 ScanSAR (L-band radar). For each reservoir it prints, for our years (Jul 2023 to Dec 2025) and for all
the years of the department's sheet (Oct 2020 to Oct 2026):
- Sentinel-2 photos at least 98% clear over the water, and the Landsat ones on other dates (QA_PIXEL: no cloud,
  dilated cloud, cirrus or shadow);
- clear days usable by reservoir_stats.py: radar within 2 days and storage on record within 2 days, each radar image
  once, from Sentinel-2 alone and with Landsat added;
- radar passes with storage on record, which a radar-only check could use whatever the weather.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\count_other_satellites.py
"""
import csv
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import ee

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_more_days as C  # noqa: E402  (NAMES, PERIODS)
import export_chips as X  # noqa: E402  (region, PROJECT)
import export_reservoirs as E  # noqa: E402  (reservoirs, size_chip, SHEET)

START, END = C.PERIODS["before"][0], C.PERIODS["after"][1]
OURS = C.PERIODS["ours"]


def day(ms):
    return date(1970, 1, 1) + timedelta(milliseconds=ms)


def s1_list(g):
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(X.region(g)).filterDate(START, END)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    t, p = s1.aggregate_array("system:time_start").getInfo(), s1.aggregate_array("platform_number").getInfo()
    return sorted({(day(a), b) for a, b in zip(t, p)})


def s2_list(g):
    col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(X.region(g)).filterDate(START, END)
    t, p = col.aggregate_array("system:time_start").getInfo(), col.aggregate_array("SPACECRAFT_NAME").getInfo()
    return sorted({(day(a), b) for a, b in zip(t, p)})


def landsat(g):
    """Landsat 8 and 9 dates that fully cover the chip, with the clear share of the JRC water (best scene per date)."""
    roi = X.region(g)
    tank = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(0)
    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterBounds(roi).filterDate(START, END))

    def stats(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(0b11110).eq(0).updateMask(tank).rename("clear")
        cover = ee.Image.constant(1).updateMask(img.select("SR_B2").mask()).unmask(0).rename("cover")
        s = clear.addBands(cover).reduceRegion(ee.Reducer.mean(), roi, 30, maxPixels=1e9)
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "clear": s.get("clear"),
                                 "cover": s.get("cover")})

    best = {}
    for f in ee.FeatureCollection(col.map(stats)).filter(ee.Filter.gte("cover", 0.999)).getInfo()["features"]:
        p = f["properties"]
        if p.get("clear") is not None and p["clear"] > best.get(p["date"], {"clear": -1})["clear"]:
            best[p["date"]] = p
    return sorted(best.values(), key=lambda p: p["date"])


def palsar2(g):
    col = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR").filterBounds(X.region(g)).filterDate(START, END)
    return sorted({day(a) for a in col.aggregate_array("system:time_start").getInfo()})


def main():
    ee.Initialize(project=X.PROJECT)
    storage = {}
    for r in csv.DictReader(open(E.SHEET, encoding="utf-8")):
        if r["reservoir"] in C.NAMES and r["storage_acft"] not in ("", "nan"):
            storage.setdefault(r["reservoir"], set()).add(date.fromisoformat(r["date"]))
    s2 = list(csv.DictReader(open(ROOT / "out/more_days_photos.csv", encoding="utf-8")))

    def on_record(name, d):
        return any(d + timedelta(days=k) in storage[name] for k in range(-2, 3))

    def usable(name, clear_days, radar, a, b):
        out = set()
        for d in clear_days:
            if a <= str(d) < b and on_record(name, d):
                near = min(radar, key=lambda r: abs((r - d).days))
                if abs((near - d).days) <= 2:
                    out.add(near)
        return len(out)

    first = True
    print("reservoir | S2 clear: ours / all | Landsat clear on other dates: ours / all | usable, S2: ours / all"
          " | usable, S2 + Landsat: ours / all | radar passes with storage: ours / all")
    for t in [t for t in E.reservoirs() if t["name"] in C.NAMES]:
        g, _, _ = E.size_chip(t)
        s1 = s1_list(g)
        radar = sorted({d for d, _ in s1})
        ls = landsat(g)
        if first:
            print("Sentinel-1 dates by year and satellite:", dict(sorted(Counter((d.year, p) for d, p in s1).items())))
            print("Sentinel-2 dates by year and satellite:",
                  dict(sorted(Counter((d.year, p[-2:]) for d, p in s2_list(g)).items())))
            print("Landsat 8/9 dates covering the chip, by year:", dict(sorted(Counter(p["date"][:4] for p in ls).items())))
            print("PALSAR-2 ScanSAR dates by year:", dict(sorted(Counter(d.year for d in palsar2(g)).items())))
            first = False
        s2c = {date.fromisoformat(r["date"]) for r in s2 if r["reservoir"] == t["name"] and float(r["clear"]) >= 0.98}
        lsc = {date.fromisoformat(p["date"]) for p in ls if p["clear"] >= 0.98} - s2c
        n_ours = sum(OURS[0] <= str(d) < OURS[1] for d in lsc)
        print(t["name"], "|", sum(OURS[0] <= str(d) < OURS[1] for d in s2c), "/", len(s2c), "|", n_ours, "/", len(lsc),
              "|", usable(t["name"], s2c, radar, *OURS), "/", usable(t["name"], s2c, radar, START, END),
              "|", usable(t["name"], s2c | lsc, radar, *OURS), "/", usable(t["name"], s2c | lsc, radar, START, END),
              "|", sum(OURS[0] <= str(d) < OURS[1] and on_record(t["name"], d) for d in radar), "/",
              sum(on_record(t["name"], d) for d in radar), flush=True)


if __name__ == "__main__":
    main()
