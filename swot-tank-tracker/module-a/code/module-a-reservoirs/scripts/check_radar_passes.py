r"""Which radar pass made each chip's radar bands? (2 Nov 2026) The chips take the mosaic of every Sentinel-1 image on
the radar date (export_chips.s1_near), and on many dates the satellite passed twice: in the morning going south
(descending) and in the evening going north (ascending), looking from opposite sides. In a mosaic the image last in
the collection is on top, and the others fill the rest. For each chip date this replays the same mosaic with the
relative orbit of each image as its value and counts the share of each orbit over the reservoir's JRC water and over
the whole chip. Writes out/radar_passes.csv and prints a summary. Chips are not changed. Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\check_radar_passes.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import ee
import rasterio

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_sources as A  # noqa: E402  (SETS)
import export_chips as X  # noqa: E402  (PROJECT)


def shares(roi, days):
    """date -> orbit shares over the JRC water and over the whole chip, and the (orbit, pass, satellite) of each image
    on that date, replaying export_chips.s1_near's mosaic."""
    tank = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(0)
    base = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))

    def one(day):
        d = ee.Date(day)
        col = base.filterDate(d, d.advance(1, "day"))
        src = col.map(lambda img: ee.Image.constant(img.get("relativeOrbitNumber_start")).toInt16()
                      .updateMask(img.select("VV").mask()).rename("orbit")).mosaic()
        water = src.updateMask(tank).reduceRegion(ee.Reducer.frequencyHistogram(), roi, 20, maxPixels=1e9)
        chip = src.reduceRegion(ee.Reducer.frequencyHistogram(), roi, 40, maxPixels=1e9)
        images = col.reduceColumns(ee.Reducer.toList(3), ["relativeOrbitNumber_start", "orbitProperties_pass",
                                                          "platform_number"]).get("list")
        return ee.Feature(None, {"date": day, "water": water.get("orbit"), "chip": chip.get("orbit"), "images": images})

    def norm(h):
        return {str(int(float(k))): v / sum(h.values()) for k, v in h.items()} if h else {}

    out = {}
    for f in ee.FeatureCollection(ee.List(sorted(days)).map(one)).getInfo()["features"]:
        p = f["properties"]
        out[p["date"]] = {"water": norm(p.get("water")), "chip": norm(p.get("chip")),
                          "images": [(str(int(o)), ps, f"Sentinel-1{pl}") for o, ps, pl in p.get("images") or []]}
    return out


def fmt(share):
    return ";".join(f"{k}:{round(100 * v)}%" for k, v in sorted(share.items(), key=lambda kv: -kv[1]))


def main():
    ee.Initialize(project=X.PROJECT)
    rows = []
    for name, (folder, pattern, chips) in A.SETS.items():
        seen = set()
        for man in sorted(folder.glob(pattern)):
            groups = {}  # one Earth Engine query per chip footprint
            for r in csv.DictReader(open(man, encoding="utf-8")):
                if r["file"] in seen:  # the 7-tank manifests share some chips
                    continue
                seen.add(r["file"])
                with rasterio.open(chips / r["file"]) as src:
                    groups.setdefault((tuple(src.bounds), src.crs.to_string()), []).append(r)
            for (bounds, crs), group in groups.items():
                got = shares(ee.Geometry.Rectangle(list(bounds), crs, False), {r["s1_date"] for r in group})
                for r in group:
                    g = got[r["s1_date"]]
                    passes = {o: ps for o, ps, _ in g["images"]}
                    top = max(g["water"], key=g["water"].get) if g["water"] else ""
                    rows.append({"set": name, "tank": r["tank"], "file": r["file"], "kind": r.get("kind", ""),
                                 "s1_date": r["s1_date"], "satellites": "+".join(sorted({s for *_, s in g["images"]})),
                                 "passes_on_date": len(passes),
                                 "orbits_on_date": "+".join(f"{o} {passes[o].lower()}" for o in sorted(passes, key=int)),
                                 "water_orbits": fmt(g["water"]), "chip_orbits": fmt(g["chip"]), "main_orbit": top,
                                 "main_pass": passes.get(top, ""),
                                 "water_mixed": bool(g["water"]) and max(g["water"].values()) < 0.995})
            print(f"{name}: {man.name} done", flush=True)
    with open(ROOT / "out/radar_passes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name in A.SETS:
        mine = [r for r in rows if r["set"] == name]
        mixed = [r for r in mine if r["water_mixed"]]
        by_tank = {}
        for r in mine:
            by_tank.setdefault(r["tank"], set()).add(r["main_pass"])
        print(f"\n{name}: {len(mine)} chips")
        print("  passes on the radar date:", dict(sorted(Counter(r["passes_on_date"] for r in mine).items())))
        print(f"  water from more than one pass: {len(mixed)} chips at {len({r['tank'] for r in mixed})} reservoirs")
        print("  main pass over the water:", dict(Counter(r["main_pass"] or "?" for r in mine).most_common()))
        print("  reservoirs whose chips mix ascending and descending over time:",
              sum(len(v - {''}) > 1 for v in by_tank.values()), "of", len(by_tank))


if __name__ == "__main__":
    main()
