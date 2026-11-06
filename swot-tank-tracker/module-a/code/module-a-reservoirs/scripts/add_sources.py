r"""Adds to every chip manifest which satellites and images made each chip (2 Nov 2026), so a problem with one satellite
can be traced later: the Sentinel-2 satellite (2A, 2B or 2C) and image ids, and the Sentinel-1 satellite (1A to 1D),
pass, relative orbit and image ids (export_reservoirs.sources). The chips are not changed. It covers manifests/ here
and the 7-tank chips in temp/module-a-tanks/chips/, and prints a summary with any chip whose images mix satellites,
passes or orbits, or whose images were not found. Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\add_sources.py
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
import export_chips as X  # noqa: E402  (PROJECT)
import export_reservoirs as E  # noqa: E402  (sources)

SETS = {"117 reservoirs": (ROOT / "manifests", "*.csv", ROOT / "chips"),
        "7 tanks": (REPO / "temp/module-a-tanks/chips", "manifest*.csv", REPO / "temp/module-a-tanks/chips")}
COLUMNS = ("s2_satellite", "s1_satellite", "s1_pass", "s1_orbit")


def main():
    ee.Initialize(project=X.PROJECT)
    for name, (folder, pattern, chips) in SETS.items():
        counts, odd, seen = {c: Counter() for c in COLUMNS}, [], set()
        for man in sorted(folder.glob(pattern)):
            rows = list(csv.DictReader(open(man, encoding="utf-8")))
            groups = {}  # a manifest can hold several tanks: one Earth Engine query per chip footprint
            for r in rows:
                with rasterio.open(chips / r["file"]) as src:
                    groups.setdefault((tuple(src.bounds), src.crs.to_string()), []).append(r)
            for (bounds, crs), group in groups.items():
                E.sources(ee.Geometry.Rectangle(list(bounds), crs, False), group)
            with open(man, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            for r in rows:
                if r["file"] in seen:  # the 7-tank manifests share some chips
                    continue
                seen.add(r["file"])
                for c in COLUMNS:
                    counts[c][r[c]] += 1
                if not r["s2_images"] or not r["s1_images"] or any("+" in r[c] for c in COLUMNS):
                    odd.append(f'{r["file"]}: ' + ", ".join(f"{c} {r[c] or 'not found'}" for c in COLUMNS))
            print(f"{name}: {man.name} done", flush=True)
        print(f"\n{name}: {len(seen)} chips")
        for c in COLUMNS:
            print(f"  {c}:", dict(counts[c].most_common()))
        print(f"  chips that mix satellites, passes or orbits, or lack an image: {len(odd)}")
        for line in odd:
            print("   ", line)


if __name__ == "__main__":
    main()
