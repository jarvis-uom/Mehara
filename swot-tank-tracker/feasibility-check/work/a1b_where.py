"""A1 follow-up: which OpenStreetMap tanks have no SWOT lake ID, and why?

Saves the OSM outlines (work/a1_osm_water.gpkg) and each lake's matched OSM area
(work/a1_lake_osm_area.csv) for A2 and A4, and prints the breakdown.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

import a1_lake_ids as a1

BASE = Path(__file__).resolve().parents[1]
WORK = BASE / "work"

zips = sorted((WORK / "swot").glob("SWOT_L2_HR_LakeSP_Prior_*.zip"))
files = [(z, z.name.split("_")[7], z.stem) for z in zips]
raw = pd.concat([gpd.read_file(f"zip://{z}").assign(pass_no=p) for z, p, _ in files], ignore_index=True)
raw = raw[(raw.p_lon >= a1.W) & (raw.p_lon <= a1.E) & (raw.p_lat >= a1.S) & (raw.p_lat <= a1.N)]
pts = {p: gpd.GeoSeries([Point(x, y) for x, y in zip(g.p_lon, g.p_lat)], crs="EPSG:4326").to_crs(a1.UTM)
       for p, g in raw.groupby("pass_no")}

pld = pd.read_csv(BASE / "results" / "A_pld_lakes.csv", dtype={"lake_id": str})
osm = a1.osm_water()
all_pts = gpd.GeoSeries([Point(x, y) for x, y in zip(pld.p_lon, pld.p_lat)], crs="EPSG:4326").to_crs(a1.UTM)
ids = []
for g in osm.geometry:
    d = all_pts.distance(g)
    ids.append(";".join(pld.lake_id[d <= a1.MATCH_M]))
osm["lake_ids"] = ids
osm["has_id"] = osm.lake_ids != ""
c = osm.geometry.centroid
for p, s in pts.items():
    osm[f"km_to_pass{p}_lake"] = [round(s.distance(pt).min() / 1000, 1) for pt in c]
c_ll = c.to_crs("EPSG:4326")
osm["lon"], osm["lat"] = c_ll.x.round(4), c_ll.y.round(4)
osm.to_file(WORK / "a1_osm_water.gpkg", driver="GPKG")

# Largest matched OSM polygon for each lake
rows = []
for _, o in osm.sort_values("area_ha").iterrows():
    for lid in o.lake_ids.split(";"):
        if lid:
            rows.append({"lake_id": lid, "osm_area_ha": round(o.area_ha, 1), "osm_name": o.osm_name})
pd.DataFrame(rows).drop_duplicates("lake_id", keep="last").to_csv(WORK / "a1_lake_osm_area.csv", index=False)

big = osm[osm.area_ha >= 6.25].copy()
bins = [6.25, 10, 25, 100, 1e9]
labels = ["6.25-10 ha", "10-25 ha", "25-100 ha", "100+ ha"]
big["size"] = pd.cut(big.area_ha, bins, labels=labels, right=False)
print("Share with a lake ID, by size (OSM outlines):")
print(big.groupby("size", observed=True).has_id.agg(["count", "sum", "mean"]).round(2).to_string())

near = [c for c in big.columns if c.startswith("km_to_pass")]
big["covered"] = (big[near].min(axis=1) <= 5)
print("\nTanks of 6.25 ha+ with no SWOT lake from either pass within 5 km (likely outside both swaths):",
      int((~big.covered).sum()), "of", len(big))
cov = big[big.covered]
print(f"Inside the two passes' coverage: {int(cov.has_id.sum())} of {len(cov)} have a lake ID "
      f"({cov.has_id.mean():.0%})")
big["lon_band"] = (np.floor(big.lon / 0.05) * 0.05).round(2)
print("\nBy longitude band (share with ID, count):")
print(big.groupby("lon_band").has_id.agg(["count", "mean"]).round(2).T.to_string())
print("\nLargest tanks without an ID:")
print(big[~big.has_id].nlargest(12, "area_ha")[["osm_name", "area_ha", "lon", "lat"] + near]
      .round(1).to_string(index=False))
stats = {
    "inside_coverage_count": len(cov), "inside_coverage_with_id": int(cov.has_id.sum()),
    "inside_coverage_share": round(float(cov.has_id.mean()), 3),
    "by_size": big.groupby("size", observed=True).has_id.agg(["count", "sum"]).to_dict(),
}
(WORK / "a1b_stats.json").write_text(json.dumps(stats, indent=1, default=str))
