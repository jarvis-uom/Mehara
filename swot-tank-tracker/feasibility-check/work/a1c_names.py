"""A1 fix: give each tank name to only one SWOT lake (the biggest that matches).

Rewrites tank_name in results/A_pld_lakes.csv and the comments in results/A_lake_ids.txt.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"

pld = pd.read_csv(RES / "A_pld_lakes.csv", dtype={"lake_id": str})
ref = dict(zip(pld.lake_id, pld.p_ref_area_ha))
dept = pd.read_csv(RES / "A1_tanks.csv", dtype=str).fillna("")

names = {}
# Department tanks first: direct matches beat matches through an OSM outline
for _, t in dept.sort_values("area_fsl_ha", key=lambda s: s.astype(float), ascending=False).iterrows():
    for field in ("lake_ids_direct", "lake_ids_via_osm"):
        cands = [i for i in t[field].split(";") if i and i not in names]
        if cands:
            names[max(cands, key=lambda i: ref.get(i, 0))] = t["name"]
            break
# Then OpenStreetMap names
import geopandas as gpd  # noqa: E402
osm = gpd.read_file(BASE / "work" / "a1_osm_water.gpkg")
for _, o in osm[osm.osm_name.fillna("") != ""].sort_values("area_ha", ascending=False).iterrows():
    cands = [i for i in str(o.lake_ids).split(";") if i and i != "None" and i not in names]
    if cands and o.osm_name not in names.values():
        names[max(cands, key=lambda i: ref.get(i, 0))] = o.osm_name

pld["tank_name"] = pld.lake_id.map(names).fillna("")
pld.to_csv(RES / "A_pld_lakes.csv", index=False)
with open(RES / "A_lake_ids.txt", "w") as f:
    f.write("# SWOT Prior Lake Database IDs in the study box 80.40-80.80E, 8.10-8.60N\n")
    f.write("# lake_id  # tank name (if matched), SWOT reference area\n")
    for _, r in pld.sort_values("p_ref_area_ha", ascending=False).iterrows():
        f.write(f"{r.lake_id}  # {r.tank_name or 'unnamed'}, {r.p_ref_area_ha:g} ha\n")
print("named lakes:", sum(1 for v in names.values() if v), "of", len(pld))
print(pld[pld.tank_name != ""].nlargest(12, "p_ref_area_ha")[["lake_id", "tank_name", "p_ref_area_ha"]].to_string(index=False))
