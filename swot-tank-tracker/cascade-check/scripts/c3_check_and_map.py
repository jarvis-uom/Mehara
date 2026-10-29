"""Check the chains from c2 against the Irrigation Department's catchments, add names and SWOT
passes, and draw the chosen eight and the chains with their flow paths on a hillshade map.

Run: python -I c3_check_and_map.py <project temp folder> <out folder>
"""
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import LineString, Point

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
temp, out = Path(sys.argv[1]), Path(sys.argv[2])
fc, cc = temp / "feasibility-check", temp / "cascade-check"
chains = json.loads((out / "chains.json").read_text(encoding="utf-8"))
cand = gpd.read_file(out / "candidates.geojson")
pld = pd.read_csv(fc / "results/A_pld_lakes.csv", dtype={"lake_id": str}).set_index("lake_id")
osm = gpd.read_file(fc / "work/a1_osm_water.gpkg").to_crs("EPSG:4326")
catch = gpd.read_file(cc / "web-sources/irrigation_layers/catchments.geojson")
CHOSEN = {"4580005172": "V1", "4580010522": "V2", "4580006282": "V3", "4580010542": "V4",
          "4580011762": "V5", "4580011422": "V6", "4580007392": "V7", "4580043692": "V8"}

rows = []
for n, ch in enumerate(chains, 1):
    for m in ch["members"]:
        p = Point(m["lon"], m["lat"])
        g = cand[cand.lake_id == m["lake_id"]].geometry.iloc[0]
        names = osm[osm.intersects(g.representative_point())]["name"].dropna().unique().tolist() if "name" in osm else []
        rows.append({"chain": n, "ends_in": ch["ends_in"][0], "lake_id": m["lake_id"], "chosen": m["chosen"],
                     "outline_ha": m["outline_ha"], "usable": m["usable"], "drains_into": m["drains_into"][0],
                     "dept_catchments": "; ".join(catch[catch.contains(p)]["Reservoir_Name"]),
                     "swot_passes": pld.loc[m["lake_id"], "passes"], "pld_name": pld.loc[m["lake_id"], "lake_name"],
                     "osm_name": "; ".join(n_ for n_ in names if n_), "lat": m["lat"], "lon": m["lon"]})
t = pd.DataFrame(rows)
t.to_csv(out / "chains_checked.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(t.drop(columns=["lat", "lon"]).to_string(index=False))

# --- map: hillshade, department reservoirs, the chosen eight, chain members, flow paths
W, S, E, N = 80.40, 8.12, 80.72, 8.47
with rasterio.open(cc / "terrain/dem_download/Copernicus_DSM_COG_10_N08_00_E080_00_DEM.tif") as src:
    win = from_bounds(W, S, E, N, src.transform)
    dem = src.read(1, window=win).astype(float)
gy, gx = np.gradient(dem, 30, 30)
slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
aspect = np.arctan2(-gx, gy)
az, alt = np.radians(315), np.radians(45)
shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)

fig, ax = plt.subplots(figsize=(9, 9.5), dpi=110)
ax.imshow(shade, cmap="Greys_r", extent=(W, E, S, N), alpha=0.55, vmin=0.4, vmax=1.0)
dept = gpd.read_file(cc / "web-sources/irrigation_layers/reservoirs.geojson").cx[W:E, S:N]
dept.plot(ax=ax, color="#9ecae1", edgecolor="#3182bd", linewidth=0.6)
for r in dept.itertuples():
    if r.Type == "Major":
        c = r.geometry.representative_point()
        ax.annotate(r.Res_Name, (c.x, c.y), fontsize=8, color="#08519c", ha="center", weight="bold")
catch.cx[W:E, S:N].boundary.plot(ax=ax, color="#756bb1", linewidth=0.7, linestyle="--")
colors = ["#e6550d", "#31a354", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
for n, ch in enumerate(chains):
    for m in ch["members"]:
        g = cand[cand.lake_id == m["lake_id"]]
        g.plot(ax=ax, color=colors[n % len(colors)], edgecolor="black", linewidth=0.4)
        if m["drains_into"][0].startswith("458"):
            to = cand[cand.lake_id == m["drains_into"][0]].geometry.iloc[0].representative_point()
            ax.annotate("", (to.x, to.y), (m["lon"], m["lat"]),
                        arrowprops=dict(arrowstyle="->", color=colors[n % len(colors)], lw=1.4))
for lid, key in CHOSEN.items():
    g = cand[cand.lake_id == lid]
    p = g.geometry.iloc[0].representative_point()
    g.boundary.plot(ax=ax, color="black", linewidth=1.6)
    ax.annotate(key, (p.x, p.y), xytext=(4, 4), textcoords="offset points", fontsize=9, weight="bold")
ax.set_xlim(W, E)
ax.set_ylim(S, N)
ax.set_title("Village tanks that pass the SWOT checks: who drains into whom\n"
             "black outline = our chosen eight (V1-V8); coloured = tanks in a chain; arrows = downhill flow; "
             "dashed = department catchments", fontsize=9)
ax.text(0.01, 0.01, "Elevation: Copernicus GLO-30 (© DLR e.V. 2010-2014 and © Airbus 2014-2018, provided under COPERNICUS by the EU and ESA).\n"
        "Tank outlines © OpenStreetMap contributors. Reservoirs and catchments: Irrigation Department of Sri Lanka.",
        transform=ax.transAxes, fontsize=6.5, va="bottom", bbox=dict(fc="white", alpha=0.8, lw=0))
fig.tight_layout()
fig.savefig(out / "cascade_map.png")
print("wrote", out / "cascade_map.png")
