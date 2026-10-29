"""Which SWOT-ready village tanks drain into each other (a real cascade)?

Candidates: the feasibility check's village-tank filters (a6_shortlist.py): SWOT reference area
6.25-25 ha, 40+ usable readings, a yearly rise and fall of 1 m or more, a median jump of 0.25 m or
less, 5% or fewer gross outliers, its own OpenStreetMap outline under 60 ha, within 12 km of one of the
big five. This gives the same 49 tanks as the feasibility check (feasibility-check/work/a6_all_candidates.csv).
Flow: priority flood on Copernicus GLO-30, as in c1_flow_trace.py, over the whole study box.

Run: python -I c2_cascade_search.py <project temp folder> <out folder>
"""
import heapq
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import Point

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
temp, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)
fc = temp / "feasibility-check"
sys.path.insert(0, str(fc / "work"))
import a2_summarize as a2  # noqa: E402  (the feasibility check's own reader for cached SWOT heights)

UTM = "EPSG:32644"
BIG = {"4580051323": "Nachchaduwa", "4580013372": "Hurulu Wewa", "4580009523": "Nuwara Wewa",
       "4580006192": "Mahakanadarawa", "4580013332": "Manankattiya"}
CHOSEN = {"4580005172": "V1", "4580010522": "V2", "4580006282": "V3", "4580010542": "V4",
          "4580011762": "V5", "4580011422": "V6", "4580007392": "V7", "4580043692": "V8"}

# --- candidates, with the same filters as a6_shortlist.py
lakes = pd.read_csv(fc / "results/A2_heights_by_lake.csv", dtype={"lake_id": str})
pld = pd.read_csv(fc / "results/A_pld_lakes.csv", dtype={"lake_id": str})
lakes = lakes.merge(pld[["lake_id", "p_lon", "p_lat", "p_ref_area_ha"]], on="lake_id")
osm = gpd.read_file(fc / "work/a1_osm_water.gpkg").to_crs(UTM)
lakes["pt"] = list(gpd.GeoSeries([Point(x, y) for x, y in zip(lakes.p_lon, lakes.p_lat)], crs="EPSG:4326").to_crs(UTM))


def own_outline(p):
    inside = osm[osm.contains(p)]
    return inside.sort_values("area_ha").iloc[0].geometry if len(inside) else None


def outlier_share(lake_id):
    s = a2.load(lake_id)
    s = s[s.usable]
    if len(s) < 10:
        return np.nan
    med = s.wse.median()
    mad = 1.4826 * (s.wse - med).abs().median()
    return float(((s.wse - med).abs() > 3.5 * mad).mean()) if mad > 0 else 0.0


big_pts = {lid: lakes.loc[lakes.lake_id == lid, "pt"].iloc[0] for lid in BIG}
big_out = {lid: own_outline(p) for lid, p in big_pts.items()}
cand = lakes[(lakes.size_class == "6.25-25 ha") & (lakes.usable_points >= 40) &
             (lakes.rise_fall_usable_m >= 1.0) & (lakes.jitter_usable_m <= 0.25)].copy()
rows = []
for _, l in cand.iterrows():
    inside = osm[osm.contains(l.pt)]
    if not len(inside):
        continue
    own = inside.sort_values("area_ha").iloc[0]
    o = own.geometry
    if own.area_ha >= 60 or any(o.equals(b) for b in big_out.values() if b is not None):
        continue
    d = {BIG[k]: o.distance(b) / 1000 for k, b in big_out.items() if b is not None}
    near = min(d, key=d.get)
    if d[near] > 12:
        continue
    osh = outlier_share(l.lake_id)
    if not (osh <= 0.05):
        continue
    rows.append({"lake_id": l.lake_id, "lon": l.p_lon, "lat": l.p_lat, "ref_ha": l.p_ref_area_ha,
                 "outline_ha": round(o.area / 1e4, 1), "usable": int(l.usable_points), "near_big": near,
                 "km_to_big": round(d[near], 1), "geometry": o})
cand = gpd.GeoDataFrame(rows, crs=UTM)
print(f"{len(cand)} candidate village tanks (the feasibility check found 49)")
missing = [k for k in CHOSEN if k not in set(cand.lake_id)]
print("chosen eight among them:", 8 - len(missing), "| not in list:", missing)

# --- elevation over the whole study box, and priority-flood receivers
W, S, E, N = 80.33, 8.10, 80.86, 8.62
with rasterio.open(temp / "cascade-check/terrain/dem_download/Copernicus_DSM_COG_10_N08_00_E080_00_DEM.tif") as src:
    win = from_bounds(W, S, E, N, src.transform)
    dem = src.read(1, window=win).astype("float64")
    tr = src.window_transform(win)
rows_, cols = dem.shape
z = dem.ravel()
rec = np.full(rows_ * cols, -2, dtype=np.int64)
heap = []
for r in range(rows_):
    for c in range(cols):
        if r in (0, rows_ - 1) or c in (0, cols - 1):
            i = r * cols + c
            rec[i] = -1
            heap.append((z[i], i))
heapq.heapify(heap)
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
while heap:
    h, i = heapq.heappop(heap)
    r, c = divmod(i, cols)
    for dr, dc in NB:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows_ and 0 <= cc < cols:
            j = rr * cols + cc
            if rec[j] == -2:
                rec[j] = i
                heapq.heappush(heap, (max(z[j], h), j))
print(f"flow directions done ({rows_} x {cols} cells)")

# id raster: big five (by OSM outline) and department reservoirs win over candidates
dept = gpd.read_file(temp / "cascade-check/web-sources/irrigation_layers/reservoirs.geojson")
labels, shapes = [], []
c4326 = cand.to_crs("EPSG:4326")
for r in c4326.itertuples():
    labels.append(("cand", r.lake_id))
    shapes.append((r.geometry, len(labels)))
for r in dept.itertuples():
    labels.append(("dept", f"{r.Res_Name} ({r.Type})"))
    shapes.append((r.geometry, len(labels)))
for lid, o in big_out.items():
    labels.append(("dept", f"{BIG[lid]} (Major)"))
    shapes.append((gpd.GeoSeries([o], crs=UTM).to_crs("EPSG:4326").iloc[0], len(labels)))
ids = rasterize(shapes, out_shape=dem.shape, transform=tr, fill=0, dtype="int32").ravel()


def cell(lon, lat):
    c, r = ~tr * (lon, lat)
    return int(r) * cols + int(c)


down = {}
for r in c4326.itertuples():
    p = r.geometry.representative_point()
    i, seq, end = cell(p.x, p.y), [], "leaves the box"
    while i >= 0:
        k = ids[i]
        if k:
            kind, name = labels[k - 1]
            if kind == "cand" and name != r.lake_id and (not seq or seq[-1] != name):
                seq.append(name)
            elif kind == "dept":
                end = name
                break
        i = rec[i]
    down[r.lake_id] = {"next_candidates": seq, "ends_in": end}

# chains: groups of candidates where one drains into another
parent = {k: k for k in down}


def find(k):
    while parent[k] != k:
        parent[k] = parent[parent[k]]
        k = parent[k]
    return k


for k, v in down.items():
    for nxt in v["next_candidates"][:1]:
        parent[find(k)] = find(nxt)
groups = {}
for k in down:
    groups.setdefault(find(k), []).append(k)
chains = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
info = cand.set_index("lake_id")
print(f"\n{len(chains)} chains with two or more SWOT-ready tanks:")
result = []
for g in chains:
    ends = {down[k]["ends_in"] for k in g}
    members = []
    for k in sorted(g, key=lambda k: -len(down[k]["next_candidates"])):
        m = info.loc[k]
        members.append({"lake_id": k, "chosen": CHOSEN.get(k, ""), "outline_ha": float(m.outline_ha),
                        "usable": int(m.usable), "lon": float(m.lon), "lat": float(m.lat),
                        "drains_into": down[k]["next_candidates"][:1] or [down[k]["ends_in"]]})
    result.append({"tanks": len(g), "ends_in": sorted(ends), "members": members})
    print(f"  {len(g)} tanks, ending in {', '.join(sorted(ends))}:")
    for m in members:
        print(f"     {m['lake_id']} {m['chosen']:>2} {m['outline_ha']:>5} ha, {m['usable']} readings, "
              f"{m['lat']:.3f} N {m['lon']:.3f} E -> {m['drains_into'][0]}")
singles = [k for g in groups.values() if len(g) == 1 for k in g]
print(f"\n{len(singles)} candidates drain alone (no other SWOT-ready tank up or downstream)")
for k in CHOSEN:
    if k in down:
        print(f"  chosen {CHOSEN[k]}: next SWOT-ready tank downstream: {down[k]['next_candidates'][:1] or 'none'}; ends in {down[k]['ends_in']}")
cand.drop(columns="geometry").assign(ends_in=[down[k]["ends_in"] for k in cand.lake_id],
                                     next_candidate=[(down[k]["next_candidates"] or [""])[0] for k in cand.lake_id]
                                     ).to_csv(out / "candidates_flow.csv", index=False)
(out / "chains.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
cand.to_crs("EPSG:4326").to_file(out / "candidates.geojson", driver="GeoJSON")
print("wrote", out / "chains.json")
