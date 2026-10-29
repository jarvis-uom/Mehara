"""Member B pilot, step 1: which of the department's 117 reservoirs does SWOT see, and how often?
Matches each reservoir's coordinates to SWOT's prior lakes, fetches missing Hydrocron series.
Writes b_matched.csv."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

WORK = Path(__file__).resolve().parents[1] / "feasibility-check" / "work"
sys.path.insert(0, str(WORK))
from a2_fetch import fetch            # noqa: E402  (our own Hydrocron fetcher, with its cache)
from a2_summarize import load         # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
W, S, E, N = 79.5, 5.85, 82.0, 9.9
UTM = "EPSG:32644"

frames = []
for z in sorted((WORK / "swot").glob("SWOT_L2_HR_LakeSP_Prior_*.zip")):
    g = gpd.read_file(f"zip://{z}", columns=["lake_id", "p_lon", "p_lat", "p_ref_area"])
    frames.append(g[g.p_lon.between(W, E) & g.p_lat.between(S, N)])
g = pd.concat(frames, ignore_index=True)
rows = []
for lid, grp in g.groupby("lake_id"):
    geoms = [x for x in grp.geometry if x is not None and not x.is_empty]
    rows.append({"lake_id": lid, "p_lon": grp.p_lon.iloc[0], "p_lat": grp.p_lat.iloc[0],
                 "ref_area_km2": float(grp.p_ref_area.iloc[0]), "geometry": unary_union(geoms) if geoms else None})
lakes = gpd.GeoDataFrame(rows, geometry="geometry", crs=g.crs).to_crs(UTM)
lakes["ref_pt"] = gpd.GeoSeries([Point(x, y) for x, y in zip(lakes.p_lon, lakes.p_lat)], crs="EPSG:4326").to_crs(UTM).values
print(len(lakes), "SWOT prior lakes on the island;", int(lakes.geometry.notna().sum()), "with an outline seen in Oct 2026")

d = pd.read_csv("storage_daily.csv")
res = d.groupby(["kind", "reservoir"]).agg(lat=("lat", "first"), lon=("lon", "first"),
                                            capacity_acft=("capacity_acft", "first")).reset_index()
pts = gpd.GeoSeries([Point(x, y) for x, y in zip(res.lon, res.lat)], crs="EPSG:4326").to_crs(UTM)
out = []
for (i, r), pt in zip(res.iterrows(), pts):
    radius = np.sqrt(lakes.ref_area_km2 * 1e6 / np.pi)
    d_ref = lakes.ref_pt.apply(lambda q: q.distance(pt))
    d_poly = lakes.geometry.apply(lambda q: q.distance(pt) if q is not None else np.inf)
    cand = lakes[(d_poly <= 500) | (d_ref <= np.maximum(1500, 1.3 * radius))].copy()
    if cand.empty:
        out.append({**r, "lake_id": None})
        continue
    cand["d"] = np.minimum(d_poly[cand.index], d_ref[cand.index])
    best = cand.sort_values(["d", "ref_area_km2"], ascending=[True, False]).iloc[0]
    # the department point may sit on the bund: prefer the biggest lake within 500 m of it
    near = cand[cand.d <= 500]
    if len(near):
        best = near.sort_values("ref_area_km2", ascending=False).iloc[0]
    depth = r.capacity_acft * 1233.48 / (best.ref_area_km2 * 1e6)
    out.append({**r, "lake_id": best.lake_id, "ref_area_km2": round(best.ref_area_km2, 3),
                "match_dist_m": round(best.d), "implied_mean_depth_m": round(depth, 1)})
m = pd.DataFrame(out)
m["match_ok"] = m.lake_id.notna() & m.implied_mean_depth_m.between(0.4, 25)
todo = m[m.match_ok].lake_id.unique().tolist()
with ThreadPoolExecutor(max_workers=4) as pool:
    status = dict(pool.map(fetch, todo))
print("Hydrocron:", pd.Series(status).value_counts().to_dict())
for i, r in m[m.match_ok].iterrows():
    df = load(r.lake_id)
    m.loc[i, "readings"] = len(df)
    m.loc[i, "usable"] = int(df.usable.sum()) if len(df) else 0
    m.loc[i, "best"] = int(df.good.sum()) if len(df) else 0
m.to_csv("b_matched.csv", index=False)
ok = m[m.match_ok]
print(f"matched {len(ok)} of {len(m)} reservoirs ({ok.kind.value_counts().to_dict()}); "
      f"unmatched or odd: {(~m.match_ok).sum()}")
print("SWOT readings per matched reservoir: median", int(ok.readings.median()), "| usable median", int(ok.usable.median()),
      "| total readings", int(ok.readings.sum()), "| total usable", int(ok.usable.sum()))
print("reservoirs with >= 20 usable readings:", int((ok.usable >= 20).sum()))
print(m[~m.match_ok][["kind", "reservoir", "lake_id", "match_dist_m", "implied_mean_depth_m"]].to_string(index=False))
