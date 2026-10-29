"""Pair Global Water Watch water areas with the department's daily storage, for the area -> storage test.
Each department reservoir (its full-supply outline from the department layer) is matched to the GWW reservoir whose
outline overlaps it most. Each GWW area reading (date in Sri Lanka time) gets that day's storage share.
Writes gww_pairs.csv: reservoir, kind, date, area_ha, frac."""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape

sys.stdout.reconfigure(encoding="utf-8")
NC = Path(__file__).resolve().parents[1] / "data-sources" / "gww-archive" / "gww_reservoir_timeseries.nc"
W, S, E, N = 79.5, 5.85, 82.0, 9.9

ds = xr.open_dataset(NC)
lon, lat = ds.longitude.values, ds.latitude.values
geo_size = ds.geometry_row_size.values
geo_off = np.concatenate([[0], np.cumsum(geo_size)[:-1]])
obs_size = ds.surface_water_area_row_size.values.astype(np.int64)
obs_off = np.concatenate([[0], np.cumsum(obs_size)[:-1]])
in_box = np.flatnonzero((lon >= W) & (lon <= E) & (lat >= S) & (lat <= N) & (geo_size > 0) & (obs_size > 0))
print("GWW reservoirs in Sri Lanka with an outline and readings:", len(in_box))
gbytes = ds.geometry_utf8
outlines = [shape(json.loads(bytes(gbytes[geo_off[i]:geo_off[i] + geo_size[i]].values).decode("utf-8"))) for i in in_box]
gww = gpd.GeoDataFrame({"i": in_box, "gww_id": ds.reservoir.values[in_box]}, geometry=outlines, crs="EPSG:4326").to_crs("EPSG:32644")
gww["gww_area_ha"] = gww.geometry.area / 1e4

lay = gpd.read_file("dept_layer.geojson").to_crs("EPSG:32644")
link = pd.read_csv("sheet_to_layer.csv")
link = link[link.dist_m <= 1000].merge(lay[["Res_Name", "geometry"]].drop_duplicates("Res_Name"),
                                        left_on="layer_name", right_on="Res_Name")
link = gpd.GeoDataFrame(link, geometry="geometry", crs="EPSG:32644")
rows = []
for r in link.itertuples():
    hits = gww[gww.intersects(r.geometry)]
    if hits.empty:
        continue
    inter = hits.geometry.intersection(r.geometry).area
    iou = inter / (hits.geometry.area + r.geometry.area - inter)
    j = iou.idxmax()
    rows.append({"reservoir": r.reservoir, "kind": r.kind, "i": hits.i[j], "gww_id": hits.gww_id[j],
                 "iou": round(float(iou[j]), 2), "gww_area_ha": round(hits.gww_area_ha[j], 1),
                 "dept_area_ha": round(r.geometry.area / 1e4, 1)})
match = pd.DataFrame(rows)
match = match[match.iou >= 0.3]
print(f"department reservoirs matched (outline overlap IoU >= 0.3): {len(match)} | IoU median {match.iou.median():.2f}")

st = pd.read_csv("storage_daily.csv", parse_dates=["date"])
st["frac"] = st.storage_acft / st.capacity_acft
area = ds.surface_water_area.values
when = ds.surface_water_area_time.values
pairs = []
for m in match.itertuples():
    o, n = obs_off[m.i], obs_size[m.i]
    t = pd.to_datetime(when[o:o + n]).tz_localize("UTC").tz_convert("Asia/Colombo").tz_localize(None).normalize()
    a = pd.DataFrame({"date": t, "area_ha": area[o:o + n] / 1e4})
    a = a[(a.date >= "2020-10-01") & (a.area_ha > 0)].groupby("date", as_index=False).area_ha.median()
    s = st[(st.reservoir == m.reservoir) & (st.kind == m.kind)].set_index("date").frac
    a["frac"] = a.date.map(s)
    a["frac"] = a.frac.fillna((a.date - pd.Timedelta(days=1)).map(s))
    a = a.dropna(subset=["frac"])
    a["reservoir"], a["kind"] = m.reservoir, m.kind
    pairs.append(a)
p = pd.concat(pairs, ignore_index=True)[["reservoir", "kind", "date", "area_ha", "frac"]]
p.to_csv("gww_pairs.csv", index=False)
match.to_csv("gww_match.csv", index=False)
print(f"pairs: {len(p):,} at {p.reservoir.nunique()} reservoirs | per reservoir: median {int(p.groupby('reservoir').size().median())}"
      f" | dates {p.date.min().date()} to {p.date.max().date()}")
