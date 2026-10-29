"""Option 1 for Member B (from an outside review): can one model, trained on many reservoirs, turn a tank's water
area into stored water better than a fitted formula, on reservoirs it never saw?

Pairs: a water area on some day (from SWOT's own area, or from Global Water Watch) and the department's storage that day.
  x = water area / full-supply area (department outline)      y = storage / capacity (share full)
Baselines, fitted on the training reservoirs only:
  - one power law for all tanks: y = c * x^k
  - "area x fixed depth": y = x * (full area * d) / capacity, with one depth d for all tanks
Model: LightGBM, never less water for more area (monotone), with the full area, outline shape, catchment and type.
Test: 5 folds by reservoir; error = mean absolute error in % of capacity. Also shown: each reservoir's own best
power law (fitted on its own pairs: a ceiling, since a new tank has no history).
Usage: python -I area_storage_test.py swot|gww"""
import sys
import warnings

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "swot"

lay = gpd.read_file("dept_layer.geojson").to_crs("EPSG:32644")
lay["perim_m"] = lay.geometry.length
lay["poly_area_m2"] = lay.geometry.area
link = pd.read_csv("sheet_to_layer.csv")
link = link[link.dist_m <= 1000]
info = link.merge(lay[["Res_Name", "Area_FSL_ha", "Catch_ha", "Type", "perim_m", "poly_area_m2", "FSL_m_MSL"]]
                  .drop_duplicates("Res_Name"), left_on="layer_name", right_on="Res_Name", how="left")
info["shape_index"] = info.perim_m / np.sqrt(4 * np.pi * info.poly_area_m2)    # 1 = a circle; bigger = more ragged
cap = pd.read_csv("storage_daily.csv").groupby(["kind", "reservoir"]).capacity_acft.first().rename("cap_acft").reset_index()
info = info.merge(cap, on=["kind", "reservoir"])

if SOURCE == "swot":
    p = pd.read_csv("b_readings_labelled.csv", parse_dates=["date"])
    p = p[p.quality_f <= 1]
    p = p.assign(area_ha=p.area_total * 100)[["reservoir", "kind", "date", "area_ha", "frac"]]
else:
    p = pd.read_csv("gww_pairs.csv", parse_dates=["date"])
d = p.merge(info, on=["kind", "reservoir"])
d["x"] = (d.area_ha / d.area_fsl_ha).clip(0, 1.5)
d["y"] = d.frac.clip(0, 1.3)
d["full_volume_ratio"] = d.area_fsl_ha * 1e4 / (d.cap_acft * 1233.48)     # 1 / mean full depth (per metre)
d = d[d.x.notna() & d.y.notna() & d.area_fsl_ha.gt(0)]
per = d.groupby("reservoir").size()
d = d[d.reservoir.isin(per[per >= 8].index)].reset_index(drop=True)
print(f"source: {SOURCE} | reservoirs: {d.reservoir.nunique()} | pairs: {len(d):,} | pairs per reservoir: median {int(d.groupby('reservoir').size().median())}")

FEATS = ["x", "log_full_area", "shape_index", "log_catch", "major", "full_volume_ratio"]
d["log_full_area"], d["log_catch"] = np.log10(d.area_fsl_ha), np.log10(d.Catch_ha.clip(lower=1))
d["major"] = (d.kind == "Major").astype(int)
pred = {k: np.full(len(d), np.nan) for k in ["power_law", "area_x_depth", "model", "own_power_law"]}
for a, b in GroupKFold(5).split(d, groups=d.reservoir):
    tr, te = d.iloc[a], d.iloc[b]
    pl = least_squares(lambda q: q[0] * tr.x.values ** q[1] - tr.y.values, [1.0, 1.3], loss="soft_l1", f_scale=0.1,
                       bounds=([0.01, 0.2], [5, 5])).x
    pred["power_law"][b] = pl[0] * te.x.values ** pl[1]
    dep = least_squares(lambda q: tr.x.values * tr.area_fsl_ha.values * 1e4 * q[0]
                        / (tr.cap_acft.values * 1233.48) - tr.y.values, [3.0], loss="soft_l1", f_scale=0.1,
                        bounds=([0.1], [30])).x
    pred["area_x_depth"][b] = te.x.values * te.area_fsl_ha.values * 1e4 * dep[0] / (te.cap_acft.values * 1233.48)
    mono = [1 if f == "x" else 0 for f in FEATS]
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                          subsample_freq=1, monotone_constraints=mono, objective="huber", alpha=0.1, verbose=-1).fit(tr[FEATS], tr.y)
    pred["model"][b] = m.predict(te[FEATS])
for r, g in d.groupby("reservoir"):          # ceiling: the reservoir's own curve (fitted on itself, so optimistic)
    q = least_squares(lambda q: q[0] * g.x.values ** q[1] - g.y.values, [1.0, 1.3], loss="soft_l1", f_scale=0.1,
                      bounds=([0.01, 0.2], [5, 5])).x
    pred["own_power_law"][g.index] = q[0] * g.x.values ** q[1]

rows = []
for k, v in pred.items():
    err = np.abs(np.clip(v, 0, 1.3) - d.y.values) * 100
    per_res = pd.Series(err).groupby(d.reservoir.values).mean()
    rows.append({"method": k, "mae_pct_capacity": round(err.mean(), 1), "median_reservoir_mae": round(per_res.median(), 1),
                 "small_lt_2km2": round(err[d.area_fsl_ha.values < 200].mean(), 1),
                 "medium_kind": round(err[d.kind.values == "Medium"].mean(), 1)})
res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(f"area_storage_{SOURCE}.csv", index=False)
