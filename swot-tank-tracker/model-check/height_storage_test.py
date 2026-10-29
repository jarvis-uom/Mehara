"""The height version of B's model (outside review, round 2): turn SWOT height, plus water area where there is one,
into % full, for reservoirs the model never saw. Run with and without B's trust screen.

Pairs: SWOT readings at 84 department reservoirs (b_readings_labelled.csv) with that day's storage share.
Inputs a village tank has: height above the tank's lowest screened reading, the height span of its readings,
full-supply area and outline shape (from the map), catchment; the area version adds the Global Water Watch area
share within 5 days. No capacity or scheme type.
Baseline: one formula for all tanks in relative height, y = c * ((h - low) / span)^k, fitted on training reservoirs.
Ceiling: each tank's own curve in height (fitted on its own pairs).
Screens: all readings; SWOT's loose flag (quality_f <= 1); B's trust screen keeping the same number per reservoir."""
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

lay = gpd.read_file("dept_layer.geojson").to_crs("EPSG:32644")
lay["shape_index"] = lay.geometry.length / np.sqrt(4 * np.pi * lay.geometry.area)
link = pd.read_csv("sheet_to_layer.csv")
link = link[link.dist_m <= 1000].merge(lay[["Res_Name", "Area_FSL_ha", "Catch_ha", "shape_index"]].drop_duplicates("Res_Name"),
                                        left_on="layer_name", right_on="Res_Name", how="left")
b = pd.read_csv("b_readings_labelled.csv", parse_dates=["date"])
b = b.merge(link[["reservoir", "kind", "Area_FSL_ha", "Catch_ha", "shape_index"]], on=["reservoir", "kind"], how="inner")
gww = pd.read_csv("gww_pairs.csv", parse_dates=["date"]).sort_values("date")
b = b.sort_values("date")
b = pd.merge_asof(b, gww[["reservoir", "kind", "date", "area_ha"]], on="date", by=["reservoir", "kind"],
                  direction="nearest", tolerance=pd.Timedelta(days=5)).reset_index(drop=True)
b["y"] = b.frac.clip(0, 1.3)


def screened(d, how):
    if how == "all readings":
        return d
    if how == "SWOT loose flag (quality_f <= 1)":
        return d[d.quality_f <= 1]
    keep = []                                   # B's trust screen: the same number per reservoir as the loose flag
    for _, g in d.groupby("lake_id"):
        keep += list(g.sort_values("risk").index[:int((g.quality_f <= 1).sum())])
    return d.loc[keep]


def run(d, with_area):
    d = d.copy()
    lo = d.groupby("lake_id").wse.transform(lambda s: s.quantile(0.05))
    hi = d.groupby("lake_id").wse.transform(lambda s: s.quantile(0.95))
    d["h_above_low"], d["h_span"] = d.wse - lo, (hi - lo).clip(lower=0.3)
    d["h_rel"] = (d.h_above_low / d.h_span).clip(-0.5, 1.5)
    d["log_full_area"], d["log_catch"] = np.log10(d.Area_FSL_ha), np.log10(d.Catch_ha.clip(lower=1))
    feats = ["h_rel", "h_above_low", "h_span", "log_full_area", "shape_index", "log_catch"]
    if with_area:
        d = d[d.area_ha.notna()].copy()
        d["area_share"] = (d.area_ha / d.Area_FSL_ha).clip(0, 1.5)
        feats.append("area_share")
    d = d.reset_index(drop=True)
    mono = [1 if f in ("h_rel", "h_above_low", "area_share") else 0 for f in feats]
    out = {"formula": np.full(len(d), np.nan), "model": np.full(len(d), np.nan), "own_curve": np.full(len(d), np.nan)}
    x = d.h_rel.clip(0.001, 1.5).values
    for a, t in GroupKFold(5).split(d, groups=d.lake_id):
        q = least_squares(lambda p: p[0] * x[a] ** p[1] - d.y.values[a], [0.8, 1.2], loss="soft_l1", f_scale=0.1,
                          bounds=([0.05, 0.2], [3, 5])).x
        out["formula"][t] = q[0] * x[t] ** q[1]
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                              subsample_freq=1, monotone_constraints=mono, objective="huber", alpha=0.1, verbose=-1)
        out["model"][t] = m.fit(d.iloc[a][feats], d.y.iloc[a]).predict(d.iloc[t][feats])
    for _, g in d.groupby("lake_id"):
        xi = x[g.index]
        q = least_squares(lambda p: p[0] * xi ** p[1] - g.y.values, [0.8, 1.2], loss="soft_l1", f_scale=0.1,
                          bounds=([0.05, 0.2], [3, 5])).x
        out["own_curve"][g.index] = q[0] * xi ** q[1]
    small = d.Area_FSL_ha.values < 200
    row = {"reservoirs": d.lake_id.nunique(), "readings": len(d)}
    per = {}
    for k, v in out.items():
        err = np.abs(np.clip(v, 0, 1.3) - d.y.values) * 100
        row[k] = round(err.mean(), 1)
        row[k + "_lt2km2"] = round(err[small].mean(), 1)
        per[k] = pd.Series(err).groupby(d.lake_id.values).mean()
    row["model_beats_formula_share"] = round((per["model"] < per["formula"]).mean(), 2)
    return row


rows = []
for with_area in (False, True):
    for how in ("all readings", "SWOT loose flag (quality_f <= 1)", "B's trust screen, same number kept"):
        r = run(screened(b, how), with_area)
        rows.append({"version": "height + area" if with_area else "height only", "readings_used": how, **r})
res = pd.DataFrame(rows)
res.to_csv("height_storage_results.csv", index=False)
with pd.option_context("display.width", 220):
    print("error in % of capacity, reservoirs never seen (5 folds by reservoir)")
    print(res.to_string(index=False))
