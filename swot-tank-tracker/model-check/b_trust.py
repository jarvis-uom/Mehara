"""Member B pilot, step 2: can a learned model pick good SWOT readings better than SWOT's own flags?
Label: how far each reading sits from its reservoir's own height-storage curve. The curve is fitted robustly to
all of that reservoir's readings against the department's storage on the same day. 'Bad' = more than 30 cm off.
Features: SWOT's own fields only (no gauge), so the model could run on any tank.
Test: 10 folds by reservoir, so every score comes from a model that never saw that reservoir."""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

WORK = Path(__file__).resolve().parents[1] / "feasibility-check" / "work"
sys.path.insert(0, str(WORK))
from a2_summarize import load  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
BAD_M = 0.30
FEATS = ["quality_f", "xovr_cal_q", "dark_frac", "partial_f", "xtrk_km", "wse_u", "area_ratio", "log_ref_area",
         "mon_sin", "mon_cos", "nb_dev", "nb_n"]

m = pd.read_csv("b_matched.csv", dtype={"lake_id": str})
m = m[m.match_ok].drop_duplicates("lake_id")
st = pd.read_csv("storage_daily.csv", parse_dates=["date"])
st["frac"] = st.storage_acft / st.capacity_acft
lv = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "timing-gauges" / "levels_daily.csv",
                 dtype={"lake_id": str}, parse_dates=["day"])

all_rows, fits = [], []
for r in m.itertuples():
    df = load(r.lake_id)
    if len(df) < 15:
        continue
    df["date"] = df.time.dt.tz_convert("Asia/Colombo").dt.tz_localize(None).dt.normalize()
    s = st[(st.reservoir == r.reservoir) & (st.kind == r.kind)].set_index("date").frac
    df["frac"] = df.date.map(s)
    df["frac"] = df.frac.fillna((df.date - pd.Timedelta(days=1)).map(s))
    df = df[df.frac.between(0.03, 0.97)].copy()
    df = df[(df.wse - df.wse.median()).abs() < 15]
    if len(df) < 15 or df.frac.quantile(.9) - df.frac.quantile(.1) < 0.10:
        continue
    x, y = df.frac.values, df.wse.values
    fit = least_squares(lambda p: p[0] + p[1] * x + p[2] * x * x - y, [np.median(y), 3.0, 0.0],
                        loss="soft_l1", f_scale=0.10)
    df["resid"] = y - (fit.x[0] + fit.x[1] * x + fit.x[2] * x * x)
    # SWOT-only features
    df["xtrk_km"] = df.xtrk_dist.abs() / 1000
    df["area_ratio"] = df.area_total / df.area_total.median()
    df["log_ref_area"] = np.log10(r.ref_area_km2)
    mon = df.date.dt.month
    df["mon_sin"], df["mon_cos"] = np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)
    t = df.date.values.astype("datetime64[D]").astype(int)
    dev, cnt = [], []
    for i in range(len(df)):
        near = (np.abs(t - t[i]) <= 16) & (np.arange(len(df)) != i)
        dev.append(abs(y[i] - np.median(y[near])) if near.any() else np.nan)
        cnt.append(int(near.sum()))
    df["nb_dev"], df["nb_n"] = dev, cnt
    # our current cleaning rule: SWOT flags plus 3.5 MAD on the usable readings
    u = df.usable
    med = df.wse[u].median()
    mad = (df.wse[u] - med).abs().median() or 0.01
    df["our_rule"] = u & ((0.6745 * (df.wse - med) / mad).abs() <= 3.5)
    df["reservoir"], df["kind"], df["ref_area_km2"] = r.reservoir, r.kind, r.ref_area_km2
    all_rows.append(df)
    fits.append({"reservoir": r.reservoir, "lake_id": r.lake_id, "n": len(df), "slope_m_per_full": round(fit.x[1] + fit.x[2], 2),
                 "median_abs_resid_m": round(np.median(np.abs(df.resid)), 3)})

d = pd.concat(all_rows, ignore_index=True)
d["bad"] = d.resid.abs() > BAD_M
fits = pd.DataFrame(fits)
print(f"reservoirs used: {d.reservoir.nunique()} | readings: {len(d):,} | bad (more than {BAD_M * 100:.0f} cm off the curve): {d.bad.mean():.0%}")
print("median distance from the curve, per reservoir: median", round(fits.median_abs_resid_m.median(), 3), "m")

# check the label against real gauge levels at the 5 big reservoirs
g = d.merge(lv[["day", "lake_id", "level_dept_only_m"]].dropna(), left_on=["lake_id", "date"], right_on=["lake_id", "day"])
if len(g):
    g["gauge_resid"] = g.wse - g.level_dept_only_m
    g["gauge_resid"] -= g.groupby("lake_id").gauge_resid.transform("median")
    print(f"label check at the 5 big reservoirs ({len(g)} readings): correlation of curve-residual with gauge-residual "
          f"{np.corrcoef(g.resid, g.gauge_resid)[0, 1]:.2f}; 'bad' agrees on {((g.resid.abs() > BAD_M) == (g.gauge_resid.abs() > BAD_M)).mean():.0%}")

d["risk"] = np.nan
for a, b in GroupKFold(10).split(d, groups=d.lake_id):
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                             subsample=0.8, subsample_freq=1, verbose=-1).fit(d.iloc[a][FEATS], d.iloc[a].bad)
    d.loc[d.index[b], "risk"] = clf.predict_proba(d.iloc[b][FEATS])[:, 1]
imp = pd.Series(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, verbose=-1)
                .fit(d[FEATS], d.bad).booster_.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
print("what the trust model uses most:", ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(6).items()))


def summary(name, keep):
    k = d[keep]
    return {"method": name, "kept": len(k), "kept_share": round(len(k) / len(d), 2), "bad_in_kept": round(k.bad.mean(), 3),
            "mae_kept_cm": round(k.resid.abs().mean() * 100, 1), "p90_kept_cm": round(k.resid.abs().quantile(.9) * 100, 1),
            "reservoirs_with_10plus": int((k.groupby("lake_id").size() >= 10).sum())}


rows = [summary("all readings", d.wse.notna())]
for name, keep in [("SWOT flag: good only (quality_f = 0)", d.quality_f == 0),
                   ("SWOT flag: good or suspect (quality_f <= 1)", d.quality_f <= 1),
                   ("our rule (flags + crossover + dark water + 3.5 MAD)", d.our_rule)]:
    rows.append(summary(name, keep))
    cut = d.risk.sort_values().iloc[int(keep.sum()) - 1]
    rows.append(summary(f"  trust model, same number kept", d.risk <= cut))
res = pd.DataFrame(rows)
res.to_csv("b_results.csv", index=False)
d.to_csv("b_readings_labelled.csv", index=False)
print(res.to_string(index=False))
print("\nAUC for spotting bad readings (1 = perfect, 0.5 = guessing):",
      "trust model", round(roc_auc_score(d.bad, d.risk), 2), "| quality_f alone", round(roc_auc_score(d.bad, d.quality_f), 2),
      "| our rule", round(roc_auc_score(d.bad, ~d.our_rule), 2))
q2 = d[d.quality_f == 2]
print(f"readings SWOT calls bad (quality_f = 2): {len(q2)}; of these within {BAD_M * 100:.0f} cm of the curve: {(~q2.bad).mean():.0%}")
small = d[d.ref_area_km2 < 2]
print(f"small reservoirs (< 2 km²): {small.lake_id.nunique()} | readings {len(small)} | bad {small.bad.mean():.0%} | "
      f"AUC trust {roc_auc_score(small.bad, small.risk):.2f} vs quality_f {roc_auc_score(small.bad, small.quality_f):.2f}")
