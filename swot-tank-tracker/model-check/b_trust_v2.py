"""Member B pilot, version 2, with the fixes from an outside review (28 Oct 2026):
- the height-storage curve must rise and bend the physical way: h = h0 + a * x^b, a > 0, 0.2 <= b <= 1,
  fitted to readings with wse_u < 0.5 m;
- department days whose storage had not changed for more than 5 days are left out (the sheet may be stale);
- reservoirs where more than half the readings are 'bad' are left out (wrong lake match or broken data);
- features use only what is known on the day: nearby readings from the past 16 days only, and area against
  SWOT's prior-lake reference area instead of the lake's own median over all years;
- the fair test: inside each reservoir, keep as many readings as the flag rule keeps there.
Also a learning curve (5, 10, 20, 40 and all training reservoirs)."""
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
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
              subsample_freq=1, verbose=-1)

m = pd.read_csv("b_matched.csv", dtype={"lake_id": str})
m = m[m.match_ok].drop_duplicates("lake_id")
st = pd.read_csv("storage_daily.csv", parse_dates=["date"]).sort_values(["kind", "reservoir", "date"])
st["frac"] = st.storage_acft / st.capacity_acft
same = (st.storage_acft.diff() == 0) & (st.date.diff() == pd.Timedelta(days=1)) & (st.reservoir == st.reservoir.shift())
st["unchanged_days"] = same.groupby((~same).cumsum()).cumsum()
lv = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "timing-gauges" / "levels_daily.csv",
                 dtype={"lake_id": str}, parse_dates=["day"])


def curve(p, x):
    return p[0] + p[1] * x ** p[2]


rows, dropped, n_frozen = [], [], 0
for r in m.itertuples():
    df = load(r.lake_id)
    if len(df) < 15:
        continue
    df["date"] = df.time.dt.tz_convert("Asia/Colombo").dt.tz_localize(None).dt.normalize()
    s = st[(st.reservoir == r.reservoir) & (st.kind == r.kind)].set_index("date")
    for col in ("frac", "unchanged_days"):
        df[col] = df.date.map(s[col])
        df[col] = df[col].fillna((df.date - pd.Timedelta(days=1)).map(s[col]))
    frozen = df.unchanged_days > 5
    n_frozen += int(frozen.sum())
    df = df[~frozen & df.frac.between(0.03, 0.97)].copy()
    df = df[(df.wse - df.wse.median()).abs() < 15]
    if len(df) < 15 or df.frac.quantile(.9) - df.frac.quantile(.1) < 0.10:
        continue
    x, y = df.frac.values, df.wse.values
    fit_on = df.wse_u.values < 0.5
    if fit_on.sum() < 10:
        fit_on[:] = True
    p0 = [np.percentile(y[fit_on], 5), max(np.ptp(y[fit_on]), 0.5), 0.6]
    fit = least_squares(lambda p: curve(p, x[fit_on]) - y[fit_on], p0, loss="soft_l1", f_scale=0.10,
                        bounds=([-np.inf, 1e-3, 0.2], [np.inf, 60, 1.0]))
    df["resid"] = y - curve(fit.x, x)
    if (df.resid.abs() > BAD_M).mean() > 0.5:
        dropped.append(r.reservoir)
        continue
    df["xtrk_km"] = df.xtrk_dist.abs() / 1000
    df["area_ratio"] = df.area_total / r.ref_area_km2
    df["log_ref_area"] = np.log10(r.ref_area_km2)
    mon = df.date.dt.month
    df["mon_sin"], df["mon_cos"] = np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)
    t = df.date.values.astype("datetime64[D]").astype(int)
    dev, cnt = [], []
    for i in range(len(df)):
        past = (t >= t[i] - 16) & (t < t[i])            # only readings SWOT had already made
        dev.append(abs(y[i] - np.median(y[past])) if past.any() else np.nan)
        cnt.append(int(past.sum()))
    df["nb_dev"], df["nb_n"] = dev, cnt
    u = df.usable
    med = df.wse[u].median()
    mad = (df.wse[u] - med).abs().median() or 0.01
    df["our_rule"] = u & ((0.6745 * (df.wse - med) / mad).abs() <= 3.5)
    df["reservoir"], df["kind"], df["ref_area_km2"] = r.reservoir, r.kind, r.ref_area_km2
    df["curve_b"] = fit.x[2]
    rows.append(df)

d = pd.concat(rows, ignore_index=True)
d["bad"] = d.resid.abs() > BAD_M
print(f"pairs left out because the department value was unchanged for more than 5 days: {n_frozen}")
print(f"reservoirs left out (more than half 'bad': wrong match or broken data): {len(dropped)} {dropped}")
print(f"reservoirs used: {d.lake_id.nunique()} | readings: {len(d):,} | "
      f"bad (more than {BAD_M * 100:.0f} cm off): {d.bad.mean():.0%}")
g = d.merge(lv[["day", "lake_id", "level_dept_only_m"]].dropna(), left_on=["lake_id", "date"], right_on=["lake_id", "day"])
g["gauge_resid"] = g.wse - g.level_dept_only_m
g["gauge_resid"] -= g.groupby("lake_id").gauge_resid.transform("median")
print(f"label check at the 5 big reservoirs ({len(g)} readings): correlation {np.corrcoef(g.resid, g.gauge_resid)[0, 1]:.2f}; "
      f"same good/bad call {((g.resid.abs() > BAD_M) == (g.gauge_resid.abs() > BAD_M)).mean():.0%}")

d["risk"] = np.nan
for a, b in GroupKFold(10).split(d, groups=d.lake_id):
    clf = lgb.LGBMClassifier(**PARAMS).fit(d.iloc[a][FEATS], d.iloc[a].bad)
    d.loc[d.index[b], "risk"] = clf.predict_proba(d.iloc[b][FEATS])[:, 1]
imp = pd.Series(lgb.LGBMClassifier(**PARAMS).fit(d[FEATS], d.bad).booster_.feature_importance("gain"), index=FEATS)
imp = imp.sort_values(ascending=False)
print("what the model uses most:", ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(6).items()))

RULES = [("SWOT flag: good only (quality_f = 0)", lambda x: x.quality_f == 0),
         ("SWOT flag: good or suspect (quality_f <= 1)", lambda x: x.quality_f <= 1),
         ("our rule (flags + crossover + dark water + 3.5 MAD)", lambda x: x.our_rule)]


def matched_keep(x, rule, score):
    """Inside each reservoir, keep as many readings as the rule keeps there, choosing the lowest scores."""
    keep = []
    for _, grp in x.groupby("lake_id"):
        keep += list(grp.sort_values(score).index[:int(rule(grp).sum())])
    return x.loc[keep]


def stats(name, k, n):
    return {"method": name, "kept": len(k), "kept_share": round(len(k) / n, 2), "bad_in_kept": round(k.bad.mean(), 3),
            "mae_kept_cm": round(k.resid.abs().mean() * 100, 1), "p90_kept_cm": round(k.resid.abs().quantile(.9) * 100, 1)}


out = [stats("all readings", d, len(d))]
for name, rule in RULES:
    out.append(stats(name, d[rule(d)], len(d)))
    out.append(stats("  trust model, same number kept in each reservoir", matched_keep(d, rule, "risk"), len(d)))
    out.append(stats("  wse_u alone, same number kept in each reservoir", matched_keep(d, rule, "wse_u"), len(d)))
res = pd.DataFrame(out)
with pd.option_context("display.width", 200):
    print(res.to_string(index=False))
au = []
for _, grp in d.groupby("lake_id"):
    if grp.bad.nunique() == 2:
        au.append({"model": roc_auc_score(grp.bad, grp.risk), "wse_u": roc_auc_score(grp.bad, grp.wse_u),
                   "quality_f": roc_auc_score(grp.bad, grp.quality_f) if grp.quality_f.nunique() > 1 else np.nan})
au = pd.DataFrame(au)
print("\nAUC inside each reservoir (median):", au.median().round(2).to_dict(),
      "| model better than quality_f in", f"{(au.model > au.quality_f).mean():.0%}", "of reservoirs")
print("AUC over all readings: model", round(roc_auc_score(d.bad, d.risk), 2),
      "| wse_u alone", round(roc_auc_score(d.bad, d.wse_u), 2))
small = d[d.ref_area_km2 < 2]
rule = RULES[1][1]
print(f"small reservoirs (< 2 km2, {small.lake_id.nunique()}): quality_f <= 1 keeps error {small[rule(small)].resid.abs().mean() * 100:.1f} cm"
      f" | trust model, same number per reservoir: {matched_keep(small, rule, 'risk').resid.abs().mean() * 100:.1f} cm")
q2 = d[d.quality_f == 2]
print(f"readings SWOT calls bad (quality_f = 2): {len(q2)}; within {BAD_M * 100:.0f} cm of the curve: {(~q2.bad).mean():.0%}")
res.to_csv("b_results.csv", index=False)
d.to_csv("b_readings_labelled.csv", index=False)

# learning curve, judged the fair way
lakes, rng, lc = d.lake_id.unique(), np.random.default_rng(0), []
for rep in range(5):
    perm = rng.permutation(lakes)
    for test_lakes in np.array_split(perm, 5):
        te = d[d.lake_id.isin(test_lakes)].copy()
        pool = np.setdiff1d(perm, test_lakes)
        for n in (5, 10, 20, 40, len(pool)):
            tr = d[d.lake_id.isin(rng.choice(pool, size=n, replace=False))]
            te["r"] = lgb.LGBMClassifier(**PARAMS).fit(tr[FEATS], tr.bad).predict_proba(te[FEATS])[:, 1]
            row = {"train_reservoirs": n if n < 60 else "all (67-68)", "train_readings": len(tr)}
            for key, rl in (("q0", RULES[0][1]), ("q01", RULES[1][1])):
                row[f"model_cm_{key}"] = matched_keep(te, rl, "r").resid.abs().mean() * 100
                row[f"flag_cm_{key}"] = te[rl(te)].resid.abs().mean() * 100
            lc.append(row)
lc = pd.DataFrame(lc).groupby("train_reservoirs", sort=False).mean().round(1)
lc.to_csv("b_learning_curve.csv")
print("\nlearning curve (mean error of kept readings, cm; same number kept per reservoir as the flag):")
print(lc.to_string())
