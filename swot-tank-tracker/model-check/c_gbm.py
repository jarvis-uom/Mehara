"""Member C pilot: does a learned model forecast storage better than 'no change' and the seasonal average?
Gradient boosting (LightGBM) on the department's daily storage, ERA5 rain and a seasonal average from the training years.
Time split: train on targets before 1 Apr 2025, test from 1 Apr 2025 to 27 Oct 2026.
'Unseen tanks' test: 5-fold by reservoir, so each test reservoir was never in training."""
import sys
import warnings

sys.path.insert(0, ".")
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from c_prep import HORIZONS, features, panel

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
BASE = ["f0", "d1", "d3", "d7", "d14", "d28", "lta0", "lta_h", "dlta", "anom", "doy_sin", "doy_cos", "log_cap", "major"]
RAIN = ["rain3", "rain7", "rain30", "rain90"]
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.9, objective="l1", verbose=-1)


def fit_predict(tr, te, cols):
    m = lgb.LGBMRegressor(**PARAMS).fit(tr[cols], tr.y)
    return m.predict(te[cols]), m


p = panel()
print("panel:", len(p), "reservoir-days;", p.groupby("kind").reservoir.nunique().to_dict())
results, preds = [], []
for h in HORIZONS:
    t = features(p, h)
    tr, te = t[~t.test], t[t.test].copy()
    te["persist"] = 0.0
    te["seasonal_change"] = te.dlta.fillna(0)
    te["seasonal_avg"] = (te.lta_h - te.f0).fillna(0)
    te["gbm"], m = fit_predict(tr, te, BASE + RAIN)
    te["gbm_no_rain"], _ = fit_predict(tr, te, BASE)
    te["gbm_future_rain"], _ = fit_predict(tr, te, BASE + RAIN + ["rain_next"])
    te["gbm_unseen"] = np.nan
    for k, (a, b) in enumerate(GroupKFold(5).split(t, groups=t.reservoir)):
        tr_k = t.iloc[a][~t.iloc[a].test]
        te_k = t.iloc[b][t.iloc[b].test]
        te.loc[te_k.index, "gbm_unseen"] = fit_predict(tr_k, te_k, BASE + RAIN)[0]
    imp = pd.Series(m.booster_.feature_importance("gain"), index=BASE + RAIN).sort_values(ascending=False)
    print(f"\nh={h} days | train rows {len(tr):,} | test rows {len(te):,} | top features: "
          + ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(5).items()))
    big = te.y.abs() >= 0.10
    for method in ["persist", "seasonal_avg", "seasonal_change", "gbm_no_rain", "gbm", "gbm_unseen", "gbm_future_rain"]:
        err = (te[method] - te.y).abs() * 100
        row = {"h_days": h, "method": method, "mae_all": err.mean(), "mae_major": err[te.kind == "Major"].mean(),
               "mae_medium": err[te.kind == "Medium"].mean(),
               "mae_maha": err[te.date.dt.month.isin([10, 11, 12, 1, 2, 3])].mean(),
               "mae_yala": err[te.date.dt.month.isin([4, 5, 6, 7, 8, 9])].mean(),
               "mae_big_moves": err[big].mean(), "share_big_moves": big.mean()}
        results.append(row)
    te["h_days"] = h
    preds.append(te[["h_days", "reservoir", "kind", "date", "f0", "y", "persist", "seasonal_avg", "seasonal_change",
                     "gbm", "gbm_unseen", "gbm_future_rain"]])

r = pd.DataFrame(results)
base = r[r.method == "persist"].set_index("h_days").mae_all
r["skill_vs_no_change"] = 1 - r.mae_all / r.h_days.map(base)
r = r.round(3)
r.to_csv("c_results.csv", index=False)
pd.concat(preds).to_csv("c_test_predictions.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print("\nMAE = mean absolute error, in % of each reservoir's capacity (test: 1 Apr 2025 - 27 Oct 2026)")
    print(r.to_string(index=False))
