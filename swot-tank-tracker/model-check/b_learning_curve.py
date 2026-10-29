"""Member B: is about 4,700 readings at 85 reservoirs enough to train the trust model?
Learning curve: hold out a fifth of the reservoirs, train on 5, 10, 20, 40 or all 68 of the rest,
and score the held-out ones. Repeated 5 times with different random splits."""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
FEATS = ["quality_f", "xovr_cal_q", "dark_frac", "partial_f", "xtrk_km", "wse_u", "area_ratio", "log_ref_area",
         "mon_sin", "mon_cos", "nb_dev", "nb_n"]
d = pd.read_csv("b_readings_labelled.csv", dtype={"lake_id": str})
lakes = d.lake_id.unique()
rng = np.random.default_rng(0)
rows = []
for rep in range(5):
    perm = rng.permutation(lakes)
    for test_lakes in np.array_split(perm, 5):
        te = d[d.lake_id.isin(test_lakes)]
        pool = np.setdiff1d(perm, test_lakes)
        flag_kept = te[te.quality_f == 0]
        for n in (5, 10, 20, 40, len(pool)):
            tr = d[d.lake_id.isin(rng.choice(pool, size=n, replace=False))]
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                     subsample=0.8, subsample_freq=1, verbose=-1).fit(tr[FEATS], tr.bad)
            risk = clf.predict_proba(te[FEATS])[:, 1]
            kept = te.iloc[np.argsort(risk)[:len(flag_kept)]]
            rows.append({"train_reservoirs": n, "train_readings": len(tr), "train_bad": int(tr.bad.sum()),
                         "auc_model": roc_auc_score(te.bad, risk), "auc_swot_flag": roc_auc_score(te.bad, te.quality_f),
                         "error_kept_model_cm": kept.resid.abs().mean() * 100,
                         "error_kept_swot_flag_cm": flag_kept.resid.abs().mean() * 100})
r = pd.DataFrame(rows)
s = r.groupby("train_reservoirs").agg(train_readings=("train_readings", "mean"), train_bad=("train_bad", "mean"),
                                      auc_model=("auc_model", "mean"), auc_low=("auc_model", lambda x: x.quantile(.1)),
                                      auc_swot_flag=("auc_swot_flag", "mean"),
                                      error_kept_model_cm=("error_kept_model_cm", "mean"),
                                      error_kept_swot_flag_cm=("error_kept_swot_flag_cm", "mean")).round(2)
s.to_csv("b_learning_curve.csv")
print("25 tests per row (5 random splits x 5 held-out groups of about 17 reservoirs)")
print(s.to_string())
per_month = d.groupby(pd.to_datetime(d.date).dt.to_period("M")).size()
print("\nreadings per month in this set (median over months):", int(per_month.median()),
      "| months covered:", len(per_month))
