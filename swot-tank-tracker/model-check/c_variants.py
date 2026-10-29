"""Member C pilot, part 3: the harder, real case for village tanks.
(a) No seasonal average (a village tank has no record to build one): the season comes only from the date and rain.
(b) Sparse, noisy inputs: the model sees a storage reading only every 10 days, with about 3% noise, as SWOT would give.
All tested on reservoirs the model never saw (5 folds by reservoir); test period 1 Apr 2025 - 27 Oct 2026."""
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
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.9, objective="l1", verbose=-1)
FULL = ["f0", "d1", "d3", "d7", "d14", "d28", "lta0", "lta_h", "dlta", "anom", "doy_sin", "doy_cos", "log_cap", "major",
        "rain3", "rain7", "rain30", "rain90"]
NO_LTA = [c for c in FULL if c not in ("lta0", "lta_h", "dlta", "anom")]


def unseen(t, cols):
    pred = pd.Series(np.nan, index=t.index)
    for a, b in GroupKFold(5).split(t, groups=t.reservoir):
        tr, te = t.iloc[a], t.iloc[b]
        tr, te = tr[~tr.test], te[te.test]
        pred[te.index] = lgb.LGBMRegressor(**PARAMS).fit(tr[cols], tr.y).predict(te[cols])
    return pred


p = panel()
rng = np.random.default_rng(0)
sparse = p.copy()
phase = sparse.reservoir.map({r: rng.integers(10) for r in sparse.reservoir.unique()})
seen = ((sparse.date - pd.Timestamp("2020-01-01")).dt.days % 10) == phase
sparse["frac"] = (p.frac + rng.normal(0, 0.03, len(p))).clip(0, 1.3).where(seen)
sparse["frac"] = sparse.groupby(["kind", "reservoir"]).frac.ffill(limit=12)
truth = p.set_index(["kind", "reservoir", "date"]).frac

rows = []
for h in HORIZONS:
    for mode, src in (("daily, exact", p), ("every 10 days, noisy", sparse)):
        t = features(src, h)
        # the target is always the true storage h days later, from what the model knew at day t
        key = pd.MultiIndex.from_arrays([t.kind, t.reservoir, t.date + pd.Timedelta(days=h)])
        t["y"] = truth.reindex(key).values - t.f0.values
        t = t[t.y.notna()]
        te = t[t.test]
        out = {"h_days": h, "inputs": mode, "test_rows": len(te),
               "no_change": (te.y.abs() * 100).mean(),
               "seasonal_change (past-years average)": ((te.dlta.fillna(0) - te.y).abs() * 100).mean()}
        for name, cols in (("model with seasonal average", FULL), ("model without seasonal average", NO_LTA)):
            out[name] = ((unseen(t, cols)[te.index] - te.y).abs() * 100).mean()
        rows.append(out)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()})
r = pd.DataFrame(rows).round(2)
r.to_csv("c_variants.csv", index=False)
print("\nMAE in % of capacity, reservoirs never seen in training:")
print(r.to_string(index=False))
