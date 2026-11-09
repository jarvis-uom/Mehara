r"""Scores series 2's development batch 1 by DEV-6's keep rules (fixed 5 Nov 2026, before any result).

Each notebook's output is downloaded to runs/dev6/<variant>/ (fold1/, fold2/). The scores, each the mean of folds 1
and 2:
- M1: S1S2-Water pooled IoU over the 5 held-out training scenes and the 5 Asian scenes (all 10 pooled per fold);
- M2: Dynamic World IoU on the held-out reservoirs' first-half clear chips, pooled and the median over reservoirs;
- M3: Sen1Floods11 validation-split IoU (89 chips);
- M4: storage error (% of capacity), the median over reservoirs. Per reservoir: the first-half passes kept by rule 3's
  filter (storage on record within 2 days, radar gap at most 5%), today's log-log line fitted on the first quarter
  (before q2_from) and scored on the second (q2_from to later_from), the median absolute error over capacity.
Then the keep rules: long against the control; lovasz against long if long is kept, else against the control.
Beside the rules (not part of them): M4's paired differences by reservoir, both folds pooled.
Writes runs/dev6/dev6_scores.csv and dev6_verdict.csv.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\score_dev6.py
"""
import csv
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/dev6"
VARIANTS = ("control", "long", "lovasz")
sys.stdout.reconfigure(encoding="utf-8")

GAIN = {"M1": 0.5, "M2 median": 0.3, "M3": 1.0, "M4": -0.2}  # DEV-6: gain if at least this (M4: down by 0.2)
LOSS = {"M1": -0.5, "M2 pooled": -0.3, "M2 median": -0.3, "M3": -1.0, "M4": 0.2}  # a loss if this far or further


def load_kernel():
    spec = importlib.util.spec_from_file_location("retrain3", ROOT / "kernel3/retrain.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    return K


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iou(tp, fp, fn):
    return 100 * tp / (tp + fp + fn) if tp + fp + fn else float("nan")


def counts(rows):
    return [sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn")]


def storage(K, fold_dir, variant, split, info, series):
    """Per reservoir: second-quarter storage error of the log-log line fitted on the first quarter."""
    out = {}
    for f in sorted(fold_dir.glob("areas_passes/*.csv")):
        rows = read(f)
        if not rows:
            continue
        tank = rows[0]["tank"]
        sp = split.get(tank)
        if not sp or not sp["later_from"]:
            continue
        s = series.get(tank, {})
        keep = []
        for r in rows:
            if r["date"] >= sp["later_from"] or float(r["radar_gap_share"]) > 0.05:
                continue
            st = K.nearest(s, date.fromisoformat(r["date"]))
            if st is not None:
                keep.append((r["date"], float(r[f"{variant}6, water"]), st))
        keep.sort()
        q1 = [k for k in keep if k[0] < sp["q2_from"]]
        q2 = [k for k in keep if k[0] >= sp["q2_from"]]
        if len(q1) < 3 or not q2:
            continue
        f_line = K.fit(np.array([k[1] for k in q1]), np.array([k[2] for k in q1]))
        if f_line is None:
            continue
        a2, s2 = np.array([k[1] for k in q2]), np.array([k[2] for k in q2])
        out[tank] = 100 * float(np.median(np.abs(f_line(a2) - s2))) / info[tank]["cap"]
    return out


def score(K, variant, split, info, series):
    folds, per_res = [], {}
    for k in (1, 2):
        d, i = RUNS / variant / f"fold{k}", k - 1
        s1 = read(d / f"dev_s1s2_fold{i}.csv")
        fl = read(d / f"dev_flood_fold{i}.csv")
        dw = read(d / f"dw_counts_fold{i}.csv")
        per = {}
        for r in dw:
            v = per.setdefault(r["reservoir"], [0, 0, 0])
            for j, c in enumerate(counts([r])):
                v[j] += c
        st = storage(K, d, variant, split, info, series)
        per_res.update(st)
        folds.append({"fold": k, "scenes": len(s1), "M1": iou(*counts(s1)),
                      "M1 held-out training": iou(*counts([r for r in s1 if r["set"] == "held-out training"])),
                      "M1 Asian": iou(*counts([r for r in s1 if r["set"] != "held-out training"])),
                      "M2 pooled": iou(*counts(dw)),
                      "M2 median": float(np.median([iou(*v) for v in per.values()])),
                      "M2 chips": len(dw), "M3": iou(*counts(fl)),
                      "M4": float(np.median(list(st.values()))), "M4 reservoirs": len(st)})
    mean = {m: float(np.mean([f[m] for f in folds])) for m in folds[0] if m != "fold"}
    return folds, mean, per_res


def judge(a, b):
    """Is a kept against b? Differences a - b, the gain and no-loss checks."""
    d = {m: a[m] - b[m] for m in ("M1", "M2 pooled", "M2 median", "M3", "M4")}
    gains = [m for m, g in GAIN.items() if (d[m] <= g if g < 0 else d[m] >= g)]
    losses = [m for m, l in LOSS.items() if (d[m] >= l if l > 0 else d[m] <= l)]
    return d, gains, losses, bool(gains) and not losses


def main():
    K = load_kernel()
    info, series, _ = K.sheet(ROOT / "kaggle-data/tables")
    split = {r["reservoir"]: r for r in read(ROOT / "kernel6-dev/split_dates.csv")}
    res, rows, per = {}, [], {}
    for v in VARIANTS:
        folds, mean, per[v] = score(K, v, split, info, series)
        res[v] = mean
        for f in folds + [{"fold": "mean", **mean}]:
            rows.append({"variant": v, **{k: round(x, 3) if isinstance(x, float) else x for k, x in f.items()}})
    K.write(RUNS / "dev6_scores.csv", rows)
    for r in rows:
        print(r)
    verdict = []
    d, g, l, kept = judge(res["long"], res["control"])
    verdict.append(("long vs control", d, g, l, kept))
    base = "long" if kept else "control"
    d2, g2, l2, kept2 = judge(res["lovasz"], res[base])
    verdict.append((f"lovasz vs {base}", d2, g2, l2, kept2))
    out = []
    for name, d, g, l, k in verdict:
        print(f"\n{name}: " + ", ".join(f"{m} {x:+.2f}" for m, x in d.items()))
        print(f"  gains: {g or 'none'}; losses: {l or 'none'} -> {'KEPT' if k else 'not kept'}")
        out.append({"comparison": name, **{m: round(x, 3) for m, x in d.items()}, "gains": "; ".join(g),
                    "losses": "; ".join(l), "kept": k})
    for a, b in (("long", "control"), ("lovasz", "control"), ("lovasz", "long")):
        names = sorted(set(per[a]) & set(per[b]))
        dd = np.array([per[a][n] - per[b][n] for n in names])
        print(f"beside: M4 {a} - {b} by reservoir: median {np.median(dd):+.2f}, better at "
              f"{int((dd < 0).sum())} of {len(dd)}")
    K.write(RUNS / "dev6_verdict.csv", out)


if __name__ == "__main__":
    main()
