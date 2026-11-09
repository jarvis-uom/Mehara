r"""The curve study E1, exactly as PROTOCOL-CURVE.md (fixed 6 Nov 2026, commit 88f9e76) says.

Each reservoir's first half of run 5's out-of-fold pass areas (rule 3's filter), fitted on the first quarter and
scored on the second. Step 1 scores the curves L0 to L5, step 2 the causal smoothing S0 to S2 on the step 1 winner;
each keeps the simplest candidate within 0.1 points of the best overall score (the median over reservoirs of the
median absolute error in % of capacity). Writes runs/e1/e1_reservoirs.csv and e1_summary.csv (errors only).
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\curve_study.py
"""
import csv
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUN5 = ROOT / "runs/2026-11-05-retrain-5"
OUT = ROOT / "runs/e1"
COL = "run 5, water"
CURVES = ("L0", "L1", "L2", "L3", "L4", "L5")  # simplest first
SMOOTH = ("S0", "S1", "S2")
WITHIN = 0.1
sys.stdout.reconfigure(encoding="utf-8")


def load_kernel():
    spec = importlib.util.spec_from_file_location("retrain3", ROOT / "kernel3/retrain.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    return K


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def capped(f, cap):
    return lambda a: np.clip(f(a), 0, cap)


def hinged(a, s):
    keep = (a > 0) & (s > 0)
    x, y = np.log(a[keep]), np.log(s[keep])
    best = None
    for h in np.percentile(x, (50, 60, 70, 80, 90)):
        if (x <= h).sum() < 3 or (x > h).sum() < 3:
            continue
        X = np.column_stack([np.ones_like(x), x, np.maximum(0, x - h)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, h, coef)
    if best is None:
        return None
    _, h, c = best

    def f(area):
        xx = np.log(np.maximum(area, 0.01))
        return np.exp(c[0] + c[1] * xx + c[2] * np.maximum(0, xx - h))
    return f


def isotonic(a, s):
    """Pool-adjacent-violators on storage sorted by area; read by linear interpolation, flat beyond the range."""
    o = np.argsort(a, kind="stable")
    x, y = a[o], s[o].astype(float)
    blocks = []  # [sum, count, x_sum]
    for xi, yi in zip(x, y):
        blocks.append([yi, 1, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1] = [blocks[-1][0] + b[0], blocks[-1][1] + b[1], blocks[-1][2] + b[2]]
    xs = np.array([b[2] / b[1] for b in blocks])
    ys = np.array([b[0] / b[1] for b in blocks])
    if len(xs) < 2:
        return None
    return lambda area: np.interp(area, xs, ys)


def fit_all(K, a, s, cap):
    """Each candidate fitted on the first quarter; L1 stands in where a candidate cannot be fitted."""
    f0 = K.fit(a, s)
    if f0 is None:
        return None, []
    out = {"L0": f0, "L1": capped(f0, cap)}
    fell = []
    b, c = np.polyfit(a, s, 1) if len(set(a)) > 1 else (None, None)
    out["L2"] = capped(lambda x: c + b * x, cap) if b is not None else None
    keep = (a > 0) & (s > 0)
    if keep.sum() >= 3 and len(set(a[keep])) > 1:
        slope, icpt, *_ = stats.theilslopes(np.log(s[keep]), np.log(a[keep]))
        out["L3"] = capped(lambda x: np.exp(icpt + slope * np.log(np.maximum(x, 0.01))), cap)
    else:
        out["L3"] = None
    h = hinged(a, s)
    out["L4"] = capped(h, cap) if h else None
    iso = isotonic(a, s)
    out["L5"] = capped(iso, cap) if iso else None
    for k in CURVES:
        if out[k] is None:
            out[k] = out["L1"]
            fell.append(k)
    return out, fell


def smooth(dates, readings, kind):
    """Causal smoothing of the readings (dates sorted): S1 the mean over the last 12 days, S2 the median over 24."""
    if kind == "S0":
        return readings
    days, agg = (12, np.mean) if kind == "S1" else (24, np.median)
    d = np.array([date.fromisoformat(x).toordinal() for x in dates])
    return np.array([agg(readings[(d <= d[i]) & (d >= d[i] - days)]) for i in range(len(d))])


def choose(scores, order):
    best = min(scores[k] for k in order)
    return next(k for k in order if scores[k] <= best + WITHIN), best


def main():
    K = load_kernel()
    info, series, _ = K.sheet(ROOT / "kaggle-data/tables")
    split = {r["reservoir"]: r for r in read(ROOT / "kernel6-dev/split_dates.csv") if r["later_from"]}
    data = []
    for f in sorted(RUN5.glob("part*/fold*/areas_passes/*.csv")):
        passes = read(f)
        if not passes or passes[0]["tank"] not in split:
            continue
        tank = passes[0]["tank"]
        sp, s, cap = split[tank], series.get(tank, {}), info[tank]["cap"]
        pts = []
        for r in passes:
            if r["date"] >= sp["later_from"] or float(r["radar_gap_share"]) > 0.05 or r[COL] == "":
                continue
            st = K.nearest(s, date.fromisoformat(r["date"]))
            if st is not None:
                pts.append((r["date"], float(r[COL]), st))
        pts.sort()
        d = [p[0] for p in pts]
        a = np.array([p[1] for p in pts])
        st = np.array([p[2] for p in pts])
        q1 = np.array([x < sp["q2_from"] for x in d])
        if q1.sum() < 3 or (~q1).sum() == 0:
            continue
        fits, fell = fit_all(K, a[q1], st[q1], cap)
        if fits is None:
            continue
        data.append({"tank": tank, "cap": cap, "d": d, "s": st, "q1": q1,
                     "read": {k: fits[k](a) for k in CURVES}, "fell": fell})
    rows = []
    for x in data:
        q2, cap = ~x["q1"], x["cap"]
        r = {"reservoir": x["tank"], "q1": int(x["q1"].sum()), "q2": int(q2.sum()), "fell_back_to_L1": " ".join(x["fell"])}
        for k in CURVES:
            r[k] = 100 * float(np.median(np.abs(x["read"][k][q2] - x["s"][q2]))) / cap
        rows.append(r)
    overall = {k: float(np.median([r[k] for r in rows])) for k in CURVES}
    win1, best1 = choose(overall, CURVES)
    for x, r in zip(data, rows):
        q2 = ~x["q1"]
        for k in SMOOTH:
            sm = smooth(x["d"], x["read"][win1], k)
            r[f"{win1}+{k}"] = 100 * float(np.median(np.abs(sm[q2] - x["s"][q2]))) / x["cap"]
    overall2 = {k: float(np.median([r[f"{win1}+{k}"] for r in rows])) for k in SMOOTH}
    win2, best2 = choose(overall2, SMOOTH)
    final = f"{win1}+{win2}"

    def near_full(read_of):
        e = np.concatenate([100 * (read_of(x)[~x["q1"]] - x["s"][~x["q1"]])[x["s"][~x["q1"]] >= 0.85 * x["cap"]] / x["cap"]
                            for x in data])
        return float(np.median(np.abs(e))), float(np.median(e)), len(e)

    summ = []
    for k in CURVES:
        dd = np.array([r[k] - r["L0"] for r in rows])
        nf = near_full(lambda x, k=k: x["read"][k])
        summ.append({"step": 1, "candidate": k, "overall": round(overall[k], 3),
                     "median_d_vs_L0": round(float(np.median(dd)), 3), "better_than_L0": int((dd < 0).sum()),
                     "near_full_error": round(nf[0], 2), "near_full_signed": round(nf[1], 2), "near_full_passes": nf[2],
                     "fell_back_to_L1": sum(k in x["fell"] for x in data), "chosen": k == win1})
    for k in SMOOTH:
        dd = np.array([r[f"{win1}+{k}"] - r["L0"] for r in rows])
        nf = near_full(lambda x, k=k: smooth(x["d"], x["read"][win1], k))
        summ.append({"step": 2, "candidate": f"{win1}+{k}", "overall": round(overall2[k], 3),
                     "median_d_vs_L0": round(float(np.median(dd)), 3), "better_than_L0": int((dd < 0).sum()),
                     "near_full_error": round(nf[0], 2), "near_full_signed": round(nf[1], 2), "near_full_passes": nf[2],
                     "fell_back_to_L1": "", "chosen": k == win2})
    OUT.mkdir(parents=True, exist_ok=True)
    K.write(OUT / "e1_reservoirs.csv", [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in rows])
    K.write(OUT / "e1_summary.csv", summ)
    print(f"{len(rows)} reservoirs")
    for r in summ:
        print(r)
    print(f"step 1: {win1} (best {best1:.3f}); step 2: {final} (best {best2:.3f}); L0 {overall['L0']:.3f}")


if __name__ == "__main__":
    main()
