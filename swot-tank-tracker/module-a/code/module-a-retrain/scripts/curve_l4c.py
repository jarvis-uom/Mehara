r"""Step 4 of the curve study: L4c, as PROTOCOL-CURVE.md's step 4 addendum (commit e39d36c) says.

L4c is L4 with the upper segment at least as steep as the lower (c2 >= 0; a hinge whose unconstrained c2 is negative
falls back to the plain log-log line, the exact constrained least-squares fit). Scored as step 1 (first quarter to
second, 103 reservoirs); L4c replaces L4 if its overall score is at most L4's + 0.1. Beside the choice: downhill upper
segments under L4 and L4c, the error near full, smoothing (S1, S2) and the morning passes (O1 to O3) re-scored with L4c.
The step 1 to 3 code is reused unchanged (curve_study.py, curve_morning.py); only the hinge fit is swapped here.
Writes runs/e1/e1_l4c_reservoirs.csv, e1_l4c_summary.csv and runs/e1/l4c-beside/ (the morning re-score).
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\curve_l4c.py
"""
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def hinge_fit(a, s, monotone):
    """L4's hinge search (curve_study.hinged) returning the coefficients; with monotone, c2 >= 0."""
    keep = (a > 0) & (s > 0)
    x, y = np.log(a[keep]), np.log(s[keep])
    best = None
    for h in np.percentile(x, (50, 60, 70, 80, 90)):
        if (x <= h).sum() < 3 or (x > h).sum() < 3:
            continue
        X = np.column_stack([np.ones_like(x), x, np.maximum(0, x - h)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if monotone and coef[2] < 0:
            c2, *_ = np.linalg.lstsq(X[:, :2], y, rcond=None)
            coef = np.array([c2[0], c2[1], 0.0])
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, h, coef)
    return best


def as_curve(best):
    if best is None:
        return None
    _, h, c = best

    def f(area):
        xx = np.log(np.maximum(area, 0.01))
        return np.exp(c[0] + c[1] * xx + c[2] * np.maximum(0, xx - h))
    return f


def main():
    C = load("curve_study", ROOT / "scripts/curve_study.py")
    K = C.load_kernel()
    info, series, _ = K.sheet(ROOT / "kaggle-data/tables")
    split = {r["reservoir"]: r for r in C.read(ROOT / "kernel6-dev/split_dates.csv") if r["later_from"]}
    rows, near = [], {"L4": [], "L4c": []}
    smooth_rows = []
    down = {"L4": 0, "L4c": 0}
    for f in sorted(C.RUN5.glob("part*/fold*/areas_passes/*.csv")):
        passes = C.read(f)
        if not passes or passes[0]["tank"] not in split:
            continue
        tank = passes[0]["tank"]
        sp, s, cap = split[tank], series.get(tank, {}), info[tank]["cap"]
        pts = []
        for r in passes:  # as curve_study.main
            if r["date"] >= sp["later_from"] or float(r["radar_gap_share"]) > 0.05 or r[C.COL] == "":
                continue
            st = K.nearest(s, date.fromisoformat(r["date"]))
            if st is not None:
                pts.append((r["date"], float(r[C.COL]), st))
        pts.sort()
        d = [p[0] for p in pts]
        a, st = np.array([p[1] for p in pts]), np.array([p[2] for p in pts])
        q1 = np.array([x < sp["q2_from"] for x in d])
        if q1.sum() < 3 or (~q1).sum() == 0:
            continue
        f0 = K.fit(a[q1], st[q1])
        if f0 is None:
            continue
        r = {"reservoir": tank}
        reads = {}
        for k, mono in (("L4", False), ("L4c", True)):
            best = hinge_fit(a[q1], st[q1], mono)
            if best is not None and best[2][1] + best[2][2] < 0:
                down[k] += 1
            fk = as_curve(best)
            fk = C.capped(fk, cap) if fk else C.capped(f0, cap)  # L1 where no hinge qualifies, as step 1
            reads[k] = fk(a)
            q2 = ~q1
            r[k] = 100 * float(np.median(np.abs(reads[k][q2] - st[q2]))) / cap
            full = q2 & (st >= 0.85 * cap)
            near[k].extend(list(100 * (reads[k][full] - st[full]) / cap))
        for sm in ("S1", "S2"):
            v = C.smooth(d, reads["L4c"], sm)
            r[f"L4c+{sm}"] = 100 * float(np.median(np.abs(v[~q1] - st[~q1]))) / cap
        rows.append(r)
    ov = {k: float(np.median([r[k] for r in rows])) for k in ("L4", "L4c", "L4c+S1", "L4c+S2")}
    choice = "L4c" if ov["L4c"] <= ov["L4"] + C.WITHIN else "L4"
    dd = np.array([r["L4c"] - r["L4"] for r in rows])
    summ = [{"candidate": k, "overall": round(ov[k], 3),
             "downhill_upper_segment": down.get(k, ""),
             "near_full_error": round(float(np.median(np.abs(near[k]))), 2) if k in near else "",
             "near_full_signed": round(float(np.median(near[k])), 2) if k in near else "",
             "chosen": (k == choice) if k in ("L4", "L4c") else "beside"} for k in ov]
    summ.append({"candidate": "L4c - L4 by reservoir", "overall": round(float(np.median(dd)), 3),
                 "downhill_upper_segment": f"better at {int((dd < 0).sum())}, worse at {int((dd > 0).sum())}",
                 "near_full_error": "", "near_full_signed": "", "chosen": ""})
    K.write(C.OUT / "e1_l4c_reservoirs.csv", [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
                                              for r in rows])
    K.write(C.OUT / "e1_l4c_summary.csv", summ)
    print(f"{len(rows)} reservoirs")
    for r in summ:
        print(r)
    print(f"step 4: {choice}")
    # beside: the morning passes re-scored with L4c (curve_morning.py unchanged, its hinge swapped, its output moved)
    C.hinged = lambda a, s: as_curve(hinge_fit(a, s, True))
    C.OUT = C.OUT / "l4c-beside"
    C.OUT.mkdir(parents=True, exist_ok=True)
    M = load("curve_morning", ROOT / "scripts/curve_morning.py")
    M.load = lambda name, path: C if name == "curve_study" else load(name, path)
    print("beside, the morning passes with L4c:")
    M.main()


if __name__ == "__main__":
    main()
