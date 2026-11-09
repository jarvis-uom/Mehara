r"""Step 3 of the curve study: the morning passes, exactly as PROTOCOL-CURVE.md's addendum (commit ad31f5e) says.

Every candidate uses L4 (the step 1 and 2 winner, no smoothing) and is scored on the main orbit's second-quarter
passes, as steps 1 and 2: O0 main only; O1 one curve on both orbits' first-quarter passes; O2 as O1 with the morning
areas scaled to the main orbit (median main / morning area over first-quarter pairs at most 6 days apart; 1 with
fewer than 3 pairs); O3 O2's curve with each main reading averaged with the latest morning reading in the 6 days
before it. The simplest within 0.1 points of the best is kept. Reservoirs without first-quarter morning passes use O0.
Reads the morning areas from runs/morning-areas/areas_morning (the Kaggle notebook's output).
Writes runs/e1/e1_morning_reservoirs.csv and e1_morning_summary.csv.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\curve_morning.py
"""
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MORNING = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/morning-areas/areas_morning"
CANDS = ("O0", "O1", "O2", "O3")  # simplest first
PAIR_DAYS = 6
sys.stdout.reconfigure(encoding="utf-8")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def l4(K, C, a, s, cap):
    """L4 as step 1 (capped hinged line), with L1 where no hinge qualifies; None if not even L0 can be fitted."""
    h = C.hinged(a, s)
    if h:
        return C.capped(h, cap)
    f0 = K.fit(a, s)
    return C.capped(f0, cap) if f0 else None


def points(K, C, path, sp, s):
    pts = []
    for r in C.read(path) if path.exists() else []:
        if r["date"] >= sp["later_from"] or float(r["radar_gap_share"]) > 0.05 or r[C.COL] == "":
            continue
        st = K.nearest(s, date.fromisoformat(r["date"]))
        if st is not None:
            pts.append((r["date"], float(r[C.COL]), st))
    return sorted(pts)


def ordinal(d):
    return date.fromisoformat(d).toordinal()


def main():
    C = load("curve_study", ROOT / "scripts/curve_study.py")
    K = C.load_kernel()
    info, series, _ = K.sheet(ROOT / "kaggle-data/tables")
    split = {r["reservoir"]: r for r in C.read(ROOT / "kernel6-dev/split_dates.csv") if r["later_from"]}
    if not MORNING.is_dir():
        raise SystemExit(f"no morning areas at {MORNING}")
    rows, beside = [], []
    for f in sorted(C.RUN5.glob("part*/fold*/areas_passes/*.csv")):
        first = C.read(f)
        if not first or first[0]["tank"] not in split:
            continue
        tank = first[0]["tank"]
        sp, s, cap = split[tank], series.get(tank, {}), info[tank]["cap"]
        main_pts = points(K, C, f, sp, s)
        morn = points(K, C, MORNING / f.name, sp, s)
        mq1 = [p for p in main_pts if p[0] < sp["q2_from"]]
        mq2 = [p for p in main_pts if p[0] >= sp["q2_from"]]
        oq1 = [p for p in morn if p[0] < sp["q2_from"]]
        oq2 = [p for p in morn if p[0] >= sp["q2_from"]]
        if len(mq1) < 3 or not mq2:
            continue
        arr = lambda pts, i: np.array([p[i] for p in pts], dtype=float)  # noqa: E731
        f0 = l4(K, C, arr(mq1, 1), arr(mq1, 2), cap)
        if f0 is None:
            continue
        truth, a2 = arr(mq2, 2), arr(mq2, 1)
        score = lambda pred: 100 * float(np.median(np.abs(pred - truth))) / cap  # noqa: E731
        r = {"reservoir": tank, "main_q1": len(mq1), "main_q2": len(mq2), "morning_q1": len(oq1), "morning_q2": len(oq2)}
        r["O0"] = score(f0(a2))
        # the factor: each main first-quarter pass with its nearest morning first-quarter pass at most 6 days away
        ratios = []
        for p in mq1:
            near = [q for q in oq1 if abs(ordinal(q[0]) - ordinal(p[0])) <= PAIR_DAYS and q[1] > 0]
            if near:
                q = min(near, key=lambda q: abs(ordinal(q[0]) - ordinal(p[0])))
                ratios.append(p[1] / q[1])
        factor = float(np.median(ratios)) if len(ratios) >= 3 else 1.0
        r.update({"pairs": len(ratios), "factor": factor})
        f1 = l4(K, C, np.r_[arr(mq1, 1), arr(oq1, 1)], np.r_[arr(mq1, 2), arr(oq1, 2)], cap) if oq1 else None
        f2 = l4(K, C, np.r_[arr(mq1, 1), factor * arr(oq1, 1)], np.r_[arr(mq1, 2), arr(oq1, 2)], cap) if oq1 else None
        if f1 is None or f2 is None:
            r.update({"O1": r["O0"], "O2": r["O0"], "O3": r["O0"], "morning_used": False})
        else:
            r["O1"] = score(f1(a2))
            p2 = f2(a2)
            r["O2"] = score(p2)
            p3 = p2.copy()
            for i, p in enumerate(mq2):
                d = ordinal(p[0])
                prev = [q for q in morn if d - PAIR_DAYS <= ordinal(q[0]) <= d]
                if prev:
                    p3[i] = (p2[i] + float(f2(np.array([factor * prev[-1][1]]))[0])) / 2
            r["O3"] = score(p3)
            r["morning_used"] = True
            if oq2:
                beside.append(100 * float(np.median(np.abs(f2(factor * arr(oq2, 1)) - arr(oq2, 2)))) / cap)
        rows.append(r)
    overall = {k: float(np.median([r[k] for r in rows])) for k in CANDS}
    win, best = C.choose(overall, CANDS)
    summ = []
    for k in CANDS:
        dd = np.array([r[k] - r["O0"] for r in rows])
        summ.append({"candidate": k, "overall": round(overall[k], 3), "median_d_vs_O0": round(float(np.median(dd)), 3),
                     "better_than_O0": int((dd < 0).sum()), "worse_than_O0": int((dd > 0).sum()), "chosen": k == win})
    summ.append({"candidate": "beside: O2 on the morning orbit's own second-quarter passes",
                 "overall": round(float(np.median(beside)), 3) if beside else "", "median_d_vs_O0": "",
                 "better_than_O0": f"{len(beside)} reservoirs", "worse_than_O0": "", "chosen": ""})
    K.write(C.OUT / "e1_morning_reservoirs.csv",
            [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in rows])
    K.write(C.OUT / "e1_morning_summary.csv", summ)
    used = sum(r["morning_used"] for r in rows)
    print(f"{len(rows)} reservoirs, {used} with morning passes used; median factor "
          f"{np.median([r['factor'] for r in rows if r['morning_used']]):.3f}")
    for r in summ:
        print(r)
    print(f"step 3: {win} (best {best:.3f})")


if __name__ == "__main__":
    main()
