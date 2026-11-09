r"""E0, the error budget for series 2 (PROTOCOL-6 section B; REVIEW-2026-11-05.md step 1). 6 Nov 2026.

Where does run 5's storage error come from: the water map, or the step from water area to storage? Development data
only: each reservoir's first half (passes before later_from in kernel6-dev/split_dates.csv), today's log-log line
fitted on the first quarter (before q2_from) and scored on the second, as DEV-6's M4. Nothing here chooses anything.

Inputs: run 5's out-of-fold pass areas (runs/2026-11-05-retrain-5), the photo's water area on every clear Sentinel-2
date (out/clear_photos, NDWI inside the same outline; a date counts when at least 98% of the outline is clear) and the
department's storage (kaggle-data/tables, private; matched within 2 days as rule 3).

Per reservoir (% of capacity, medians), then the median over reservoirs:
- radar: run 5's areas, first quarter to second;
- photo: the photo areas, the same date boundaries; the "map is perfect" case as far as a photo can tell;
- same days: passes with a clear photo within 1 day, both measures fitted and scored on the same dates only;
- in-sample: the line fitted and scored on the whole first half (what the line's shape and the noise allow);
- exact day: the radar score with storage on the pass day only;
- steady: consecutive passes with storage changing by under 0.5% of capacity; the line's change between them (noise);
- by fill level and outside the first quarter's storage range: pooled second-quarter errors.
Writes runs/e0/e0_reservoirs.csv and e0_summary.csv (errors only, no storage).
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\error_budget.py
"""
import csv
import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN5 = ROOT / "runs/2026-11-05-retrain-5"
OUT = ROOT / "runs/e0"
COL = "run 5, water"
BANDS = ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.85), (0.85, 9))
sys.stdout.reconfigure(encoding="utf-8")


def load_kernel():
    spec = importlib.util.spec_from_file_location("retrain3", ROOT / "kernel3/retrain.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    return K


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def err(K, fit_pts, score_pts, cap):
    """Median absolute error (% of capacity) of the log-log line fitted on fit_pts, read at score_pts; and the
    signed errors. Points are (date, area, storage)."""
    if len(fit_pts) < 3 or not score_pts:
        return np.nan, []
    f = K.fit(np.array([p[1] for p in fit_pts]), np.array([p[2] for p in fit_pts]))
    if f is None:
        return np.nan, []
    e = 100 * (f(np.array([p[1] for p in score_pts])) - np.array([p[2] for p in score_pts])) / cap
    return float(np.median(np.abs(e))), list(e)  # the list is signed: the line's reading minus storage


def main():
    K = load_kernel()
    info, series, _ = K.sheet(ROOT / "kaggle-data/tables")
    split = {r["reservoir"]: r for r in read(ROOT / "kernel6-dev/split_dates.csv") if r["later_from"]}
    rows, pooled = [], {"band": {b: [] for b in BANDS}, "inside": [], "outside": []}
    for f in sorted(RUN5.glob("part*/fold*/areas_passes/*.csv")):
        passes = read(f)
        if not passes or passes[0]["tank"] not in split:
            continue
        tank = passes[0]["tank"]
        sp, s, cap = split[tank], series.get(tank, {}), info[tank]["cap"]
        radar, exact = [], []
        for r in passes:
            if r["date"] >= sp["later_from"] or float(r["radar_gap_share"]) > 0.05 or r[COL] == "":
                continue
            d = date.fromisoformat(r["date"])
            st = K.nearest(s, d)
            if st is not None:
                radar.append((r["date"], float(r[COL]), st))
                if d in s:
                    exact.append((r["date"], float(r[COL]), s[d]))
        radar.sort()
        photo = []
        pf = ROOT / "out/clear_photos" / f.name
        for r in read(pf) if pf.exists() else []:
            if r["date"] >= sp["later_from"] or int(r["mask_pixels"]) == 0:
                continue
            if int(r["clear"]) < 0.98 * int(r["mask_pixels"]):
                continue
            st = K.nearest(s, date.fromisoformat(r["date"]))
            if st is not None:
                photo.append((r["date"], int(r["water"]) * 0.01, st))  # 10 m pixels: 0.01 ha each
        photo.sort()
        q1 = lambda pts: [p for p in pts if p[0] < sp["q2_from"]]  # noqa: E731
        q2 = lambda pts: [p for p in pts if p[0] >= sp["q2_from"]]  # noqa: E731
        e_radar, e_list = err(K, q1(radar), q2(radar), cap)
        e_photo, _ = err(K, q1(photo), q2(photo), cap)
        # same days: a pass and a clear photo at most 1 day apart
        pd = {p[0]: p for p in photo}
        pairs = []
        for p in radar:
            d = date.fromisoformat(p[0])
            for k in (0, -1, 1):
                ph = pd.get((d + timedelta(days=k)).isoformat())
                if ph:
                    pairs.append((p, ph))
                    break
        r_same = [a for a, _ in pairs]
        p_same = [(a[0], b[1], a[2]) for a, b in pairs]  # the photo's area, the pass's storage
        e_same_r, _ = err(K, q1(r_same), q2(r_same), cap)
        e_same_p, _ = err(K, q1(p_same), q2(p_same), cap)
        e_in, _ = err(K, radar, radar, cap)
        e_exact, _ = err(K, q1(exact), q2(exact), cap)
        # steady storage: consecutive passes, storage change under 0.5% of capacity; the line's change between them
        steady = []
        if len(q1(radar)) >= 3:
            fl = K.fit(np.array([p[1] for p in q1(radar)]), np.array([p[2] for p in q1(radar)]))
            if fl:
                for a, b in zip(radar, radar[1:]):
                    if abs(a[2] - b[2]) < 0.005 * cap:
                        steady.append(100 * abs(float(fl(b[1]) - fl(a[1]))) / cap)
        # by fill level and outside the first quarter's range (second quarter, radar)
        lo = min((p[2] for p in q1(radar)), default=np.nan)
        hi = max((p[2] for p in q1(radar)), default=np.nan)
        for p, e in zip(q2(radar), e_list):
            share = p[2] / cap
            for b in BANDS:
                if b[0] <= share < b[1]:
                    pooled["band"][b].append(e)
            pooled["inside" if lo <= p[2] <= hi else "outside"].append(abs(e))
        rows.append({"reservoir": tank, "cap_mcm": round(cap, 2), "radar_q1": len(q1(radar)), "radar_q2": len(q2(radar)),
                     "photo_q1": len(q1(photo)), "photo_q2": len(q2(photo)), "same_day": len(pairs), "exact_q2": len(q2(exact)),
                     "radar": e_radar, "photo": e_photo, "same_radar": e_same_r, "same_photo": e_same_p,
                     "in_sample": e_in, "exact_day": e_exact,
                     "steady_noise": float(np.median(steady)) if len(steady) >= 3 else np.nan, "steady_pairs": len(steady),
                     "q2_near_full": float(np.mean([p[2] >= 0.85 * cap for p in q2(radar)])) if q2(radar) else np.nan,
                     "q2_outside_q1": float(np.mean([not (lo <= p[2] <= hi) for p in q2(radar)])) if q2(radar) else np.nan})
    OUT.mkdir(parents=True, exist_ok=True)
    K.write(OUT / "e0_reservoirs.csv", [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
                                        for r in rows])

    def med(key, sub=None):
        v = [r[key] for r in (sub or rows) if np.isfinite(r[key])]
        return (float(np.median(v)) if v else np.nan), len(v)

    summ = []
    for key, what in (("radar", "radar (run 5), first quarter to second"), ("photo", "photo, the same split"),
                      ("in_sample", "radar, line fitted and scored on the whole first half"),
                      ("exact_day", "radar, storage on the pass day only"),
                      ("steady_noise", "radar noise: the line's change between passes at steady storage"),
                      ("q2_near_full", "share of second-quarter passes at 85% full or more"),
                      ("q2_outside_q1", "share of second-quarter passes outside the first quarter's storage range")):
        m, n = med(key)
        summ.append({"measure": what, "median": round(m, 3), "reservoirs": n})
    both = [r for r in rows if np.isfinite(r["radar"]) and np.isfinite(r["photo"])]
    d = np.array([r["radar"] - r["photo"] for r in both])
    summ.append({"measure": "radar minus photo, paired (reservoirs with both)", "median": round(float(np.median(d)), 3),
                 "reservoirs": len(d)})
    same = [r for r in rows if np.isfinite(r["same_radar"]) and np.isfinite(r["same_photo"])]
    for key, what in (("same_radar", "same days: radar"), ("same_photo", "same days: photo")):
        summ.append({"measure": what, "median": round(float(np.median([r[key] for r in same])), 3),
                     "reservoirs": len(same)})
    for b in BANDS:
        v = pooled["band"][b]
        summ.append({"measure": f"second-quarter radar error, storage {int(b[0]*100)} to {min(int(b[1]*100), 100)}% full "
                                f"(pooled passes)", "median": round(float(np.median(np.abs(v))), 3) if v else "",
                     "reservoirs": f"{len(v)} passes"})
        summ.append({"measure": f"  the same, signed (the line's reading minus storage)",
                     "median": round(float(np.median(v)), 3) if v else "", "reservoirs": f"{len(v)} passes"})
    for k in ("inside", "outside"):
        v = pooled[k]
        summ.append({"measure": f"second-quarter radar error, {k} the first quarter's storage range (pooled passes)",
                     "median": round(float(np.median(v)), 3) if v else "", "reservoirs": f"{len(v)} passes"})
    K.write(OUT / "e0_summary.csv", summ)
    for r in summ:
        print(f"{r['median']!s:>8}  {r['measure']} [{r['reservoirs']}]")


if __name__ == "__main__":
    main()
