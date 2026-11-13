r"""Step 5 of the curve study: one season-aware correction to L4c, as PROTOCOL-CURVE.md's step 5 addendum (commit
4d3afc1, fixed 13 Nov 2026 before any candidate was scored) says.

The seasonal value m(d): the median of the reservoir's daily storage record (before q2_from, as cleaned by rule 3's
code) on the days whose day of the year is within 15 days of d's, leaving out every day within 182 days of d.
Candidates, simplest first, all capped, all reading L4c where they cannot be fitted:
- Z0: L4c;
- Z1: (1 - w) x L4c + w x m(d) everywhere, one weight per reservoir fitted on the first quarter by least squares,
  clipped to 0..1 (0 with fewer than 5 passes);
- Z2: the same blend only where L4c reads 60% of capacity or more;
- Z3: L4c's form plus one yearly wave in log storage (at least 12 first-quarter passes spanning 365 days).
Scored as steps 1 and 4 (first quarter to second, 103 reservoirs). The rule: the best candidate, or the simplest
within 0.1 of it, is kept only if it is at least 0.1 better than L4c overall and a one-sided Wilcoxon on (L4c - it) > 0
over reservoirs has p < 0.05. Beside the rule: Hodges-Lehmann gains, the weights, November to January, the error by
L4c's own reading, the seasonal value alone, and the T1 scorer's development trial for the taken candidate.
Writes runs/e1/e1_season_reservoirs.csv, e1_season_summary.csv and e1_season_t1_trial.csv.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\curve_season.py
"""
import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDS = ("Z1", "Z2", "Z3")  # simplest first
ZONE, MIN_W, MIN_Z3, SPAN_Z3, NEAR, APART = 0.60, 5, 12, 365, 15, 182
sys.stdout.reconfigure(encoding="utf-8")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def doy(d):
    return d.timetuple().tm_yday


class Season:
    """The seasonal value of one reservoir, from its daily storage record before a date."""

    def __init__(self, series, before):
        days = sorted(d for d in series if d < before)
        self.o = np.array([d.toordinal() for d in days])
        self.y = np.array([doy(d) for d in days])
        self.v = np.array([series[d] for d in days], dtype=float)

    def __call__(self, d):
        if not len(self.o):
            return np.nan
        gap = np.abs(self.y - doy(d))
        ok = (np.minimum(gap, 365 - gap) <= NEAR) & (np.abs(self.o - d.toordinal()) > APART)
        return float(np.median(self.v[ok])) if ok.any() else np.nan


def weight(s, c, m, use):
    """Least squares of the storage on the blend over the passes in `use`, clipped to 0..1."""
    use = use & np.isfinite(m)
    if use.sum() < MIN_W:
        return 0.0
    den = float(((m[use] - c[use]) ** 2).sum())
    if den == 0:
        return 0.0
    return float(np.clip(((s[use] - c[use]) * (m[use] - c[use])).sum() / den, 0, 1))


def blend(c, m, w, where):
    ok = where & np.isfinite(m)
    out = c.copy()
    out[ok] = (1 - w) * c[ok] + w * m[ok]
    return out


def wave_fit(a, s, t):
    """Z3: L4c's hinge search with a yearly wave added; b2 >= 0 as L4c. Returns a reader of (area, t) or None."""
    keep = (a > 0) & (s > 0)
    x, y, tt = np.log(a[keep]), np.log(s[keep]), t[keep]
    wave = np.column_stack([np.sin(2 * np.pi * tt), np.cos(2 * np.pi * tt)])
    best = None
    for h in np.percentile(x, (50, 60, 70, 80, 90)):
        if (x <= h).sum() < 3 or (x > h).sum() < 3:
            continue
        X = np.column_stack([np.ones_like(x), x, np.maximum(0, x - h), wave])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if coef[2] < 0:
            c4, *_ = np.linalg.lstsq(np.delete(X, 2, axis=1), y, rcond=None)
            coef = np.array([c4[0], c4[1], 0.0, c4[2], c4[3]])
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, h, coef)
    if best is None:  # no hinge qualifies: the plain log-log line plus the wave
        if len(x) < 5 or len(set(x)) < 2:
            return None
        X = np.column_stack([np.ones_like(x), x, wave])
        c4, *_ = np.linalg.lstsq(X, y, rcond=None)
        best = (0.0, 0.0, np.array([c4[0], c4[1], 0.0, c4[2], c4[3]]))
    _, h, c = best

    def f(area, t):
        xx = np.log(np.maximum(area, 0.01))
        return np.exp(c[0] + c[1] * xx + c[2] * np.maximum(0, xx - h) + c[3] * np.sin(2 * np.pi * t)
                      + c[4] * np.cos(2 * np.pi * t))
    return f


def readings(K, C, L, cap, dates, a, s, q1, season):
    """Every candidate's reading at every pass, fitted on the first quarter. dates are date objects."""
    f0 = K.fit(a[q1], s[q1])
    if f0 is None:
        return None
    fc = L.as_curve(L.hinge_fit(a[q1], s[q1], True))
    c = (C.capped(fc, cap) if fc else C.capped(f0, cap))(a)
    m = np.array([season(d) for d in dates])
    t = np.array([doy(d) / 365.25 for d in dates])
    zone = c >= ZONE * cap
    w1, w2 = weight(s, c, m, q1), weight(s, c, m, q1 & zone)
    out = {"Z0": c, "Z1": blend(c, m, w1, np.ones(len(c), bool)), "Z2": blend(c, m, w2, zone)}
    kept = q1 & (a > 0) & (s > 0)
    span = (max(d for d, k in zip(dates, kept) if k) - min(d for d, k in zip(dates, kept) if k)).days if kept.any() else 0
    fz = wave_fit(a[q1], s[q1], t[q1]) if kept.sum() >= MIN_Z3 and span >= SPAN_Z3 else None
    out["Z3"] = np.clip(fz(a, t), 0, cap) if fz else c
    hist_median = float(np.median(season.v)) if len(season.v) else np.nan
    out["seasonal alone"] = np.where(np.isfinite(m), m, hist_median)
    return out, {"w1": w1, "w2": w2, "z3_fitted": fz is not None, "m": m, "zone": zone}


def err(read, s, use, cap):
    return 100 * float(np.median(np.abs(read[use] - s[use]))) / cap


def main():
    C = load("curve_study", ROOT / "scripts/curve_study.py")
    L = load("curve_l4c", ROOT / "scripts/curve_l4c.py")
    S5 = load("score_run5", ROOT / "scripts/score_run5.py")
    T = load("score_t1", ROOT / "scripts/score_t1.py")
    K = C.load_kernel()
    info, series, _ = K.sheet(ROOT / "kaggle-data/tables")
    split = {r["reservoir"]: r for r in C.read(ROOT / "kernel6-dev/split_dates.csv") if r["later_from"]}
    keys = ("Z0",) + CANDS + ("seasonal alone",)
    rows, t1_rows, pool = [], [], {k: [] for k in keys}
    pool_c, pool_month = [], []
    for f in sorted(C.RUN5.glob("part*/fold*/areas_passes/*.csv")):
        passes = C.read(f)
        if not passes or passes[0]["tank"] not in split:
            continue
        tank = passes[0]["tank"]
        sp, s_all, cap = split[tank], series.get(tank, {}), info[tank]["cap"]
        pts = []
        for r in passes:  # as curve_study.main
            if r["date"] >= sp["later_from"] or float(r["radar_gap_share"]) > 0.05 or r[C.COL] == "":
                continue
            st = K.nearest(s_all, date.fromisoformat(r["date"]))
            if st is not None:
                pts.append((r["date"], float(r[C.COL]), st))
        pts.sort()
        dates = [date.fromisoformat(p[0]) for p in pts]
        a, s = np.array([p[1] for p in pts]), np.array([p[2] for p in pts])
        q1 = np.array([p[0] < sp["q2_from"] for p in pts])
        if q1.sum() < 3 or (~q1).sum() == 0:
            continue
        season = Season(s_all, date.fromisoformat(sp["q2_from"]))
        got = readings(K, C, L, cap, dates, a, s, q1, season)
        if got is None:
            continue
        read, note = got
        q2 = ~q1
        r = {"reservoir": tank, "q1": int(q1.sum()), "q2": int(q2.sum()), "w_Z1": round(note["w1"], 3),
             "w_Z2": round(note["w2"], 3), "Z3_fitted": note["z3_fitted"],
             "q2_without_seasonal_value": int((q2 & ~np.isfinite(note["m"])).sum())}
        for k in keys:
            r[k] = err(read[k], s, q2, cap)
            pool[k].extend(list(100 * np.abs(read[k][q2] - s[q2]) / cap))
        pool_c.extend(list(read["Z0"][q2] / cap))
        pool_month.extend(d.month for d, k in zip(dates, q2) if k)
        wet = q2 & np.array([d.month in (11, 12, 1) for d in dates])
        if wet.sum() >= 3:
            for k in keys:
                r[f"{k} Nov-Jan"] = err(read[k], s, wet, cap)
        rows.append(r)
        # the T1 scorer's development trial (score_t1.py --dev): gap-free main-orbit passes, T1's truth rule
        orbit = T.main_orbit(passes)
        q1_end = (date.fromisoformat(sp["q2_from"]) - timedelta(days=1)).isoformat()
        hist = T.history_points(K, C, passes, s_all, q1_end)
        win = [(x["date"], float(x[C.COL]), T.truth_on(s_all, date.fromisoformat(x["date"])))
               for x in passes if sp["q2_from"] <= x["date"] < sp["later_from"] and x["orbit"] == orbit
               and float(x["radar_gap_share"]) == 0 and x[C.COL] != ""]
        win = [x for x in win if x[2] is not None]
        if len(hist) >= 3 and len(win) >= T.MIN_DATES:
            hd = [date.fromisoformat(p[0]) for p in hist] + [date.fromisoformat(p[0]) for p in win]
            ha = np.array([p[1] for p in hist] + [p[1] for p in win])
            hs = np.array([p[2] for p in hist] + [p[2] for p in win])
            hq = np.array([True] * len(hist) + [False] * len(win))
            f0 = K.fit(ha[hq], hs[hq])
            got = readings(K, C, L, cap, hd, ha, hs, hq, season) if f0 is not None else None
            if got is not None:
                tr = {"reservoir": tank, "scored_dates": len(win), "I": err(f0(ha), hs, ~hq, cap)}
                for k in ("Z0",) + CANDS:
                    tr[k] = err(got[0][k], hs, ~hq, cap)
                t1_rows.append(tr)

    ov = {k: float(np.median([r[k] for r in rows])) for k in keys}
    best = min(CANDS, key=lambda k: ov[k])
    taken = next(k for k in CANDS if ov[k] <= ov[best] + C.WITHIN)
    z0 = np.array([r["Z0"] for r in rows])
    summ = []
    for k in keys:
        d = z0 - np.array([r[k] for r in rows])
        hl = S5.hodges_lehmann(d) if k != "Z0" else (0.0, 0.0, 0.0)
        wet = [(r["Z0 Nov-Jan"], r[f"{k} Nov-Jan"]) for r in rows if "Z0 Nov-Jan" in r]
        summ.append({"candidate": k, "overall": round(ov[k], 3), "gain_over_L4c": round(ov["Z0"] - ov[k], 3),
                     "HL_gain": round(hl[0], 3), "HL_CI": f"{hl[1]:+.3f} to {hl[2]:+.3f}",
                     "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                     "p_better": S5.one_sided(d) if k != "Z0" else "",
                     "Nov-Jan (median over reservoirs)": round(float(np.median([w[1] for w in wet])), 3),
                     "Nov-Jan reservoirs": len(wet)})
    p_taken = S5.one_sided(z0 - np.array([r[taken] for r in rows]))
    kept = bool(ov["Z0"] - ov[taken] >= C.WITHIN and p_taken < 0.05)
    K.write(C.OUT / "e1_season_reservoirs.csv",
            [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in rows])
    pc = np.array(pool_c)
    bands = (("under 60%", pc < 0.6), ("60 to 85%", (pc >= 0.6) & (pc < 0.85)), ("85% and over", pc >= 0.85))
    beside = [{"candidate": f"by L4c's reading, {name} ({int(use.sum())} passes)",
               **{k: round(float(np.median(np.array(pool[k])[use])), 2) for k in keys}} for name, use in bands]
    K.write(C.OUT / "e1_season_summary.csv", summ + [{"candidate": b["candidate"],
                                                      "overall": " | ".join(f"{k} {b[k]}" for k in keys)} for b in beside])

    print(f"{len(rows)} reservoirs; L4c {ov['Z0']:.3f}")
    for r in summ:
        print(r)
    for b in beside:
        print(b)
    w1, w2 = np.array([r["w_Z1"] for r in rows]), np.array([r["w_Z2"] for r in rows])
    print(f"weights: Z1 median {np.median(w1):.2f}, zero at {int((w1 == 0).sum())}; Z2 median {np.median(w2):.2f}, "
          f"zero at {int((w2 == 0).sum())}; Z3 fitted at {sum(r['Z3_fitted'] for r in rows)} of {len(rows)}; "
          f"second-quarter passes without a seasonal value: {sum(r['q2_without_seasonal_value'] for r in rows)}")
    print(f"the rule: best {best} ({ov[best]:.3f}), taken {taken} ({ov[taken]:.3f}; {ov['Z0'] - ov[taken]:+.3f} "
          f"against L4c, p {p_taken:.4f}) -> {'KEPT' if kept else 'not kept: L4c stays'}")
    # beside: the T1 trial for the taken candidate, by PROTOCOL-6's step 1 against I, and against L4c
    K.write(C.OUT / "e1_season_t1_trial.csv",
            [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in t1_rows])
    i, c1, z = (np.array([r[k] for r in t1_rows]) for k in ("I", "Z0", taken))
    hl_i, hl_c = S5.hodges_lehmann(i - z), S5.hodges_lehmann(c1 - z)
    print(f"beside, the T1 trial ({len(t1_rows)} reservoirs): I {np.median(i):.2f}, L4c {np.median(c1):.2f}, "
          f"{taken} {np.median(z):.2f}; {taken} against I: no-loss p {S5.one_sided(i + T.MARGIN - z):.4f}, gain p "
          f"{S5.one_sided(i - z):.4f}, HL {hl_i[0]:+.2f} ({hl_i[1]:+.2f} to {hl_i[2]:+.2f}); against L4c: HL "
          f"{hl_c[0]:+.2f} ({hl_c[1]:+.2f} to {hl_c[2]:+.2f}), p {S5.one_sided(c1 - z):.4f}")
    for k in CANDS:
        zz = np.array([r[k] for r in t1_rows])
        print(f"  T1 trial, {k}: {np.median(zz):.2f}")


if __name__ == "__main__":
    main()
