r"""T1, the prospective storage test of PROTOCOL-6 (part 1, fixed 5 Nov 2026), and its step 1 rule (C1 against I).

Written and tried on development data on 7 Nov 2026, before any window pass is measured. As PROTOCOL-6 T1 says:
- reservoirs: run 5's 103; evaluation dates: passes from 1 Nov 2026 to 31 Jan 2027 on the reservoir's main orbit (the
  orbit of its history), scored only with no radar gap inside the outline in VV and VH (radar_gap_share_both == 0);
- curves: fitted once per reservoir on all history passes to 31 Oct 2026 (radar gap at most 5%, storage within 2
  days, as rule 3), on run 5's out-of-fold areas; never refitted. I uses today's line (L0); C1 uses PROTOCOL-CURVE's
  choice (L4c: the hinged line with the upper segment at least as steep as the lower, capped);
- the truth: the department's daily sheets saved by water-data-logger (reservoir-status/major and /medium), effective
  storage against rule 3's capacity. "The sheet of the pass date" is the sheet whose Update Date is that date: on every
  complete day from 2 to 6 Nov its rows equal the department's history tab (rule 3's source) for that day at 74 of 74
  major reservoirs, while the sheet's own "Date" column lags for some reservoirs. If that sheet is missing: the mean
  of the day before and the day after if both exist, else the one that exists, else the date is not scored;
- error at a reservoir: the median absolute error over its scored dates, % of capacity; reservoirs with fewer than 3
  scored dates are left out for every pipeline and counted; a reservoir with no row at all in the logger's sheets (a
  name spelt differently there) is named in the output (a guard added on 10 Nov 2026; no rule changes);
- baselines: seasonal (median history-pass storage within 15 days of the same day of the year) and no model;
- step 1: no loss if a one-sided Wilcoxon on (I + 0.5 - C1) > 0 has p < 0.05; a gain (only then) if a one-sided
  Wilcoxon on (I - C1) > 0 has p < 0.05 and the Hodges-Lehmann estimate is at least 0.5.
Modes:
    --check-truth   compares the logger's sheets with storage_daily.csv on the days both hold (must agree);
    --dev           the trial on development data: history = each reservoir's first quarter, window = its second
                    quarter, truth from storage_daily.csv by the same rule; VV gap only (run 5's areas have no VH
                    column), so this mode uses radar_gap_share == 0;
    (no flag)       the real test, with --window <folder of window-pass areas per reservoir>. Not before Feb 2027.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\score_t1.py --dev
"""
import argparse
import csv
import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT.parent
LOGGER = TEMP / "repos/water-data-logger/reservoir-status"
START, END, HISTORY_END = "2026-11-01", "2027-01-31", "2026-10-31"
MARGIN, MIN_DATES = 0.5, 3
sys.stdout.reconfigure(encoding="utf-8")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def num(s):
    s = (s or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def logger_truth(K):
    """{reservoir: {date: storage in million m3}} from the sheets, keyed by their Update Date; the latest save wins."""
    out = {}
    for kind in ("major", "medium"):
        for f in sorted((LOGGER / kind).glob("*.csv")):  # names sort by save day, so later saves overwrite earlier ones
            with open(f, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    try:
                        m, d, y = (int(x) for x in r["Update Date"].strip().split("/"))
                        day = date(y, m, d)
                    except (ValueError, KeyError, AttributeError):
                        continue
                    v = num(r.get("Effective Storage_Acft"))
                    if v is not None:
                        out.setdefault(r["Reservoir"].strip(), {})[day] = v * K.ACFT_TO_MCM
    return out


def no_truth(lt, names):
    """A guard: run 5's reservoirs with no row at all in the logger's sheets (a name spelt differently there would
    otherwise be left out of T1 without a word)."""
    return sorted(n for n in names if not lt.get(n))


def truth_on(s, d):
    """PROTOCOL-6's rule: the pass date; else the mean of the day before and after if both; else the one; else None."""
    if d in s:
        return s[d]
    a, b = s.get(d - timedelta(days=1)), s.get(d + timedelta(days=1))
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else b


def curves(K, C, L, a, s, cap):
    """I's and C1's curves on the history points: L0, and L4c (L1 where no hinge qualifies, as the curve study)."""
    f0 = K.fit(a, s)
    if f0 is None:
        return None
    fc = L.as_curve(L.hinge_fit(a, s, True))
    return {"I": f0, "C1": C.capped(fc, cap) if fc else C.capped(f0, cap)}


def score(K, C, L, S5, reservoirs):
    """reservoirs: name -> (cap, history points [(date, area, storage)], window points [(date, area, truth or None)])."""
    rows, left_out = [], []
    for name, (cap, hist, win) in sorted(reservoirs.items()):
        win = [w for w in win if w[2] is not None]
        if len(hist) < 3 or len(win) < MIN_DATES:
            left_out.append(name)
            continue
        ha, hs = np.array([p[1] for p in hist]), np.array([p[2] for p in hist])
        cv = curves(K, C, L, ha, hs, cap)
        if cv is None:
            left_out.append(name)
            continue
        wa, wt = np.array([w[1] for w in win]), np.array([w[2] for w in win])
        r = {"reservoir": name, "history_passes": len(hist), "scored_dates": len(win)}
        for k, f in cv.items():
            r[k] = 100 * float(np.median(np.abs(f(wa) - wt))) / cap
        doy = np.array([date.fromisoformat(p[0]).timetuple().tm_yday for p in hist])
        seas = []
        for w in win:
            dd = date.fromisoformat(w[0]).timetuple().tm_yday
            near = hs[np.minimum(np.abs(doy - dd), 365 - np.abs(doy - dd)) <= 15]
            seas.append(np.median(near) if len(near) else np.median(hs))
        r["seasonal"] = 100 * float(np.median(np.abs(np.array(seas) - wt))) / cap
        r["no model"] = 100 * float(np.median(np.abs(np.median(hs) - wt))) / cap
        rows.append(r)
    i, c = np.array([r["I"] for r in rows]), np.array([r["C1"] for r in rows])
    p_ni = S5.one_sided(i + MARGIN - c)
    p_gain = S5.one_sided(i - c)
    hl = S5.hodges_lehmann(i - c)
    no_loss = p_ni < 0.05
    gain = bool(no_loss and p_gain < 0.05 and hl[0] >= MARGIN)
    summary = {"reservoirs": len(rows), "left_out": len(left_out),
               **{f"median {k}": round(float(np.median([r[k] for r in rows])), 3)
                  for k in ("I", "C1", "seasonal", "no model")},
               "p_no_loss": p_ni, "p_gain": p_gain, "HL_gain": round(hl[0], 3),
               "HL_CI": f"{hl[1]:.3f} to {hl[2]:.3f}", "no_loss": no_loss, "gain": gain,
               "pipeline": "C1" if no_loss else "I"}
    return rows, summary, left_out


def history_points(K, C, passes, s, before):
    pts = []
    for r in passes:
        if r["date"] > before or float(r["radar_gap_share"]) > 0.05 or r[C.COL] == "":
            continue
        st = K.nearest(s, date.fromisoformat(r["date"]))
        if st is not None:
            pts.append((r["date"], float(r[C.COL]), st))
    return sorted(pts)


def main_orbit(passes):
    orbits = [r["orbit"] for r in passes]
    return max(set(orbits), key=orbits.count) if orbits else None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true")
    g.add_argument("--check-truth", action="store_true")
    g.add_argument("--window", help="folder of window-pass areas, one CSV per reservoir (run 5's columns + "
                                    "radar_gap_share_both)")
    ap.add_argument("--out", default=str(ROOT / "runs/t1"))
    args = ap.parse_args()
    C = load("curve_study", ROOT / "scripts/curve_study.py")
    L = load("curve_l4c", ROOT / "scripts/curve_l4c.py")
    S5 = load("score_run5", ROOT / "scripts/score_run5.py")
    K = C.load_kernel()
    info, series, _ = K.sheet(ROOT / "kaggle-data/tables")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.check_truth:
        lt, raw = logger_truth(K), {}
        with open(ROOT / "kaggle-data/tables/storage_daily.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["reservoir"] and r["storage_acft"]:
                    raw.setdefault(r["reservoir"], {})[date.fromisoformat(r["date"])] = \
                        float(r["storage_acft"]) * K.ACFT_TO_MCM
        same = diff = 0
        bad = []
        for n, days in lt.items():
            for d, v in days.items():
                if d in raw.get(n, {}):
                    if abs(raw[n][d] - v) < 1e-6:
                        same += 1
                    else:
                        diff += 1
                        bad.append(f"{n} {d}: sheet {v:.4f}, history {raw[n][d]:.4f}")
        days = sorted({d for v in lt.values() for d in v})
        print(f"logger sheets: {len(lt)} reservoirs, Update Dates {days[0]} to {days[-1]}; "
              f"days also in storage_daily.csv: {same} equal, {diff} different")
        for b in bad[:10]:
            print("  ", b)
        names = {C.read(f)[0]["tank"] for f in C.RUN5.glob("part*/fold*/areas_passes/*.csv")}
        missing = no_truth(lt, names)
        print(f"run 5's {len(names)} reservoirs with no truth in the logger's sheets: {', '.join(missing) or 'none'}")
        return

    split = {r["reservoir"]: r for r in C.read(ROOT / "kernel6-dev/split_dates.csv") if r["later_from"]}
    lt = None if args.dev else logger_truth(K)
    reservoirs = {}
    for f in sorted(C.RUN5.glob("part*/fold*/areas_passes/*.csv")):
        passes = C.read(f)
        if not passes:
            continue
        name = passes[0]["tank"]
        cap, s, orbit = info[name]["cap"], series.get(name, {}), main_orbit(passes)
        if args.dev:
            if name not in split:
                continue
            sp = split[name]
            q1_end = (date.fromisoformat(sp["q2_from"]) - timedelta(days=1)).isoformat()
            hist = history_points(K, C, passes, s, q1_end)
            win = [(r["date"], float(r[C.COL]), truth_on(s, date.fromisoformat(r["date"])))
                   for r in passes if sp["q2_from"] <= r["date"] < sp["later_from"] and r["orbit"] == orbit
                   and float(r["radar_gap_share"]) == 0 and r[C.COL] != ""]
        else:
            hist = history_points(K, C, passes, s, HISTORY_END)
            wf = Path(args.window) / f.name
            wrows = C.read(wf) if wf.exists() else []
            tr = lt.get(name, {})
            win = [(r["date"], float(r[C.COL]), truth_on(tr, date.fromisoformat(r["date"])))
                   for r in wrows if START <= r["date"] <= END and r["orbit"] == orbit
                   and float(r["radar_gap_share_both"]) == 0 and r[C.COL] != ""]
        reservoirs[name] = (cap, hist, win)
    rows, summary, left_out = score(K, C, L, S5, reservoirs)
    if lt is not None:  # the guard: a reservoir with no truth at all is named, not only counted
        missing = no_truth(lt, reservoirs)
        summary["no_truth"] = " ".join(missing) or "none"
        if missing:
            print(f"WARNING: no truth in the logger's sheets for {len(missing)} reservoirs: {', '.join(missing)}")
    tag = "dev" if args.dev else "window"
    K.write(out / f"t1_{tag}_reservoirs.csv", [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
                                              for r in rows])
    K.write(out / f"t1_{tag}_summary.csv", [{"measure": k, "value": v} for k, v in summary.items()]
            + [{"measure": "left out", "value": " ".join(left_out)}])
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("left out:", ", ".join(left_out) or "none")


if __name__ == "__main__":
    main()
