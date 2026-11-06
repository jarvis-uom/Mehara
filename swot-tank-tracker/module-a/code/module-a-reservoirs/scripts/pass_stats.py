r"""Water area against storage on every radar pass (2 Nov 2026): the radar-only check, step 3 of 3.

For each reservoir measured by measure_passes.py it pairs each pass with the department's storage (nearest within
2 days) and leaves out passes where more than 5% of the outline has no radar. For each method (the radar-only U-Net of
run 2 and the three run 6 seeds when measured, each also in the wide outline of measure_passes.py, and VV < -16.75 dB):
- Spearman of area against storage over all passes, and over the passes whose nearest photo (within 2 days) shows the
  water clear (at least 98%) or cloudy (under 70%);
- the storage error on later dates, as the system would run (fit on the past, then read new passes): a power law
  (storage = c * area^b) fitted on the first half of the passes by date reads storage from the area on the second
  half (median, % of capacity). The baseline with no model reads every later pass as the median storage of the first
  half.
Reservoirs are flagged as in reservoir_stats.py (water body size, same point, storage range under 10% of capacity) and
with fewer than 20 passes. Writes passes/out/pass_scores.csv, pass_flags.csv, pass_summary.csv and pass_examples.png
(storage read from the radar against the department's, for a few reservoirs), and prints the medians next to the
first check's (out/reservoir_scores.csv) at the reservoirs used in both.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\pass_stats.py
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir_stats as S  # noqa: E402  (sheet, nearest, jrc_areas, fit, write, colours)

sys.stdout.reconfigure(encoding="utf-8")
PASSES = ROOT / "passes"
OUT = PASSES / "out"
MAIN = "radar U-Net r2"
WIDE = "radar U-Net r2, wide outline"
BASELINE = "no model (earlier median storage)"
MIN_PASSES = 20
MEASURES = ("spearman_all", "spearman_clear", "spearman_cloudy", "later_error_pct")


def rho(a, s):
    return float(stats.spearmanr(a, s)[0]) if len(a) >= 5 else np.nan


def score(rows, cap, m):
    """rows are in date order."""
    a, s = np.array([r[m] for r in rows]), np.array([r["storage"] for r in rows])
    clear = np.array([r["photo_clear"] is not None and r["photo_clear"] >= 0.98 for r in rows])
    cloudy = np.array([r["photo_clear"] is not None and r["photo_clear"] < 0.7 for r in rows])
    half = len(rows) // 2
    f = S.fit(a[:half], s[:half])
    return {"spearman_all": rho(a, s), "spearman_clear": rho(a[clear], s[clear]),
            "spearman_cloudy": rho(a[cloudy], s[cloudy]),
            "later_error_pct": 100 * float(np.median(np.abs(f(a[half:]) - s[half:]))) / cap if f else np.nan,
            "passes": len(rows), "clear_passes": int(clear.sum()), "cloudy_passes": int(cloudy.sum())}


def baseline(rows, cap):
    s = np.array([r["storage"] for r in rows])
    half = len(rows) // 2
    return {"later_error_pct": 100 * float(np.median(np.abs(np.median(s[:half]) - s[half:]))) / cap,
            "passes": len(rows)}


def median(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), 2) if v else None


def plot_examples(series_by, names, info):
    names = [n for n in names if n in series_by]
    if not names:
        return
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.4 * len(names)), sharex=True, facecolor=S.SURFACE)
    for ax, name in zip(np.atleast_1d(axes), names):
        rows, cap = series_by[name], info[name]["cap"]
        d = [date.fromisoformat(r["date"]) for r in rows]
        s = np.array([r["storage"] for r in rows])
        half = len(rows) // 2
        ax.set_facecolor(S.SURFACE)
        ax.plot(d, 100 * s / cap, color=S.INK2, lw=2, label="Department's storage")
        for m, colour, label in ((MAIN, S.BLUE, "Read from the radar, outline"),
                                 (WIDE, S.ORANGE, "Read from the radar, wide outline")):
            if m not in rows[0]:
                continue
            a = np.array([r[m] for r in rows])
            f = S.fit(a[:half], s[:half])
            if f:
                ax.plot(d, 100 * f(a) / cap, color=colour, lw=0, marker="o", ms=3, label=label)
        ax.axvline(d[half], color=S.GRID, lw=1)
        ax.text(d[half], 112, "  curve fitted before this line, tested after it", color=S.INK2, fontsize=8, va="center")
        ax.set_ylim(0, 120)
        ax.set_ylabel("% of capacity", color=S.INK2)
        ax.set_title(f"{name} ({cap:.0f} million m3)", loc="left", color=S.INK, fontsize=10)
        ax.grid(axis="y", color=S.GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (2.4 * len(names))))
    fig.savefig(OUT / "pass_examples.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="Huruluwewa,Nuwara Wewa,Balaharuwa,Saddhathissa",
                    help="reservoirs for pass_examples.png")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    info, series = S.sheet()
    jrc = S.jrc_areas()
    at_point = {}
    for n, v in info.items():
        at_point.setdefault((v["lat"], v["lon"]), []).append(n)
    scores, flags, series_by, methods = [], [], {}, []
    for path in sorted((PASSES / "areas").glob("*.csv")):
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        if not rows:
            continue
        name, res = rows[0]["tank"], info[rows[0]["tank"]]
        here = [k for k in rows[0] if k.startswith("radar U-Net") or k.startswith("VV <")]
        methods += [m for m in here if m not in methods]
        keep = []
        for r in rows:
            s = S.nearest(series.get(name, {}), date.fromisoformat(r["date"]))
            if s is None or float(r["radar_gap_share"]) > 0.05:
                continue
            keep.append({"date": r["date"], "storage": s, "photo_clear": float(r["photo_clear"]) if r["photo_clear"] else None,
                         **{m: float(r[m]) for m in here}})
        keep.sort(key=lambda r: r["date"])
        expected = 50 * res["cap"] ** 0.9  # rough area of a Sri Lankan tank of this capacity (ha)
        ratio = jrc.get(name, np.nan) / expected
        st = np.array([r["storage"] for r in keep]) if keep else np.array([0.0])
        why = []
        if not 0.2 <= ratio <= 5:
            why.append(f"JRC area {jrc.get(name, 0):.0f} ha against about {expected:.0f} ha expected")
        others = [n for n in at_point[(res["lat"], res["lon"])] if n != name]
        if others:
            why.append(f"same point in the sheet as {', '.join(others)}")
        if (st.max() - st.min()) < 0.1 * res["cap"]:
            why.append("storage hardly changed")
        if len(keep) < MIN_PASSES:
            why.append(f"only {len(keep)} passes")
        flags.append({"reservoir": name, "kind": res["kind"], "cap_mcm": round(res["cap"], 2), "passes": len(keep),
                      "first": keep[0]["date"] if keep else "", "last": keep[-1]["date"] if keep else "",
                      "storage_range_pct": round(100 * (st.max() - st.min()) / res["cap"], 1),
                      "used": not why, "why_left_out": "; ".join(why)})
        if why:
            continue
        series_by[name] = keep
        for m in here + [BASELINE]:
            s = baseline(keep, res["cap"]) if m == BASELINE else score(keep, res["cap"], m)
            scores.append({"reservoir": name, "kind": res["kind"], "cap_mcm": round(res["cap"], 2), "method": m,
                           **{k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}})
    used = sorted(series_by)
    print(f"{len(flags)} reservoirs measured, {len(used)} used; left out:")
    for f in flags:
        if not f["used"]:
            print(f"   {f['reservoir']}: {f['why_left_out']}")
    print(f"passes per reservoir used (median): {median([f['passes'] for f in flags if f['used']])}")

    sizes = {"all": used, "under 5 million m3": [r for r in used if info[r]["cap"] < 5],
             "5 to 30 million m3": [r for r in used if 5 <= info[r]["cap"] < 30],
             "30 million m3 and more": [r for r in used if info[r]["cap"] >= 30]}
    summary = []
    for group, subset in sizes.items():
        for m in methods + [BASELINE]:
            mine = [s for s in scores if s["reservoir"] in subset and s["method"] == m]
            summary.append({"group": group, "reservoirs": len(subset), "method": m,
                            **{k: median([s.get(k) for s in mine]) for k in MEASURES}})
    print("\nMedian over reservoirs (Spearman all passes, clear passes, cloudy passes; later storage error, % of capacity):")
    for group, subset in sizes.items():
        print(f"  {group} ({len(subset)} reservoirs)")
        for s in (x for x in summary if x["group"] == group):
            v = ["-" if s[k] is None else s[k] for k in MEASURES]
            print(f"     {s['method']:36s} {v[0]!s:>6} {v[1]!s:>6} {v[2]!s:>6} {v[3]!s:>7}")

    tests = []
    for a, b, k in ((MAIN, "VV < -16.75 dB", "spearman_all"), (MAIN, "VV < -16.75 dB", "later_error_pct"),
                    (MAIN, BASELINE, "later_error_pct"), (WIDE, MAIN, "spearman_all"), (WIDE, MAIN, "later_error_pct"),
                    (WIDE, BASELINE, "later_error_pct")):
        va = {s["reservoir"]: s.get(k) for s in scores if s["method"] == a}
        vb = {s["reservoir"]: s.get(k) for s in scores if s["method"] == b}
        both = [r for r in used if va.get(r) is not None and vb.get(r) is not None
                and np.isfinite(va[r]) and np.isfinite(vb[r])]
        d = np.array([va[r] - vb[r] for r in both]) * (-1 if k == "later_error_pct" else 1)
        if len(d) >= 5:
            tests.append({"first": a, "second": b, "measure": k, "reservoirs": len(d),
                          "median_gain": round(float(np.median(d)), 3), "share_first_better": round(float((d > 0).mean()), 2),
                          "wilcoxon_p": float(f"{stats.wilcoxon(d)[1]:.2g}")})
    print("\nPaired Wilcoxon tests over reservoirs (gain > 0: the first method is better):")
    for t in tests:
        print(f"   {t['first']} vs {t['second']}, {t['measure']}: median gain {t['median_gain']}, first better at "
              f"{t['share_first_better']:.0%} of {t['reservoirs']}, p = {t['wilcoxon_p']}")

    first = {r["reservoir"]: r for r in csv.DictReader(open(ROOT / "out/reservoir_scores.csv", encoding="utf-8"))
             if r["method"] == MAIN}
    now = {s["reservoir"]: s for s in scores if s["method"] == MAIN}
    wide = {s["reservoir"]: s for s in scores if s["method"] == WIDE}
    both = [r for r in used if r in first]
    print(f"\nThe radar U-Net (run 2) at the {len(both)} reservoirs used in both checks (medians):")
    print(f"   first check (about 30 chips each): Spearman clear {median([float(first[r]['spearman_clear']) for r in both])},"
          f" cloudy {median([float(first[r]['spearman_cloudy']) for r in both])}")
    for label, v in (("every pass", now), ("every pass, wide outline", wide)):
        print(f"   {label}: Spearman all {median([v[r]['spearman_all'] for r in both if r in v])},"
              f" clear {median([v[r]['spearman_clear'] for r in both if r in v])},"
              f" cloudy {median([v[r]['spearman_cloudy'] for r in both if r in v])}")
    back = [r for r in used if r not in first]
    if back:
        print(f"   reservoirs scored now but not in the first check ({len(back)}): " + ", ".join(
            f"{r} {now[r]['spearman_all']:.2f}" for r in back))

    for fname, table in (("pass_flags", flags), ("pass_scores", scores), ("pass_summary", summary), ("pass_tests", tests)):
        S.write(OUT / f"{fname}.csv", table)
    plot_examples(series_by, args.examples.split(","), info)


if __name__ == "__main__":
    main()
