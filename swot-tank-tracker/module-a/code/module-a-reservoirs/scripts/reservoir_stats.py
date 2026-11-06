r"""Water area against official storage at every reservoir (2 Nov 2026). The label-free check at scale, step 2 of 2.

For each reservoir measured by measure_areas.py it pairs each chip with the department's storage on the photo date
(within 2 days; temp/model-check/storage_daily.csv) and leaves out chips where more than 5% of the outline has no
radar. A radar rule is tuned for each reservoir: the VV threshold (3 x 3 median) whose area best matches the
radar + optical U-Net's area on that reservoir's clear days (storage is not used). Each radar image is used once: of
the photos that share one, the nearest in time is kept, and a cloudy photo whose radar image is also among the clear
days is left out. For each method:
- Spearman of area against storage on the clear days and on the cloudy days;
- the storage error on cloudy days when the method's own clear-day power law (storage = c * area^b) reads storage
  from the cloudy-day area (median, % of capacity). The baseline without a model reads every cloudy day as the
  median storage of the clear days.
Paired Wilcoxon tests over reservoirs compare the main methods (gain > 0: the first method is better, that is a higher
Spearman or a lower error).
Reservoirs are flagged, and left out of the summary, when the JRC water body found is far from the size the capacity
suggests (the sheet's point may have found the wrong water), when the sheet gives another reservoir the same point
(the water found belongs to one of them at most), when storage hardly changed over the chips' dates (range under 10%
of capacity), or with fewer than 6 clear or 4 cloudy days.
Writes out/reservoir_scores.csv, out/reservoir_flags.csv, out/reservoir_summary.csv, out/paired_tests.csv,
out/map_radar_unet.png and out/methods_clear_vs_cloudy.png. With --no-size-check it also keeps the reservoirs left
out only for their size and prints the results without writing anything (a sensitivity check). With --one-pass it
keeps, at each reservoir, only the chips whose radar over the water came from that reservoir's usual pass (morning
or evening; out/radar_passes.csv from check_radar_passes.py), prints the results and compares them with the saved
scores at the reservoirs that lost chips, without writing anything.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\reservoir_stats.py
"""
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_areas as A  # noqa: E402  (methods, thresholds)

sys.stdout.reconfigure(encoding="utf-8")
SHEET = REPO / "temp/model-check/storage_daily.csv"
OUT = ROOT / "out"
TUNED = "radar rule tuned per reservoir"
R6 = ["radar U-Net r6 s0", "radar U-Net r6 s1", "radar U-Net r6 s2"]
METHODS = list(A.MODELS) + [A.SWITCH, "NDWI >= -0.06", "VV < -16.75 dB", TUNED]
BASELINE = "no model (clear-day median storage)"
PAIRS = [("radar U-Net r2", "fusion U-Net r3", "spearman_cloudy"),
         ("radar U-Net r2", "NDWI >= -0.06", "spearman_cloudy"),
         ("radar U-Net r2", A.SWITCH, "spearman_cloudy"),
         ("radar U-Net r2", TUNED, "spearman_cloudy"),
         ("fusion U-Net r3", "radar U-Net r2", "spearman_clear"),
         ("fusion U-Net r3", "NDWI >= -0.06", "spearman_clear"),
         ("radar U-Net r2", "fusion U-Net r3", "cloudy_error_pct"),
         ("radar U-Net r2", BASELINE, "cloudy_error_pct")]
INK, INK2, SURFACE, BLUE, ORANGE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834", "#e6e5e0"


def sheet():
    info, series = {}, {}
    for r in csv.DictReader(open(SHEET, encoding="utf-8")):
        if not r["reservoir"]:
            continue
        info.setdefault(r["reservoir"], {"kind": r["kind"], "lat": float(r["lat"]), "lon": float(r["lon"]),
                                         "cap": float(r["capacity_acft"]) * 1233.48 / 1e6})
        if r["storage_acft"]:
            series.setdefault(r["reservoir"], {})[date.fromisoformat(r["date"])] = float(r["storage_acft"]) * 1233.48 / 1e6
    return info, series


def nearest(s, d, days=2):
    for k in range(days + 1):
        for dd in (d - timedelta(days=k), d + timedelta(days=k)):
            if dd in s:
                return s[dd]
    return None


def jrc_areas():
    out = {}
    for log in sorted((ROOT / "logs").glob("progress_part*.csv")):
        for r in csv.DictReader(open(log, encoding="utf-8")):
            if r["status"] == "done" and r["jrc_area_ha"]:
                out[r["reservoir"]] = float(r["jrc_area_ha"])
    return out


def fit(a, s):
    keep = (s > 0) & (a > 0)
    if keep.sum() < 3:
        return None
    b, c = np.polyfit(np.log(a[keep]), np.log(s[keep]), 1)
    return lambda area: np.exp(c + b * np.log(np.maximum(area, 0.01)))


def score(rows, cap, m):
    clear = [r for r in rows if r["kind"] == "clear"]
    cloudy = [r for r in rows if r["kind"] == "cloudy"]
    ac, sc = np.array([r[m] for r in clear]), np.array([r["storage"] for r in clear])
    ay, sy = np.array([r[m] for r in cloudy]), np.array([r["storage"] for r in cloudy])
    f = fit(ac, sc)
    return {"spearman_clear": stats.spearmanr(ac, sc)[0], "spearman_cloudy": stats.spearmanr(ay, sy)[0],
            "cloudy_error_pct": 100 * float(np.median(np.abs(f(ay) - sy))) / cap if f else np.nan}


def baseline(rows, cap):
    """No model: every cloudy day read as the median storage of the clear days."""
    sc = np.array([r["storage"] for r in rows if r["kind"] == "clear"])
    sy = np.array([r["storage"] for r in rows if r["kind"] == "cloudy"])
    return {"spearman_clear": np.nan, "spearman_cloudy": np.nan,
            "cloudy_error_pct": 100 * float(np.median(np.abs(np.median(sc) - sy))) / cap}


def one_per_radar_image(rows):
    """Of the photos that share a radar image keep the nearest in time; leave out a cloudy photo whose radar image is
    also among the clear days, so the clear-day fit never sees a cloudy day's radar."""
    best = {}
    for r in sorted(rows, key=lambda r: r["days_apart"]):
        best.setdefault((r["kind"], r["s1_date"]), r)
    clear = {r["s1_date"] for r in best.values() if r["kind"] == "clear"}
    return [r for r in best.values() if r["kind"] == "clear" or r["s1_date"] not in clear]


def storage_range_pct(rows, kind, cap):
    st = [r["storage"] for r in rows if r["kind"] == kind]
    return round(100 * (max(st) - min(st)) / cap, 1) if st else None


def paired(scores, used):
    out = []
    for a, b, k in PAIRS:
        va = {s["reservoir"]: s[k] for s in scores if s["method"] == a}
        vb = {s["reservoir"]: s[k] for s in scores if s["method"] == b}
        both = [r for r in used if np.isfinite(va.get(r, np.nan)) and np.isfinite(vb.get(r, np.nan))]
        d = np.array([va[r] - vb[r] for r in both]) * (-1 if k == "cloudy_error_pct" else 1)  # lower error is better
        out.append({"first": a, "second": b, "measure": k, "reservoirs": len(both),
                    "median_gain": round(float(np.median(d)), 3), "share_first_better": round(float((d > 0).mean()), 2),
                    "wilcoxon_p": float(f"{stats.wilcoxon(d)[1]:.2g}")})
    return out


def usual_passes():
    """chip file -> the pass that made its radar over the water, and reservoir -> its most common pass."""
    main_pass, tally = {}, {}
    for r in csv.DictReader(open(OUT / "radar_passes.csv", encoding="utf-8")):
        if r["set"] == "117 reservoirs":
            main_pass[r["file"]] = r["main_pass"]
            if r["main_pass"]:
                tally.setdefault(r["tank"], Counter())[r["main_pass"]] += 1
    return main_pass, {t: c.most_common(1)[0][0] for t, c in tally.items()}


def compare_saved(scores, changed):
    """The one-pass scores against the saved ones, at the reservoirs that lost chips."""
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    saved = {(r["reservoir"], r["method"]): r for r in csv.DictReader(open(OUT / "reservoir_scores.csv", encoding="utf-8"))}
    new = {(s["reservoir"], s["method"]): s for s in scores}
    print(f"\nOne pass against the saved scores, at the {len(changed)} reservoirs that lost chips:")
    for m in ("radar U-Net r2", "fusion U-Net r3", "NDWI >= -0.06"):
        for k in ("spearman_clear", "spearman_cloudy", "cloudy_error_pct"):
            pairs = [(num(saved[(r, m)][k]), num(new[(r, m)][k])) for r in sorted(changed)
                     if (r, m) in saved and (r, m) in new]
            pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
            if not pairs:
                continue
            a, b = np.array(pairs).T
            d = b - a
            p = float(f"{stats.wilcoxon(d)[1]:.2g}") if len(d) >= 5 and np.any(d != 0) else None
            print(f"   {m:16s} {k:17s} {len(d):3d} reservoirs: saved {np.median(a):.2f}, one pass {np.median(b):.2f},"
                  f" median change {np.median(d):+.3f}, p = {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-size-check", action="store_true",
                    help="keep reservoirs left out only for their size; print the results, write nothing")
    ap.add_argument("--one-pass", action="store_true",
                    help="keep only the chips whose radar came from each reservoir's usual pass; print, write nothing")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    info, series = sheet()
    jrc = jrc_areas()
    at_point = {}
    for n, v in info.items():
        at_point.setdefault((v["lat"], v["lon"]), []).append(n)
    main_pass, usual = usual_passes() if args.one_pass else ({}, {})
    scores, flags, changed = [], [], set()
    for path in sorted((ROOT / "areas").glob("*.csv")):
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        counts = np.load(path.with_name(path.stem + "_vv.npy"))
        name = rows[0]["tank"]
        res = info[name]
        keep = []
        for r, c in zip(rows, counts):
            if args.one_pass and main_pass.get(r["file"]) != usual.get(name):
                changed.add(name)
                continue
            s = nearest(series.get(name, {}), date.fromisoformat(r["s2_date"]))
            if s is None or float(r["radar_gap_share"]) > 0.05:
                continue
            row = {"kind": r["kind"], "s1_date": r["s1_date"], "days_apart": int(r["days_apart"]), "storage": s,
                   "counts": c, **{m: float(r[m]) for m in METHODS if m != TUNED}}
            keep.append(row)
        keep = one_per_radar_image(keep)
        clear = [r for r in keep if r["kind"] == "clear"]
        # the tuned radar rule: the threshold whose radar area best matches the radar + optical U-Net on clear days
        if clear:
            cost = [sum(abs(r["counts"][i] - 100 * r["fusion U-Net r3"]) for r in clear) for i in range(len(A.THRESHOLDS))]
            i_best = int(np.argmin(cost))
            for r in keep:
                r[TUNED] = r["counts"][i_best] * 0.01
        expected = 50 * res["cap"] ** 0.9  # rough area of a Sri Lankan tank of this capacity (ha)
        ratio = jrc.get(name, np.nan) / expected
        st = np.array([r["storage"] for r in keep]) if keep else np.array([0.0])
        n_clear, n_cloudy = len(clear), sum(r["kind"] == "cloudy" for r in keep)
        why = []
        if not 0.2 <= ratio <= 5 and not args.no_size_check:
            why.append(f"JRC area {jrc.get(name, 0):.0f} ha against about {expected:.0f} ha expected")
        others = [n for n in at_point[(res["lat"], res["lon"])] if n != name]
        if others:
            why.append(f"same point in the sheet as {', '.join(others)}")
        if (st.max() - st.min()) < 0.1 * res["cap"]:
            why.append("storage hardly changed")
        if n_clear < 6 or n_cloudy < 4:
            why.append(f"only {n_clear} clear and {n_cloudy} cloudy days")
        flags.append({"reservoir": name, "kind": res["kind"], "cap_mcm": round(res["cap"], 2),
                      "jrc_area_ha": jrc.get(name, ""), "expected_ha": round(expected), "clear_days": n_clear,
                      "cloudy_days": n_cloudy, "storage_range_clear_pct": storage_range_pct(keep, "clear", res["cap"]),
                      "storage_range_cloudy_pct": storage_range_pct(keep, "cloudy", res["cap"]),
                      "used": not why, "why_left_out": "; ".join(why)})
        if why:
            continue
        for m in METHODS + [BASELINE]:
            s = baseline(keep, res["cap"]) if m == BASELINE else score(keep, res["cap"], m)
            scores.append({"reservoir": name, "kind": res["kind"], "cap_mcm": round(res["cap"], 2),
                           "lat": res["lat"], "lon": res["lon"], "method": m,
                           **{k: round(v, 3) for k, v in s.items()}})
    used = sorted({s["reservoir"] for s in scores})
    print(f"{len(flags)} reservoirs measured, {len(used)} used; left out:")
    for f in flags:
        if not f["used"]:
            print(f"   {f['reservoir']}: {f['why_left_out']}")
    ranges = {k: np.median([f[f"storage_range_{k}_pct"] for f in flags if f["used"]]) for k in ("clear", "cloudy")}
    print(f"storage range over the chips' dates (median, % of capacity): clear days {ranges['clear']:.0f}, "
          f"cloudy days {ranges['cloudy']:.0f}")

    # summary: medians over reservoirs, overall and by size; the three run 6 radar seeds also as one method (mean)
    def per_method(m, subset):
        if m == "radar U-Net r6 (mean of 3 seeds)":
            vals = {k: [np.mean([s[k] for s in scores if s["reservoir"] == r and s["method"] in R6]) for r in subset]
                    for k in ("spearman_clear", "spearman_cloudy", "cloudy_error_pct")}
        else:
            vals = {k: [s[k] for s in scores if s["reservoir"] in subset and s["method"] == m]
                    for k in ("spearman_clear", "spearman_cloudy", "cloudy_error_pct")}
        out = {}
        for k, v in vals.items():
            v = [x for x in v if np.isfinite(x)]
            out[k] = round(float(np.median(v)), 2) if v else None
        return out

    sizes = {"all": used, "under 5 million m3": [r for r in used if info[r]["cap"] < 5],
             "5 to 30 million m3": [r for r in used if 5 <= info[r]["cap"] < 30],
             "30 million m3 and more": [r for r in used if info[r]["cap"] >= 30]}
    summary = []
    for group, subset in sizes.items():
        for m in METHODS + ["radar U-Net r6 (mean of 3 seeds)", BASELINE]:
            summary.append({"group": group, "reservoirs": len(subset), "method": m, **per_method(m, subset)})
    print("\nMedian over reservoirs (Spearman clear, Spearman cloudy, cloudy storage error % of capacity):")
    for group, subset in sizes.items():
        print(f"  {group} ({len(subset)} reservoirs)")
        for s in (x for x in summary if x["group"] == group):
            v = ["-" if s[k] is None else s[k] for k in ("spearman_clear", "spearman_cloudy", "cloudy_error_pct")]
            print(f"     {s['method']:44s} {v[0]!s:>6} {v[1]!s:>6} {v[2]!s:>7}")
    tests = paired(scores, used)
    print("\nPaired Wilcoxon tests over reservoirs (gain > 0: the first method is better):")
    for t in tests:
        print(f"   {t['first']} vs {t['second']}, {t['measure']}: median gain {t['median_gain']}, first better at "
              f"{t['share_first_better']:.0%} of {t['reservoirs']}, p = {t['wilcoxon_p']}")
    if args.one_pass:
        compare_saved(scores, changed)
    if args.no_size_check or args.one_pass:
        return  # a sensitivity check: nothing written
    for fname, table in (("reservoir_flags", flags), ("reservoir_scores", scores), ("reservoir_summary", summary),
                       ("paired_tests", tests)):
        write(OUT / f"{fname}.csv", table)
    plot_methods(summary)
    plot_map(scores, flags, info)


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_methods(summary):
    """One axis: median Spearman per method, clear days (blue) and cloudy days (orange)."""
    rows = [s for s in summary if s["group"] == "all" and s["method"] not in R6 and s["spearman_cloudy"] is not None]
    rows.sort(key=lambda s: s["spearman_cloudy"] if s["spearman_cloudy"] is not None else -1)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(rows))
    for s, yy in zip(rows, y):
        ax.plot([s["spearman_clear"], s["spearman_cloudy"]], [yy, yy], color=GRID, linewidth=2, zorder=1)
    ax.scatter([s["spearman_clear"] for s in rows], y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="clear days")
    ax.scatter([s["spearman_cloudy"] for s in rows], y, s=70, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="cloudy days")
    ax.set_yticks(y, [s["method"] for s in rows], fontsize=9, color=INK)
    ax.set_xlim(-0.1, 1.0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("Spearman of water area against official storage (median over reservoirs)", color=INK2)
    n = rows[0]["reservoirs"]
    ax.set_title(f"How well each method's water area follows storage, {n} reservoirs", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "methods_clear_vs_cloudy.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def island():
    """Sri Lanka's outline (USDOS LSIB, from Earth Engine), kept in out/ after the first call."""
    path = OUT / "sri_lanka.geojson"
    if not path.exists():
        import ee
        sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
        import export_chips as X
        ee.Initialize(project=X.PROJECT)
        fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "Sri Lanka"))
        path.write_text(json.dumps(fc.geometry().simplify(200).getInfo()), encoding="utf-8")
    def rings(g):
        if g["type"] == "Polygon":
            return [np.array(g["coordinates"][0])]
        if g["type"] == "MultiPolygon":
            return [np.array(p[0]) for p in g["coordinates"]]
        if g["type"] == "GeometryCollection":  # what simplify() can return
            return [r for part in g["geometries"] for r in rings(part)]
        return []
    return rings(json.loads(path.read_text(encoding="utf-8")))


def plot_map(scores, flags, info):
    """Each reservoir coloured by how well the radar-only U-Nets (run 6, mean of 3 seeds) follow storage on cloudy days."""
    by_res = {}
    for s in scores:
        if s["method"] in R6:
            by_res.setdefault(s["reservoir"], []).append(s["spearman_cloudy"])
    fig, ax = plt.subplots(figsize=(6.5, 8.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    try:
        for ring in island():
            ax.fill(ring[:, 0], ring[:, 1], color="#efeee9", edgecolor="#c9c8c2", linewidth=0.8, zorder=1)
    except Exception as e:  # the map still works without the outline
        print("no island outline:", e)
    left = [f["reservoir"] for f in flags if not f["used"]]
    ax.scatter([info[r]["lon"] for r in left], [info[r]["lat"] for r in left], s=28, facecolor="none",
               edgecolor=INK2, linewidth=0.8, zorder=2, label="left out (see reservoir_flags.csv)")
    names = sorted(by_res)
    v = [float(np.mean(by_res[r])) for r in names]
    # Blues without its white end, which would read as "left out"
    blues = matplotlib.colors.ListedColormap(plt.get_cmap("Blues")(np.linspace(0.15, 1, 256)))
    sc = ax.scatter([info[r]["lon"] for r in names], [info[r]["lat"] for r in names], c=v, cmap=blues, vmin=0, vmax=1,
                    s=60, edgecolor=INK2, linewidth=0.6, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman with storage on cloudy days", color=INK2)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"Radar-only U-Net on cloudy days, {len(names)} reservoirs", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "map_radar_unet.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
