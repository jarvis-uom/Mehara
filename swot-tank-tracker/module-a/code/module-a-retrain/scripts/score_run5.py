r"""Scores run 5 against run 3's photo-label control by the four rules of ../PROTOCOL-5.md (fixed 5 Nov 2026).

Run 5 ran on Kaggle in three parts, each downloaded to its own folder (part1/, part2/, part3/ under --run). Before
scoring, the parts are joined and checked, as PROTOCOL-5 says; the script stops unless:
- folds 1 to 5 are each found exactly once, each with its 3 weights, its Sen1Floods11 scores, its pass areas and its
  Dynamic World counts;
- each of the 20 S1S2-Water scenes has exactly 5 run 5 rows, one for each fold.
The rules are scored as score_run4.py scores run 4, with PROTOCOL-5's fixed bars:
- rule 1: S1S2-Water's 15 test scenes against the control in the S1S2-Water check;
- rule 2: Dynamic World on run 4's 931 chips (the same chips are required);
- rule 3: storage error per reservoir, pass by pass against run 3's control areas;
- rule 4: the mean of the five folds' Sen1Floods11 test-split IoU.
Reported beside the rules: the Hodges-Lehmann estimate and 95% confidence interval of the paired differences for
rules 1 and 3, and run 4 scored the same way (beside_run4.csv). Writes into --run: s1s2_summary.csv, dw_pooled.csv,
dw_reservoirs.csv, pass_scores.csv, pass_summary.csv, flood.csv, beside_run4.csv and protocol5_check.csv.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\score_run5.py
Self-check on run 4's outputs (must give run 4's own numbers; writes only into --out):
    ... score_run5.py --run runs\2026-11-04-retrain-4 --s1s2 runs\2026-11-05-run4-s1s2 --name "run 4" --tag photo4
        --out tmp\score5-selfcheck --allow-missing-weights
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT.parent
CTRL = "run 3 control"
sys.stdout.reconfigure(encoding="utf-8")

# PROTOCOL-5's fixed bars
BAR = {"1a": 68.3, "1b": 73.6, "1c": 8, "2a": 91.25, "2b": 89.35, "2c": 84.98, "3a": 8.78, "3b": 0.25, "4": 65.6}
CONTROL_DW = (91.75, 90.35, 85.98)  # the control on the 931 chips: pooled, median reservoir, low water
CONTROL_ERR = 8.28


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


def pooled(rows):
    a = [sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn")]
    return {"iou": iou(*a), "precision": 100 * a[0] / (a[0] + a[1]) if a[0] + a[1] else float("nan"),
            "recall": 100 * a[0] / (a[0] + a[2]) if a[0] + a[2] else float("nan")}


def per_reservoir(rows):
    res = {}
    for r in rows:
        v = res.setdefault(r["reservoir"], [0, 0, 0])
        for i, k in enumerate(("tp", "fp", "fn")):
            v[i] += int(r[k])
    return {n: iou(*v) for n, v in res.items()}


def one_sided(x):
    """One-sided Wilcoxon signed-rank test that the values x are above zero (unrounded)."""
    x = np.array([v for v in x if np.isfinite(v)])
    if len(x) < 5 or not np.any(x != 0):
        return float("nan")
    return float(stats.wilcoxon(x, alternative="greater")[1])


def hodges_lehmann(d, conf=0.95):
    """Hodges-Lehmann estimate of the paired differences d (the median of the Walsh averages) and its confidence
    interval from the exact null distribution of the signed-rank statistic (no correction for ties)."""
    d = np.array([v for v in d if np.isfinite(v)], dtype=float)
    n = len(d)
    i, j = np.triu_indices(n)
    w = np.sort((d[i] + d[j]) / 2)
    m = n * (n + 1) // 2
    c = np.zeros(m + 1)
    c[0] = 1
    for k in range(1, n + 1):
        c = c + np.concatenate([np.zeros(k), c[:-k]])
    cdf = np.cumsum(c) / 2.0 ** n
    ka = int(np.searchsorted(cdf, (1 - conf) / 2, side="right"))  # P(T <= ka - 1) <= alpha / 2
    return float(np.median(w)), float(w[ka]), float(w[m - ka - 1])


def fmt_hl(hl, unit=""):
    return f"{hl[0]:+.2f}{unit} (95% CI {hl[1]:+.2f} to {hl[2]:+.2f})"


def find_folds(run, tag, allow_missing_weights):
    """The fold folders under --run (directly or in part*/), checked: each of folds 1 to 5 once and complete."""
    found, problems = {}, []
    for p in sorted(run.glob("fold[1-5]")) + sorted(run.glob("part*/fold[1-5]")):
        found.setdefault(int(p.name[4:]), []).append(p)
    for k in range(1, 6):
        if len(found.get(k, [])) != 1:
            problems.append(f"fold {k}: found {len(found.get(k, []))} times")
            continue
        f, i = found[k][0], k - 1
        n_w = len(list((f / "weights").glob(f"fold{i}_{tag}_seed*.pt")))
        if n_w != 3 and not allow_missing_weights:
            problems.append(f"fold {k}: {n_w} weights, not 3")
        for need in (f"flood_fold{i}.csv", f"dw_counts_fold{i}.csv"):
            if not (f / need).exists():
                problems.append(f"fold {k}: {need} missing")
        if not list((f / "areas_passes").glob("*.csv")):
            problems.append(f"fold {k}: no pass areas")
    return {k: v[0] for k, v in found.items() if len(v) == 1}, problems


def scene_rows(dirs, name):
    """Run rows from every scenes/ folder given, joined per scene; checked: 20 scenes, each with folds 1 to 5 once."""
    rows = [r for d in dirs for f in sorted(Path(d).glob("*.csv")) for r in read(f) if r["method"] == name]
    per = {}
    for r in rows:
        per.setdefault(r["scene"], []).append(int(r["fold"]))
    problems = [f"scene {s}: folds {sorted(v)}" for s, v in sorted(per.items()) if sorted(v) != [1, 2, 3, 4, 5]]
    if len(per) != 20:
        problems.append(f"{len(per)} scenes, not 20")
    return rows, problems


def rule1(rows, check, name):
    """As score_s1s2.py's summary(): pooled IoU is the mean of the five folds' pooled IoU; a scene's IoU is the mean
    of its five folds."""
    rows = rows + [r for f in sorted((Path(check) / "scenes").glob("*.csv")) for r in read(f) if r["method"] == CTRL]
    table, per = [], {}
    for st in ("main", "extra"):
        for m in (name, CTRL):
            folds = {}
            for r in (r for r in rows if r["set"] == st and r["method"] == m):
                v = folds.setdefault(r["fold"], [0, 0, 0])
                for i, k in enumerate(("tp", "fp", "fn")):
                    v[i] += int(r[k])
                per.setdefault((st, m), {}).setdefault(r["scene"], []).append(float(r["iou"]))
            pp = [pooled([{"tp": a[0], "fp": a[1], "fn": a[2]}]) for a in folds.values()]
            sc = [float(np.mean(v)) for v in per.get((st, m), {}).values()]
            table.append({"set": st, "method": m, "scenes": len(per.get((st, m), {})), "folds": len(folds),
                          **{k: round(float(np.mean([p[k] for p in pp])), 2) for k in ("iou", "precision", "recall")},
                          "median_scene_iou": round(float(np.median([x for x in sc if np.isfinite(x)])), 2)})
    a = {s: float(np.mean(v)) for s, v in per[("main", name)].items()}
    c = {s: float(np.mean(v)) for s, v in per[("main", CTRL)].items()}
    d = np.array([a[s] - c[s] for s in sorted(set(a) & set(c))])
    get = lambda st, m, k: next(r[k] for r in table if r["set"] == st and r["method"] == m)  # noqa: E731
    n_folds, p = get("main", name, "folds"), one_sided(d)
    checks = [("1a. pooled IoU at least 68.3", get("main", name, "iou") >= BAR["1a"] and n_folds == 5,
               f"{name} {get('main', name, 'iou'):.2f} over {n_folds} folds, control {get('main', CTRL, 'iou'):.2f}"),
              ("1b. median scene IoU at least 73.6", get("main", name, "median_scene_iou") >= BAR["1b"],
               f"{name} {get('main', name, 'median_scene_iou'):.2f}, control {get('main', CTRL, 'median_scene_iou'):.2f}"),
              ("1c. better at 8 or more of the 15 scenes", int((d > 0).sum()) >= BAR["1c"] and len(d) == 15,
               f"better at {int((d > 0).sum())} of {len(d)}, median gain {np.median(d):+.2f}"),
              ("1d. one-sided Wilcoxon p < 0.05", p < 0.05, f"p = {p:.2g}")]
    hl = hodges_lehmann(d)
    extra = (f"rule 1 reported: Hodges-Lehmann gain {fmt_hl(hl, ' IoU points')}; precision "
             f"{get('main', name, 'precision'):.2f} vs {get('main', CTRL, 'precision'):.2f}, recall "
             f"{get('main', name, 'recall'):.2f} vs {get('main', CTRL, 'recall'):.2f}; 5 Asian scenes "
             f"{get('extra', name, 'iou'):.2f} vs {get('extra', CTRL, 'iou'):.2f}")
    summary = {"pooled": get("main", name, "iou"), "median_scene": get("main", name, "median_scene_iou"),
               "better": f"{int((d > 0).sum())} of {len(d)}", "p": p, "hl": hl,
               "asian": get("extra", name, "iou")}
    return checks, extra, table, summary


def rule2(folds, photo, name, chips931):
    run = [r for k in sorted(folds) for f in sorted(folds[k].glob("dw_counts_fold*.csv")) for r in read(f)]
    ctrl = [r for f in sorted((photo / "reservoirs").glob("*_counts.csv")) for r in read(f)
            if r["reference"] == "B Dynamic World" and r["method"] == CTRL]
    chip = {(c["reservoir"], c["file"]): c for c in read(photo / "chips.csv")}
    kr = {(r["reservoir"], r["file"]) for r in run}
    ctrl = [r for r in ctrl if (r["reservoir"], r["file"]) in chips931]
    out, rows = {"chips": len(kr), "same_931": kr == chips931}, []
    for gname, keep in (("all", lambda c: True), ("low water (under 20% full)", lambda c: c["low_water"] == "True")):
        for m, data in ((name, run), (CTRL, ctrl)):
            sub = [r for r in data if keep(chip[(r["reservoir"], r["file"])])]
            per = per_reservoir(sub)
            p = pooled(sub)
            med = float(np.median([v for v in per.values() if np.isfinite(v)]))
            rows.append({"group": gname, "method": m, "reservoirs": len(per), "chips": len(sub),
                         **{k: round(v, 2) for k, v in p.items()}, "median_reservoir_iou": round(med, 2)})
            out[(gname, m)] = {**p, "median": med, "per": per}
    a, c = out[("all", name)], out[("all", CTRL)]
    lo, loc = out[("low water (under 20% full)", name)], out[("low water (under 20% full)", CTRL)]
    got = (round(c["iou"], 2), round(c["median"], 2), round(loc["iou"], 2))
    if got != CONTROL_DW:
        raise SystemExit(f"the control on the 931 chips gives {got}, not PROTOCOL-5's {CONTROL_DW}")
    d = [a["per"][n] - c["per"][n] for n in sorted(set(a["per"]) & set(c["per"]))]
    same = out["same_931"]
    checks = [("2a. pooled IoU at least 91.25", a["iou"] >= BAR["2a"] and same,
               f"{name} {a['iou']:.2f}, control {c['iou']:.2f}; chips {out['chips']}, the 931: {same}"),
              ("2b. median reservoir IoU at least 89.35", a["median"] >= BAR["2b"] and same,
               f"{name} {a['median']:.2f}, control {c['median']:.2f}"),
              ("2c. low-water IoU at least 84.98", lo["iou"] >= BAR["2c"] and same,
               f"{name} {lo['iou']:.2f}, control {loc['iou']:.2f}")]
    extra = (f"rule 2 reported: precision {a['precision']:.2f} vs {c['precision']:.2f}, recall {a['recall']:.2f} vs "
             f"{c['recall']:.2f}; {name} better at {np.mean([x > 0 for x in d]):.0%} of {len(d)} reservoirs, "
             f"median gain {np.median(d):+.2f}")
    res_rows = [{"reservoir": n, name: round(a["per"][n], 2), "control": round(c["per"][n], 2)} for n in sorted(a["per"])]
    summary = {"dw_pooled": a["iou"], "median_reservoir": a["median"], "low_water": lo["iou"]}
    return rows, res_rows, checks, extra, summary


def rule3(K, folds, run3, tables, name):
    info, series, _ = K.sheet(Path(tables))
    ctrl_rows = {}
    for f in sorted(run3.glob("part-*/areas_passes/*.csv")):
        for r in read(f):
            ctrl_rows[(r["tank"], r["date"], r["orbit"], r["pass"])] = r
    per, report = {}, {"reservoirs": 0, "passes": 0, "passes_without_control": 0}
    for k in sorted(folds):
        for f in sorted(folds[k].glob("areas_passes/*.csv")):
            rows = read(f)
            if not rows:
                continue
            tank = rows[0]["tank"]
            merged = []
            for r in rows:
                c = ctrl_rows.get((tank, r["date"], r["orbit"], r["pass"]))
                if c is None:
                    report["passes_without_control"] += 1
                    continue
                base = {k2: r[k2] for k2 in ("tank", "date", "orbit", "pass", "photo_date", "photo_clear", "outline_ha",
                                              "radar_gap_share")}
                merged.append({**base, K.NEW: r[f"{name}, water"], K.CTRL: c[K.CTRL]})  # K.NEW holds this run here
            report["reservoirs"] += 1
            report["passes"] += len(merged)
            sc = K.score_passes(merged, series.get(tank, {}), info[tank]["cap"])
            if sc:
                per[tank] = sc
    names = sorted(per)
    err = np.array([per[n][K.NEW]["later_error_pct"] for n in names])
    errc = np.array([per[n][K.CTRL]["later_error_pct"] for n in names])
    if round(float(np.median(errc)), 2) != CONTROL_ERR:
        raise SystemExit(f"the control's median storage error is {np.median(errc):.2f}%, not PROTOCOL-5's {CONTROL_ERR}%")
    d = err - errc
    p_ni = one_sided(errc + 0.5 - err)
    rows = [{"reservoir": n, "cap_mcm": round(info[n]["cap"], 2), "passes": per[n]["passes"],
             f"{name} later_error_pct": round(per[n][K.NEW]["later_error_pct"], 3),
             "control later_error_pct": round(per[n][K.CTRL]["later_error_pct"], 3),
             f"{name} spearman_all": round(per[n][K.NEW]["spearman_all"], 3)} for n in names]
    summary = [{"method": m, "reservoirs": len(names),
                **{k: K.med([per[n][key].get(k, np.nan) for n in names]) for k in K.MEASURES}}
               for m, key in ((name, K.NEW), ("run 3 photo-label control", K.CTRL))]
    for b in K.BASELINES:
        summary.append({"method": b, "reservoirs": len(names),
                        "later_error_pct": K.med([per[n][b]["later_error_pct"] for n in names])})
    checks = [("3a. median storage error at most 8.78%", np.median(err) <= BAR["3a"],
               f"{name} {np.median(err):.2f}%, control {np.median(errc):.2f}%"),
              ("3b. median paired difference at most +0.25", np.median(d) <= BAR["3b"],
               f"median d {np.median(d):+.3f} points; {name} better at {np.mean(d < 0):.0%} of {len(d)}"),
              ("3c. non-inferior within 0.5 (one-sided Wilcoxon p < 0.05)", p_ni < 0.05, f"p = {p_ni:.2g}")]
    hl = hodges_lehmann(d)
    extra = f"rule 3 reported: Hodges-Lehmann difference in storage error {fmt_hl(hl, ' points')} ({len(d)} reservoirs)"
    s = {"median_error": float(np.median(err)), "median_d": float(np.median(d)), "p_ni": p_ni, "hl3": hl}
    return rows, summary, checks, report, extra, s


def rule4(folds, name):
    flood = [r for k in sorted(folds) for f in sorted(folds[k].glob("flood_fold*.csv")) for r in read(f)]
    test = [float(r[f"test: {name}, water"]) for r in flood]
    bol = [float(r[f"bolivia: {name}, water"]) for r in flood]
    m = float(np.mean(test))
    checks = [("4. Sen1Floods11 test split at least 65.6", len(test) == 5 and m >= BAR["4"],
               f"{name} {m:.3f} over {len(test)} folds (folds {min(test):.2f} to {max(test):.2f})")]
    extra = f"rule 4 reported: Bolivia {np.mean(bol):.2f} over {len(bol)} folds (control 48.5)"
    return flood, checks, extra, {"flood_test": m, "bolivia": float(np.mean(bol))}


def score(K, args, run, s1s2_dirs, name, tag):
    folds, problems = find_folds(run, tag, args.allow_missing_weights)
    s_rows, s_problems = scene_rows(s1s2_dirs, name)
    if problems + s_problems:
        raise SystemExit(f"{name}: not scored, the parts are incomplete:\n  " + "\n  ".join(problems + s_problems))
    chips931 = {(r["reservoir"], r["file"]) for f in sorted(Path(args.run4).glob("fold*/dw_counts_fold*.csv"))
                for r in read(f)}
    if len(chips931) != 931:
        raise SystemExit(f"run 4's Dynamic World chips: {len(chips931)}, not 931")
    c1, x1, t1, s1 = rule1(s_rows, args.s1s2_check, name)
    dw_rows, dw_res, c2, x2, s2 = rule2(folds, Path(args.photo), name, chips931)
    st_rows, st_sum, c3, rep3, x3, s3 = rule3(K, folds, Path(args.run3), args.tables, name)
    flood, c4, x4, s4 = rule4(folds, name)
    return {"checks": c1 + c2 + c3 + c4, "extra": [x1, x2, x3, x4], "summary": {**s1, **s2, **s3, **s4},
            "tables": {"s1s2_summary.csv": t1, "dw_pooled.csv": dw_rows, "dw_reservoirs.csv": dw_res,
                       "pass_scores.csv": st_rows, "pass_summary.csv": st_sum, "flood.csv": flood},
            "folds": {k: str(v.relative_to(run)) for k, v in folds.items()}, "passes": rep3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/2026-11-05-retrain-5"), help="holds part1/ to part3/")
    ap.add_argument("--s1s2", nargs="*", default=None,
                    help="S1S2-Water output folders (default: every scenes/ folder under --run)")
    ap.add_argument("--name", default="run 5")
    ap.add_argument("--tag", default="photo5", help="the weights' name: fold<i>_<tag>_seed<s>.pt")
    ap.add_argument("--out", default=None, help="where to write (default: --run)")
    ap.add_argument("--allow-missing-weights", action="store_true", help="only for the self-check on run 4")
    ap.add_argument("--run4", default=str(ROOT / "runs/2026-11-04-retrain-4"))
    ap.add_argument("--run4-s1s2", default=str(ROOT / "runs/2026-11-05-run4-s1s2"))
    ap.add_argument("--run3", default=str(ROOT / "runs/2026-11-03-retrain-3"))
    ap.add_argument("--photo", default=str(TEMP / "module-a-photo-check/runs/2026-11-04-photo-check"))
    ap.add_argument("--tables", default=str(ROOT / "kaggle-data/tables"))
    ap.add_argument("--s1s2-check", default=str(TEMP / "s1s2water-check/runs/2026-11-04-s1s2water"))
    args = ap.parse_args()
    K = load_kernel()
    run = Path(args.run)
    out = Path(args.out or run)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) / "scenes" for d in args.s1s2] if args.s1s2 else sorted(run.glob("part*/s1s2/scenes"))
    main_run = score(K, args, run, dirs, args.name, args.tag)
    for fname, rows in main_run["tables"].items():
        K.write(out / fname, rows)
    checks = main_run["checks"]
    verdict = "WIN (all four rules hold)" if all(h for _, h, _ in checks) else "not a win"
    print("folds:", json.dumps(main_run["folds"]), "| rule 3 passes:", json.dumps(main_run["passes"]))
    for rule, holds, text in checks:
        print(f"{'yes' if holds else 'NO '}  {rule}: {text}")
    for x in main_run["extra"]:
        print(x)
    print(f"verdict by PROTOCOL-5 ({args.name}):", verdict)
    beside = []
    if args.name != "run 4":
        r4 = score(K, args, Path(args.run4), [Path(args.run4_s1s2) / "scenes"], "run 4", "photo4")
        a, b = main_run["summary"], r4["summary"]
        for key, label in (("pooled", "1a S1S2-Water pooled IoU"), ("median_scene", "1b median scene IoU"),
                           ("better", "1c scenes better than the control"), ("p", "1d p"),
                           ("hl", "rule 1 Hodges-Lehmann gain (95% CI)"), ("asian", "5 Asian scenes IoU"),
                           ("dw_pooled", "2a Dynamic World pooled IoU"), ("median_reservoir", "2b median reservoir IoU"),
                           ("low_water", "2c low-water IoU"), ("median_error", "3a median storage error %"),
                           ("median_d", "3b median d"), ("p_ni", "3c p"), ("hl3", "rule 3 Hodges-Lehmann d (95% CI)"),
                           ("flood_test", "4 Sen1Floods11 test"), ("bolivia", "Bolivia")):
            show = lambda v: (fmt_hl(v) if key.startswith("hl") else v if isinstance(v, str)  # noqa: E731
                              else f"{v:.2g}" if key in ("p", "p_ni") else f"{v:.2f}")
            beside.append({"measure": label, args.name: show(a[key]), "run 4": show(b[key])})
        K.write(out / "beside_run4.csv", beside)
        print("beside run 4:")
        for r in beside:
            print(f"  {r['measure']}: {r[args.name]} | run 4 {r['run 4']}")
    K.write(out / "protocol5_check.csv", [{"rule": r, "holds": h, "numbers": t} for r, h, t in checks]
            + [{"rule": "reported", "holds": "", "numbers": x} for x in main_run["extra"]]
            + [{"rule": "verdict", "holds": "", "numbers": verdict}])


if __name__ == "__main__":
    main()
