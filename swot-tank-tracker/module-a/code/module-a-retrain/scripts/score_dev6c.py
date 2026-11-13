r"""Scores series 2's development batch 3 by DEV-6's addendum of 11 Nov 2026 (fixed before any run, commit cdbc642).

The notebooks' outputs are downloaded to runs/dev6c/<variant>/ (fold1/, fold2/). The recipes, simplest first:
- R0 control (batch 2's models, scored again), R1 edge, R2 aug, R3 edgeaug, R5 longedgeaug: runs/dev6c/;
- R4 long: batch 1's score files in runs/dev6/long/ (plain maps only, floods pooled only). If long was trained
  again (runs/dev6c/long/), that run gives R4's grown maps, and its plain files are checked against batch 1's.
Scores, each paired with R0's:
- S: the IoU of each of the 10 development scenes (the mean of the two folds' IoUs);
- F: the pooled IoU of the 89 flood validation chips (the mean of the two folds);
- H: each held-out reservoir's Dynamic World IoU, pooled over its first-half clear chips (the whole chip);
- T: each reservoir's storage error with Curve 2, as score_t1.py --dev (fitted on the first quarter, scored on the
  second quarter's main-orbit passes with no radar gap), on the recipe's own areas.
Keep rules: no loss (a one-sided Wilcoxon on S + 2, on H + 1 and on T + 0.5 each with p < 0.05; F down by less than
1.0) and a gain (the Hodges-Lehmann estimate of S or of H is +0.5 or more, or F is up by 1.0 or more). The choice: the
kept recipe with the highest mean scene IoU, or the simplest kept recipe within 0.5 of it. Then the grown maps (G) of
the chosen recipe (R0 if none) against its plain maps, by the same rules.
Beside the rules: per-fold scores, precision and recall, floods per event, Dynamic World inside the outline, the
storage error with Curve 1, the intervals, the grown maps of every recipe, the edge clean-up's pixel counts, and a
check that R0's plain score files equal batch 2's byte for byte.
Writes runs/dev6c/dev6c_scenes.csv, dev6c_floods.csv, dev6c_reservoirs.csv and dev6c_verdict.csv.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\score_dev6c.py
"""
import csv
import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/dev6c"
RECIPES = {"R0": ("control", RUN), "R1": ("edge", RUN), "R2": ("aug", RUN), "R3": ("edgeaug", RUN),
           "R4": ("long", ROOT / "runs/dev6"), "R5": ("longedgeaug", RUN)}  # simplest first
FOLDS = (1, 2)
MARGIN_S, MARGIN_H, MARGIN_T, LOSS_F = 2.0, 1.0, 0.5, 1.0
GAIN_S, GAIN_H, GAIN_F, WITHIN = 0.5, 0.5, 1.0, 0.5
sys.stdout.reconfigure(encoding="utf-8")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def tally(rows):
    return [sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn")]


def iou(c):
    return 100 * c[0] / (c[0] + c[1] + c[2]) if c[0] + c[1] + c[2] else float("nan")


def pr(c):
    return 100 * c[0] / max(c[0] + c[1], 1), 100 * c[0] / max(c[0] + c[2], 1)


def by(rows, key):
    out = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def storage(M, d, col, grown, info, series, split):
    """Per reservoir of one fold: the storage error with Curve 2 ("C1") and Curve 1 ("I"), as score_t1.py --dev, on
    the areas in column col (the grown areas come from areas_grown/, joined by date)."""
    K, T1 = M["K"], M["T1"]
    out = {}
    for f in sorted((d / "areas_passes").glob("*.csv")):
        passes = read(f)
        if not passes or passes[0]["tank"] not in split:
            continue
        name = passes[0]["tank"]
        if grown:
            g = {r["date"]: r[col] for r in read(d / "areas_grown" / f.name)}
            passes = [{**r, col: g[r["date"]]} for r in passes]
        sp, cap, s, orbit = split[name], info[name]["cap"], series.get(name, {}), T1.main_orbit(passes)
        q1_end = (date.fromisoformat(sp["q2_from"]) - timedelta(days=1)).isoformat()
        hist = []
        for r in passes:
            if r["date"] > q1_end or float(r["radar_gap_share"]) > 0.05:
                continue
            st = K.nearest(s, date.fromisoformat(r["date"]))
            if st is not None:
                hist.append((r["date"], float(r[col]), st))
        hist.sort()
        win = [(float(r[col]), T1.truth_on(s, date.fromisoformat(r["date"]))) for r in passes
               if sp["q2_from"] <= r["date"] < sp["later_from"] and r["orbit"] == orbit
               and float(r["radar_gap_share"]) == 0]
        win = [w for w in win if w[1] is not None]
        if len(hist) < 3 or len(win) < T1.MIN_DATES:
            continue
        cv = T1.curves(K, M["C"], M["L"], np.array([p[1] for p in hist]), np.array([p[2] for p in hist]), cap)
        if cv is None:
            continue
        wa, wt = np.array([w[0] for w in win]), np.array([w[1] for w in win])
        out[name] = {k: 100 * float(np.median(np.abs(f_(wa) - wt))) / cap for k, f_ in cv.items()}
    return out


def scores(M, variant, base, grown, info, series, split):
    """One recipe's scores with its plain or grown maps; None where its files do not hold them."""
    kind = "grown" if grown else "plain"
    old = not (base / variant / "fold1" / "dev3_s1s2_fold0.csv").exists()  # batch 1's files: plain, floods pooled
    if grown and old:
        return None
    S, P, F, E, H, Ho, T = {}, {}, [], {}, {}, {}, {}
    for k in FOLDS:
        d, i = base / variant / f"fold{k}", k - 1
        s_rows = read(d / f"dev_s1s2_fold{i}.csv") if old else \
            [r for r in read(d / f"dev3_s1s2_fold{i}.csv") if r["map"] == kind]
        for r in s_rows:
            S.setdefault((r["set"], r["scene"]), []).append(iou(tally([r])))
            P.setdefault((r["set"], r["scene"]), []).append(pr(tally([r])))
        if old:
            F.append(iou(tally(read(d / f"dev_flood_fold{i}.csv"))))
            dw = {"chip": read(d / f"dw_counts_fold{i}.csv")}
        else:
            fl = [r for r in read(d / f"dev3_flood_fold{i}.csv") if r["map"] == kind]
            if len(fl) != 89:
                raise SystemExit(f"{variant} fold {k}: {len(fl)} flood chips, not 89")
            F.append(iou(tally(fl)))
            for ev, rows in by(fl, lambda r: r["chip"].split("_")[0]).items():
                E.setdefault(ev, []).append(iou(tally(rows)))
            dw = by([r for r in read(d / f"dev3_dw_fold{i}.csv") if r["map"] == kind], lambda r: r["region"])
        for name, rows in by(dw["chip"], lambda r: r["reservoir"]).items():
            H[name] = iou(tally(rows))
        for name, rows in by(dw.get("outline", []), lambda r: r["reservoir"]).items():
            Ho[name] = iou(tally(rows))
        T.update(storage(M, d, f"{variant}6, {'grown' if grown else 'water'}", grown, info, series, split))
    if len(S) != 10 or any(len(v) != len(FOLDS) for v in S.values()):
        raise SystemExit(f"{variant}: incomplete scene scores ({len(S)} scenes)")
    return {"S": {k: float(np.mean(v)) for k, v in S.items()}, "S folds": S,
            "P": {k: tuple(np.mean(v, axis=0)) for k, v in P.items()}, "F": float(np.mean(F)), "F folds": F,
            "E": {k: float(np.mean(v)) for k, v in E.items()}, "H": H, "H outline": Ho, "T": T}


def paired(a, b, sign=1.0):
    """The paired differences sign * (a - b) over the units both hold, as an array."""
    return np.array([sign * (a[k] - b[k]) for k in sorted(set(a) & set(b))
                     if np.isfinite(a[k]) and np.isfinite(b[k])])


def judge(M, c, b):
    """The addendum's keep rules for scores c against b. Returns the row for the verdict table and kept or not."""
    S5 = M["S5"]
    dS, dH = paired(c["S"], b["S"]), paired(c["H"], b["H"])
    dT = paired({k: v["C1"] for k, v in b["T"].items()}, {k: v["C1"] for k, v in c["T"].items()})  # above 0 = better
    dF = c["F"] - b["F"]
    p = {"S": S5.one_sided(dS + MARGIN_S), "H": S5.one_sided(dH + MARGIN_H), "T": S5.one_sided(dT + MARGIN_T)}
    hl = {"S": S5.hodges_lehmann(dS), "H": S5.hodges_lehmann(dH), "T": S5.hodges_lehmann(dT)}
    losses = [k for k in ("S", "H", "T") if not p[k] < 0.05] + (["F"] if dF <= -LOSS_F else [])
    gains = [k for k, lim in (("S", GAIN_S), ("H", GAIN_H)) if hl[k][0] >= lim] + (["F"] if dF >= GAIN_F else [])
    kept = bool(gains) and not losses
    row = {"S_mean": round(float(np.mean(list(c["S"].values()))), 3), "S_mean_difference": round(float(dS.mean()), 3),
           "S_HL": round(hl["S"][0], 3), "S_HL_CI": f"{hl['S'][1]:+.2f} to {hl['S'][2]:+.2f}",
           "S_scenes_up": int((dS > 0).sum()), "S_lowest": round(float(dS.min()), 3), "S_p_no_loss": p["S"],
           "F_difference": round(dF, 3),
           "H_reservoirs": len(dH), "H_HL": round(hl["H"][0], 3), "H_HL_CI": f"{hl['H'][1]:+.2f} to {hl['H'][2]:+.2f}",
           "H_up": int((dH > 0).sum()), "H_p_no_loss": p["H"],
           "T_reservoirs": len(dT), "T_HL_gain": round(hl["T"][0], 3),
           "T_HL_CI": f"{hl['T'][1]:+.2f} to {hl['T'][2]:+.2f}", "T_better": int((dT > 0).sum()),
           "T_p_no_loss": p["T"], "gains": "; ".join(gains), "losses": "; ".join(losses), "kept": kept}
    return row, kept


def beside(M, c, b):
    """Not rules: Dynamic World inside the outline and the storage error with Curve 1, paired."""
    S5, out = M["S5"], {}
    if c["H outline"] and b["H outline"]:
        d = paired(c["H outline"], b["H outline"])
        out["H outline"] = S5.fmt_hl(S5.hodges_lehmann(d)) + f", up at {int((d > 0).sum())} of {len(d)}"
    d = paired({k: v["I"] for k, v in b["T"].items()}, {k: v["I"] for k, v in c["T"].items()})
    out["T with Curve 1 (gain)"] = S5.fmt_hl(S5.hodges_lehmann(d)) + f", better at {int((d > 0).sum())} of {len(d)}"
    return out


def say(name, row, extra):
    print(f"\n{name}: S mean {row['S_mean']:.2f} ({row['S_mean_difference']:+.2f}; HL {row['S_HL']:+.2f}, "
          f"{row['S_HL_CI']}; up at {row['S_scenes_up']} of 10, lowest {row['S_lowest']:+.2f}; no-loss p "
          f"{row['S_p_no_loss']:.3f})")
    print(f"  F {row['F_difference']:+.2f}; H HL {row['H_HL']:+.2f} ({row['H_HL_CI']}; up at {row['H_up']} of "
          f"{row['H_reservoirs']}; no-loss p {row['H_p_no_loss']:.4f}); T gain HL {row['T_HL_gain']:+.2f} "
          f"({row['T_HL_CI']}; better at {row['T_better']} of {row['T_reservoirs']}; no-loss p {row['T_p_no_loss']:.4f})")
    print(f"  gains: {row['gains'] or 'none'}; losses: {row['losses'] or 'none'} -> {'KEPT' if row['kept'] else 'not kept'}")
    for k, v in extra.items():
        print(f"  beside: {k}: {v}")


def main():
    M = {"C": load("curve_study", ROOT / "scripts/curve_study.py"), "L": load("curve_l4c", ROOT / "scripts/curve_l4c.py"),
         "S5": load("score_run5", ROOT / "scripts/score_run5.py"), "T1": load("score_t1", ROOT / "scripts/score_t1.py")}
    M["K"] = M["C"].load_kernel()
    info, series, _ = M["K"].sheet(ROOT / "kaggle-data/tables")
    split = {r["reservoir"]: r for r in read(ROOT / "kernel6-dev/split_dates.csv") if r["later_from"]}

    # the check: R0's plain score files against batch 2's, byte for byte
    same = diff = 0
    for k in FOLDS:
        a, b = RUN / "control" / f"fold{k}", ROOT / "runs/dev6b" / f"fold{k}"
        files = [f"dev_s1s2_fold{k - 1}.csv", f"dev_flood_fold{k - 1}.csv", f"dw_counts_fold{k - 1}.csv"] + \
            [f"areas_passes/{f.name}" for f in sorted((b / "areas_passes").glob("*.csv"))]
        for f in files:
            ok = (a / f).exists() and (a / f).read_bytes() == (b / f).read_bytes()
            same, diff = same + ok, diff + (not ok)
            if not ok:
                print(f"  differs from batch 2: fold {k} {f}")
    print(f"check: R0's plain score files against batch 2's: {same} equal, {diff} different")

    plain = {c: scores(M, v, base, False, info, series, split) for c, (v, base) in RECIPES.items()}
    grown = {c: scores(M, v, base, True, info, series, split) for c, (v, base) in RECIPES.items()}
    again = {}  # recipes trained again in this batch: their grown maps are judged against that run's plain maps
    if "R4" in RECIPES and (RUN / "long" / "fold1" / "dev3_s1s2_fold0.csv").exists():
        same = diff = 0
        for k in FOLDS:
            a, b = RUN / "long" / f"fold{k}", ROOT / "runs/dev6/long" / f"fold{k}"
            for f in [f"dev_s1s2_fold{k - 1}.csv", f"dev_flood_fold{k - 1}.csv", f"dw_counts_fold{k - 1}.csv"] +                     [f"areas_passes/{f.name}" for f in sorted((b / "areas_passes").glob("*.csv"))]:
                ok = (a / f).exists() and (a / f).read_bytes() == (b / f).read_bytes()
                same, diff = same + ok, diff + (not ok)
        print(f"check: long trained again, its plain score files against batch 1's: {same} equal, {diff} different")
        again["R4"] = scores(M, "long", RUN, False, info, series, split)
        grown["R4"] = scores(M, "long", RUN, True, info, series, split)
    scenes = sorted(plain["R0"]["S"], key=lambda t: (t[0], int(t[1])))
    codes = list(RECIPES)
    print(f"\n{'scene':24s}" + "".join(f"{c:>8s}" for c in codes))
    out = []
    for sc in scenes:
        print(f"{sc[0][:17] + ' ' + sc[1]:24s}" + "".join(f"{plain[c]['S'][sc]:8.2f}" for c in codes))
        r = {"set": sc[0], "scene": sc[1]}
        for c in codes:
            r[c] = round(plain[c]["S"][sc], 3)
            r[f"{c} folds"] = " ".join(f"{x:.2f}" for x in plain[c]["S folds"][sc])
            r[f"{c} precision"], r[f"{c} recall"] = (round(x, 2) for x in plain[c]["P"][sc])
            if grown[c]:
                r[f"{c} grown"] = round(grown[c]["S"][sc], 3)
        out.append(r)
    write(RUN / "dev6c_scenes.csv", out)
    print(f"{'mean of the 10 scenes':24s}" + "".join(f"{np.mean(list(plain[c]['S'].values())):8.2f}" for c in codes))
    print(f"{'floods (F)':24s}" + "".join(f"{plain[c]['F']:8.2f}" for c in codes))
    print(f"{'H median reservoir':24s}" + "".join(f"{np.nanmedian(list(plain[c]['H'].values())):8.2f}" for c in codes))
    print(f"{'T median, Curve 2':24s}" + "".join(f"{np.median([v['C1'] for v in plain[c]['T'].values()]):8.2f}"
                                                 for c in codes))
    events = sorted(plain["R0"]["E"])
    fl = [{"event": e, **{c: round(plain[c]["E"][e], 2) for c in codes if plain[c]["E"]},
           **{f"{c} grown": round(grown[c]["E"][e], 2) for c in codes if grown[c]}} for e in events]
    fl.append({"event": "all 89 chips", **{c: round(plain[c]["F"], 3) for c in codes},
               **{f"{c} grown": round(grown[c]["F"], 3) for c in codes if grown[c]}})
    write(RUN / "dev6c_floods.csv", fl)
    for r in fl[:-1]:
        print(f"{'  flood ' + r['event']:24s}" + "".join(f"{r[c]:8.2f}" if c in r else f"{'':8s}" for c in codes))
    names = sorted(plain["R0"]["H"])
    write(RUN / "dev6c_reservoirs.csv",
          [{"reservoir": n,
            **{f"{c} H": round(plain[c]["H"][n], 2) for c in codes if n in plain[c]["H"]},
            **{f"{c} H outline": round(plain[c]["H outline"][n], 2) for c in codes if n in plain[c]["H outline"]},
            **{f"{c} T": round(plain[c]["T"][n]["C1"], 3) for c in codes if n in plain[c]["T"]},
            **{f"{c} T Curve 1": round(plain[c]["T"][n]["I"], 3) for c in codes if n in plain[c]["T"]},
            **{f"{c} grown H": round(grown[c]["H"][n], 2) for c in codes if grown[c] and n in grown[c]["H"]},
            **{f"{c} grown T": round(grown[c]["T"][n]["C1"], 3) for c in codes if grown[c] and n in grown[c]["T"]}}
           for n in names])

    verdict, kept = [], []
    for c in codes[1:]:
        row, k = judge(M, plain[c], plain["R0"])
        say(f"{c} ({RECIPES[c][0]}) against R0", row, beside(M, plain[c], plain["R0"]))
        verdict.append({"comparison": f"{c} against R0", **row})
        if k:
            kept.append(c)
    choice = "R0"
    if kept:
        best = max(np.mean(list(plain[c]["S"].values())) for c in kept)
        choice = next(c for c in kept if np.mean(list(plain[c]["S"].values())) >= best - WITHIN)
    print(f"\nkept: {', '.join(kept) or 'none'}; the choice: {choice} ({RECIPES[choice][0]})")
    verdict.append({"comparison": "choice", "kept": choice})
    if grown[choice] is None:
        print(f"G: {choice} has no grown maps in its files; it is trained again first (the addendum)")
        verdict.append({"comparison": f"G on {choice}", "kept": "not scored yet"})
    else:
        row, k = judge(M, grown[choice], again.get(choice, plain[choice]))
        say(f"G on {choice}: grown against plain maps", row,
            beside(M, grown[choice], again.get(choice, plain[choice])))
        verdict.append({"comparison": f"G on {choice}", **row})
    print("\nbeside: the grown maps of the other recipes, against their own plain maps")
    for c in codes:
        if c != choice and grown[c]:
            row, _ = judge(M, grown[c], again.get(c, plain[c]))
            print(f"  {c}: S mean {row['S_mean_difference']:+.2f} (lowest {row['S_lowest']:+.2f}), F "
                  f"{row['F_difference']:+.2f}, H HL {row['H_HL']:+.2f}, T gain HL {row['T_HL_gain']:+.2f}")
            verdict.append({"comparison": f"beside: G on {c}", **row})
    write(RUN / "dev6c_verdict.csv", verdict)
    for c, (v, base) in RECIPES.items():
        for k in FOLDS:
            f = base / v / f"fold{k}" / f"run_fold{k - 1}.json"
            e = json.loads(f.read_text(encoding="utf-8")).get("edge") if f.exists() else None
            if e:
                print(f"beside: {c} fold {k}: the clean-up's part 1 moved {e['plants_by_water_px']} label pixels, "
                      f"{100 * e['share_of_outline']:.1f}% of the outlines' pixels")


if __name__ == "__main__":
    main()
