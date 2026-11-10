r"""Scores series 2's development batch 2 by DEV-6's addendum of 10 Nov 2026 (fixed before any run, commit f0858af).

The notebook's output is downloaded to runs/dev6b/ (fold1/, fold2/, ens_part<i>.csv). Candidates, against A0:
- A0: each fold's 3 seeds; a scene's IoU is the mean of the folds' IoUs; the flood IoU is the mean of the folds';
- A1: all folds' models in one map; A2: A1 with the 8 turns and flips.
Scores: S, the IoU of each of the 10 development scenes; F, the pooled IoU of the 89 flood validation chips.
Keep rules: a gain (the mean of the 10 per-scene differences is +0.3 or more, or F is up by 1.0 or more) and no loss
(no scene down by 1.0 or more; F down by less than 1.0). Among the kept, A1 unless A2's mean scene IoU is more than
0.1 above A1's.
Beside the rules: A0 with flips, precision and recall, the flood IoU per event, batch 1's control, and a check that
this run's A0 counts equal retrain6.py's own per-fold counts.
Writes runs/dev6b/dev6b_scenes.csv, dev6b_floods.csv and dev6b_verdict.csv.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\score_dev6b.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/dev6b"
FOLDS = (1, 2)
GAIN_S, GAIN_F, LOSS_S, LOSS_F, WITHIN = 0.3, 1.0, 1.0, 1.0, 0.1
sys.stdout.reconfigure(encoding="utf-8")


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def iou(c):
    return 100 * c[0] / (c[0] + c[1] + c[2]) if c[0] + c[1] + c[2] else float("nan")


def pr(c):
    return (100 * c[0] / max(c[0] + c[1], 1), 100 * c[0] / max(c[0] + c[2], 1))


def main():
    rows = [r for f in sorted(RUN.glob("ens_part*.csv")) for r in read(f)]
    acc = {}  # (set, scene, candidate) -> [tp, fp, fn]; flood chips are pooled
    event = {}  # (event, candidate) -> [tp, fp, fn]
    for r in rows:
        c = [int(r[k]) for k in ("tp", "fp", "fn")]
        for store, key in ((acc, (r["set"], r["scene"], r["candidate"])),
                           (event, (r["chip"].split("_")[0], r["candidate"])) if r["set"] == "flood" else (None, None)):
            if store is not None:
                a = store.setdefault(key, [0, 0, 0])
                for j in range(3):
                    a[j] += c[j]
    scenes = sorted({(s, sc) for s, sc, _ in acc if s != "flood"}, key=lambda t: (t[0], int(t[1])))
    if len(scenes) != 10 or sum(r["set"] == "flood" and r["candidate"] == "A1" for r in rows) != 89:
        raise SystemExit(f"incomplete output: {len(scenes)} scenes, "
                         f"{sum(r['set'] == 'flood' and r['candidate'] == 'A1' for r in rows)} flood chips")
    # the check: this run's A0 counts against retrain6.py's own, per fold
    worst = 0.0
    for k in FOLDS:
        for r in read(RUN / f"fold{k}" / f"dev_s1s2_fold{k - 1}.csv"):
            own = [int(r[c]) for c in ("tp", "fp", "fn")]
            mine = acc[(r["set"], r["scene"], f"A0 fold{k}")]
            worst = max(worst, abs(iou(own) - iou(mine)))
        fl = read(RUN / f"fold{k}" / f"dev_flood_fold{k - 1}.csv")[0]
        worst = max(worst, abs(iou([int(fl[c]) for c in ("tp", "fp", "fn")]) - iou(acc[("flood", "valid", f"A0 fold{k}")])))
    print(f"check: A0 against retrain6.py's own scores, largest IoU difference {worst:.3f} points")

    def score(cand, key):
        if cand in ("A0", "A0+flips"):
            return float(np.mean([iou(acc[(*key, f"{cand} fold{k}")]) for k in FOLDS]))
        return iou(acc[(*key, cand)])

    cands = ("A0", "A0+flips", "A1", "A2")
    out = []
    for s, sc in scenes:
        r = {"set": s, "scene": sc, **{c: round(score(c, (s, sc)), 3) for c in cands}}
        for c in ("A1", "A2"):
            p, rc = pr(acc[(s, sc, c)])
            r[f"{c} precision"], r[f"{c} recall"] = round(p, 2), round(rc, 2)
        out.append(r)
    write(RUN / "dev6b_scenes.csv", out)
    S = {c: np.array([score(c, k) for k in scenes]) for c in cands}
    Fl = {c: score(c, ("flood", "valid")) for c in cands}
    print(f"{'scene':26s}" + "".join(f"{c:>10s}" for c in cands))
    for (s, sc), i in zip(scenes, range(10)):
        print(f"{s + ' ' + sc:26s}" + "".join(f"{S[c][i]:10.2f}" for c in cands))
    print(f"{'mean of the 10 scenes':26s}" + "".join(f"{S[c].mean():10.2f}" for c in cands))
    print(f"{'floods (F)':26s}" + "".join(f"{Fl[c]:10.2f}" for c in cands))
    events = sorted({e for e, _ in event})
    fl_rows = [{"event": e, **{c: round(float(np.mean([iou(event[(e, f'{c} fold{k}')]) for k in FOLDS]))
                                  if c in ("A0", "A0+flips") else iou(event[(e, c)]), 2) for c in cands}} for e in events]
    write(RUN / "dev6b_floods.csv", fl_rows + [{"event": "all 89 chips", **{c: round(Fl[c], 3) for c in cands}}])
    for r in fl_rows:
        print(f"{'  flood ' + r['event']:26s}" + "".join(f"{r[c]:10.2f}" for c in cands))
    verdict, kept = [], []
    for c in ("A1", "A2"):
        d = S[c] - S["A0"]
        dF = Fl[c] - Fl["A0"]
        gain = [g for g, ok in (("S mean", d.mean() >= GAIN_S), ("F", dF >= GAIN_F)) if ok]
        loss = [g for g, ok in (("a scene down by 1.0 or more", d.min() <= -LOSS_S), ("F", dF <= -LOSS_F)) if ok]
        k = bool(gain) and not loss
        if k:
            kept.append(c)
        print(f"\n{c} against A0: mean scene difference {d.mean():+.3f} (median {np.median(d):+.3f}, lowest {d.min():+.3f}, "
              f"highest {d.max():+.3f}, up at {int((d > 0).sum())} of 10); F {dF:+.3f}")
        print(f"  gains: {gain or 'none'}; losses: {loss or 'none'} -> {'KEPT' if k else 'not kept'}")
        verdict.append({"candidate": c, "mean_scene_difference": round(float(d.mean()), 3),
                        "median": round(float(np.median(d)), 3), "lowest": round(float(d.min()), 3),
                        "highest": round(float(d.max()), 3), "scenes_up": int((d > 0).sum()), "F_difference": round(dF, 3),
                        "gains": "; ".join(gain), "losses": "; ".join(loss), "kept": k})
    choice = "none"
    if kept:
        choice = "A2" if "A2" in kept and ("A1" not in kept or S["A2"].mean() > S["A1"].mean() + WITHIN) else "A1"
    print(f"\nthe choice: {choice}")
    verdict.append({"candidate": "choice", "kept": choice})
    write(RUN / "dev6b_verdict.csv", verdict)


if __name__ == "__main__":
    main()
