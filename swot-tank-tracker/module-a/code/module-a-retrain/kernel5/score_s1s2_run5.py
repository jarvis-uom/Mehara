r"""Rule 1 of ../PROTOCOL-5.md (5 Nov 2026): run 5's fold ensembles (3 seeds each) on S1S2-Water's scenes.

As ../kernel4-s1s2/score_s1s2_run4.py: it uses the S1S2-Water check's own code (score_s1s2.py, PROTOCOL-S1S2 as
amended) and scores only run 5. Two differences, so that each Kaggle part scores its own folds:
- it scores the folds it is given (--folds, 1 to 5), each as its own ensemble, as run 4's scoring did;
- with --part i --parts n it scores every n-th scene from the i-th, so two GPUs share the scenes.
Each scene's pixel counts are per (scene, fold) and do not depend on the other folds, so the parts' rows are simply put
together on the laptop. Writes scenes/<scene>.csv (method "run 5") and run_part<i>.json to --out, each scene as it
finishes; a scene already written is skipped.
    python -u score_s1s2_run5.py --input <datasets> --weights <folder with fold*_photo5_seed*.pt> --folds 1,2 --out <folder>
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_s1s2 as S  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RUN5 = "run 5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--folds", required=True, help="1 to 5, for example 1,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="first scene only, a 1100 x 1300 window; 1 seed per fold")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    folds = [int(f) - 1 for f in args.folds.split(",")]
    weights = {}
    for w in S.find(args.weights, "fold*_photo5_seed*.pt", ""):
        weights.setdefault(int(w.stem.split("_")[0][4:]), {})[w.name] = w  # by name: no duplicates
    need = 1 if args.dry_run else 3
    found = {k + 1: len(weights.get(k, {})) for k in folds}
    if any(v != need for v in found.values()):
        raise RuntimeError(f"need {need} seeds for each fold, found {found}")
    groups, mean, std = [], None, None
    for k in folds:
        models = []
        for name in sorted(weights[k]):
            model, ck = S.load(weights[k][name], device)
            if ck["mode"] != "photo5" or ck["fold"] != k:
                raise RuntimeError(f"{name}: mode {ck['mode']}, fold {ck['fold']}")
            models.append(model)
            mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
        groups.append((RUN5, k + 1, "not land", models))
    if args.dry_run:
        scenes_dir = Path(args.input) / "temp/s1s2water-check/gee"
    else:
        scenes_dir = [p for p in S.find(args.input, "scenes.csv", "s1s2water") if "gee" in p.parent.name][0].parent
    with open(scenes_dir / "scenes.csv", encoding="utf-8") as f:
        scenes = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    scenes = sorted(scenes, key=lambda r: (r["set"] != "main", int(r["scene"])))[args.part::args.parts]
    scenes = scenes[:1] if args.dry_run else scenes
    S.log(f"PROTOCOL-5 rule 1 on {gpu}: folds {[k + 1 for k in folds]}, scenes part {args.part + 1} of {args.parts} "
          f"({len(scenes)} scenes), {sum(len(g[3]) for g in groups)} models")
    for row in scenes:
        sid = row["scene"]
        if (out / "scenes" / f"{sid}.csv").exists():
            S.log(f"scene {sid}: already scored, skipped")
            continue
        t1 = time.time()
        res = S.score_scene(scenes_dir / f"{sid}_radar.tif", scenes_dir / f"{sid}_truth.tif", groups, mean, std,
                            device, (5000, 5000, 1100, 1300) if args.dry_run else None)
        keep = {c: row[c] for c in ("set", "scene", "lat", "lon", "landcover", "date_s1", "mask_m", "shift_dy",
                                    "shift_dx")}
        rows = [{**keep, **res["info"], "method": m, "fold": fd, **S.scores(*a), "tp": a[0], "fp": a[1], "fn": a[2],
                 "tn": a[3]} for (m, fd), a in sorted(res["acc"].items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
                if m == RUN5]
        S.write(out / "scenes" / f"{sid}.csv", rows)
        S.log(f"scene {sid} ({row['set']}) in {time.time() - t1:.0f} s: IoU by fold {[r['iou'] for r in rows]}")
    with open(out / f"run_part{args.part}.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "PROTOCOL-5.md rule 1, scored with score_s1s2.py (PROTOCOL-S1S2 as amended)",
                   "gpu": gpu, "folds": [k + 1 for k in folds], "part": args.part, "parts": args.parts,
                   "scenes": [r["scene"] for r in scenes], "minutes": round((time.time() - t0) / 60, 1),
                   "dry_run": args.dry_run}, f, indent=2)
    S.log(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
