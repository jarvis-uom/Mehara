r"""Builds run 4's inputs (4 Nov 2026), under ../PROTOCOL-4.md (fixed), from run 3's prep output (res.pkl).

It writes run4-inputs/, the private Kaggle dataset is4990-module-a-run4-inputs:
- res4.pkl: run 3's 103 reservoirs, chip lists and folds, with every storage number removed (cap, lo, hi, s,
  s_photo, s_radar, transfer). Each clear chip and each radar pass gets "dry" (the dry-bed weighting).
- labels4/<reservoir>.npz: the photo label of every clear chip (labels.py photo_label, plus never-water land:
  photo unsure and land_out -> land), made once so that the GPU reads only radar and labels.
- folds.csv; run 6's three weights; retrain4.py.
Dry bed (PROTOCOL-4 item 4): a clear chip whose outline is at least half cloud-free, with bare ground on at least 30%
of the outline's cloud-free pixels; a pass is dry if its photo_near chip is dry and at most 2 days away.
Light: CPU, reads the clear chips once.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\prep4.py
"""
import csv
import pickle
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import labels as L  # noqa: E402

RUN3 = ROOT / "runs/2026-11-03-retrain-3/prep"
FIRST = ROOT / "kaggle-data/first-check"
OUT = ROOT / "run4-inputs"
W6 = REPO / "temp/module-a-tanks/weights/run-6"
DROP = {"cap", "lo", "hi", "known"}
DROP_CHIP = {"s", "s_photo", "s_radar", "transfer"}


def read_full(path):
    with rasterio.open(path) as src:
        a = src.read().astype(np.float32)
    s1, s2 = a[:2], a[2:12]
    s1[s1 == -32768] = np.nan
    s1 /= 100
    s2[s2 == 0] = np.nan
    return np.concatenate([s1, s2]), a[13] / 100


def label4(photo, ctx):
    y = L.photo_label(photo, ctx)
    y[(photo == L.P_NONE) & ctx["land_out"]] = L.LAND
    return y


def main():
    with open(RUN3 / "res.pkl", "rb") as f:
        d = pickle.load(f)
    (OUT / "labels4").mkdir(parents=True, exist_ok=True)
    res4, n_dry, n_dry_pass, n_lab = {}, 0, 0, 0
    for name in d["used"]:
        R = d["res"][name]
        labs, dry = {}, {}
        for e in R["clear"]:
            x, cs = read_full(FIRST / e["path"][1])
            photo = L.photo_class(x, cs)
            labs[e["file"]] = label4(photo, R["ctx"])
            ins = R["inside"]
            free = ins & (photo != L.P_NONE)
            dry[e["file"]] = bool(free.sum() >= 0.5 * ins.sum() and free.any()
                                  and (photo == L.P_BARE)[free].mean() >= 0.3)
        np.savez_compressed(OUT / "labels4" / f"{R['safe']}.npz", **{k.replace(".tif", ""): v for k, v in labs.items()})
        n_lab += len(labs)
        clear = [{**{k: v for k, v in e.items() if k not in DROP_CHIP}, "dry": dry[e["file"]]} for e in R["clear"]]
        passes = []
        for e in R["passes"]:
            pn = e["photo_near"]["file"] if e.get("photo_near") else None
            gap = abs((date.fromisoformat(e["date"]) - date.fromisoformat(e["photo_near"]["s2_date"])).days) if pn else None
            passes.append({**{k: v for k, v in e.items() if k not in DROP_CHIP | {"photo_near"}}, "photo_near": pn,
                           "near_gap": gap, "dry": bool(pn and dry[pn] and gap <= 2)})
        cloudy = [{k: v for k, v in e.items() if k not in DROP_CHIP} for e in R["cloudy"]]
        res4[name] = {**{k: v for k, v in R.items() if k not in DROP | {"clear", "passes", "cloudy"}},
                      "clear": clear, "passes": passes, "cloudy": cloudy}
        n_dry += sum(c["dry"] for c in clear)
        n_dry_pass += sum(p["dry"] for p in passes)
    bad = [k for R in res4.values() for k in ("cap", "lo", "hi") if k in R]
    bad += [k for R in res4.values() for e in R["clear"] + R["passes"] + R["cloudy"] for k in DROP_CHIP if k in e]
    assert not bad, f"storage numbers left: {set(bad)}"
    with open(OUT / "res4.pkl", "wb") as f:
        pickle.dump({"res": res4, "fold": d["fold"], "used": d["used"], "labels_version": L.VERSION + " + land_out",
                     "note": "run 3's prep output without storage numbers (PROTOCOL-4)"}, f)
    shutil.copy(RUN3 / "folds.csv", OUT / "folds.csv")
    (OUT / "run-6").mkdir(exist_ok=True)
    for n in ("unet_radar_best.pt", "unet_radar_seed1_best.pt", "unet_radar_seed2_best.pt"):
        shutil.copy(W6 / n, OUT / "run-6" / n)
    with open(OUT / "prep4.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reservoir", "fold", "clear", "dry_clear", "passes_with_photo", "dry_passes"])
        for n, R in res4.items():
            w.writerow([n, d["fold"][n], len(R["clear"]), sum(c["dry"] for c in R["clear"]),
                        sum(bool(p["photo_near"]) for p in R["passes"]), sum(p["dry"] for p in R["passes"])])
    print(f"{len(res4)} reservoirs; {n_lab} labelled clear chips ({n_dry} dry-bed); {n_dry_pass} dry-bed passes")


if __name__ == "__main__":
    main()
