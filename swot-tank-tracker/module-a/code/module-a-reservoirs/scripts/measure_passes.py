r"""Water area on every radar pass (2 Nov 2026): the radar-only check, step 2 of 3.

For each reservoir exported by export_passes.py it measures the water inside the reservoir's outline (from its
first-check chip: the JRC water body that holds the chip centre, widened by 30 m, as in measure_areas.py) with the
radar-only U-Net of run 2 and VV < -16.75 dB, and records the share of the outline without radar. --seeds adds the
three run 6 seeds. Each U-Net is also measured in a wide outline: the outline widened by 300 m more, counting only
water connected to water inside the outline. The JRC outline (1984 to 2021, from Landsat photos) can be smaller than
the full lake, which spreads further at high water (at Akathimuruppu, 165 ha just outside a 279 ha outline at 94%
storage), so inside it the area stops growing when the lake is full. Writes passes/areas/<reservoir>.csv and skips
reservoirs already measured. With --wait it keeps going beside the export, measuring each reservoir once its export
is done, until all are measured.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\measure_passes.py --wait
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_passes as P  # noqa: E402  (folders, first_chip)
import export_reservoirs as E  # noqa: E402  (reservoirs, safe)
import measure_areas as A  # noqa: E402  (load_model, MODELS)
import run_model as M  # noqa: E402  (load, predict, tank_mask, BANDS)

sys.stdout.reconfigure(encoding="utf-8")
AREAS = P.PASSES / "areas"
WIDE = "wide outline"
WIDEN_PX = 30  # 300 m


def measure(models, path, mask, wide):
    with rasterio.open(path) as src:
        s1 = src.read().astype(np.float32)
    s1[s1 == -32768] = np.nan
    x = np.full((len(M.BANDS),) + s1.shape[1:], np.nan, dtype=np.float32)
    x[M.BANDS.index("VV")], x[M.BANDS.index("VH")] = s1[0] / 100, s1[1] / 100  # stored as dB x 100
    vv = x[M.BANDS.index("VV")]
    row = {"outline_ha": round(float(mask.sum()) * 0.01, 1), "wide_outline_ha": round(float(wide.sum()) * 0.01, 1),
           "radar_gap_share": round(float(np.isnan(vv[mask]).mean()), 3)}
    for name, (model, ck) in models.items():
        w = M.predict(model, ck, x).astype(bool)
        lab, _ = ndimage.label(w & wide)
        joined = np.isin(lab, np.unique(lab[w & mask])) & (lab > 0)  # water in the wide outline that joins the lake
        row[name] = round(float((w & mask).sum()) * 0.01, 1)
        row[f"{name}, {WIDE}"] = round(float(joined.sum()) * 0.01, 1)
    with np.errstate(invalid="ignore"):
        row["VV < -16.75 dB"] = round(float(((vv < -16.75) & mask).sum()) * 0.01, 1)
    return row


def export_status():
    """reservoir -> its last status in the export logs."""
    return {r["reservoir"]: r["status"] for log in sorted(P.LOGS.glob("progress_part*.csv"))
            for r in csv.DictReader(open(log, encoding="utf-8"))}


def measured(name):
    """Measured on every date of its manifest (a test run measures only a few), with the wide outline too."""
    a, m = AREAS / f"{E.safe(name)}.csv", P.MANIFESTS / f"{E.safe(name)}.csv"
    if not (a.exists() and m.exists()):
        return False
    lines = open(a, encoding="utf-8").readlines()
    return WIDE in lines[0] and len(lines) == len(open(m, encoding="utf-8").readlines())


def measure_one(models, name):
    man = list(csv.DictReader(open(P.MANIFESTS / f"{E.safe(name)}.csv", encoding="utf-8")))
    _, occ = M.load(P.first_chip(name))
    mask = M.tank_mask(occ)
    wide = ndimage.binary_dilation(mask, iterations=WIDEN_PX)
    rows = [{**{k: r[k] for k in ("tank", "date", "orbit", "pass", "photo_date", "photo_clear")},
             **measure(models, P.CHIPS / r["file"], mask, wide)} for r in man]
    tmp = AREAS / f"{E.safe(name)}.part"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(AREAS / f"{E.safe(name)}.csv")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", action="store_true", help="also the three run 6 seeds")
    ap.add_argument("--wait", action="store_true", help="keep measuring beside the export until all are done")
    ap.add_argument("--only", default="", help="comma list of reservoir names (for a test)")
    args = ap.parse_args()
    AREAS.mkdir(parents=True, exist_ok=True)
    names = ["radar U-Net r2"] + ([n for n in A.MODELS if " r6 " in n] if args.seeds else [])
    models = {n: A.load_model(A.MODELS[n]) for n in names}
    every = [t["name"] for t in E.reservoirs()]
    started = time.time()
    while True:
        status = export_status()
        todo = args.only.split(",") if args.only else [n for n in every if status.get(n) == "done" and not measured(n)]
        for name in todo:
            k = measure_one(models, name)
            print(f"[{(time.time() - started) / 60:6.1f} min] {name}: {k} dates", flush=True)
        if args.only or not args.wait:
            break
        status = export_status()  # all exports finished (done or failed) and every done one measured
        if len(status) >= len(every) and all(measured(n) for n in every if status.get(n) == "done"):
            break
        if not todo:
            time.sleep(60)
    print("measuring finished", flush=True)


if __name__ == "__main__":
    main()
