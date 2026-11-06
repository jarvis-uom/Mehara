r"""Module A's water maps against the clear photo at our 103 reservoirs (4 Nov 2026), under ../PROTOCOL-PHOTO.md
(fixed before any scoring).

It only runs models that already exist (no training). For every clear first-check chip it scores:
- r2, r6 (3 seeds averaged), and run 3's retrained and photo-control ensembles of the reservoir's own fold;
- an Otsu radar threshold (run 2's otsu()).

They are scored against two references:
- A: the photo's water test (labels.py photo_class);
- B: Dynamic World (the private dataset is4990-module-a-dynamic-world, made by ../scripts/fetch_dynamic_world.py).

Each reservoir's counts are written as soon as it is done. It runs on Kaggle's CPU (no GPU hours). Quick local check
(2 reservoirs, 2 chips each):
    python -I photo_check.py --dry-run --input <project folder> --out <folder>
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy import ndimage, stats


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


try:
    import rasterio
except ImportError:
    pip_install("rasterio")
    import rasterio
try:
    import segmentation_models_pytorch as smp
except ImportError:
    pip_install("--no-deps", "segmentation-models-pytorch==0.6.0")
    import segmentation_models_pytorch as smp

import torch
import torch.nn.functional as F

R2, R6, NEW, CTRL, OTSU = "r2", "r6", "run 3", "run 3 control", "Otsu"
METHODS = (R2, R6, NEW, CTRL, OTSU)
REFS = ("A photo test", "B Dynamic World")
DW_WATER, DW_LAND, DW_FLOODED = (0,), (1, 2, 4, 5, 6, 7), 3
ACFT_TO_MCM = 1233.48 / 1e6
L = None  # labels.py, imported once its folder is known


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")  # as retrain.safe


def find(base, pattern, must=""):
    hits = sorted(p for p in Path(base).rglob(pattern) if must in p.as_posix())
    if not hits:
        raise FileNotFoundError(f"{pattern} ({must}) under {base}")
    return hits


def paths(args):
    if args.dry_run:
        b = Path(args.input)
        r6 = b / "temp/module-a-tanks/weights/run-6"
        return {"first": b / "temp/module-a-retrain/kaggle-data/first-check",
                "tables": b / "temp/module-a-retrain/kaggle-data/tables",
                "code": b / "temp/module-a-retrain/scripts",
                "folds": b / "temp/module-a-retrain/runs/2026-11-03-retrain-3/prep/folds.csv",
                "r2": b / "temp/module-a-tanks/weights/run-2-radar/unet_radar_best.pt",
                "r6": [r6 / "unet_radar_best.pt", r6 / "unet_radar_seed1_best.pt", r6 / "unet_radar_seed2_best.pt"],
                "run3": b / "temp/module-a-retrain/runs/2026-11-03-retrain-3", "run3_must": "part-",
                "dw": b / "temp/module-a-photo-check/dw"}
    i = Path(args.input)
    r6 = find(i, "unet_radar_seed1_best.pt", "module-a-run-6")[0].parent
    return {"first": find(i, "Akathimuruppu_2023-07-04.tif")[0].parents[1],
            "tables": find(i, "storage_daily.csv")[0].parent, "code": find(i, "labels.py")[0].parent,
            "folds": find(i, "folds.csv", "retrain-3-prep")[0],
            "r2": find(i, "unet_radar_best.pt", "module-a-run-2")[0],
            "r6": [r6 / "unet_radar_best.pt", r6 / "unet_radar_seed1_best.pt", r6 / "unet_radar_seed2_best.pt"],
            "run3": i, "run3_must": "retrain-3", "dw": find(i, "chips.csv", "dynamic-world")[0].parent}


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    classes = len(ck.get("classes") or (0, 0))
    model = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=len(ck["bands"]), classes=classes)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def run3_models(p, device):
    """{(fold, mode): [3 models]} from run 3's weights (Colab's copy of fold 4 left out, as in merge_run3.py)."""
    out = {}
    for w in find(p["run3"], "fold*_*_seed*.pt", p["run3_must"]):
        if "colab" in w.as_posix():
            continue
        fold, mode, _ = w.stem.split("_")
        out.setdefault((int(fold[4:]), mode), []).append(w)
    if len(out) != 10 or any(len(v) != 3 for v in out.values()):
        raise RuntimeError(f"run 3 weights: {[(k, len(v)) for k, v in sorted(out.items())]}")
    return {k: [load(w, device)[0] for w in sorted(v)] for k, v in out.items()}


# ---------- chips, as run 3's retrain.py ----------

def read_full(path):
    """VV, VH (dB) and 10 Sentinel-2 bands (NaN = no data); JRC occurrence; Cloud Score+."""
    with rasterio.open(path) as src:
        a = src.read().astype(np.float32)
    s1, s2 = a[:2], a[2:12]
    s1[s1 == -32768] = np.nan
    s1 /= 100
    s2[s2 == 0] = np.nan
    return np.concatenate([s1, s2]), a[12], a[13] / 100


def tank_mask(occ):
    """The JRC water body the export centred the chip on, widened by 30 m (as retrain.tank_mask)."""
    lab, n = ndimage.label(occ > 0)
    h, w = occ.shape
    if n == 0:
        return np.zeros(occ.shape, bool)
    sizes = ndimage.sum(np.ones(occ.shape), lab, index=np.arange(1, n + 1))
    off = [np.hypot((s[0].start + s[0].stop - 1) / 2 - (h - 1) / 2, (s[1].start + s[1].stop - 1) / 2 - (w - 1) / 2)
           for s in ndimage.find_objects(lab)]
    near = [i for i in range(n) if off[i] <= 5]
    k = 1 + (max(near, key=lambda i: sizes[i]) if near else int(np.argmin(off)))
    return ndimage.binary_dilation(lab == k, iterations=3)


def otsu(v):
    """Run 2's otsu(): 256 bins over -30 to 5 dB."""
    v = v[np.isfinite(v)]
    if v.size < 100:
        return -16.75
    hist, edges = np.histogram(np.clip(v, -30, 5), bins=256)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return float(edges[int(np.argmax(between)) + 1])


def water_maps(x, groups, mean, std, device):
    """Each method's water on one chip."""
    z = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)
    h, w = z.shape[1:]
    zt = F.pad(torch.from_numpy(z.astype(np.float32))[None], (0, (-w) % 32, 0, (-h) % 32), mode="reflect").to(device)
    out = {}
    with torch.no_grad():
        for method, rule, models in groups:
            pr = torch.stack([torch.softmax(m(zt).float()[0, :, :h, :w], 0) for m in models]).mean(0).cpu().numpy()
            out[method] = pr[1] > 0.5 if rule == "water" else pr[0] < 0.5
    vv = ndimage.median_filter(np.nan_to_num(x[0], nan=0.0), size=3)
    out[OTSU] = vv < otsu(np.where(np.isfinite(x[0]), vv, np.nan))
    return out


def counts(m, t, ok):
    return [int((m & t & ok).sum()), int((m & ~t & ok).sum()), int((~m & t & ok).sum()), int((~m & ~t & ok).sum())]


def scores(tp, fp, fn, tn):
    f = lambda a, b: round(100 * a / b, 2) if b else float("nan")  # noqa: E731
    return {"iou": f(tp, tp + fp + fn), "f1": f(2 * tp, 2 * tp + fp + fn), "precision": f(tp, tp + fp),
            "recall": f(tp, tp + fn)}


def storage_table(tables):
    cap, s = {}, {}
    with open(tables / "storage_daily.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["reservoir"]:
                continue
            cap[r["reservoir"]] = float(r["capacity_acft"])
            if r["storage_acft"]:
                s.setdefault(r["reservoir"], {})[date.fromisoformat(r["date"])] = float(r["storage_acft"])
    return cap, s


def share_on(series, cap, d, days=2):
    best = min(((abs((k - d).days), v) for k, v in series.items() if abs((k - d).days) <= days), default=None)
    return None if best is None or not cap else best[1] / cap


def size_group(mcm):
    return "under 5 million m3" if mcm < 5 else "5 to 30 million m3" if mcm < 30 else "30 million m3 and more"


# ---------- the job ----------

def score_reservoir(name, fold, p, models, base, mean, std, device, cap, series, limit, dw_index):
    groups = base + [(NEW, "not land", models[(fold, "rule")]), (CTRL, "not land", models[(fold, "photo")])]
    with open(p["tables"] / "manifests" / f"{safe(name)}.csv", encoding="utf-8") as f:
        chips = [r for r in csv.DictReader(f) if r["kind"] == "clear"]
    chips = chips[:limit] if limit else chips
    info, cnt = [], []
    for r in chips:
        x, occ, cs = read_full(p["first"] / safe(name) / r["file"])
        inside = tank_mask(occ)
        photo = L.photo_class(x, cs)
        radar = np.isfinite(x[0]) & np.isfinite(x[1])
        dw = None
        if r["file"] in dw_index:  # chips with no Dynamic World image (a cloudier tile) are not scored against B
            with rasterio.open(dw_index[r["file"]]) as src:
                dw = src.read(1)
        maps = water_maps(x[:2], groups, mean, std, device)
        refs = {REFS[0]: (photo == L.P_WATER, radar & (photo != L.P_NONE))}
        if dw is not None:
            scored = np.isin(dw, DW_WATER + DW_LAND) & (cs >= 0.6) & radar
            refs[REFS[1]] = (np.isin(dw, DW_WATER), scored)
        s = share_on(series.get(name, {}), cap.get(name), date.fromisoformat(r["s2_date"]))
        flooded = (dw == DW_FLOODED) & (cs >= 0.6) & radar if dw is not None else np.zeros(radar.shape, bool)
        row = {"reservoir": name, "fold": fold, "file": r["file"], "s2_date": r["s2_date"], "s1_date": r["s1_date"],
               "days_apart": r["days_apart"], "storage_share": None if s is None else round(s, 3),
               "low_water": s is not None and s < 0.2,
               "plants_share": round(float((photo == L.P_PLANTS)[inside].mean()), 3) if inside.any() else None,
               "dw": dw is not None, "dw_flooded_px": int(flooded.sum()),
               **{f"flooded called water: {m}": int((maps[m] & flooded).sum()) for m in METHODS}}
        info.append(row)
        for ref, (t, ok) in refs.items():
            for m in METHODS:
                tp, fp, fn, tn = counts(maps[m], t, ok)
                cnt.append({"reservoir": name, "file": r["file"], "reference": ref, "method": m,
                            "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return info, cnt


def write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summary(out, cap_mcm):
    """Pooled scores by reference, group and method; the median over reservoirs; the protocol's Wilcoxon tests."""
    info, cnt = [], []
    for f in sorted((out / "reservoirs").glob("*_chips.csv")):
        with open(f, encoding="utf-8") as fh:
            info += list(csv.DictReader(fh))
    for f in sorted((out / "reservoirs").glob("*_counts.csv")):
        with open(f, encoding="utf-8") as fh:
            cnt += list(csv.DictReader(fh))
    chip = {(r["reservoir"], r["file"]): r for r in info}
    groups = {"all": lambda c: True,
              "low water (under 20% full)": lambda c: c["low_water"] == "True",
              "plant-covered (plants 20% or more of the outline)": lambda c: c["plants_share"] not in ("", None)
              and float(c["plants_share"]) >= 0.2}
    for g in ("under 5 million m3", "5 to 30 million m3", "30 million m3 and more"):
        groups[g] = lambda c, g=g: size_group(cap_mcm.get(c["reservoir"], 0)) == g
    pooled, per_res = [], {}
    for ref in REFS:
        for gname, keep in groups.items():
            for m in METHODS:
                rows = [r for r in cnt if r["reference"] == ref and r["method"] == m
                        and keep(chip[(r["reservoir"], r["file"])])]
                a = [sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn", "tn")]
                res = {}
                for r in rows:
                    v = res.setdefault(r["reservoir"], [0, 0, 0, 0])
                    for i, k in enumerate(("tp", "fp", "fn", "tn")):
                        v[i] += int(r[k])
                ious = [scores(*v)["iou"] for v in res.values()]
                ious = [x for x in ious if np.isfinite(x)]
                pooled.append({"reference": ref, "group": gname, "method": m, "reservoirs": len(res),
                               "chips": len({(r["reservoir"], r["file"]) for r in rows}), **scores(*a),
                               "median_reservoir_iou": round(float(np.median(ious)), 2) if ious else None})
                if gname == "all":
                    per_res[(ref, m)] = {k: scores(*v)["iou"] for k, v in res.items()}
    tests = []
    for ref in REFS:
        for a, b in ((NEW, R6), (CTRL, NEW), (NEW, OTSU)):
            common = sorted(set(per_res.get((ref, a), {})) & set(per_res.get((ref, b), {})))
            d = [per_res[(ref, a)][k] - per_res[(ref, b)][k] for k in common
                 if np.isfinite(per_res[(ref, a)][k]) and np.isfinite(per_res[(ref, b)][k])]
            if len(d) >= 5:
                p = float(stats.wilcoxon(d)[1]) if any(x != 0 for x in d) else 1.0
                tests.append({"reference": ref, "first": a, "second": b, "reservoirs": len(d),
                              "median_gain": round(float(np.median(d)), 2),
                              "first_better_share": round(float(np.mean([x > 0 for x in d])), 2),
                              "wilcoxon_p": float(f"{p:.2g}")})
    fl = {m: sum(int(c[f"flooded called water: {m}"]) for c in info) for m in METHODS}
    total = sum(int(c["dw_flooded_px"]) for c in info)
    flooded = [{"method": m, "dw_flooded_px": total, "called_water_share": round(fl[m] / total, 3) if total else None}
               for m in METHODS]
    write(out / "pooled.csv", pooled)
    write(out / "tests.csv", tests)
    write(out / "flooded_vegetation.csv", flooded)
    write(out / "chips.csv", info)
    return {"pooled_all": [r for r in pooled if r["group"] == "all"], "tests": tests, "flooded": flooded,
            "chips": len(info), "chips_with_dw": sum(c["dw"] == "True" for c in info)}


def main():
    global L
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    p = paths(args)
    sys.path.insert(0, str(p["code"]))
    import labels
    L = labels
    out = Path(args.out)
    (out / "reservoirs").mkdir(parents=True, exist_ok=True)
    with open(p["folds"], encoding="utf-8") as f:
        folds = {r["reservoir"]: int(r["fold"]) for r in csv.DictReader(f)}
    names = sorted(folds)[:2] if args.dry_run else sorted(folds)
    r2, ck = load(p["r2"], device)
    base = [(R2, "water", [r2]), (R6, "water", [load(w, device)[0] for w in p["r6"]])]
    models = run3_models(p, device)
    mean, std = np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)
    cap, series = storage_table(p["tables"])
    cap_mcm = {k: v * ACFT_TO_MCM for k, v in cap.items()}
    dw_index = {f.name: f for f in Path(p["dw"]).rglob("*.tif")}  # wherever Kaggle unpacked them
    if not args.dry_run and len(dw_index) < 1000:
        raise RuntimeError(f"only {len(dw_index)} Dynamic World files found (expected 1,004)")
    log(f"protocol: PROTOCOL-PHOTO.md; labels.py {L.VERSION}; {len(names)} reservoirs; device {device}, "
        f"{torch.get_num_threads()} threads; {len(dw_index)} Dynamic World files")
    for i, name in enumerate(names, 1):
        t1 = time.time()
        info, cnt = score_reservoir(name, folds[name], p, models, base, mean, std, device, cap, series,
                                    2 if args.dry_run else 0, dw_index)
        write(out / "reservoirs" / f"{safe(name)}_chips.csv", info)
        write(out / "reservoirs" / f"{safe(name)}_counts.csv", cnt)
        log(f"{i}/{len(names)} {name}: {len(info)} chips in {time.time() - t1:.0f} s")
    res = summary(out, cap_mcm)
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "PROTOCOL-PHOTO.md", "labels_version": L.VERSION, "device": str(device),
                   "minutes": round((time.time() - t0) / 60, 1), "dry_run": args.dry_run, **res}, f, indent=2)
    for r in res["pooled_all"]:
        log(r["reference"], r["method"], "IoU", r["iou"], "F1", r["f1"], "P", r["precision"], "R", r["recall"],
            "median reservoir IoU", r["median_reservoir_iou"])
    log("tests:", res["tests"])
    log("flooded vegetation:", res["flooded"])
    log(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
