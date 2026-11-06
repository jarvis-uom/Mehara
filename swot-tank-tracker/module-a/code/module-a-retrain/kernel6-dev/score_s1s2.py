r"""Module A on S1S2-Water's expert labels (3 Nov 2026), under ../PROTOCOL-S1S2.md (fixed before any scoring, commit
a2a101d; amendment 1, commit c05991d).

It only runs models that already exist (no training):
- r2, and r6 (3 seeds averaged);
- run 3's retrained and photo-control ensembles (5 folds x 3 seeds);
- an Otsu radar threshold (run 2's otsu()).

They are scored on S1S2-Water's test scenes (scene 31 left out) and on 5 Asian extras, with Earth Engine radar for
each scene's own Sentinel-1 products and S1S2-Water's expert masks resampled to 10 m and registered. Both come from
the private dataset is4990-s1s2water-gee, made by ../scripts/fetch_gee_radar.py. Then comes the speckle check:
Sen1Floods11's test split and Bolivia, scored as they are and after a 3 x 3 median.
With two GPUs, two threads share the scenes, one per GPU.
On Kaggle it runs with no arguments. Quick local check on the CPU (a 1100 x 1300 window of the first scene, 3
Sen1Floods11 chips per split):
    python -I score_s1s2.py --dry-run --cpu --input <project folder> --out <folder>
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

T, M = 1024, 64  # tile and margin: each tile is scored on its centre only
BATCH = 4
STRIDE = 16  # preview pictures: every 16th pixel
R2, R6, NEW, CTRL, OTSU = "r2", "r6", "run 3", "run 3 control", "Otsu"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------- 1. inputs ----------

def find(base, pattern, must):
    hits = sorted(p for p in Path(base).rglob(pattern) if must in p.as_posix())
    if not hits:
        raise FileNotFoundError(f"{pattern} ({must}) under {base}")
    return hits


def input_paths(args):
    """Weights, Sen1Floods11 and the prepared scenes: on Kaggle from the notebook outputs and the private datasets;
    locally from the project folder."""
    if args.dry_run:
        b = Path(args.input)
        r6 = b / "temp/module-a-tanks/weights/run-6"
        p = {"r2": b / "temp/module-a-tanks/weights/run-2-radar/unet_radar_best.pt",
             "r6": [r6 / "unet_radar_best.pt", r6 / "unet_radar_seed1_best.pt", r6 / "unet_radar_seed2_best.pt"],
             "run3": b / "temp/module-a-retrain/runs/2026-11-03-retrain-3",
             "s1f": b / "temp/module-a-kaggle/downloads/sen1floods11/v1.1",
             "scenes": b / "temp/s1s2water-check/gee"}
    else:
        i = Path(args.input)
        r6 = find(i, "unet_radar_seed1_best.pt", "module-a-run-6")[0].parent
        p = {"r2": find(i, "unet_radar_best.pt", "module-a-run-2")[0],
             "r6": [r6 / "unet_radar_best.pt", r6 / "unet_radar_seed1_best.pt", r6 / "unet_radar_seed2_best.pt"],
             "run3": i, "s1f": find(i, "flood_train_data.csv", "")[0].parents[2],
             "scenes": find(i, "scenes.csv", "s1s2water")[0].parent}
    run3 = {}
    for w in find(p["run3"], "fold*_*_seed*.pt", "part-" if args.dry_run else "retrain-3"):
        if "colab" in w.as_posix():
            continue
        fold, mode, _ = w.stem.split("_")
        run3.setdefault((int(fold[4:]), mode), []).append(w)
    p["groups"] = run3
    return p


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    classes = len(ck.get("classes") or (0, 0))
    model = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=len(ck["bands"]), classes=classes)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def load_groups(p, device):
    """Each scored ensemble: (method, fold, rule for water, models)."""
    groups = [(R2, "", "water", [load(p["r2"], device)[0]]), (R6, "", "water", [load(w, device)[0] for w in p["r6"]])]
    for (fold, mode), ws in sorted(p["groups"].items()):
        if len(ws) != 3:
            raise RuntimeError(f"fold {fold} {mode}: {len(ws)} seeds")
        groups.append((NEW if mode == "rule" else CTRL, fold + 1, "not land", [load(w, device)[0] for w in sorted(ws)]))
    ck = torch.load(p["r6"][0], map_location="cpu", weights_only=False)
    return groups, np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)


def water_maps(groups, zt):
    """Water of each ensemble on a batch (B, 2, h, w): mean softmax; water is p(water) > 0.5 for the two-class
    Sen1Floods11 models and p(land) < 0.5 for run 3's three-class models."""
    out = {}
    with torch.no_grad(), torch.autocast(zt.device.type, dtype=torch.float16, enabled=zt.device.type == "cuda"):
        for method, fold, rule, models in groups:
            pr = torch.stack([torch.softmax(m(zt).float(), 1) for m in models]).mean(0)
            out[(method, fold)] = pr[:, 1] > 0.5 if rule == "water" else pr[:, 0] < 0.5
    return out


# ---------- 2. scoring ----------

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


def counts(m, t, ok):
    return [int((m & t & ok).sum()), int((m & ~t & ok).sum()), int((~m & t & ok).sum()), int((~m & ~t & ok).sum())]


def scores(tp, fp, fn, tn):
    f = lambda a, b: round(100 * a / b, 2) if b else float("nan")  # noqa: E731
    return {"iou": f(tp, tp + fp + fn), "f1": f(2 * tp, 2 * tp + fp + fn), "precision": f(tp, tp + fp),
            "recall": f(tp, tp + fn)}


def tile_input(img, r0, c0, mean, std):
    """The tile with its margin, normalised (no data = 0), reflect-padded to (T + 2M) square at the scene's edges;
    and its VV in dB (NaN = no data)."""
    H, W = img.shape[1:]
    ra, rb, ca, cb = max(r0 - M, 0), min(r0 + T + M, H), max(c0 - M, 0), min(c0 + T + M, W)
    x = img[:, ra:rb, ca:cb].astype(np.float32)
    x[x == -32768] = np.nan
    x /= 100
    z = np.nan_to_num((x - mean[:, None, None]) / std[:, None, None], nan=0.0)
    pt, pb, pl, pr = ra - (r0 - M), (r0 + T + M) - rb, ca - (c0 - M), (c0 + T + M) - cb
    return np.pad(z, ((0, 0), (pt, pb), (pl, pr)), mode="reflect"), np.pad(x[0], ((pt, pb), (pl, pr)), mode="reflect")


def score_scene(radar_path, truth_path, groups, mean, std, device, crop=None):
    """Pixel counts of every method on one scene, its preview arrays and its Otsu threshold."""
    with rasterio.open(radar_path) as src:
        img = src.read()
    with rasterio.open(truth_path) as src:
        tr = src.read(1)
    if crop:
        r, c, h, w = crop
        img, tr = img[:, r:r + h, c:c + w], tr[r:r + h, c:c + w]
    ok = (tr != 255) & (img != -32768).all(0)
    truth = tr == 1
    H, W = ok.shape
    vvf = np.zeros((H, W), np.float16)
    prev = {k: np.zeros((-(-H // STRIDE), -(-W // STRIDE)), np.float32) for k in ("vv", "truth", R6, NEW, OTSU)}
    acc = {}
    tiles = [(r0, c0) for r0 in range(0, H, T) for c0 in range(0, W, T) if ok[r0:r0 + T, c0:c0 + T].any()]
    for b in range(0, len(tiles), BATCH):
        zs, keep = [], []
        for r0, c0 in tiles[b:b + BATCH]:
            h, w = min(T, H - r0), min(T, W - c0)
            z, vv = tile_input(img, r0, c0, mean, std)
            zs.append(z)
            vvf[r0:r0 + h, c0:c0 + w] = ndimage.median_filter(np.nan_to_num(vv, nan=0.0), size=3)[M:M + h, M:M + w]
            keep.append((r0, c0, h, w))
        maps = water_maps(groups, torch.from_numpy(np.stack(zs)).to(device))
        for j, (r0, c0, h, w) in enumerate(keep):
            t, o = truth[r0:r0 + h, c0:c0 + w], ok[r0:r0 + h, c0:c0 + w]
            pr = slice(r0 // STRIDE, r0 // STRIDE + -(-h // STRIDE))
            pc = slice(c0 // STRIDE, c0 // STRIDE + -(-w // STRIDE))
            for key, m in maps.items():
                mm = m[j, M:M + h, M:M + w].cpu().numpy()
                a = acc.setdefault(key, [0, 0, 0, 0])
                for i, v in enumerate(counts(mm, t, o)):
                    a[i] += v
                if key in ((R6, ""), (NEW, 1)):
                    prev[key[0]][pr, pc] = mm[::STRIDE, ::STRIDE]
            prev["truth"][pr, pc] = np.where(o, t, np.nan)[::STRIDE, ::STRIDE]
            prev["vv"][pr, pc] = np.where(img[0, r0:r0 + h, c0:c0 + w] == -32768, np.nan,
                                          img[0, r0:r0 + h, c0:c0 + w] / 100)[::STRIDE, ::STRIDE]
    vv_ok = ok & np.isfinite(vvf)
    thr = otsu(vvf[vv_ok].astype(np.float32))
    om = vvf.astype(np.float32) < thr
    acc[(OTSU, "")] = counts(om, truth, ok)
    prev[OTSU] = om[::STRIDE, ::STRIDE].astype(np.float32)
    info = {"scored_pixels": int(ok.sum()), "truth_water_share": round(float(truth[ok].mean()), 4) if ok.any() else None,
            "otsu_db": round(thr, 2)}
    return {"acc": acc, "prev": prev, "info": info}


def preview(path, row, res):
    """Radar, expert water, r6, run 3 fold 1 and Otsu, every 16th pixel (Figure without pyplot: thread-safe)."""
    from matplotlib.figure import Figure
    p, acc = res["prev"], res["acc"]
    iou = {k: scores(*v)["iou"] for k, v in acc.items()}
    panels = [("radar VV (dB)", p["vv"], "gray", (-25, 0)), ("expert water", p["truth"], "Blues", (0, 1)),
              (f"r6: IoU {iou[(R6, '')]}", p[R6], "Blues", (0, 1)),
              (f"run 3 fold 1: IoU {iou[(NEW, 1)]}", p[NEW], "Blues", (0, 1)),
              (f"Otsu: IoU {iou[(OTSU, '')]}", p[OTSU], "Blues", (0, 1))]
    fig = Figure(figsize=(20, 4.4))
    ax = fig.subplots(1, 5)
    for a, (title, arr, cmap, lim) in zip(ax, panels):
        a.imshow(arr, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
        a.set_title(title, fontsize=11)
        a.axis("off")
    fig.suptitle(f"S1S2-Water scene {row['scene']} ({float(row['lat']):.1f}, {float(row['lon']):.1f}), "
                 f"{row['landcover'][:55]}, {row['date_s1']}. Contains modified Copernicus Sentinel data (Earth "
                 f"Engine); expert water: Wieland et al. (2024)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def median3(x):
    """A 3 x 3 median on each band; no-data pixels stay no data."""
    out = np.empty_like(x)
    for b in range(x.shape[0]):
        bad = ~np.isfinite(x[b])
        filled = np.where(bad, np.nanmedian(x[b]) if (~bad).any() else 0.0, x[b])
        out[b] = np.where(bad, np.nan, ndimage.median_filter(filled, size=3))
    return out


def speckle_check(groups, mean, std, device, root, limit):
    """Sen1Floods11's test split and Bolivia, as they are and after a 3 x 3 median, with every model."""
    hand = Path(root) / "data/flood_events/HandLabeled"
    rows = []
    for split in ("bolivia", "test"):
        with open(Path(root) / f"splits/flood_handlabeled/flood_{split}_data.csv", encoding="utf-8") as f:
            ids = [ln.split(",")[0].strip().replace("_S1Hand.tif", "") for ln in f if ln.strip()]
        ids = ids[:limit] if limit else ids
        tot = {}
        for c in ids:
            with rasterio.open(hand / "S1Hand" / f"{c}_S1Hand.tif") as src:
                x = src.read([1, 2]).astype(np.float32)
            x[~np.isfinite(x)] = np.nan
            with rasterio.open(hand / "LabelHand" / f"{c}_LabelHand.tif") as src:
                y = src.read(1)
            ok, t = y >= 0, y == 1
            for version, xv in (("as is", x), ("3 x 3 median", median3(x))):
                z = np.nan_to_num((xv - mean[:, None, None]) / std[:, None, None], nan=0.0)
                h, w = z.shape[1:]
                zp = np.pad(z, ((0, 0), (0, (-h) % 32), (0, (-w) % 32)), mode="reflect")
                maps = water_maps(groups, torch.from_numpy(zp[None]).to(device))
                for key, m in maps.items():
                    a = tot.setdefault((version, key), [0, 0, 0, 0])
                    for i, v in enumerate(counts(m[0, :h, :w].cpu().numpy(), t, ok)):
                        a[i] += v
        for (version, (method, fold)), a in tot.items():
            rows.append({"split": split, "chips": len(ids), "version": version, "method": method, "fold": fold,
                         **scores(*a), "tp": a[0], "fp": a[1], "fn": a[2]})
    return rows


# ---------- 3. the jobs ----------

def worker(args, k, n, device, scenes):
    """Scores every n-th scene from the k-th on one device. Worker 0 then runs the speckle check."""
    p = input_paths(args)
    groups, mean, std = load_groups(p, device)
    log(f"worker {k}/{n} on {device}: {len(groups)} ensembles, {sum(len(g[3]) for g in groups)} models")
    out = Path(args.out)
    for row in scenes[k::n]:
        t0 = time.time()
        sid = row["scene"]
        res = score_scene(p["scenes"] / f"{sid}_radar.tif", p["scenes"] / f"{sid}_truth.tif", groups, mean, std,
                          device, (5000, 5000, 1100, 1300) if args.dry_run else None)
        keep = {c: row[c] for c in ("set", "scene", "lat", "lon", "landcover", "date_s1", "mask_m", "shift_dy",
                                    "shift_dx")}
        rows = [{**keep, **res["info"], "method": m, "fold": fd, **scores(*a), "tp": a[0], "fp": a[1], "fn": a[2],
                 "tn": a[3]} for (m, fd), a in sorted(res["acc"].items(), key=lambda kv: (kv[0][0], str(kv[0][1])))]
        write(out / "scenes" / f"{sid}.csv", rows)
        preview(out / "previews" / f"scene_{sid}.png", row, res)
        best = {r["method"]: r["iou"] for r in rows if r["fold"] in ("", 1)}
        log(f"scene {sid} ({row['set']}) in {time.time() - t0:.0f} s: water {res['info']['truth_water_share']}; "
            f"IoU {best}")
    if k == 0:
        rows = speckle_check(groups, mean, std, device, p["s1f"], 3 if args.dry_run else 0)
        write(out / "speckle.csv", rows)
        log("speckle check:", {(r["split"], r["version"], r["method"], r["fold"]): r["iou"] for r in rows
                               if r["fold"] in ("", 1)})


def summary(out, prepared):
    """Pooled scores by set and method, fold means for run 3 and its control, and the protocol's tests."""
    rows = []
    for f in sorted((out / "scenes").glob("*.csv")):
        with open(f, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    pooled, per_scene = [], {}
    for st in ("main", "extra"):
        sr = [r for r in rows if r["set"] == st]
        for m, fd in sorted({(r["method"], r["fold"]) for r in sr}):
            a = [sum(int(r[c]) for r in sr if (r["method"], r["fold"]) == (m, fd)) for c in ("tp", "fp", "fn", "tn")]
            pooled.append({"set": st, "method": m, "fold": fd, "scenes": len({r["scene"] for r in sr}), **scores(*a)})
        for r in sr:
            per_scene.setdefault((st, r["method"]), {}).setdefault(r["scene"], []).append(float(r["iou"]))
    table = []
    for st in ("main", "extra"):
        for m in (R2, R6, NEW, CTRL, OTSU):
            ps = [r for r in pooled if r["set"] == st and r["method"] == m]
            if not ps:
                continue
            row = {"set": st, "method": m, "scenes": ps[0]["scenes"], "folds": len(ps)}
            for s in ("iou", "f1", "precision", "recall"):
                v = [r[s] for r in ps]
                row[s] = round(float(np.mean(v)), 2)
                if len(v) > 1:
                    row[f"{s}_range"] = f"{min(v):.1f} to {max(v):.1f}"
            sc = [np.mean(v) for v in per_scene.get((st, m), {}).values()]
            sc = [x for x in sc if np.isfinite(x)]
            row["median_scene_iou"] = round(float(np.median(sc)), 2) if sc else None
            table.append(row)
    tests = []
    main = {m: {s: float(np.mean(v)) for s, v in per_scene.get(("main", m), {}).items()} for m in (R2, R6, NEW, CTRL, OTSU)}
    for a, b in ((NEW, R6), (NEW, CTRL), (R2, OTSU), (R6, OTSU), (NEW, OTSU), (CTRL, OTSU)):
        common = sorted(set(main[a]) & set(main[b]))
        d = [main[a][s] - main[b][s] for s in common if np.isfinite(main[a][s]) and np.isfinite(main[b][s])]
        if len(d) >= 5:
            p = float(stats.wilcoxon(d)[1]) if any(x != 0 for x in d) else 1.0
            tests.append({"first": a, "second": b, "scenes": len(d), "median_gain": round(float(np.median(d)), 2),
                          "first_better_share": round(float(np.mean([x > 0 for x in d])), 2),
                          "wilcoxon_p": float(f"{p:.2g}")})
    get = lambda m: next((r["iou"] for r in table if r["set"] == "main" and r["method"] == m), None)  # noqa: E731
    answers = {}
    if get(NEW) is not None and get(R6) is not None:
        answers["1. run 3 keeps water-mapping skill (pooled IoU >= r6 - 2)"] = (
            f"{'no loss' if get(NEW) >= get(R6) - 2 else 'a loss'}: run 3 {get(NEW)}, r6 {get(R6)}")
    if get(OTSU) is not None:
        answers["2. U-Nets against Otsu (pooled IoU)"] = {m: get(m) for m in (R2, R6, NEW, CTRL, OTSU)}
    if get(NEW) is not None and get(CTRL) is not None:
        answers["3. run 3 against its control (pooled IoU)"] = round(get(NEW) - get(CTRL), 2)
    left = [{"scene": r["scene"], "status": r["status"]} for r in prepared if r.get("status") != "ok"]
    write(out / "pooled_by_fold.csv", pooled)
    write(out / "summary.csv", table)
    write(out / "tests.csv", tests)
    write(out / "left_out.csv", left)
    return {"summary": table, "tests": tests, "answers": answers, "left_out": left}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--cpu", action="store_true", help="never use a GPU (the local check)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("scenes", "previews"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gpus = [] if args.cpu else [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    devices = [torch.device(f"cuda:{i}") for i in range(len(gpus))] or [torch.device("cpu")]
    if args.dry_run:
        devices = devices[:1]
    p = input_paths(args)
    with open(p["scenes"] / "scenes.csv", encoding="utf-8") as f:
        prepared = list(csv.DictReader(f))
    scenes = [r for r in prepared if r["status"] == "ok"]
    scenes = sorted(scenes, key=lambda r: (r["set"] != "main", int(r["scene"])))
    if args.dry_run:
        scenes = scenes[:1]
    log(f"protocol: PROTOCOL-S1S2.md (a2a101d, amendment 1 c05991d); {len(scenes)} scenes; {len(devices)} worker(s); "
        f"GPUs {gpus}")
    errors = []
    with ThreadPoolExecutor(len(devices)) as pool:  # one thread per GPU; the heavy work runs outside Python's lock
        jobs = [pool.submit(worker, args, k, len(devices), d, scenes) for k, d in enumerate(devices)]
        for k, job in enumerate(jobs):
            try:
                job.result()
            except Exception as e:
                errors.append(f"worker {k}: {e!r}")
                log(f"worker {k} failed: {e!r}")
    res = summary(out, prepared)
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "PROTOCOL-S1S2.md, commit a2a101d; amendment 1, commit c05991d", "workers": len(devices),
                   "gpus": gpus, "errors": errors, "minutes": round((time.time() - t0) / 60, 1),
                   "dry_run": args.dry_run, **res}, f, indent=2)
    log("summary:", json.dumps(res["summary"], indent=1))
    log("tests:", json.dumps(res["tests"], indent=1))
    log("answers:", res["answers"], "left out:", res["left_out"])
    log(f"done in {(time.time() - t0) / 60:.0f} min; errors {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
