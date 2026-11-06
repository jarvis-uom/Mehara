r"""Radar chips on every Sentinel-1 pass of every reservoir, for the radar-only check (2 Nov 2026), step 1 of 3.

The first check (export_reservoirs.py) took up to 20 clear and 20 cloudy photo days per reservoir, each with radar
within 2 days. Radar does not need a photo, so this takes every pass instead: from the start of the department's
storage records (Oct 2020 for 73 reservoirs, mid-2023 for the others) to Oct 2026, on dates with storage on record
within 2 days. Each reservoir uses one relative orbit only, so one direction and one viewing angle: the orbit that made
most of its radar over the water in the first check (out/radar_passes.csv). Each chip is the mosaic of that orbit's
images on the date, on the grid of the reservoir's first-check chips: VV and VH (dB x 100, -32768 = no data).
For each date it also notes the nearest Sentinel-2 photo within 2 days and the share of the water it shows clear
(Cloud Score+, as in export_reservoirs.py), so the check can split clear and cloudy days.
Writes passes/chips/<reservoir>/<date>.tif (git-ignored), passes/manifests/<reservoir>.csv and one line per reservoir
in passes/logs/progress_part<k>.csv. It skips finished reservoirs and saved chips, so it can be stopped and started
again. Run from the project folder; parts can run at the same time:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\export_passes.py --part 1/4
With --orbit second (PROTOCOL-6, 5 Nov 2026) it takes each reservoir's second orbit instead: the most frequent orbit of
the other direction on its first-check radar dates (the morning pass for most), on the same grid and dates rule, and
writes to passes-second/ in the same layout.
"""
import argparse
import csv
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import ee
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_chips as X  # noqa: E402  (pixels, region, candidates, PROJECT)
import export_reservoirs as E  # noqa: E402  (reservoirs, safe, retry, candidates_mosaic, SHEET, MAX_REQUEST)

sys.stdout.reconfigure(encoding="utf-8")
PASSES = ROOT / "passes"
CHIPS, MANIFESTS, LOGS = PASSES / "chips", PASSES / "manifests", PASSES / "logs"
START, END = "2020-10-01", "2026-10-31"  # the years of the department's sheet
DAYS = 2  # storage on record, and the photo for the weather, within this many days


def storage_dates():
    out = {}
    for r in csv.DictReader(open(E.SHEET, encoding="utf-8")):
        if r["reservoir"] and r["storage_acft"] not in ("", "nan"):
            out.setdefault(r["reservoir"], set()).add(date.fromisoformat(r["date"]))
    return out


def usual_orbits():
    """reservoir -> the relative orbit that made most of its radar over the water in the first check."""
    tally = {}
    for r in csv.DictReader(open(ROOT / "out/radar_passes.csv", encoding="utf-8")):
        if r["set"] == "117 reservoirs" and r["main_orbit"]:
            tally.setdefault(r["tank"], Counter())[r["main_orbit"]] += 1
    return {t: int(c.most_common(1)[0][0]) for t, c in tally.items()}


def second_orbits():
    """reservoir -> the most frequent orbit of the other direction (ascending or descending) than its usual orbit, on its
    first-check radar dates (out/radar_passes.csv, orbits_on_date). Ties go to the lower orbit number."""
    usual, tally = usual_orbits(), {}
    for r in csv.DictReader(open(ROOT / "out/radar_passes.csv", encoding="utf-8")):
        if r["set"] == "117 reservoirs":
            for o in r["orbits_on_date"].split("+"):
                if o.strip():
                    num, ps = o.split()
                    tally.setdefault(r["tank"], Counter())[(int(num), ps)] += 1
    out = {}
    for t, c in tally.items():
        main = [ps for (num, ps) in c if num == usual.get(t)]
        other = [(n, (num, ps)) for (num, ps), n in c.items() if main and ps != main[0]]
        if other:
            out[t] = min(other, key=lambda v: (-v[0], v[1][0]))[1][0]
    return out


def first_chip(name):
    man = list(csv.DictReader(open(ROOT / "manifests" / f"{E.safe(name)}.csv", encoding="utf-8")))
    return ROOT / "chips" / man[0]["file"]


def grid_of(name):
    """The Earth Engine grid of the reservoir's first-check chips."""
    with rasterio.open(first_chip(name)) as src:
        a, w, h, crs = src.transform, src.width, src.height, src.crs.to_string()
    return {"dimensions": {"width": w, "height": h}, "crsCode": crs,
            "affineTransform": {"scaleX": a.a, "shearX": a.b, "translateX": a.c, "shearY": a.d, "scaleY": a.e,
                                "translateY": a.f}}


def orbit_images(g, orbit, a, b):
    """date -> [(image id, satellite, pass)] of the orbit's images over the chip."""
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(X.region(g)).filterDate(a, b)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)))
    cols = ["system:index", "system:time_start", "platform_number", "orbitProperties_pass"]
    out = {}
    for i, t, sat, ps in E.retry(s1.reduceColumns(ee.Reducer.toList(len(cols)), cols).get("list").getInfo):
        out.setdefault(str(date(1970, 1, 1) + timedelta(milliseconds=t)), []).append((i, f"Sentinel-1{sat}", ps))
    return out


def save(g, ids, path):
    img = (ee.ImageCollection([ee.Image(f"COPERNICUS/S1_GRD/{i}") for i in ids]).select(["VV", "VH"]).mosaic()
           .multiply(100).round().toInt16().unmask(-32768))
    step = max(1, int(E.MAX_REQUEST // (g["dimensions"]["width"] * g["dimensions"]["height"] * 3)))
    parts, profile = [], None
    for k in range(0, 2, step):
        arr, profile = E.retry(X.pixels, img.select(list(range(k, min(k + step, 2)))), g)
        parts.append(arr)
    profile.update(driver="GTiff", compress="deflate", count=2, dtype="int16")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(np.concatenate(parts).astype(np.int16))
        dst.descriptions = ("VV", "VH")
    tmp.replace(path)  # a stopped run never leaves a half-written chip behind


def one(t, orbit, stored):
    started = time.time()
    g = grid_of(t["name"])
    have = sorted(stored)
    a = str(max(date.fromisoformat(START), have[0] - timedelta(days=DAYS)))
    b = str(min(date.fromisoformat(END), have[-1] + timedelta(days=DAYS + 1)))
    by_day = orbit_images(g, orbit, a, b)
    X.START, X.END = a, b
    photos = {date.fromisoformat(p["date"]): p["clear"]
              for p in E.retry(X.candidates, g, "tank", 0.0, "cs_cdf", 1.01) or E.retry(E.candidates_mosaic, g)}
    rows = []
    for day in sorted(by_day):
        d = date.fromisoformat(day)
        if not any(d + timedelta(days=k) in stored for k in range(-DAYS, DAYS + 1)):
            continue
        items = by_day[day]
        path = CHIPS / E.safe(t["name"]) / f"{day}.tif"
        if not path.exists():
            save(g, [i for i, _, _ in items], path)
        near = [p for p in photos if abs((p - d).days) <= DAYS]
        photo = min(near, key=lambda p: abs((p - d).days)) if near else None
        rows.append({"tank": t["name"], "date": day, "file": f"{E.safe(t['name'])}/{day}.tif", "orbit": orbit,
                     "pass": "+".join(sorted({ps for *_, ps in items})),
                     "satellite": "+".join(sorted({sat for _, sat, _ in items})),
                     "images": ";".join(i for i, _, _ in items), "photo_date": str(photo) if photo else "",
                     "photo_clear": round(photos[photo], 3) if photo else ""})
    if rows:
        with open(MANIFESTS / f"{E.safe(t['name'])}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return {"status": "done", "orbit": orbit, "pass": rows[0]["pass"] if rows else "", "orbit_dates": len(by_day),
            "saved": len(rows), "with_photo": sum(bool(r["photo_date"]) for r in rows),
            "minutes": round((time.time() - started) / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="1/1", help="k/n: do every n-th reservoir, starting with the k-th")
    ap.add_argument("--only", default="", help="comma list of reservoir names (for a test)")
    ap.add_argument("--max-dates", type=int, default=0, help="save at most this many dates per reservoir (a test)")
    ap.add_argument("--orbit", choices=("usual", "second"), default="usual", help="second: the other direction's orbit")
    args = ap.parse_args()
    global PASSES, CHIPS, MANIFESTS, LOGS
    if args.orbit == "second":
        PASSES = ROOT / "passes-second"
        CHIPS, MANIFESTS, LOGS = PASSES / "chips", PASSES / "manifests", PASSES / "logs"
    k, n = (int(v) for v in args.part.split("/"))
    for d in (CHIPS, MANIFESTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=X.PROJECT)
    stored, orbits = storage_dates(), (second_orbits() if args.orbit == "second" else usual_orbits())
    todo = [t for i, t in enumerate(E.reservoirs()) if i % n == k - 1]
    if args.only:
        todo = [t for t in E.reservoirs() if t["name"] in args.only.split(",")]
    log = LOGS / f"progress_part{k}.csv"
    done = {r["reservoir"] for r in csv.DictReader(open(log, encoding="utf-8"))
            if r["status"] == "done"} if log.exists() and not args.only else set()
    print(f"part {k}/{n}: {len(todo)} reservoirs, {len(done)} already done", flush=True)
    for t in todo:
        if t["name"] in done:
            continue
        try:
            if t["name"] not in orbits:
                res = {"status": f"no {args.orbit} orbit in out/radar_passes.csv"}
            else:
                st = stored[t["name"]]
                if args.max_dates:  # a test: keep the first dates only
                    st = set(sorted(st)[:12 * args.max_dates])
                res = one(t, orbits[t["name"]], st)
        except Exception as e:
            res = {"status": f"error: {str(e)[:200]}"}
        line = {"reservoir": t["name"], "status": res["status"],
                **{c: res.get(c, "") for c in ("orbit", "pass", "orbit_dates", "saved", "with_photo", "minutes")},
                "finished": time.strftime("%Y-%m-%d %H:%M")}
        if args.only:
            print(line, flush=True)
            continue
        new = not log.exists()
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(line))
            if new:
                w.writeheader()
            w.writerow(line)
        print(" ".join(f"{v}" for v in line.values()), flush=True)
    print(f"part {k}/{n} finished", flush=True)


if __name__ == "__main__":
    main()
