r"""Why the reading flattens near full (2 Nov 2026): a check after pass_stats.py.

At many reservoirs the storage read from the radar stops rising at about 60% of capacity (Huruluwewa and
Akathimuruppu in passes/out/pass_examples.png). Two checks, at the reservoirs pass_stats.py used:
- every pass (passes/areas): the share of the outline that the radar U-Net (run 2) calls water at middle storage
  (40 to 60% of capacity) and near full (85% and more), near full also in the wide outline, and on the later half of
  the passes near full the median of storage read minus storage (power law fitted on the first half, as in
  pass_stats.py; below 0 the reading is too low);
- the first check's clear days (areas/): the share of the outline that is water in the photo (NDWI >= -0.06) and on
  the radar within 2 days, by storage on the day. The photo shows how much of the outline the lake really covers.
Writes passes/out/high_water.csv (per reservoir) and passes/out/photo_vs_radar.csv (by storage on the day).
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\high_water.py
"""
import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir_stats as S  # noqa: E402  (sheet, nearest, fit, write)

sys.stdout.reconfigure(encoding="utf-8")
OUT = ROOT / "passes/out"
MAIN = "radar U-Net r2"
WIDE = "radar U-Net r2, wide outline"
PHOTO = "NDWI >= -0.06"
BANDS = ((0, 0.2, "under 20%"), (0.2, 0.4, "20 to 40%"), (0.4, 0.6, "40 to 60%"), (0.6, 0.85, "60 to 85%"),
         (0.85, 9, "85% and more"))
SIZES = (("all", 0, 1e9), ("under 5 million m3", 0, 5), ("5 to 30 million m3", 5, 30),
         ("30 million m3 and more", 30, 1e9))
FULL = 0.85
LOW_BY = 15  # reading too low near full by this many points of capacity or more


def band(share):
    return next(label for lo, hi, label in BANDS if lo <= share < hi)


def med(values, nd=2):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), nd) if v else None


def every_pass(info, series, used):
    out = []
    for path in sorted((ROOT / "passes/areas").glob("*.csv")):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        name = rows[0]["tank"]
        if name not in used:
            continue
        cap, o = info[name]["cap"], float(rows[0]["outline_ha"])
        keep = []
        for r in rows:
            s = S.nearest(series.get(name, {}), date.fromisoformat(r["date"]))
            if s is not None and float(r["radar_gap_share"]) <= 0.05:
                keep.append((r["date"], s / cap, float(r[MAIN]) / o, float(r[WIDE]) / o))
        keep.sort()
        s, a, w = (np.array([k[i] for k in keep]) for i in (1, 2, 3))
        half = len(keep) // 2
        f = S.fit(a[:half], s[:half])
        mid, full = (s >= 0.4) & (s < 0.6), s >= FULL
        later = full & (np.arange(len(s)) >= half)
        out.append({"reservoir": name, "cap_mcm": round(cap, 2), "outline_ha": o,
                    "passes_mid": int(mid.sum()), "passes_full": int(full.sum()),
                    "water_share_mid": med(a[mid]) if mid.sum() >= 3 else None,
                    "water_share_full": med(a[full]) if full.sum() >= 3 else None,
                    "wide_share_full": med(w[full]) if full.sum() >= 3 else None,
                    "later_full_passes": int(later.sum()),
                    "later_full_error_pct": med(100 * (f(a[later]) - s[later]), 1) if f and later.sum() >= 3 else None})
    return out


def photo_vs_radar(info, series, used):
    days = []
    for path in sorted((ROOT / "areas").glob("*.csv")):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["kind"] != "clear" or r["tank"] not in used:
                    continue
                s = S.nearest(series.get(r["tank"], {}), date.fromisoformat(r["s2_date"]))
                if s is not None:
                    cap, o = info[r["tank"]]["cap"], float(r["outline_ha"])
                    days.append({"cap": cap, "band": band(s / cap), "photo": float(r[PHOTO]) / o,
                                 "radar": float(r[MAIN]) / o})
    table = []
    for group, lo, hi in SIZES:
        for *_, label in BANDS:
            x = [d for d in days if d["band"] == label and lo <= d["cap"] < hi]
            table.append({"group": group, "storage": label, "clear_days": len(x),
                          "photo_water_share": med([d["photo"] for d in x]),
                          "radar_water_share": med([d["radar"] for d in x])})
    return table


def main():
    info, series = S.sheet()
    with open(OUT / "pass_flags.csv", encoding="utf-8") as f:
        used = {r["reservoir"] for r in csv.DictReader(f) if r["used"] == "True"}
    lakes = every_pass(info, series, used)
    print(f"Every pass, radar U-Net r2, {len(lakes)} reservoirs (medians): share of the outline that is water at 40 to 60%"
          f" storage, at 85% and more, in the wide outline at 85% and more; storage read minus storage near full"
          f" (later half, % of capacity); reservoirs reading {LOW_BY} points or more too low near full")
    for group, lo, hi in SIZES:
        x = [r for r in lakes if lo <= r["cap_mcm"] < hi]
        e = [r["later_full_error_pct"] for r in x if r["later_full_error_pct"] is not None]
        print(f"   {group:24s} {med([r['water_share_mid'] for r in x])!s:>5} {med([r['water_share_full'] for r in x])!s:>5}"
              f" {med([r['wide_share_full'] for r in x])!s:>5} {med(e, 1)!s:>6}   {sum(v <= -LOW_BY for v in e)} of {len(e)}")
    table = photo_vs_radar(info, series, used)
    print("\nFirst check's clear days: share of the outline that is water in the photo and on the radar, by storage:")
    for t in table:
        print(f"   {t['group']:24s} {t['storage']:13s} {t['clear_days']:4d} days   photo {t['photo_water_share']!s:>5}"
              f"   radar {t['radar_water_share']!s:>5}")
    S.write(OUT / "high_water.csv", lakes)
    S.write(OUT / "photo_vs_radar.csv", table)


if __name__ == "__main__":
    main()
