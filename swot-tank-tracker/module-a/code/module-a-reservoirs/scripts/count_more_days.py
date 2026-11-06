r"""Counts only, saves no chips (2 Nov 2026). Would more years or looser rules give enough clear days to the 7
reservoirs left out for too few (reservoir_stats.py needs 6)?

It lists every Sentinel-2 photo of each one from Oct 2020 to Oct 2026 (the years of the department's sheet), with
the clear share of the water (as in export_reservoirs.py), the nearest radar date, and whether storage is on record
within 2 days, in out/more_days_photos.csv. Then it prints the clear days each would have under each rule, counting
each radar image once. Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\count_more_days.py
"""
import csv
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import ee

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_chips as X  # noqa: E402  (candidates, region)
import export_reservoirs as E  # noqa: E402  (reservoirs, size_chip)

NAMES = ["Angamuwa", "Huruluwewa", "Nachchaduwa", "Nuwara Wewa", "Rajangana", "Thissa wewa", "Usgala Siyabalan"]
PERIODS = {"before": ("2020-10-01", "2023-07-01"), "ours": ("2023-07-01", "2026-01-01"),
           "after": ("2026-01-01", "2026-10-31")}


def s1_dates(g):
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(X.region(g)).filterDate(X.START, X.END)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    return sorted({date(1970, 1, 1) + timedelta(milliseconds=m) for m in s1.aggregate_array("system:time_start").getInfo()})


def usable(rows, name, periods, clear_min, gap_max):
    """Clear days the stats would use: storage within 2 days, radar within gap_max days, each radar image once."""
    return len({r["radar"] for r in rows if r["reservoir"] == name and r["period"] in periods and r["storage"]
                and r["clear"] >= clear_min and r["gap"] <= gap_max})


def main():
    X.START, X.END = PERIODS["before"][0], PERIODS["after"][1]
    ee.Initialize(project=X.PROJECT)
    storage = {}
    for r in csv.DictReader(open(E.SHEET, encoding="utf-8")):
        if r["reservoir"] in NAMES and r["storage_acft"] not in ("", "nan"):
            storage.setdefault(r["reservoir"], set()).add(date.fromisoformat(r["date"]))
    rows = []
    for t in [t for t in E.reservoirs() if t["name"] in NAMES]:
        g, _, _ = E.size_chip(t)
        photos = X.candidates(g, "tank", 0.0, "cs_cdf", 1.01)
        radar = s1_dates(g)
        for p in photos:
            d = date.fromisoformat(p["date"])
            near = min(radar, key=lambda r: abs((r - d).days))
            rows.append({"reservoir": t["name"], "date": p["date"], "clear": round(p["clear"], 3), "radar": str(near),
                         "gap": abs((near - d).days),
                         "period": next(k for k, (a, b) in PERIODS.items() if a <= p["date"] < b),
                         "storage": any(d + timedelta(days=k) in storage[t["name"]] for k in range(-2, 3))})
        print(t["name"], len(photos), "photos;", "radar dates by year:",
              dict(sorted(Counter(r.year for r in radar).items())), flush=True)
    with open(ROOT / "out/more_days_photos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nclear days usable (need 6):")
    print("reservoir | photos ours: >=98% / 70-98% | now (ours, 98%, 2 d) | ours, 95%, 3 d | all years, 98%, 2 d"
          " | all years, 95%, 3 d")
    for n in NAMES:
        ours = [r for r in rows if r["reservoir"] == n and r["period"] == "ours"]
        print(n, "|", sum(r["clear"] >= 0.98 for r in ours), "/", sum(0.7 <= r["clear"] < 0.98 for r in ours), "|",
              usable(rows, n, {"ours"}, 0.98, 2), "|", usable(rows, n, {"ours"}, 0.95, 3), "|",
              usable(rows, n, set(PERIODS), 0.98, 2), "|", usable(rows, n, set(PERIODS), 0.95, 3))


if __name__ == "__main__":
    main()
