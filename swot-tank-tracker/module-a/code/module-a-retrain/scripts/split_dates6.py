r"""Each reservoir's first-half dates for series 2's development (PROTOCOL-6, section B; 5 Nov 2026).

Rule 3 sorts a reservoir's scored passes by date (storage on record within 2 days, radar gap at most 5%) and tests on
the later half. Series 2 may develop only on the first half, and chooses the storage method by fitting on the first
quarter and checking on the second. This writes, per reservoir, the date where the later half starts ("later_from")
and where the second quarter starts ("q2_from"), from run 5's pass list, with the same filter as rule 3
(kernel3/retrain.py: sheet, nearest, MIN_PASSES).

The file holds dates and counts only, no storage, so it can go into the private Kaggle code dataset: the development
notebooks score only passes and clear chips dated before later_from.
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\split_dates6.py
"""
import csv
import importlib.util
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN5 = ROOT / "runs/2026-11-05-retrain-5"
OUT = ROOT / "kernel6-dev/split_dates.csv"


def load_kernel():
    spec = importlib.util.spec_from_file_location("retrain3", ROOT / "kernel3/retrain.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    return K


def main():
    K = load_kernel()
    info, series, _ = K.sheet(ROOT / "kaggle-data/tables")
    rows = []
    for f in sorted(RUN5.glob("part*/fold[1-5]/areas_passes/*.csv")):
        passes = list(csv.DictReader(open(f, encoding="utf-8")))
        if not passes:
            continue
        tank, fold = passes[0]["tank"], int(f.parents[1].name[4:])
        s = series.get(tank, {})
        keep = sorted((r for r in passes if K.nearest(s, date.fromisoformat(r["date"])) is not None
                       and float(r["radar_gap_share"]) <= 0.05), key=lambda r: r["date"])
        if len(keep) < K.MIN_PASSES:
            rows.append({"reservoir": tank, "fold": fold, "scored_passes": len(keep), "later_from": "", "q2_from": ""})
            continue
        half = len(keep) // 2
        rows.append({"reservoir": tank, "fold": fold, "scored_passes": len(keep), "first_half": half,
                     "later_from": keep[half]["date"], "q2_from": keep[half // 2]["date"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["reservoir", "fold", "scored_passes", "first_half", "q2_from", "later_from"])
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["later_from"]]
    print(f"{len(rows)} reservoirs, {len(ok)} with a split; later halves start from "
          f"{min(r['later_from'] for r in ok)} to {max(r['later_from'] for r in ok)}; wrote {OUT}")


if __name__ == "__main__":
    main()
