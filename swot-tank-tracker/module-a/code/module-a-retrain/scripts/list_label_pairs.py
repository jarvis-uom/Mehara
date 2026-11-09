r"""Every training image of runs 4 and 5 and where its label came from, as a readable table (8 Nov 2026).

Reads run4-inputs/res4.pkl (the training input of runs 4 and 5; it holds no storage numbers) and writes
out/label_pairs.csv, one row per training image:
- kind "clear photo": a clear Sentinel-2 photo (at least 98% of the water clear); its label is the photo's own water
  test; the radar of the same chip (within 2 days) learns it;
- kind "radar pass": an every-pass radar image; it learns the label of the nearest clear photo at most 5 days away
  (kernel3/retrain.py prepare(), "photo_near"; no storage check in runs 4 and 5).
Only reservoirs in training count; each reservoir's fold says which fold model tested it (and so did not train on it).
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\list_label_pairs.py
"""
import csv
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with open(ROOT / "run4-inputs/res4.pkl", "rb") as f:
        d = pickle.load(f)
    rows = []
    for n in sorted(d["used"]):
        R, k = d["res"][n], d["fold"][n] + 1
        for c in R["clear"]:
            rows.append({"reservoir": n, "fold": k, "kind": "clear photo", "radar_date": c["s1_date"],
                         "photo_date": c["s2_date"], "days_apart": c["days_apart"], "photo_clear": c["clear"],
                         "dry_bed": c.get("dry", False), "file": c["file"]})
        for p in R["passes"]:
            if p.get("photo_near"):
                photo = p["photo_near"]
                rows.append({"reservoir": n, "fold": k, "kind": "radar pass", "radar_date": p["date"],
                             "photo_date": photo.rsplit("_", 1)[1][:-4], "days_apart": p["near_gap"],
                             "photo_clear": "", "dry_bed": p.get("dry", False), "file": p["file"]})
    out = ROOT / "out/label_pairs.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    clear = sum(r["kind"] == "clear photo" for r in rows)
    print(f"{len(rows)} training images at {len(d['used'])} reservoirs: {clear} clear photos, "
          f"{len(rows) - clear} radar passes; wrote {out}")


if __name__ == "__main__":
    main()
