r"""Folders for the three private Kaggle datasets of the retraining (2 Nov 2026).

Kaggle runs everything (the team's choice), so it needs our chips and tables. This builds three
dataset folders in temp/module-a-retrain/kaggle-data/ from hard links (no extra disk space; the originals stay as
they are). The Kaggle CLI zips each top-level folder on upload, so each reservoir gets its own folder.
- tables: the department's daily storage, and the manifests, areas and flags of the first check and the every-pass
  check;
- first-check: the 3,843 chips of temp/module-a-reservoirs/chips (14 bands: VV, VH, 10 Sentinel-2 bands, JRC
  occurrence, Cloud Score+ x 100), one folder per reservoir;
- passes: the 16,552 radar chips of temp/module-a-reservoirs/passes/chips (VV, VH as dB x 100), one folder per
  reservoir.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-retrain\scripts\make_kaggle_data.py
Then upload with the commands printed at the end (TEMP on drive D:, as C: is nearly full).
"""
import csv
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RES = REPO / "temp/module-a-reservoirs"
OUT = ROOT / "kaggle-data"
OWNER = "mehararothila"
SETS = {"tables": ("is4990-module-a-tables", "IS4990 Module A tables"),
        "first-check": ("is4990-module-a-first-check-chips", "IS4990 Module A first-check chips"),
        "passes": ("is4990-module-a-pass-chips", "IS4990 Module A every-pass chips"),
        "passes-second": ("is4990-module-a-morning-pass-chips", "IS4990 Module A second-orbit pass chips")}


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def metadata(key):
    slug, title = SETS[key]
    folder = OUT / key
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump({"title": title, "id": f"{OWNER}/{slug}", "licenses": [{"name": "other"}]}, f, indent=2)
    return folder


def size_gb(folder):
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e9


def main():
    t = metadata("tables")
    for src in (REPO / "temp/model-check/storage_daily.csv", RES / "out/reservoir_flags.csv",
                RES / "passes/out/pass_flags.csv"):
        link(src, t / src.name)
    for sub, src in (("manifests", RES / "manifests"), ("areas", RES / "areas"),
                     ("pass_manifests", RES / "passes/manifests"), ("pass_areas", RES / "passes/areas")):
        for p in sorted(src.glob("*.csv")):
            link(p, t / sub / p.name)

    fc = metadata("first-check")
    n = 0
    for man in sorted((RES / "manifests").glob("*.csv")):
        with open(man, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                link(RES / "chips" / r["file"], fc / man.stem / r["file"])
                n += 1

    ps = metadata("passes")
    m = 0
    for folder in sorted((RES / "passes/chips").iterdir()):
        for p in sorted(folder.glob("*.tif")):
            link(p, ps / folder.name / p.name)
            m += 1

    p2 = metadata("passes-second")  # 6 Nov: the second (mostly morning) orbit, PROTOCOL-6; manifests included
    m2 = 0
    for folder in sorted((RES / "passes-second/chips").iterdir()):
        for q in sorted(folder.glob("*.tif")):
            link(q, p2 / folder.name / q.name)
            m2 += 1
    for q in sorted((RES / "passes-second/manifests").glob("*.csv")):
        link(q, p2 / "manifests" / q.name)
    print(f"second orbit {m2} chips, {size_gb(p2):.1f} GB")
    print(f"tables {size_gb(t):.3f} GB; first-check {n} chips, {size_gb(fc):.1f} GB; passes {m} chips, "
          f"{size_gb(ps):.1f} GB")
    kaggle = REPO / "temp/feasibility-check/.venv/Scripts/kaggle.exe"
    for key in SETS:
        print(f'"{kaggle}" datasets create -p "{OUT / key}" --dir-mode zip')
    if shutil.disk_usage(OUT).free < 5e9:
        print("warning: under 5 GB free on this drive")


if __name__ == "__main__":
    main()
