r"""Pictures for checking reservoirs by eye (2 Nov 2026). It only reads chips and results and writes PNGs to previews/.
- previews/left_out_for_size.png: the reservoirs left out because their JRC water body is far smaller than the
  capacity suggests. A clear photo with the outline and the sheet's point (red cross), and the JRC water map.
- previews/worst_sites.png: reservoirs where the methods do not follow storage. The clear photos at the lowest and
  the highest storage, with the areas that the radar + optical U-Net (run 3) and the radar U-Net (run 2) found.
Run from the project folder (after reservoir_stats.py):
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\look_sites.py
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "temp/module-a-tanks/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reservoir_stats as S  # noqa: E402  (sheet, nearest)
import run_model as M  # noqa: E402  (load, tank_mask, photo)

sys.stdout.reconfigure(encoding="utf-8")
INK, SURFACE = "#0b0b0b", "#fcfcfb"
WORST = "Batalagoda,Sugaladewi,Saddhathissa,Balaharuwa"


def save(fig, name):
    out = ROOT / "previews" / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=80, facecolor=SURFACE)
    plt.close(fig)
    print("saved", out)


def left_out_for_size():
    flags = [f for f in csv.DictReader(open(ROOT / "out/reservoir_flags.csv", encoding="utf-8"))
             if "JRC area" in f["why_left_out"]]
    fig, axes = plt.subplots(2, len(flags), figsize=(3 * len(flags), 7.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, f in enumerate(flags):
        man = ROOT / "manifests" / f"{f['reservoir'].replace(' ', '_')}.csv"
        rows = list(csv.DictReader(open(man, encoding="utf-8")))
        r = max(rows, key=lambda r: (r["kind"] == "clear", float(r["clear"])))
        x, occ = M.load(ROOT / "chips" / r["file"])
        h, w = occ.shape
        axes[0, j].imshow(M.photo(x))
        axes[0, j].contour(M.tank_mask(occ), levels=[0.5], colors="yellow", linewidths=0.8)
        axes[1, j].imshow(occ, cmap="Blues", vmin=0, vmax=100)
        for ax in axes[:, j]:
            ax.plot(w / 2, h / 2, "r+", ms=10)
            ax.set_xticks([]), ax.set_yticks([])
        axes[0, j].set_title(f"{f['reservoir']}, {f['cap_mcm']} million m3\noutline {f['jrc_area_ha']} ha, about "
                             f"{f['expected_ha']} ha expected\n{r['s2_date']}", fontsize=8, color=INK)
        axes[1, j].set_title("JRC water, 1984 to 2021\n(darker = seen more often)", fontsize=8, color=INK)
    fig.suptitle("Left out for size: the outline (yellow) against what the capacity suggests", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "left_out_for_size.png")


def worst_sites(names):
    info, series = S.sheet()
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 8.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, name in enumerate(names):
        areas = ROOT / "areas" / f"{name.replace(' ', '_')}.csv"
        rows = [r for r in csv.DictReader(open(areas, encoding="utf-8")) if r["kind"] == "clear"]
        for r in rows:
            r["storage"] = S.nearest(series[name], date.fromisoformat(r["s2_date"]))
        rows = sorted((r for r in rows if r["storage"] is not None), key=lambda r: r["storage"])
        for i, r in enumerate((rows[0], rows[-1])):
            x, occ = M.load(ROOT / "chips" / r["file"])
            ax = axes[i, j]
            ax.imshow(M.photo(x))
            ax.contour(M.tank_mask(occ), levels=[0.5], colors="yellow", linewidths=0.7)
            pct = 100 * r["storage"] / info[name]["cap"]
            ax.set_title(f"{name}, {r['s2_date']}\nstorage {pct:.0f}% of capacity\n"
                         f"run 3: {r['fusion U-Net r3']} ha, radar U-Net: {r['radar U-Net r2']} ha",
                         fontsize=8, color=INK)
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Where the methods do not follow storage: clear days at the lowest (top) and highest (bottom) storage",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "worst_sites.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worst", default=WORST, help="comma list of reservoir names for worst_sites.png")
    args = ap.parse_args()
    left_out_for_size()
    worst_sites(args.worst.split(","))


if __name__ == "__main__":
    main()
