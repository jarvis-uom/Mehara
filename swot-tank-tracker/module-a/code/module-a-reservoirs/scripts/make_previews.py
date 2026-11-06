r"""Preview pictures of downloaded reservoir chips: for each reservoir, a clear day (photo and radar) and its cloudiest
day (photo and radar). It only reads the chips and writes new PNGs to previews/.
Run from the project folder:
    temp\feasibility-check\.venv\Scripts\python.exe -I temp\module-a-reservoirs\scripts\make_previews.py
Other reservoirs: ... make_previews.py --names Badagiriya,Batalagoda --out previews/more.png
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
INK, SURFACE = "#0b0b0b", "#fcfcfb"
PICKS = "Basavakkulama_Abaya_wewa,Ambalan_Oya,Badagiriya,Dambarawa,Angamuwa,Batalagoda"


def read(path):
    with rasterio.open(path) as src:
        a = src.read().astype(np.float32)
    rgb = np.stack([a[4], a[3], a[2]], -1)  # B4, B3, B2 (bands 3 to 12 are the optical ones)
    rgb = np.clip(np.where(rgb > 0, rgb, np.nan) / 2500, 0, 1) ** 0.8
    vv = np.where(a[0] == -32768, np.nan, a[0] / 100)
    return np.nan_to_num(rgb), np.clip((np.nan_to_num(vv, nan=-25) + 25) / 20, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=PICKS)
    ap.add_argument("--out", default="previews/reservoirs_sample.png")
    args = ap.parse_args()
    names = args.names.split(",")
    fig, axes = plt.subplots(len(names), 4, figsize=(12, 3.1 * len(names)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, name in zip(axes, names):
        rows = list(csv.DictReader(open(ROOT / "manifests" / f"{name}.csv", encoding="utf-8")))
        clear = [r for r in rows if r["kind"] == "clear"]
        cloudy = [r for r in rows if r["kind"] == "cloudy"]
        picks = [("clear", clear[len(clear) // 2] if clear else None),
                 ("cloudy", min(cloudy, key=lambda r: float(r["clear"])) if cloudy else None)]
        for k, (kind, r) in enumerate(picks):
            ax_photo, ax_radar = row[2 * k], row[2 * k + 1]
            if r is None:
                for ax in (ax_photo, ax_radar):
                    ax.axis("off")
                continue
            rgb, vv = read(ROOT / "chips" / r["file"])
            ax_photo.imshow(rgb)
            ax_radar.imshow(vv, cmap="gray")
            share = f", {100 * float(r['clear']):.0f}% of the water clear" if kind == "cloudy" else ""
            ax_photo.set_title(f"{r['tank']}\n{r['s2_date']}: {kind} photo{share}", fontsize=9, color=INK)
            ax_radar.set_title(f"\nradar {r['s1_date']} (water is dark)", fontsize=9, color=INK)
        for ax in row:
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Downloaded reservoir chips: a clear day and the cloudiest day, each with its radar image",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=70, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
