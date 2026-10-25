"""A4 pictures: true-colour Sentinel-2 views of the 3 test tanks at high and low water.

Yellow line: tank outline (OpenStreetMap). Cyan: open water found by NDWI > 0.
Writes results/A_tank_images.png.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.windows import from_bounds

import a4_s2 as a4
import a4b_extra as a4b

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"


def read_bands(item, geom_utm, buffer_m, names):
    epsg = a4b.epsg_of(item)
    g = gpd.GeoSeries([geom_utm.buffer(buffer_m)], crs=a4.UTM).to_crs(epsg).iloc[0]
    out, tr = [], None
    with rasterio.Env(**a4.GDAL_ENV):
        for n in names:
            with rasterio.open(item.assets[n].href) as src:
                win = from_bounds(*g.bounds, transform=src.transform).round_offsets().round_lengths()
                out.append(src.read(1, window=win, boundless=True, fill_value=0).astype("float32"))
                tr = src.window_transform(win)
    h, w = out[0].shape
    extent = (tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f)
    return out, extent, epsg


def main():
    picks = a4.pick_tanks()
    pairs = pd.read_csv(RES / "A4_pairs.csv", dtype={"lake_id": str})
    scenes = pd.read_csv(RES / "A4_scenes.csv", dtype={"lake_id": str})
    cover = pd.read_csv(RES / "A4_village_cover.csv", dtype={"lake_id": str})
    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 5.2 * len(picks)))
    for row, (lake, outline, _) in zip(axes, picks):
        name = lake.tank_name if isinstance(lake.tank_name, str) and lake.tank_name else f"Village tank (lake {lake.lake_id})"
        items = a4b.items_for(outline.geometry)
        p = pairs[pairs.lake_id == lake.lake_id]
        for ax, (label, sel) in zip(row, (("High water", p.loc[p.wse.idxmax()]), ("Low water", p.loc[p.wse.idxmin()]))):
            sc = scenes[(scenes.lake_id == lake.lake_id) & (scenes.date == sel.s2_date)]
            item = items[sc["item"].iloc[0]]
            (r, g, b, n), extent, epsg = read_bands(item, outline.geometry, 300, ["red", "green", "blue", "nir"])
            rgb = np.clip(np.dstack([r, g, b]) / 3000.0, 0, 1) ** 0.8
            water = ((g - n) / np.maximum(g + n, 1) > 0) & (g > 0)
            ax.imshow(rgb, extent=extent)
            ax.imshow(np.ma.masked_where(~water, water), extent=extent, cmap=ListedColormap(["cyan"]), alpha=0.45)
            gpd.GeoSeries([outline.geometry], crs=a4.UTM).to_crs(epsg).boundary.plot(ax=ax, color="yellow", lw=1.2)
            extra = ""
            c = cover[(cover.lake_id == lake.lake_id) & (cover.s2_date == sel.s2_date)]
            if len(c):
                extra = (f"\nopen water {c.open_water_share.iloc[0]:.0%}, "
                         f"green plants {c.vegetation_share.iloc[0]:.0%} of the outline")
            ax.set_title(f"{name}, {label.lower()}\n{sel.s2_date}, SWOT height {sel.wse:.2f} m, "
                         f"NDWI water {sel.area_ha:.1f} ha{extra}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Sentinel-2 true colour. Yellow: tank outline. Cyan: open water found by NDWI > 0", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "A_tank_images.png", dpi=100)
    print("saved", RES / "A_tank_images.png")


if __name__ == "__main__":
    main()
