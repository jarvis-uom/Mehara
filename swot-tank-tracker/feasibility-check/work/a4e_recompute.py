"""A4 fix: recompute water area for every pair with the corrected NDWI code.

Keeps the cloud screening from A4. Recomputes area inside the outline buffered by 100 m
(the brief) and inside the outline only (no buffer, so flooded paddy next to the tank
cannot count). Rewrites results/A4_pairs.csv and results/A_area_vs_height.png.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

import a4_s2 as a4
import a4b_extra as a4b

BASE = Path(__file__).resolve().parents[1]
WORK, RES = BASE / "work", BASE / "results"


def main():
    picks = a4.pick_tanks()
    pairs = pd.read_csv(RES / "A4_pairs.csv", dtype={"lake_id": str})
    scenes = pd.read_csv(RES / "A4_scenes.csv", dtype={"lake_id": str})
    nuw = pd.read_csv(RES / "A4_nuwara_usable_pairs.csv")
    stats, out = {}, []
    for lake, outline, kind in picks:
        name = lake.tank_name if isinstance(lake.tank_name, str) and lake.tank_name else f"Village tank (lake {lake.lake_id})"
        items = a4b.items_for(outline.geometry)
        p = pairs[pairs.lake_id == lake.lake_id].copy()
        if kind == "good":  # Nuwara Wewa: use the usable-point pairs too (quality 0-1)
            extra = nuw.rename(columns={"s2_date": "s2_date"})[["swot_time", "wse", "s2_date"]]
            extra = extra.assign(tank=p.tank.iloc[0], lake_id=lake.lake_id)
            p = pd.concat([p, extra], ignore_index=True).drop_duplicates("swot_time")
        sc = scenes[scenes.lake_id == lake.lake_id].set_index("date")
        p["item"] = [sc.loc[d, "item"] for d in p.s2_date]
        with ThreadPoolExecutor(max_workers=6) as pool:
            p["area_ha"] = list(pool.map(lambda i: a4.water_area(items[i], outline.geometry, 100), p["item"]))
            p["area_outline_ha"] = list(pool.map(lambda i: a4.water_area(items[i], outline.geometry, 0), p["item"]))
        rho = spearmanr(p.area_ha, p.wse).statistic
        rho0 = spearmanr(p.area_outline_ha, p.wse).statistic
        stats[name] = {"lake_id": lake.lake_id, "outline_ha": round(outline.area_ha, 1), "pairs": len(p),
                       "spearman_buffer100": round(float(rho), 2), "spearman_outline": round(float(rho0), 2),
                       "area_ha_range": [round(p.area_outline_ha.min(), 1), round(p.area_outline_ha.max(), 1)]}
        print(name, stats[name])
        p["tank"] = name
        out.append(p.drop(columns="item"))
    allp = pd.concat(out, ignore_index=True)
    allp.to_csv(RES / "A4_pairs.csv", index=False)
    old = json.loads((WORK / "a4_stats.json").read_text())
    old["fixed_pairs"] = stats
    (WORK / "a4_stats.json").write_text(json.dumps(old, indent=1, default=str))

    fig, axes = plt.subplots(1, len(stats), figsize=(5 * len(stats), 4.2))
    for ax, (name, s) in zip(axes, stats.items()):
        p = allp[allp.tank == name]
        ax.scatter(p.wse, p.area_outline_ha, s=20)
        ax.set_xlabel("SWOT water height (m)")
        ax.set_ylabel("Sentinel-2 open water inside the outline (ha)")
        ax.set_title(f"{name}\n{s['pairs']} pairs, Spearman {s['spearman_outline']}", fontsize=9)
    fig.suptitle("Water area from Sentinel-2 (NDWI > 0) against SWOT height, pairs at most 3 days apart",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "A_area_vs_height.png", dpi=110)
    print("plot saved")


if __name__ == "__main__":
    main()
