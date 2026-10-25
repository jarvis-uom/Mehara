"""A4 follow-up.

1. Nuwara Wewa again, pairing usable SWOT points (quality 0-1), to get more than 5 pairs.
2. Village tanks: on each paired date, the share of the tank outline that is open water
   (NDWI > 0) and green vegetation (NDVI > 0.3). If vegetation is high when SWOT says the
   tank is high, the water is under plants and NDWI cannot see it.
Writes results/A4_nuwara_usable_pairs.csv, results/A4_village_cover.csv and work/a4b_stats.json.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pystac_client import Client
from scipy.stats import spearmanr
from shapely.geometry import mapping

import a2_summarize as a2
import a4_s2 as a4

BASE = Path(__file__).resolve().parents[1]
WORK, RES = BASE / "work", BASE / "results"


def epsg_of(item):
    return f"EPSG:{item.properties.get('proj:epsg') or item.properties.get('proj:code', '').split(':')[-1]}"


def cover(item, geom_utm):
    """Open-water and vegetation shares inside the tank outline (no buffer)."""
    e = epsg_of(item)
    g, inside = a4.read_window(item.assets["green"].href, geom_utm, 0, e)
    r, _ = a4.read_window(item.assets["red"].href, geom_utm, 0, e)
    n, _ = a4.read_window(item.assets["nir"].href, geom_utm, 0, e)
    g, r, n = (x.astype("float32") for x in (g, r, n))
    ok = inside & (g > 0) & (r > 0) & (n > 0)
    ndwi = (g - n) / np.maximum(g + n, 1)
    ndvi = (n - r) / np.maximum(n + r, 1)
    tot = ok.sum()
    return {"open_water_share": float(((ndwi > 0) & ok).sum() / tot) if tot else np.nan,
            "vegetation_share": float(((ndvi > 0.3) & ok).sum() / tot) if tot else np.nan}


def items_for(geom_utm):
    ll = gpd.GeoSeries([geom_utm.buffer(500)], crs=a4.UTM).to_crs("EPSG:4326").iloc[0]
    client = Client.open("https://earth-search.aws.element84.com/v1")
    return {it.id: it for it in client.search(collections=["sentinel-2-l2a"], intersects=mapping(ll),
                                               datetime="2023-07-01/2026-10-25", max_items=2000).items()}


def main():
    picks = a4.pick_tanks()
    scenes = pd.read_csv(RES / "A4_scenes.csv", dtype={"lake_id": str})
    stats = {}

    # 1. Nuwara Wewa with usable SWOT points
    lake, outline, _ = picks[0]
    items = items_for(outline.geometry)
    sc = scenes[(scenes.lake_id == lake.lake_id) & (scenes.cloud_buffer < 0.30) & (scenes.cloud_tank <= 0.05)].copy()
    sc["date"] = pd.to_datetime(sc.date)
    swot = a2.load(lake.lake_id)
    swot = swot[swot.usable]
    pairs = []
    for _, p in swot.iterrows():
        d = (sc.date - p.time.tz_convert(None).normalize()).abs()
        if len(d) and d.min() <= pd.Timedelta(days=3):
            s = sc.loc[d.idxmin()]
            pairs.append({"swot_time": p.time, "wse": p.wse, "quality_f": p.quality_f,
                          "s2_date": s.date.date(), "item": s["item"]})
    pairs = pd.DataFrame(pairs).drop_duplicates(["swot_time"])
    with ThreadPoolExecutor(max_workers=6) as pool:
        pairs["area_ha"] = list(pool.map(lambda i: a4.water_area(items[i], outline.geometry), pairs["item"]))
    rho = spearmanr(pairs.area_ha, pairs.wse).statistic
    pairs.drop(columns="item").to_csv(RES / "A4_nuwara_usable_pairs.csv", index=False)
    stats["nuwara_usable"] = {"pairs": len(pairs), "spearman": round(float(rho), 2)}
    print("Nuwara Wewa, usable SWOT points:", stats["nuwara_usable"])

    # 2. Village tanks: open water vs vegetation on the paired dates
    pairs_all = pd.read_csv(RES / "A4_pairs.csv", dtype={"lake_id": str})
    rows = []
    for lake, outline, kind in picks[1:]:
        items = items_for(outline.geometry)
        sc = scenes[scenes.lake_id == lake.lake_id].set_index("date")
        pv = pairs_all[pairs_all.lake_id == lake.lake_id]
        for _, p in pv.iterrows():
            item_id = sc.loc[p.s2_date, "item"] if p.s2_date in sc.index else None
            if item_id is None or item_id not in items:
                continue
            c = cover(items[item_id], outline.geometry)
            rows.append({"lake_id": lake.lake_id, "outline_ha": round(outline.area_ha, 1), "s2_date": p.s2_date,
                         "wse": p.wse, "ndwi_area_ha": p.area_ha, **c})
    cov = pd.DataFrame(rows)
    cov.to_csv(RES / "A4_village_cover.csv", index=False)
    for lid, g in cov.groupby("lake_id"):
        hi = g[g.wse >= g.wse.median()]
        stats[f"village_{lid}"] = {
            "dates": len(g),
            "median_open_water_share": round(float(g.open_water_share.median()), 2),
            "median_vegetation_share": round(float(g.vegetation_share.median()), 2),
            "high_water_median_vegetation_share": round(float(hi.vegetation_share.median()), 2),
            "spearman_height_vs_vegetation": round(float(spearmanr(g.wse, g.vegetation_share).statistic), 2),
        }
        print(lid, stats[f"village_{lid}"])
    (WORK / "a4b_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
