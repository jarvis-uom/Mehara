"""A4: Can we map water area from Sentinel-2? (no login)

Earth Search STAC, sentinel-2-l2a, since July 2023, windowed reads only.
Tanks: Nuwara Wewa and the 2 village tanks (6.25-25 ha) with the most usable SWOT points
that have an OpenStreetMap outline.
- Cloud: SCL classes 3, 8, 9, 10 inside a 500 m buffer; a scene is clear if under 30%.
- Water: NDWI = (green - nir) / (green + nir) > 0 inside the outline buffered by 100 m,
  on clear scenes within 3 days of a SWOT point, with at most 5% cloud on the tank itself.
Writes results/A_area_vs_height.png, results/A4_scenes.csv and results/A4_pairs.csv.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from scipy.stats import spearmanr
from shapely.geometry import Point, mapping

import a2_summarize as a2

BASE = Path(__file__).resolve().parents[1]
WORK, RES = BASE / "work", BASE / "results"
UTM = "EPSG:32644"
CLOUD = [3, 8, 9, 10]
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                VSI_CACHE="TRUE", VSI_CACHE_SIZE=str(64 * 1024 * 1024), GDAL_HTTP_MAX_RETRY="4",
                GDAL_HTTP_RETRY_DELAY="2", AWS_NO_SIGN_REQUEST="YES")


def pick_tanks():
    lakes = pd.read_csv(RES / "A2_heights_by_lake.csv", dtype={"lake_id": str})
    pld = pd.read_csv(RES / "A_pld_lakes.csv", dtype={"lake_id": str})
    lakes = lakes.merge(pld[["lake_id", "p_lon", "p_lat"]], on="lake_id")
    osm = gpd.read_file(WORK / "a1_osm_water.gpkg")  # in UTM

    def outline(lake):
        p = gpd.GeoSeries([Point(lake.p_lon, lake.p_lat)], crs="EPSG:4326").to_crs(UTM).iloc[0]
        inside = osm[osm.contains(p)]
        return inside.sort_values("area_ha").iloc[0] if len(inside) else None

    picks = []
    nuwara = lakes[lakes.tank_name.fillna("").str.contains("Nuwara")].iloc[0]
    picks.append((nuwara, outline(nuwara), "good"))
    village = lakes[(lakes.size_class == "6.25-25 ha")].sort_values("usable_points", ascending=False)
    for _, lake in village.iterrows():
        o = outline(lake)
        if o is not None and o.area_ha < 60:  # its own outline, not a bigger neighbour's
            picks.append((lake, o, "usable"))
        if len(picks) == 3:
            break
    return picks


def scene_list(geom_ll):
    client = Client.open("https://earth-search.aws.element84.com/v1")
    search = client.search(collections=["sentinel-2-l2a"], intersects=mapping(geom_ll),
                           datetime="2023-07-01/2026-10-25", max_items=2000)
    return list(search.items())


def read_window(href, geom_utm, buffer_m, epsg):
    """Read a window around geom (in the raster CRS); return array and a mask of pixels inside."""
    g = gpd.GeoSeries([geom_utm.buffer(buffer_m)], crs=UTM).to_crs(epsg).iloc[0]
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            win = from_bounds(*g.bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win, boundless=True, fill_value=0)
            tr = src.window_transform(win)
    inside = ~geometry_mask([mapping(g)], out_shape=arr.shape, transform=tr)
    return arr, inside


def scl_stats(item, geom_utm):
    epsg = f"EPSG:{item.properties.get('proj:epsg') or item.properties.get('proj:code', '').split(':')[-1]}"
    scl, inside = read_window(item.assets["scl"].href, geom_utm, 500, epsg)
    vals = scl[inside]
    valid = vals[(vals != 0) & (vals != 1)]
    if vals.size == 0 or valid.size < 0.5 * vals.size:
        return {"cloud_buffer": np.nan, "valid_share": valid.size / max(vals.size, 1)}
    scl_t, inside_t = read_window(item.assets["scl"].href, geom_utm, 0, epsg)
    vt = scl_t[inside_t]
    vt = vt[(vt != 0) & (vt != 1)]
    return {"cloud_buffer": float(np.isin(valid, CLOUD).mean()),
            "cloud_tank": float(np.isin(vt, CLOUD).mean()) if vt.size else np.nan,
            "valid_share": valid.size / vals.size}


def water_area(item, geom_utm, buffer_m=100):
    epsg = f"EPSG:{item.properties.get('proj:epsg') or item.properties.get('proj:code', '').split(':')[-1]}"
    g, inside = read_window(item.assets["green"].href, geom_utm, buffer_m, epsg)
    n, _ = read_window(item.assets["nir"].href, geom_utm, buffer_m, epsg)
    g, n = g.astype("float32"), n.astype("float32")
    # Raw values: 0 means no data. The +1000 offset of newer files is the same in both
    # bands, so it does not change the sign of NDWI; do not subtract it (that would drop
    # dark water pixels whose near-infrared value is close to the offset).
    ok = inside & (g > 0) & (n > 0)
    ndwi = np.where(ok, (g - n) / np.maximum(g + n, 1), -1)
    return float(((ndwi > 0) & ok).sum() / 100.0)  # 10 m pixels -> ha


def main():
    os.environ.update(GDAL_ENV)
    picks = pick_tanks()
    all_scenes, all_pairs, stats = [], [], {}
    for lake, outline, kind in picks:
        name = lake.tank_name if isinstance(lake.tank_name, str) and lake.tank_name else f"lake {lake.lake_id}"
        geom_utm = outline.geometry
        geom_ll = gpd.GeoSeries([geom_utm.buffer(500)], crs=UTM).to_crs("EPSG:4326").iloc[0]
        items = scene_list(geom_ll)
        print(f"{name}: {len(items)} Sentinel-2 items, outline {outline.area_ha:.1f} ha")
        with ThreadPoolExecutor(max_workers=6) as pool:
            sc = list(pool.map(lambda it: scl_stats(it, geom_utm), items))
        rows = []
        for it, s in zip(items, sc):
            rows.append({"tank": name, "lake_id": lake.lake_id, "item": it.id,
                         "date": it.datetime.date(), **s})
        scenes = pd.DataFrame(rows).sort_values(["date", "cloud_buffer"]).drop_duplicates("date")
        scenes["clear"] = scenes.cloud_buffer < 0.30
        all_scenes.append(scenes)

        swot = a2.load(lake.lake_id)
        swot = swot[swot.good] if kind == "good" else swot[swot.usable]
        clear = scenes[scenes.clear & (scenes.cloud_tank <= 0.05)].copy()
        clear["date"] = pd.to_datetime(clear.date)
        pairs = []
        for _, p in swot.iterrows():
            d = (clear.date - p.time.tz_convert(None).normalize()).abs()
            if len(d) and d.min() <= pd.Timedelta(days=3):
                pairs.append((p, clear.loc[d.idxmin()]))
        by_id = {it.id: it for it in items}
        with ThreadPoolExecutor(max_workers=6) as pool:
            areas = list(pool.map(lambda pc: water_area(by_id[pc[1]["item"]], geom_utm), pairs))
        prs = pd.DataFrame([{"tank": name, "lake_id": lake.lake_id, "swot_time": p.time, "wse": p.wse,
                             "s2_date": s.date.date(), "days_apart": abs((s.date - p.time.tz_convert(None).normalize()).days),
                             "area_ha": a} for (p, s), a in zip(pairs, areas)])
        all_pairs.append(prs)
        rho = spearmanr(prs.area_ha, prs.wse).statistic if len(prs) >= 5 else np.nan
        stats[name] = {"lake_id": lake.lake_id, "swot_points": kind, "outline_ha": round(outline.area_ha, 1),
                       "s2_dates": len(scenes), "clear_dates": int(scenes.clear.sum()),
                       "pairs": len(prs), "spearman": round(float(rho), 2) if rho == rho else None}
        print(" ", stats[name])

    scenes = pd.concat(all_scenes, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)
    scenes.to_csv(RES / "A4_scenes.csv", index=False)
    pairs.to_csv(RES / "A4_pairs.csv", index=False)
    scenes["month"] = pd.to_datetime(scenes.date).dt.month
    per_month = (scenes.groupby(["tank", "month"]).clear.sum() /
                 scenes.groupby("tank").date.apply(lambda d: pd.to_datetime(d).dt.year.nunique())).unstack()
    stats["clear_per_month_avg"] = per_month.round(1).to_dict(orient="index")
    print(per_month.round(1).to_string())

    fig, axes = plt.subplots(1, len(stats) - 1, figsize=(5 * (len(stats) - 1), 4.2))
    for ax, (name, s) in zip(np.atleast_1d(axes), [(k, v) for k, v in stats.items() if k != "clear_per_month_avg"]):
        p = pairs[pairs.tank == name]
        ax.scatter(p.wse, p.area_ha, s=18)
        ax.set_xlabel("SWOT water height (m)")
        ax.set_ylabel("Sentinel-2 water area (ha)")
        ax.set_title(f"{name} ({s['swot_points']} SWOT points)\n{s['pairs']} pairs, Spearman {s['spearman']}",
                     fontsize=9)
    fig.suptitle("Water area from Sentinel-2 (NDWI > 0) against SWOT height, pairs at most 3 days apart",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "A_area_vs_height.png", dpi=110)
    (WORK / "a4_stats.json").write_text(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
