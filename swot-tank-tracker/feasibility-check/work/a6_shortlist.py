"""Option B: shortlist 5-8 village tanks next to the 5 big reservoirs, and draw a study-area map.

A candidate village tank (6.25-25 ha by SWOT reference area) needs a SWOT lake ID, at least
40 usable readings, a yearly rise and fall of at least 1 m, a median jump of 0.25 m or less
between readings up to 10 days apart, at most 5% gross outliers, its own OpenStreetMap
outline, and must lie within 12 km of one of the big five. Each candidate is checked on its
clearest Sentinel-2 scene from February-April 2026 (high water) for open water and plants.
Writes results/A6_village_shortlist.csv and results/A_study_area_map.png.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pystac_client import Client
from shapely.geometry import Point, mapping

import a2_summarize as a2
import a4_s2 as a4
import a4b_extra as a4b

BASE = Path(__file__).resolve().parents[1]
WORK, RES = BASE / "work", BASE / "results"
BIG = {"4580051323": "Nachchaduwa", "4580013372": "Hurulu Wewa", "4580009523": "Nuwara Wewa",
       "4580006192": "Mahakanadarawa", "4580013332": "Manankattiya"}


def outlier_share(lake_id):
    s = a2.load(lake_id)
    s = s[s.usable]
    if len(s) < 10:
        return np.nan
    med = s.wse.median()
    mad = 1.4826 * (s.wse - med).abs().median()
    return float(((s.wse - med).abs() > 3.5 * mad).mean()) if mad > 0 else 0.0


def best_scene_cover(outline_utm):
    ll = gpd.GeoSeries([outline_utm.buffer(500)], crs=a4.UTM).to_crs("EPSG:4326").iloc[0]
    items = list(Client.open("https://earth-search.aws.element84.com/v1").search(
        collections=["sentinel-2-l2a"], intersects=mapping(ll),
        datetime="2026-02-01/2026-04-30", max_items=100).items())
    best = None
    for it in sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))[:12]:
        try:
            s = a4.scl_stats(it, outline_utm)
        except Exception:
            continue
        ct = s.get("cloud_tank", np.nan)
        if ct == ct and ct <= 0.05 and s.get("valid_share", 0) > 0.9:
            best = it
            break
    if best is None:
        return {"s2_date": None, "open_water_share": np.nan, "plant_share": np.nan}
    c = a4b.cover(best, outline_utm)
    return {"s2_date": best.datetime.date(), "open_water_share": c["open_water_share"],
            "plant_share": c["vegetation_share"]}


def main():
    lakes = pd.read_csv(RES / "A2_heights_by_lake.csv", dtype={"lake_id": str})
    pld = pd.read_csv(RES / "A_pld_lakes.csv", dtype={"lake_id": str})
    lakes = lakes.merge(pld[["lake_id", "p_lon", "p_lat"]], on="lake_id")
    osm = gpd.read_file(WORK / "a1_osm_water.gpkg")
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(lakes.p_lon, lakes.p_lat)], crs="EPSG:4326").to_crs(a4.UTM)
    lakes["geom_pt"] = list(pts)

    def own_outline(p):
        inside = osm[osm.contains(p)]
        return inside.sort_values("area_ha").iloc[0] if len(inside) else None

    big_outlines = {lid: own_outline(lakes.loc[lakes.lake_id == lid, "geom_pt"].iloc[0]) for lid in BIG}
    cand = lakes[(lakes.size_class == "6.25-25 ha") & (lakes.usable_points >= 40) &
                 (lakes.rise_fall_usable_m >= 1.0) & (lakes.jitter_usable_m <= 0.25)].copy()
    rows = []
    for _, l in cand.iterrows():
        o = own_outline(l.geom_pt)
        if o is None or o.area_ha >= 60:
            continue
        d = {lid: big_outlines[lid].geometry.distance(o.geometry) / 1000 for lid in BIG if big_outlines[lid] is not None}
        near = min(d, key=d.get)
        if d[near] > 12:
            continue
        rows.append({"lake_id": l.lake_id, "name": l.tank_name if isinstance(l.tank_name, str) else "",
                     "osm_name": o.osm_name or "", "ref_area_ha": l.ref_area_ha, "outline_ha": round(o.area_ha, 1),
                     "lon": round(l.p_lon, 4), "lat": round(l.p_lat, 4),
                     "near_big": BIG[near], "km_to_big": round(d[near], 1),
                     "usable_points": l.usable_points, "rise_fall_m": l.rise_fall_usable_m,
                     "jitter_m": l.jitter_usable_m, "outline_geom": o.geometry})
    cand = pd.DataFrame(rows)
    cand["outlier_share"] = [outlier_share(i) for i in cand.lake_id]
    cand = cand[cand.outlier_share <= 0.05].reset_index(drop=True)
    print(f"{len(cand)} candidates pass the SWOT filters")
    with ThreadPoolExecutor(max_workers=4) as pool:
        covers = list(pool.map(best_scene_cover, cand.outline_geom))
    cand = pd.concat([cand, pd.DataFrame(covers)], axis=1)
    cand = cand[cand.s2_date.notna()]
    cand["water_type"] = np.where(cand.plant_share >= 0.5, "plant-covered",
                                  np.where(cand.open_water_share >= 0.5, "open water", "mixed"))

    # Pick the big reservoir with the most candidates nearby, then a tight group with a mix of types
    hub = cand.groupby("near_big").size().idxmax()
    group = cand[cand.near_big == hub].sort_values("km_to_big")
    picks = []
    for kind in ("open water", "plant-covered", "mixed"):
        picks += group[group.water_type == kind].head(3).index.tolist()
    picks = group.loc[sorted(set(picks), key=lambda i: group.loc[i, "km_to_big"])].head(8)
    if len(picks) < 5:
        picks = group.head(8)
    out = picks.drop(columns="outline_geom").round(2)
    out.insert(0, "no", range(1, len(out) + 1))
    out.to_csv(RES / "A6_village_shortlist.csv", index=False)
    cand.drop(columns="outline_geom").round(2).to_csv(WORK / "a6_all_candidates.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"Hub: {hub}. Candidates by hub:", cand.groupby("near_big").size().to_dict())
    print(out[["no", "name", "osm_name", "lake_id", "outline_ha", "km_to_big", "usable_points", "rise_fall_m",
               "open_water_share", "plant_share", "water_type", "s2_date"]].to_string(index=False))

    # Study-area map
    fig, ax = plt.subplots(figsize=(9, 9))
    osm.plot(ax=ax, color="#d9e6f2", edgecolor="#b0c4d8", lw=0.3)
    for lid, name in BIG.items():
        o = big_outlines[lid]
        gpd.GeoSeries([o.geometry], crs=a4.UTM).plot(ax=ax, color="#2b6cb0", edgecolor="#1a4480", lw=0.6)
        c = o.geometry.centroid
        ax.annotate(name, (c.x, c.y), fontsize=9, fontweight="bold", ha="center", va="center", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#1a4480", ec="none", alpha=0.85))
    for _, r in picks.iterrows():
        gpd.GeoSeries([r.outline_geom], crs=a4.UTM).plot(ax=ax, color="#e8833a", edgecolor="#9c4a10", lw=0.6)
        c = r.outline_geom.centroid
        n = out.loc[out.lake_id == r.lake_id, "no"].iloc[0]
        ax.annotate(str(n), (c.x, c.y), xytext=(6, 6), textcoords="offset points", fontsize=9, color="#9c4a10",
                    fontweight="bold")
    city = gpd.GeoSeries([Point(80.4037, 8.3114)], crs="EPSG:4326").to_crs(a4.UTM).iloc[0]
    ax.plot(city.x, city.y, "k*", ms=10)
    ax.annotate("Anuradhapura", (city.x, city.y), xytext=(6, -12), textcoords="offset points", fontsize=8)
    x0, x1 = ax.get_xlim()
    y0, _ = ax.get_ylim()
    ax.plot([x0 + 2000, x0 + 12000], [y0 + 2000, y0 + 2000], "k-", lw=2)
    ax.annotate("10 km", (x0 + 7000, y0 + 2600), ha="center", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Study area, option B: the 5 big reservoirs (blue) and the village tanks shortlisted near "
                 f"{hub} (orange, numbered)\nLight blue: other mapped water bodies (OpenStreetMap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "A_study_area_map.png", dpi=110)
    print("map saved")


if __name__ == "__main__":
    main()
