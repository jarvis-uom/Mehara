"""Study-area map in the deck's style, drawn at its slide size (6.2 x 5.15 in) so labels read on screen.

Water outlines from OpenStreetMap (feasibility work/a1_osm_water.gpkg); the 13 study tanks are the
OpenStreetMap outlines that contain each tank's SWOT centre point (else the nearest within 300 m).
Writes img/study_map.png.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
FEAS = ROOT.parent / "feasibility-check"
UTM = "EPSG:32644"
EXTENT = (80.385, 8.150, 80.790, 8.470)
RESERVOIRS = {"4580009523": ("Nuwara Wewa", (-0.004, 0.032), "center"),
              "4580006192": ("Mahakanadarawa", (0.0, 0.042), "center"),
              "4580051323": ("Nachchaduwa", (0.0, -0.040), "center"),
              "4580013332": ("Manankattiya", (-0.028, 0.035), "center"),
              "4580013372": ("Hurulu Wewa", (0.0, 0.058), "center")}
INK, MUTED, WATER, LATERITE, OTHER = "#142430", "#56656F", "#2A78D6", "#EB6834", "#D3DCE2"
HALO = [pe.withStroke(linewidth=3, foreground="white")]
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10.5})


def outline(osm, lon, lat):
    p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(osm.crs).iloc[0]
    inside = osm[osm.contains(p)]
    if len(inside):
        return inside.sort_values("area_ha").iloc[-1].geometry
    d = osm.distance(p)
    return osm.loc[d.idxmin()].geometry if d.min() < 300 else p.buffer(120)


def main():
    osm = gpd.read_file(FEAS / "work" / "a1_osm_water.gpkg")
    pld = pd.read_csv(FEAS / "results" / "A_pld_lakes.csv", dtype={"lake_id": str}).set_index("lake_id")
    vil = pd.read_csv(FEAS / "results" / "A6_village_shortlist.csv", dtype={"lake_id": str})
    big = {i: outline(osm, pld.loc[i, "p_lon"], pld.loc[i, "p_lat"]) for i in RESERVOIRS}
    small = {int(r.no): outline(osm, r.lon, r.lat) for r in vil.itertuples()}
    chosen = list(big.values()) + list(small.values())
    others = osm[~osm.geometry.apply(lambda g: any(g.equals(c) for c in chosen))]

    fig = plt.figure(figsize=(6.2, 5.15), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ext = gpd.GeoSeries.from_xy([EXTENT[0], EXTENT[2]], [EXTENT[1], EXTENT[3]], crs="EPSG:4326").to_crs(UTM)
    (x0, x1), (y0, y1) = ext.x.values, ext.y.values
    others.to_crs(UTM).plot(ax=ax, color=OTHER, lw=0, zorder=1)
    gpd.GeoSeries(list(big.values()), crs=osm.crs).to_crs(UTM).plot(ax=ax, color=WATER, lw=0, zorder=2)
    gpd.GeoSeries(list(small.values()), crs=osm.crs).to_crs(UTM).plot(ax=ax, color=LATERITE, lw=0, zorder=3)
    for lake_id, (name, (dx, dy), ha) in RESERVOIRS.items():
        c = big[lake_id].centroid
        lab = gpd.GeoSeries([Point(pld.loc[lake_id, "p_lon"] + dx, pld.loc[lake_id, "p_lat"] + dy)],
                            crs="EPSG:4326").to_crs(UTM).iloc[0]
        ax.text(lab.x, lab.y, name, ha=ha, va="center", fontsize=11, fontweight="semibold", color=INK,
                path_effects=HALO, zorder=6)
    offsets = {1: (-1, 1), 2: (1, 0), 3: (1, -1), 4: (-1, -1), 5: (1, 0), 6: (-1, 0), 7: (-1, 0), 8: (1, 0)}
    for no, g in small.items():
        c = g.centroid
        ox, oy = offsets[no]
        ax.text(c.x + ox * 650, c.y + oy * 650, str(no), ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=LATERITE, path_effects=HALO, zorder=7)
    city = gpd.GeoSeries([Point(80.4037, 8.3114)], crs="EPSG:4326").to_crs(UTM).iloc[0]
    ax.plot(city.x, city.y, marker="*", ms=11, color=INK, zorder=6)
    ax.text(city.x + 600, city.y - 1500, "Anuradhapura", fontsize=9.5, color=MUTED, path_effects=HALO, zorder=6)
    # scale bar and north arrow
    sx, sy = x0 + 2500, y0 + 2300
    ax.plot([sx, sx + 10_000], [sy, sy], color=INK, lw=2.2, solid_capstyle="butt", zorder=6)
    ax.text(sx + 5_000, sy + 700, "10 km", ha="center", va="bottom", fontsize=9.5, color=INK, path_effects=HALO, zorder=6)
    ax.annotate("N", xy=(x0 + 2600, y1 - 2000), xytext=(x0 + 2600, y1 - 6200), ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    handles = [Patch(color=WATER, label="Major reservoir (daily gauge)"),
               Patch(color=LATERITE, label="Village tank (numbered)"),
               Patch(color=OTHER, label="Other tank or water body"),
               Line2D([], [], marker="*", ls="", ms=10, color=INK, label="Anuradhapura city")]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=9, frameon=True, borderpad=0.6, handlelength=1.3)
    leg.get_frame().set_edgecolor("#D5DDE3")
    ax.text(x1 - 800, y0 + 800, "Water outlines © OpenStreetMap contributors", ha="right", va="bottom",
            fontsize=7.5, color=MUTED, path_effects=HALO, zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_color("#D5DDE3")
    fig.savefig(ROOT / "img" / "study_map.png", facecolor="white")
    plt.close(fig)
    print("study map saved; village tanks:", sorted(small))


if __name__ == "__main__":
    main()
