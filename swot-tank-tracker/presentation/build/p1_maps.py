"""Maps for the deck (PNG on the background colour of the slide they sit on).

1. swot_passes.png   Sri Lanka, the 7 SWOT passes and the study box (dark slide).
2. swot_heights.png  Every lake SWOT measured in the study box on one pass (white slide).
3. ranking.png       The 187 big lakes of Sri Lanka by best-quality readings (white slide).

Swath bands come from the lakes each pass covers: the pass's lakes fix the ground-track
direction (main axis) and the nadir (the middle of the empty strip between the swaths, else
the middle of the pass); each swath is then drawn 10-60 km either side of the nadir.
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap, Normalize
from shapely.geometry import LineString, Point, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
FEAS = ROOT.parent / "feasibility-check"
IMG, DATA = ROOT / "img", ROOT / "data"
SWOT_DIR = FEAS / "work" / "swot"
UTM = "EPSG:32644"
FILL = -1e10
HDR = {"User-Agent": "IS4990-presentation/1.0 (student project)"}
LKA_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/LKA/ADM0/"
           "geoBoundaries-LKA-ADM0_simplified.geojson")
STUDY = (80.40, 8.10, 80.80, 8.60)
OURS = {"4580051323": "Nachchaduwa", "4580013372": "Hurulu Wewa", "4580009523": "Nuwara Wewa",
        "4580006192": "Mahakanadarawa", "4580013332": "Manankattiya"}
OUR_PASSES = ("092", "273")

NAVY, INK, MUTED, WATER, LATERITE, SKY = "#0B2230", "#142430", "#56656F", "#2A78D6", "#EB6834", "#8FC1F2"
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11})


def sri_lanka():
    path = DATA / "lka_adm0.geojson"
    if not path.exists():
        r = requests.get(LKA_URL, headers=HDR, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return gpd.read_file(path).to_crs("EPSG:4326")


def prior_files():
    files = {}
    for p in sorted(SWOT_DIR.glob("SWOT_L2_HR_LakeSP_Prior_*_AS_*.zip")):
        files[p.name.split("_")[6]] = p
    return files


def swath_bands(path):
    """Left and right swath polygons (EPSG:4326) of one pass over the Sri Lanka region, and the
    nadir line."""
    g = gpd.read_file(f"zip://{path}", ignore_geometry=True, columns=["p_lon", "p_lat", "wse"])
    g = g[g.p_lon.between(77.5, 84.0) & g.p_lat.between(4.0, 12.5) & (g.wse > FILL)]
    pts = gpd.GeoSeries(gpd.points_from_xy(g.p_lon, g.p_lat), crs="EPSG:4326").to_crs(UTM)
    xy = np.c_[pts.x, pts.y]
    mean = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - mean, full_matrices=False)
    along, across = vt[0], vt[1]
    c = np.sort((xy - mean) @ across)
    lo, hi = np.percentile(c, [0.5, 99.5])
    mid = (lo + hi) / 2
    gaps = np.diff(c)
    near = np.abs((c[:-1] + c[1:]) / 2 - mid) < 30_000
    i = int(np.argmax(np.where(near, gaps, 0)))
    nadir = (c[i] + c[i + 1]) / 2 if gaps[i] > 8_000 else mid
    centre = mean + nadir * across
    bands = []
    for a, b in ((-60_000, -10_000), (10_000, 60_000)):
        quad = [centre + t * along + s * across for t, s in
                ((-900_000, a), (900_000, a), (900_000, b), (-900_000, b))]
        bands.append(Polygon(quad))
    line = LineString([centre - 900_000 * along, centre + 900_000 * along])
    return (gpd.GeoSeries(bands, crs=UTM).to_crs("EPSG:4326"),
            gpd.GeoSeries([line], crs=UTM).to_crs("EPSG:4326").iloc[0])


def map_passes(lka):
    fig, ax = plt.subplots(figsize=(3.6, 5.45), dpi=300)
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    frame = (79.30, 5.75, 82.10, 10.0)
    lka.plot(ax=ax, color="#173B4F", lw=0, zorder=1)
    centre = Point((STUDY[0] + STUDY[2]) / 2, (STUDY[1] + STUDY[3]) / 2)
    for pass_no, path in sorted(prior_files().items()):
        bands, nadir = swath_bands(path)
        ours = pass_no in OUR_PASSES
        covers = any(b.contains(centre) for b in bands)
        print(f"pass {pass_no}: covers study-box centre: {covers}")
        bands.clip(box(*frame)).plot(ax=ax, color="#5AA6F0" if ours else "#5B87A3",
                                     alpha=0.42 if ours else 0.22, lw=0, zorder=2)
        if ours:
            foot = nadir.intersection(LineString([(frame[0], 6.05), (frame[2], 6.05)]))
            if not foot.is_empty:
                ax.text(foot.x, 5.95, f"Pass {pass_no}", ha="center", va="top", color=SKY,
                        fontsize=11, fontweight="semibold", zorder=6)
    lka.boundary.plot(ax=ax, color="#A9C4D3", lw=0.7, zorder=3)
    sx = [STUDY[0], STUDY[2], STUDY[2], STUDY[0], STUDY[0]]
    sy = [STUDY[1], STUDY[1], STUDY[3], STUDY[3], STUDY[1]]
    ax.plot(sx, sy, color=LATERITE, lw=2.2, zorder=4)
    ax.annotate("Our study\narea", xy=(STUDY[2], STUDY[1] + 0.12), xytext=(81.30, 7.55), color="white",
                fontsize=12, fontweight="semibold", zorder=5, ha="left",
                arrowprops=dict(arrowstyle="-", color=LATERITE, lw=1.2))
    ax.set_xlim(frame[0], frame[2])
    ax.set_ylim(frame[1], frame[3])
    ax.set_aspect(1 / np.cos(np.radians(8)))
    ax.axis("off")
    fig.savefig(IMG / "swot_passes.png", bbox_inches="tight", pad_inches=0.03, facecolor=NAVY)
    plt.close(fig)


def map_heights():
    path = prior_files()["273"]
    g = gpd.read_file(f"zip://{path}")
    g = g.cx[STUDY[0]:STUDY[2], STUDY[1]:STUDY[3]].copy()
    g["seen"] = g.wse > FILL
    date = pd.to_datetime(g.loc[g.seen, "time_str"]).dt.date.mode()[0]
    seen = g[g.seen]
    print(f"heights map: pass 273 on {date}: {len(seen)} of {len(g)} lakes measured; "
          f"wse {seen.wse.min():.1f}-{seen.wse.max():.1f} m")
    fig = plt.figure(figsize=(4.05, 5.6), dpi=300, facecolor="white")
    ax = fig.add_axes([0.01, 0.115, 0.98, 0.88])
    ax.set_facecolor("#F5F7F8")
    g[~g.seen].plot(ax=ax, color="#C9D0D4", lw=0, zorder=1)
    norm = Normalize(vmin=80, vmax=150)
    seen.plot(ax=ax, column="wse", cmap=RAMP, norm=norm, lw=0.15, edgecolor="#0d366b", zorder=2)
    bands, _ = swath_bands(path)
    both = bands.iloc[0].union(bands.iloc[1])
    gap = gpd.GeoSeries([both.convex_hull.difference(both)], crs="EPSG:4326").clip(box(*STUDY))
    gap.plot(ax=ax, facecolor="none", edgecolor="#8A979F", hatch="////", lw=0, zorder=3)
    ax.text(STUDY[0] + 0.012, STUDY[3] - 0.012, "Blind strip:\nno readings",
            ha="left", va="top", fontsize=9, color=MUTED, zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C9D0D4", lw=0.5))
    offsets = {"Nuwara Wewa": (0.03, 0.05, "bottom"), "Mahakanadarawa": (0.045, 0.055, "bottom"),
               "Nachchaduwa": (0.05, 0.03, "bottom"), "Manankattiya": (-0.07, -0.07, "top"),
               "Hurulu Wewa": (-0.085, 0.10, "bottom")}
    for lake_id, name in OURS.items():
        r = g[g.lake_id == lake_id]
        if not len(r):
            continue
        r = r.iloc[0]
        p = r.geometry.representative_point()
        dx, dy, va = offsets[name]
        label = f"{name}\n{r.wse:.2f} m" if r.seen else f"{name}\nnot measured"
        ax.annotate(label, xy=(p.x, p.y), xytext=(p.x + dx, p.y + dy), fontsize=10,
                    fontweight="semibold", color=INK, zorder=7, ha="left", va=va,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C9D0D4", lw=0.6),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=0, shrinkB=2))
    ax.set_xlim(STUDY[0], STUDY[2])
    ax.set_ylim(STUDY[1], STUDY[3])
    ax.set_aspect(1 / np.cos(np.radians(8.35)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_color("#C9D0D4")
    cax = fig.add_axes([0.12, 0.062, 0.76, 0.017])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=RAMP, norm=norm), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=MUTED, length=0)
    cb.set_label("Water height (m above sea level)", fontsize=9.5, color=MUTED, labelpad=3)
    fig.savefig(IMG / "swot_heights.png", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    meta = {"date": str(date), "measured": int(len(seen)), "lakes": int(len(g)),
            "heights": {OURS[i]: (round(float(r.wse), 2) if r.seen else None)
                        for i, r in g[g.lake_id.isin(OURS)].set_index("lake_id").iterrows()}}
    (DATA / "swot_heights_meta.json").write_text(json.dumps(meta, indent=1))


def map_ranking(lka):
    d = pd.read_csv(FEAS / "results" / "A7_srilanka_lakes.csv", dtype={"lake_id": str})
    d = d[d["rank"].notna()].sort_values("best").reset_index(drop=True)
    fig = plt.figure(figsize=(3.8, 5.55), dpi=300, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    lka.plot(ax=ax, color="#EEF3F6", edgecolor="#AEBAC1", lw=0.6, zorder=1)
    norm = Normalize(vmin=0, vmax=85)
    size = 10 + 12 * np.sqrt(d.ref_area_km2.clip(upper=40))
    ax.scatter(d.p_lon, d.p_lat, s=size, c=d.best, cmap=RAMP, norm=norm, edgecolor="white", lw=0.4, zorder=2)
    ours = d[d.lake_id.isin(OURS)]
    ax.scatter(ours.p_lon, ours.p_lat, s=size[ours.index] + 45, facecolor="none", edgecolor=LATERITE,
               lw=1.8, zorder=3)
    lagoon = d[d.lake_id == "4580004952"].iloc[0]
    ax.annotate("1. Batticaloa\nLagoon (not\na tank)", xy=(lagoon.p_lon, lagoon.p_lat),
                xytext=(81.62, 8.30), ha="left", va="bottom", fontsize=9, color=MUTED, zorder=4,
                path_effects=HALO, arrowprops=dict(arrowstyle="-", color="#8A979F", lw=0.8))
    labels = {"4580035322": ("2. Tabbowa", (-7, -3), "right"),
              "4580026082": ("3. Ibbankatuwa", (7, -4), "left"),
              "4580018923": ("4. Udawalawe", (-7, 2), "right"),
              "4580018912": ("5. Mau Ara", (7, -6), "left")}
    for _, r in d[d.lake_id.isin(labels)].iterrows():
        text, off, ha = labels[r.lake_id]
        ax.annotate(text, xy=(r.p_lon, r.p_lat), xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=9.5, color=INK, zorder=4, path_effects=HALO)
    ax.annotate("Our 5\nreservoirs", xy=(80.70, 8.27), xytext=(81.02, 9.10),
                fontsize=10.5, fontweight="semibold", color=INK, zorder=4, path_effects=HALO,
                arrowprops=dict(arrowstyle="-", color=LATERITE, lw=1.2))
    ax.set_xlim(79.4, 82.25)
    ax.set_ylim(5.80, 9.95)
    ax.set_aspect(1 / np.cos(np.radians(8)))
    ax.axis("off")
    cax = fig.add_axes([0.60, 0.93, 0.36, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=RAMP, norm=norm), cax=cax, orientation="horizontal",
                      ticks=[0, 20, 40, 60, 80])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, colors=MUTED, length=0)
    cb.set_label("Best-quality readings", fontsize=9, color=MUTED, labelpad=3)
    cax.xaxis.set_label_position("top")
    fig.savefig(IMG / "ranking.png", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    lka = sri_lanka()
    map_passes(lka)
    map_heights()
    map_ranking(lka)
    print("maps saved")
