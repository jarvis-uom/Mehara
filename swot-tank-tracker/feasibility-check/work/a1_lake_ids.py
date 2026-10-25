"""A1: Are the study-box tanks in SWOT's lake list?

Downloads the latest LakeSP Prior files for passes 273 and 092 (Asia), keeps the
lakes inside the study box, and matches them to Irrigation Department tanks and
OpenStreetMap water polygons.

Writes results/A_lake_ids.txt, results/A_pld_lakes.csv and results/A1_tanks.csv.
"""
import json
import time
from pathlib import Path

import earthaccess
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

BASE = Path(__file__).resolve().parents[1]
WORK, RES = BASE / "work", BASE / "results"
DEPT_CSV = BASE.parent / "data-checks" / "results" / "swot_coverage_anuradhapura_tanks.csv"
W, S, E, N = 80.40, 8.10, 80.80, 8.60  # study box
UTM = "EPSG:32644"
HDR = {"User-Agent": "IS4990-feasibility-check/1.0 (student project)"}
MATCH_M = 300
FILL = -1e10


def latest_prior(pass_no):
    r = requests.get(
        "https://cmr.earthdata.nasa.gov/search/granules.json",
        params={
            "short_name": "SWOT_L2_HR_LakeSP_D",
            "readable_granule_name": f"SWOT_L2_HR_LakeSP_Prior_*_{pass_no}_AS_*",
            "options[readable_granule_name][pattern]": "true",
            "sort_key": "-start_date",
            "page_size": 3,
        },
        headers=HDR, timeout=60)
    r.raise_for_status()
    entry = r.json()["feed"]["entry"][0]
    url = next(l["href"] for l in entry["links"]
               if l["href"].startswith("https") and l["href"].endswith(".zip"))
    return entry["title"], url


def download_priors():
    auth = earthaccess.login(strategy="netrc")
    assert auth.authenticated, "Earthdata login failed"
    out = WORK / "swot"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for pass_no in ("273", "092"):
        title, url = latest_prior(pass_no)
        path = out / Path(url).name
        if not path.exists():
            earthaccess.download([url], local_path=str(out))
        print(f"pass {pass_no}: {title} ({path.stat().st_size / 1e6:.1f} MB)")
        files.append((path, pass_no, title))
    return files


def load_pld(files):
    frames = []
    for path, pass_no, _ in files:
        g = gpd.read_file(f"zip://{path}")
        g["pass"] = pass_no
        frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    print("columns:", ", ".join(c for c in g.columns if c != "geometry"))
    g = g[(g.p_lon >= W) & (g.p_lon <= E) & (g.p_lat >= S) & (g.p_lat <= N)].copy()
    g["observed"] = g.wse > FILL
    rows = []
    for lake_id, grp in g.groupby("lake_id"):
        first = grp.iloc[0]
        row = {
            "lake_id": lake_id,
            "lake_name": first.get("lake_name", ""),
            "p_lon": first.p_lon,
            "p_lat": first.p_lat,
            "p_ref_area_ha": round(float(first.p_ref_area) * 100, 2),
            "passes": "+".join(sorted(grp["pass"])),
        }
        for _, r in grp.iterrows():
            p = r["pass"]
            row[f"wse_{p}"] = r.wse if r.observed else None
            row[f"area_total_{p}"] = r.area_total if r.observed else None
            row[f"quality_f_{p}"] = r.quality_f if r.observed else None
            row[f"xtrk_dist_{p}"] = r.xtrk_dist if r.observed else None
            row[f"time_{p}"] = r.time_str if r.observed else None
        geoms = [x for x in grp.geometry if x is not None and not x.is_empty]
        row["geometry"] = unary_union(geoms) if geoms else None
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=g.crs)


def osm_water():
    q = f"""[out:json][timeout:180];
(
  way["natural"="water"]({S},{W},{N},{E});
  relation["natural"="water"]({S},{W},{N},{E});
  way["landuse"="reservoir"]({S},{W},{N},{E});
  relation["landuse"="reservoir"]({S},{W},{N},{E});
);
out geom;"""
    cache = WORK / "a1_overpass.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        data, errors = None, []
        mirrors = ["https://overpass-api.de/api/interpreter",
                   "https://overpass.kumi.systems/api/interpreter",
                   "https://overpass.private.coffee/api/interpreter"]
        for attempt in range(3):
            for url in mirrors:
                try:
                    r = requests.post(url, data={"data": q}, headers=HDR, timeout=300)
                    r.raise_for_status()
                    data = r.json()
                    break
                except (requests.RequestException, ValueError) as e:
                    errors.append(f"{url}: {e}")
            if data:
                break
            time.sleep(20)
        if data is None:
            raise RuntimeError("Overpass failed: " + " | ".join(errors[-3:]))
        cache.write_text(json.dumps(data), encoding="utf-8")
    rows = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "way":
            pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(pts) < 4 or pts[0] != pts[-1]:
                continue
            geom = Polygon(pts)
        else:
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                     for m in el.get("members", [])
                     if m.get("role") == "outer" and m.get("geometry")]
            polys = list(polygonize(unary_union(lines))) if lines else []
            if not polys:
                continue
            geom = unary_union(polys)
        if not geom.is_valid:
            geom = geom.buffer(0)
        rows.append({"osm_id": f"{el['type']}/{el['id']}",
                     "osm_name": tags.get("name:en") or tags.get("name") or "",
                     "geometry": geom})
    g = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(UTM)
    g["area_ha"] = g.area / 1e4
    g = g[g.area_ha >= 1].sort_values("area_ha", ascending=False).reset_index(drop=True)
    # Drop polygons that mostly sit inside a bigger one (way + relation duplicates)
    geoms = list(g.geometry)
    tree = STRtree(geoms)
    keep = [True] * len(geoms)
    for i, gi in enumerate(geoms):
        for j in tree.query(gi):
            if j < i and keep[j] and gi.intersection(geoms[j]).area > 0.5 * gi.area:
                keep[i] = False
                break
    return g[keep].reset_index(drop=True)


def main():
    files = download_priors()
    pld = load_pld(files)
    pld_utm = pld.to_crs(UTM)
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(pld.p_lon, pld.p_lat)], crs="EPSG:4326").to_crs(UTM)
    shapes = [g if g is not None and not g.is_empty else p for g, p in zip(pld_utm.geometry, pts)]
    tree = STRtree(shapes)

    def lakes_near(geom, dist=MATCH_M):
        hits = []
        for j in tree.query(geom.buffer(dist)):
            d = min(geom.distance(shapes[j]), geom.distance(pts.iloc[j]))
            if d <= dist:
                hits.append((d, pld.lake_id.iloc[j]))
        return [lid for _, lid in sorted(hits)]

    # OpenStreetMap water polygons of 1 ha or more
    osm = osm_water()
    osm["lake_ids"] = [";".join(lakes_near(g)) for g in osm.geometry]
    osm["has_id"] = osm.lake_ids != ""

    # Irrigation Department tanks in the box
    dept = pd.read_csv(DEPT_CSV)
    dept = dept[(dept.lon >= W) & (dept.lon <= E) & (dept.lat >= S) & (dept.lat <= N)].copy()
    dpts = gpd.GeoSeries([Point(x, y) for x, y in zip(dept.lon, dept.lat)], crs="EPSG:4326").to_crs(UTM)
    direct, via_osm = [], []
    for p in dpts:
        direct.append(";".join(lakes_near(p)))
        near_osm = osm[osm.distance(p) <= MATCH_M]
        via_osm.append(";".join(i for ids in near_osm.lake_ids for i in ids.split(";") if i))
    dept["lake_ids_direct"] = direct
    dept["lake_ids_via_osm"] = via_osm
    dept["has_id"] = (dept.lake_ids_direct != "") | (dept.lake_ids_via_osm != "")

    # Name each PLD lake after the tank it matches, where possible
    names = {}
    for _, t in dept.iterrows():
        for lid in (t.lake_ids_direct + ";" + t.lake_ids_via_osm).split(";"):
            if lid and lid not in names:
                names[lid] = t["name"]
    for _, o in osm.iterrows():
        for lid in o.lake_ids.split(";"):
            if lid and lid not in names and o.osm_name:
                names[lid] = o.osm_name
    pld["tank_name"] = pld.lake_id.map(names).fillna("")

    # Save
    RES.mkdir(exist_ok=True)
    out = pld.drop(columns="geometry").sort_values("p_ref_area_ha", ascending=False)
    out.to_csv(RES / "A_pld_lakes.csv", index=False)
    with open(RES / "A_lake_ids.txt", "w") as f:
        f.write("# SWOT Prior Lake Database IDs in the study box 80.40-80.80E, 8.10-8.60N\n")
        for _, r in out.iterrows():
            f.write(f"{r.lake_id}  # {r.tank_name or 'unnamed'}, {r.p_ref_area_ha:g} ha\n")
    dept.to_csv(RES / "A1_tanks.csv", index=False)
    osm.drop(columns="geometry").to_csv(WORK / "a1_osm_water.csv", index=False)

    # Numbers for the summary
    big = osm[osm.area_ha >= 6.25]
    d_big = dept[dept.area_fsl_ha >= 6.25]
    share = big.has_id.mean() if len(big) else float("nan")
    stats = {
        "pld_lakes_in_box": len(pld),
        "pld_by_size_ha": {
            ">=25": int((pld.p_ref_area_ha >= 25).sum()),
            "6.25-25": int(((pld.p_ref_area_ha >= 6.25) & (pld.p_ref_area_ha < 25)).sum()),
            "1-6.25": int(((pld.p_ref_area_ha >= 1) & (pld.p_ref_area_ha < 6.25)).sum()),
            "<1": int((pld.p_ref_area_ha < 1).sum()),
        },
        "pld_seen_in_latest_file": {p: int(pld[f"wse_{p}"].notna().sum()) if f"wse_{p}" in pld else 0
                                    for p in ("273", "092")},
        "osm_water_ge_1ha": len(osm),
        "osm_ge_6.25ha": len(big),
        "osm_ge_6.25ha_with_id": int(big.has_id.sum()),
        "share_ge_6.25ha_with_id": round(share, 3),
        "osm_ge_25ha": int((osm.area_ha >= 25).sum()),
        "osm_ge_25ha_with_id": int(osm[osm.area_ha >= 25].has_id.sum()),
        "osm_1_6.25ha": int(((osm.area_ha >= 1) & (osm.area_ha < 6.25)).sum()),
        "osm_1_6.25ha_with_id": int(osm[(osm.area_ha >= 1) & (osm.area_ha < 6.25)].has_id.sum()),
        "dept_tanks_in_box": len(dept),
        "dept_ge_6.25ha": len(d_big),
        "dept_ge_6.25ha_with_id": int(d_big.has_id.sum()),
        "dept_without_id": dept.loc[~dept.has_id, "name"].tolist(),
        "files": [t for _, _, t in files],
        "verdict": "PASS" if share >= 0.7 else ("RISK" if share >= 0.4 else "FAIL"),
    }
    (WORK / "a1_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
