"""A7: Are our 5 big reservoirs the best SWOT lakes in the whole of Sri Lanka?

Downloads the latest LakeSP Prior file for every SWOT pass that crosses Sri Lanka, lists the
island's lakes, fetches the SWOT series of every lake of 50 ha or more from Hydrocron (same
fetch and quality rules as A2) and ranks the lakes of 1 km² or more by best-quality readings.
Names come from SWOT's lake list or, when it has none, from the Irrigation Department layer.

Writes results/A7_srilanka_lakes.csv.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import earthaccess
import geopandas as gpd
import pandas as pd
import requests

import a1_lake_ids as a1
from a2_fetch import fetch
from a2_summarize import jitter, load

BASE = Path(__file__).resolve().parents[1]
WORK, RES = BASE / "work", BASE / "results"
PASSES = ("023", "092", "273", "301", "398", "551", "579")  # CMR: every pass crossing Sri Lanka, cycles 056-057
W, S, E, N = 79.5, 5.85, 82.0, 9.9  # Sri Lanka; west of 79.5 E or north of 9.9 N is India
MIN_FETCH_KM2, MIN_RANK_KM2 = 0.5, 1.0
OUR_FIVE = {"4580051323": "Nachchaduwa", "4580013372": "Hurulu Wewa", "4580009523": "Nuwara Wewa",
            "4580006192": "Mahakanadarawa", "4580013332": "Manankattiya"}
DEPT = ("https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/arcgis/rest/services/"
        "Scheme_Data_2/FeatureServer/10/query")


def download():
    auth = earthaccess.login(strategy="netrc")
    assert auth.authenticated, "Earthdata login failed"
    out = WORK / "swot"
    files = []
    for pass_no in PASSES:
        title, url = a1.latest_prior(pass_no)
        path = out / Path(url).name
        if not path.exists():
            earthaccess.download([url], local_path=str(out))
        assert path.exists(), f"download failed: {title}"
        print(f"pass {pass_no}: {title} ({path.stat().st_size / 1e6:.1f} MB)")
        files.append((path, pass_no))
    return files


def island_lakes(files):
    frames = []
    for path, pass_no in files:
        g = gpd.read_file(f"zip://{path}", ignore_geometry=True,
                          columns=["lake_id", "lake_name", "p_lon", "p_lat", "p_ref_area"])
        g = g[g.p_lon.between(W, E) & g.p_lat.between(S, N)].copy()
        g["pass"] = pass_no
        frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    g["lake_name"] = g.lake_name.where(g.lake_name.notna() & (g.lake_name != "no_data"), "")
    return g.groupby("lake_id").agg(
        lake_name=("lake_name", "max"), p_lon=("p_lon", "first"), p_lat=("p_lat", "first"),
        ref_area_km2=("p_ref_area", "first"),
        passes=("pass", lambda s: "+".join(sorted(set(s))))).reset_index()


def dept_reservoir(lon, lat):
    """Irrigation Department reservoir at the lake's centre point, else within about 300 m (biggest
    if several). The layer is in EPSG:5235, which rejects distance buffers, so the fallback is a box."""
    sr = {"spatialReference": {"wkid": 4326}}
    d = 0.003
    tries = [("esriGeometryPoint", {"x": lon, "y": lat, **sr}),
             ("esriGeometryEnvelope", {"xmin": lon - d, "ymin": lat - d, "xmax": lon + d, "ymax": lat + d, **sr})]
    feats = []
    for gtype, geom in tries:
        params = {"geometry": json.dumps(geom), "geometryType": gtype,
                  "spatialRel": "esriSpatialRelIntersects",
                  "outFields": "Res_Name,Type,Rivr_Basin,Cap_FSL_MCM,Area_FSL_ha",
                  "returnGeometry": "false", "f": "json"}
        try:
            feats = requests.get(DEPT, params=params, headers=a1.HDR, timeout=60).json().get("features", [])
        except (requests.RequestException, ValueError):
            feats = []
        if feats:
            break
    if not feats:
        return {}
    a = max((f["attributes"] for f in feats), key=lambda x: x.get("Area_FSL_ha") or 0)
    return {"dept_name": a["Res_Name"], "dept_type": a["Type"], "basin": a["Rivr_Basin"],
            "capacity_mcm": a["Cap_FSL_MCM"]}


def main():
    lakes = island_lakes(download())
    big = lakes[lakes.ref_area_km2 >= MIN_FETCH_KM2].copy()
    print(f"{len(lakes)} SWOT lakes in Sri Lanka; {len(big)} of 50 ha or more; "
          f"{(lakes.ref_area_km2 >= MIN_RANK_KM2).sum()} of 1 km² or more")
    with ThreadPoolExecutor(max_workers=4) as pool:
        status = dict(pool.map(fetch, big.lake_id))
    counts = pd.Series(status).map(lambda s: s if s in ("ok", "cached", "no data") else "failed")
    print("Hydrocron:", counts.value_counts().to_dict())

    rows = []
    for lake in big.itertuples():
        df = load(lake.lake_id)
        good = df[df.good] if len(df) else df
        n, b = len(df), len(good)
        rows.append({
            "lake_id": lake.lake_id, "swot_name": lake.lake_name,
            "ref_area_km2": round(lake.ref_area_km2, 2), "passes": lake.passes,
            "readings": n, "best": b, "usable": int(df.usable.sum()) if n else 0,
            "best_share": round(b / n, 2) if n else None,
            "jitter_best_m": jitter(good) if b else None,
            "p_lon": lake.p_lon, "p_lat": lake.p_lat,
            "in_study_box": a1.W <= lake.p_lon <= a1.E and a1.S <= lake.p_lat <= a1.N,
            "our_pick": OUR_FIVE.get(lake.lake_id, ""),
            "fetch": status.get(lake.lake_id, "")})
    out = pd.DataFrame(rows)
    ranked = out[out.ref_area_km2 >= MIN_RANK_KM2].sort_values(["best", "usable"], ascending=False)
    out["rank"] = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    out = out.sort_values(["rank", "best"], ascending=[True, False], na_position="last")

    named = out[(out["rank"] <= 25) | (out.our_pick != "")]
    dept = {i: dept_reservoir(r.p_lon, r.p_lat) for i, r in named.iterrows()}
    for col in ("dept_name", "dept_type", "basin", "capacity_mcm"):
        out[col] = pd.Series({i: d.get(col) for i, d in dept.items()})
    first = ["rank", "lake_id", "swot_name", "dept_name", "our_pick", "ref_area_km2", "passes",
             "readings", "best", "usable", "best_share", "jitter_best_m"]
    out = out[first + [c for c in out.columns if c not in first]]
    out.to_csv(RES / "A7_srilanka_lakes.csv", index=False)

    show = ["rank", "lake_id", "swot_name", "dept_name", "basin", "ref_area_km2", "passes",
            "readings", "best", "usable", "best_share", "in_study_box", "our_pick"]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(out[out["rank"] <= 20][show].to_string(index=False))
        print(out[out.our_pick != ""][show].to_string(index=False))
    stats = {"island_lakes": len(lakes), "fetched": len(big),
             "ranked_1km2": int(out["rank"].notna().sum()), "hydrocron": counts.value_counts().to_dict()}
    (WORK / "a7_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
