"""
Data access checks for the three IS4990 project candidates.

Run on your own computer (normal internet), for example from a terminal:
    pip install requests
    python check_access.py

Everything is saved in ./results next to this script. No login is needed
for any step here; steps that need a login (Earthdata, Earth Engine) are
listed in README.md.
"""

import csv
import io
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

TIMEOUT = 60
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "IS4990-data-check/1.0 (student project)"

# Areas of interest: (west, south, east, north)
SRI_LANKA = (79.5, 5.9, 82.0, 9.9)
ANURADHAPURA = (80.30, 8.20, 80.50, 8.40)   # SWOT tank cascade candidate
GAMPAHA = (79.90, 7.00, 80.10, 7.20)        # Paddy Land Watch
KALUTARA = (79.95, 6.50, 80.20, 6.70)       # Paddy Land Watch
SEASONS = {
    "Yala 2025": ("2025-04-01T00:00:00Z", "2025-09-30T23:59:59Z"),
    "Maha 2025/26": ("2025-10-01T00:00:00Z", "2026-03-31T23:59:59Z"),
    "Yala 2026": ("2026-04-01T00:00:00Z", "2026-09-30T23:59:59Z"),
}

summary = []


def note(line):
    print(line)
    summary.append(line)


def bbox_str(b):
    return ",".join(str(v) for v in b)


def get(url, **kw):
    r = SESSION.get(url, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------- SWOT
def swot_catalogue():
    """Count SWOT Lake Single-Pass (version D) files and list passes."""
    note("\n## SWOT Lake Single-Pass (CMR, no login)")
    base = "https://cmr.earthdata.nasa.gov/search/granules.json"
    for name, box in [("Sri Lanka", SRI_LANKA), ("Anuradhapura", ANURADHAPURA)]:
        titles, search_after = [], None
        while True:
            headers = {"CMR-Search-After": search_after} if search_after else {}
            r = SESSION.get(base, params={
                "short_name": "SWOT_L2_HR_LakeSP_D",
                "bounding_box": bbox_str(box),
                "page_size": 2000,
                "sort_key": "start_date",
            }, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            entries = r.json()["feed"]["entry"]
            titles += [e["title"] for e in entries]
            search_after = r.headers.get("CMR-Search-After")
            if not entries or not search_after:
                break
        passes = sorted({t.split("_")[6] for t in titles if t.count("_") > 7})
        first = titles[0] if titles else "none"
        last = titles[-1] if titles else "none"
        note(f"- {name}: {len(titles)} files; passes {', '.join(passes)}")
        note(f"  first {first}")
        note(f"  last  {last}")
        with open(OUT / f"swot_lakesp_files_{name.replace(' ', '_')}.txt", "w") as f:
            f.write("\n".join(titles))


def hydrocron(lake_ids_file=HERE / "lake_ids.txt"):
    """Pull SWOT height/area series for each lake ID listed in lake_ids.txt."""
    note("\n## Hydrocron (no login)")
    ids = []
    if lake_ids_file.exists():
        ids = [l.split("#")[0].strip() for l in lake_ids_file.read_text().splitlines()]
        ids = [i for i in ids if i.isdigit()]
    if not ids:
        ids = ["6350036102"]  # example lake from the PO.DAAC webinar, to prove the API answers
        note("- lake_ids.txt is empty, so this run uses the documentation's example lake.")
    url = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
    for lake in ids:
        r = SESSION.get(url, params={
            "feature": "PriorLake",
            "feature_id": lake,
            "start_time": "2023-07-01T00:00:00Z",
            "end_time": time.strftime("%Y-%m-%dT00:00:00Z"),
            "output": "csv",
            "fields": "lake_id,time_str,wse,area_total,quality_f",
        }, timeout=TIMEOUT)
        if r.status_code != 200:
            note(f"- {lake}: HTTP {r.status_code}")
            continue
        try:
            text = r.json()["results"]["csv"]
        except (ValueError, KeyError, TypeError):
            text = r.text
        rows = list(csv.DictReader(io.StringIO(text)))
        clean, seen = [], set()
        for row in rows:
            key = (row.get("time_str"), row.get("wse"))
            if key in seen or row.get("time_str") in (None, "", "no_data"):
                continue
            try:
                if float(row.get("wse", "nan")) < -1e10:   # fill value
                    continue
            except ValueError:
                continue
            seen.add(key)
            clean.append(row)
        with open(OUT / f"hydrocron_{lake}.csv", "w", newline="") as f:
            if clean:
                w = csv.DictWriter(f, fieldnames=list(clean[0].keys()))
                w.writeheader()
                w.writerows(clean)
        note(f"- {lake}: {len(rows)} rows returned, {len(clean)} after removing duplicates and fill values")


def irrigation_tank_layer():
    """Download every tank in the Irrigation Department Scheme_Data_2 layer."""
    note("\n## Irrigation Department tank layer (ArcGIS, no login)")
    url = ("https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/arcgis/rest/services/"
           "Scheme_Data_2/FeatureServer/10/query")
    rows, offset = [], 0
    while True:
        data = get(url, params={
            "where": "1=1", "outFields": "*", "returnGeometry": "false", "f": "json",
            "resultOffset": offset, "resultRecordCount": 1000,
        }).json()
        feats = data.get("features", [])
        rows += [f["attributes"] for f in feats]
        if not feats or not data.get("exceededTransferLimit"):
            break
        offset += len(feats)
    if rows:
        with open(OUT / "irrigation_tanks.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    by_type = {}
    for r in rows:
        by_type[r.get("Type")] = by_type.get(r.get("Type"), 0) + 1
    with_cap = sum(1 for r in rows if (r.get("Cap_FSL_MCM") or 0) > 0)
    note(f"- {len(rows)} tanks; by type {by_type}; {with_cap} have a capacity value")


def irrigation_dashboard_sheet():
    """Save the public Google Sheet tabs behind the reservoir dashboards."""
    note("\n## Irrigation Department reservoir status sheet (public Google Sheet)")
    base = ("https://docs.google.com/spreadsheets/d/e/"
            "2PACX-1vTcSGhi9RESl7CMCl1TQnrKe07Gx5Q696YiSB9jneIHqIP9lifpqSErgI3D5k9KtQXSdW5JpycIIr5e/pub")
    for label, gid in [("major", "1238978530"), ("medium", "698704031")]:
        try:
            r = get(base, params={"gid": gid, "single": "true", "output": "csv"})
            path = OUT / f"reservoir_status_{label}_{time.strftime('%Y%m%d')}.csv"
            path.write_bytes(r.content)
            n = max(len(r.text.splitlines()) - 1, 0)
            note(f"- {label}: {n} rows saved to {path.name}; keep running daily to build a history")
        except requests.RequestException as e:
            note(f"- {label}: failed ({e})")


# ---------------------------------------------------------------- Sentinel and NISAR
def sentinel2_counts():
    """Count Sentinel-2 L2A scenes and clear scenes (<20% cloud) per season."""
    note("\n## Sentinel-2 L2A scenes (Earth Search, no login)")
    url = "https://earth-search.aws.element84.com/v1/search"
    note("| Site | Season | All scenes | Under 20% cloud |")
    note("| --- | --- | --- | --- |")
    for site, box in [("Gampaha", GAMPAHA), ("Kalutara", KALUTARA), ("Anuradhapura", ANURADHAPURA)]:
        for season, (t0, t1) in SEASONS.items():
            counts = []
            for query in (None, json.dumps({"eo:cloud_cover": {"lt": 20}})):
                params = {"collections": "sentinel-2-l2a", "bbox": bbox_str(box),
                          "datetime": f"{t0}/{t1}", "limit": 1}
                if query:
                    params["query"] = query
                d = get(url, params=params).json()
                counts.append(d.get("numberMatched", d.get("context", {}).get("matched")))
            note(f"| {site} | {season} | {counts[0]} | {counts[1]} |")


def asf_search(platform, box, start, end, extra=None, label=""):
    point = f"POINT({(box[0] + box[2]) / 2:.3f} {(box[1] + box[3]) / 2:.3f})"
    params = {"platform": platform, "intersectsWith": point, "start": start, "end": end,
              "output": "csv", "maxResults": 2000}
    if extra:
        params.update(extra)
    r = get("https://api.daac.asf.alaska.edu/services/search/param", params=params)
    path = OUT / f"asf_{platform}_{label}.csv"
    path.write_bytes(r.content)
    return list(csv.DictReader(io.StringIO(r.text)))


def sentinel1_dates():
    note("\n## Sentinel-1 GRD over Gampaha (ASF, no login)")
    rows = asf_search("Sentinel-1", GAMPAHA, "2026-04-01T00:00:00Z",
                      time.strftime("%Y-%m-%dT00:00:00Z"),
                      {"processingLevel": "GRD_HD"}, "gampaha")
    dates = sorted({(r.get("Acquisition Date") or r.get("Start Time") or "")[:10] for r in rows})
    gaps = []
    for a, b in zip(dates, dates[1:]):
        try:
            gaps.append((time.mktime(time.strptime(b, "%Y-%m-%d")) - time.mktime(time.strptime(a, "%Y-%m-%d"))) / 86400)
        except ValueError:
            pass
    note(f"- {len(rows)} scenes since 1 Apr 2026 on {len(dates)} dates; gaps in days: {sorted(set(int(g) for g in gaps))}")


def nisar_products():
    note("\n## NISAR products (ASF search, no login; downloads need Earthdata)")
    for site, box in [("Gampaha", GAMPAHA), ("Anuradhapura", ANURADHAPURA)]:
        rows = asf_search("NISAR", box, "2025-07-30T00:00:00Z",
                          time.strftime("%Y-%m-%dT00:00:00Z"), None, site.lower())
        kinds = {}
        for r in rows:
            k = r.get("Processing Level") or r.get("processingLevel") or "?"
            kinds[k] = kinds.get(k, 0) + 1
        note(f"- {site}: {len(rows)} products; by type {kinds}")


# ---------------------------------------------------------------- OpenStreetMap
def osm_counts():
    note("\n## OpenStreetMap counts for Moratuwa (Overpass, no login)")
    q = """[out:json][timeout:60];
area["name"="Moratuwa"]->.a;
way(area.a)["power"~"^(line|minor_line)$"];out count;
node(area.a)["power"="pole"];out count;
node(area.a)["natural"="tree"];out count;
way(area.a)["landuse"="farmland"];out count;"""
    r = SESSION.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=120)
    r.raise_for_status()
    labels = ["power lines (ways)", "power poles", "trees", "farmland areas"]
    for label, el in zip(labels, r.json().get("elements", [])):
        note(f"- {label}: {el.get('tags', {}).get('total', '?')}")


def main():
    steps = [swot_catalogue, hydrocron, irrigation_tank_layer, irrigation_dashboard_sheet,
             sentinel2_counts, sentinel1_dates, nisar_products, osm_counts]
    for step in steps:
        try:
            step()
        except Exception as e:  # keep going so one blocked service does not stop the rest
            note(f"- {step.__name__} failed: {e}")
    (OUT / "summary.md").write_text("# Data access check, " + time.strftime("%d %b %Y") + "\n" + "\n".join(summary) + "\n")
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
