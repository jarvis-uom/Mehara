"""Fetch OSM waterway lines (rivers, streams, canals, drains, ditches),
dams, weirs and dykes for 80.30-80.90 E, 8.00-8.70 N from the Overpass API.

The answer is cached in osm/download/. If a cached file exists it is reused.
Run:  python -I 01_fetch_overpass.py
"""
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "download")
os.makedirs(OUT_DIR, exist_ok=True)

# south, west, north, east
BBOX = "8.00,80.30,8.70,80.90"

QUERIES = {
    # every waterway feature (lines, plus any nodes and relations)
    "waterways": f"""
[out:json][timeout:600];
(
  way["waterway"]({BBOX});
  relation["waterway"]({BBOX});
  node["waterway"]({BBOX});
  way["man_made"~"^(dyke|embankment)$"]({BBOX});
);
out body geom;
""",
    # water areas in the wider box (to see tanks outside the study box too)
    "water_areas": f"""
[out:json][timeout:600];
(
  way["natural"="water"]({BBOX});
  relation["natural"="water"]({BBOX});
  way["landuse"="reservoir"]({BBOX});
  relation["landuse"="reservoir"]({BBOX});
  way["water"="reservoir"]({BBOX});
);
out body geom;
""",
}

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {"User-Agent": "IS4990-student-project-cascade-check/1.0 (academic use)"}


def fetch(name, query):
    path = os.path.join(OUT_DIR, f"overpass_{name}.json")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"[{name}] cached: {path} ({os.path.getsize(path)/1e6:.1f} MB)")
        return path
    last_err = None
    for url in ENDPOINTS:
        for attempt in range(3):
            try:
                print(f"[{name}] querying {url} (attempt {attempt+1}) ...")
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=900)
                print(f"[{name}] HTTP {r.status_code}, {len(r.content)/1e6:.2f} MB")
                if r.status_code == 200:
                    data = r.json()
                    if data.get("remark"):
                        print(f"[{name}] remark: {data['remark']}")
                        if "runtime error" in data["remark"].lower():
                            raise RuntimeError(data["remark"])
                    meta = {
                        "endpoint": url,
                        "bbox_s_w_n_e": BBOX,
                        "query": query,
                        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "osm_base": data.get("osm3s", {}).get("timestamp_osm_base"),
                        "n_elements": len(data.get("elements", [])),
                    }
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                    with open(path.replace(".json", "_meta.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    print(f"[{name}] saved {meta['n_elements']} elements -> {path}")
                    return path
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                print(last_err)
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)
                print(f"[{name}] error: {last_err}")
            time.sleep(15 * (attempt + 1))
    raise SystemExit(f"[{name}] all endpoints failed: {last_err}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(QUERIES)
    for n in names:
        fetch(n, QUERIES[n])
