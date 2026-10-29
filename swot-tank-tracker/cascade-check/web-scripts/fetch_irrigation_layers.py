"""Download Irrigation Department (Sri Lanka) open ArcGIS layers for the study box.

Source org: https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/arcgis/rest/services
(no login). Saves GeoJSON (EPSG:4326) into ../web-sources/irrigation_layers/.
"""
import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/arcgis/rest/services"
OUT = Path(__file__).resolve().parent.parent / "web-sources" / "irrigation_layers"
OUT.mkdir(parents=True, exist_ok=True)

# a little wider than the study box 80.40-80.80 E, 8.10-8.60 N
BOX = (80.30, 8.00, 80.90, 8.70)

LAYERS = {
    "tank_points": "Scheme_Data_2/FeatureServer/3",
    "pooling_tank_points": "Scheme_Data_2/FeatureServer/4",
    "canals": "Scheme_Data_2/FeatureServer/8",
    "irrigable_area": "Scheme_Data_2/FeatureServer/9",
    "reservoirs": "Scheme_Data_2/FeatureServer/10",
    "pooling_tanks": "Scheme_Data_2/FeatureServer/11",
    "catchments": "Scheme_Data_2/FeatureServer/12",
    "malwathu_watersheds": "Malwathuoya_watersheds_with_IE_division/FeatureServer/0",
    "malwathu_streams": "Malwathuoya_Streams/FeatureServer/0",
}


def fetch(name, path):
    url = f"{BASE}/{path}/query"
    feats, offset = [], 0
    while True:
        params = {
            "where": "1=1",
            "geometry": ",".join(map(str, BOX)),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "outSR": 4326,
            "returnGeometry": "true",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": 1000,
        }
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            print(name, "ERROR", d["error"])
            break
        got = d.get("features", [])
        feats += got
        more = d.get("exceededTransferLimit") or d.get("properties", {}).get("exceededTransferLimit")
        if not got or not more:
            break
        offset += len(got)
        time.sleep(0.5)
    fc = {"type": "FeatureCollection", "features": feats}
    (OUT / f"{name}.geojson").write_text(json.dumps(fc), encoding="utf-8")
    print(f"{name}: {len(feats)} features")


if __name__ == "__main__":
    names = sys.argv[1:] or list(LAYERS)
    for n in names:
        try:
            fetch(n, LAYERS[n])
        except Exception as e:  # keep going with the other layers
            print(n, "FAILED", repr(e))
