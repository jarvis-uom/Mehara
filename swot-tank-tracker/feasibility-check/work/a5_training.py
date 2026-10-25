"""A5: Training data is reachable (check only, no download).

S1S2-Water: Zenodo record 8314175 (doi 10.5281/zenodo.8314175).
Sen1Floods11: public Google Cloud bucket "sen1floods11".
Writes work/a5_stats.json.
"""
import json
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parents[1]
HDR = {"User-Agent": "IS4990-feasibility-check/1.0 (student project)"}
stats = {}

# S1S2-Water on Zenodo
r = requests.get("https://zenodo.org/api/records/8314175", headers=HDR, timeout=60)
r.raise_for_status()
rec = r.json()
files = rec.get("files", [])
stats["s1s2_water"] = {
    "title": rec.get("metadata", {}).get("title"),
    "license": (rec.get("metadata", {}).get("license") or {}).get("id"),
    "files": len(files),
    "total_gb": round(sum(f.get("size", 0) for f in files) / 1e9, 2),
    "largest": sorted(((f.get("key"), round(f.get("size", 0) / 1e9, 2)) for f in files),
                      key=lambda x: -x[1])[:5],
}
# Confirm one file link answers (headers only)
if files:
    link = files[0].get("links", {}).get("self")
    h = requests.head(link, headers=HDR, timeout=60, allow_redirects=True)
    stats["s1s2_water"]["first_file_http"] = h.status_code

# Sen1Floods11 bucket listing
total, count, token, prefixes = 0, 0, None, {}
for _ in range(100):
    params = {"maxResults": 1000, "fields": "items(name,size),nextPageToken"}
    if token:
        params["pageToken"] = token
    r = requests.get("https://storage.googleapis.com/storage/v1/b/sen1floods11/o", params=params,
                     headers=HDR, timeout=60)
    r.raise_for_status()
    d = r.json()
    for it in d.get("items", []):
        size = int(it.get("size", 0))
        total += size
        count += 1
        top = "/".join(it["name"].split("/")[:3])
        prefixes[top] = prefixes.get(top, 0) + size
    token = d.get("nextPageToken")
    if not token:
        break
stats["sen1floods11"] = {
    "objects": count, "total_gb": round(total / 1e9, 2),
    "biggest_folders_gb": sorted(((k, round(v / 1e9, 2)) for k, v in prefixes.items()), key=lambda x: -x[1])[:6],
}
(BASE / "work" / "a5_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
