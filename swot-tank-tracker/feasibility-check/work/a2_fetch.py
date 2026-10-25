"""A2 step 1: download every study-box lake's SWOT series from Hydrocron (no login).

Reads results/A_pld_lakes.csv and saves one CSV per lake in work/hydrocron/.
"""
import csv
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "work" / "hydrocron"
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
FIELDS = "lake_id,time_str,wse,wse_u,area_total,quality_f,xovr_cal_q,dark_frac,partial_f,xtrk_dist"
END = time.strftime("%Y-%m-%dT00:00:00Z")
HDR = {"User-Agent": "IS4990-feasibility-check/1.0 (student project)"}


def fetch(lake_id):
    path = OUT / f"{lake_id}.csv"
    if path.exists():
        return lake_id, "cached"
    params = {"feature": "PriorLake", "feature_id": lake_id,
              "start_time": "2023-07-01T00:00:00Z", "end_time": END,
              "output": "csv", "fields": FIELDS,
              "collection_name": "SWOT_L2_HR_LakeSP_D"}
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, headers=HDR, timeout=90)
        except requests.RequestException as e:
            status = f"error {type(e).__name__}"
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                text = r.json()["results"]["csv"]
            except (ValueError, KeyError, TypeError):
                text = r.text
            path.write_text(text, encoding="utf-8")
            return lake_id, "ok"
        if r.status_code == 400 and "collection_name" in params:
            params.pop("collection_name")  # older API: fall back to the default collection
            continue
        if r.status_code == 400 and "not found" in r.text.lower():
            path.write_text("", encoding="utf-8")
            return lake_id, "no data"
        status = f"HTTP {r.status_code}: {r.text[:120]}"
        time.sleep(3 * (attempt + 1))
    return lake_id, status


def main():
    ids = pd.read_csv(BASE / "results" / "A_pld_lakes.csv", dtype={"lake_id": str}).lake_id.tolist()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, ids))
    counts = {}
    for _, s in results:
        key = s if s in ("ok", "cached", "no data") else "failed"
        counts[key] = counts.get(key, 0) + 1
    failed = [(i, s) for i, s in results if s not in ("ok", "cached", "no data")]
    (BASE / "work" / "a2_fetch_log.json").write_text(json.dumps({"counts": counts, "failed": failed}, indent=1))
    print(f"{len(ids)} lakes in {time.time() - t0:.0f} s: {counts}")
    for i, s in failed[:10]:
        print("  failed", i, s)


if __name__ == "__main__":
    main()
