"""Hurulu Wewa height-area pairs for the deck's scatter chart.

Every Sentinel-2 scene over Hurulu Wewa (Jul 2023 - Oct 2026) with under 2% cloud in the box,
paired with a best-quality SWOT reading at most 3 days away; water area = NDWI > 0 inside the
OpenStreetMap outline buffered by 250 m (the same rule as p2_images.py).
Writes data/hurulu_pairs.csv.
"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import p2_images as P


def main():
    items = P.s2_items(P.HURULU, "2023-07-15", "2026-10-25", max_cloud=101)
    with ThreadPoolExecutor(6) as pool:
        shares = list(pool.map(lambda it: P.cloud_share(it, P.HURULU), items))
    clear = [it for it, c in zip(items, shares) if c < 0.02]
    outline, ocrs = P.hurulu_outline()
    rows = []
    for it in clear:
        day = it.datetime.date()
        swot_day, wse = P.swot_near(day, days=3)
        if wse is None:
            continue
        km2, _, _ = P.water_area(it, outline, ocrs)
        rows.append({"s2_date": day.isoformat(), "item": it.id, "swot_date": swot_day, "wse": wse,
                     "water_km2": round(float(km2), 2)})
        print(rows[-1])
    out = pd.DataFrame(rows).drop_duplicates("s2_date").sort_values("wse")
    out.to_csv(P.DATA / "hurulu_pairs.csv", index=False)
    print(f"{len(items)} scenes, {len(clear)} clear, {len(out)} pairs")


if __name__ == "__main__":
    main()
