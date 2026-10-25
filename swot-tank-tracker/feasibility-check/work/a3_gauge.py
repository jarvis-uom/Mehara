"""A3: Is SWOT reading the right level? (sanity check, not pass/fail)

Gauge level (m above MSL) = FSL_m_MSL - (FSD_ft - depth_ft) * 0.3048, from today's
Irrigation Department sheet and its ArcGIS tank layer. Compared with each tank's latest
SWOT reading (good if there is one, else usable). Writes results/A3_gauge_vs_swot.csv.
"""
import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
WORK, RES = BASE / "work", BASE / "results"
DC = BASE.parent / "data-checks" / "results"
GENERIC = r"\b(wewa|wawa|tank|reservoir|lake|kulama|kulam)\b"


def skeleton(name):
    """Spelling-tolerant key: drop generic words, spaces and vowels."""
    s = re.sub(GENERIC, "", str(name).lower())
    s = re.sub(r"[^a-z]", "", s)
    s = re.sub(r"wewa$", "", s)
    return re.sub(r"[aeiouy]", "", s)


def read_sheet(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    return pd.DataFrame([dict(zip(head, r[:len(head)])) for r in rows[1:] if len(r) >= len(head)])


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return np.nan


def main():
    sheets = sorted(DC.glob("reservoir_status_*_*.csv"))
    latest_day = max(p.stem.split("_")[-1] for p in sheets)
    sheet = pd.concat([read_sheet(p).assign(kind=p.stem.split("_")[2]) for p in sheets
                       if p.stem.endswith(latest_day)], ignore_index=True)
    sheet = sheet[sheet.Range.str.upper().str.contains("ANURAD")].copy()
    layer = pd.read_csv(DC / "irrigation_tanks.csv")
    layer["key"] = layer.Res_Name.map(skeleton)
    lakes = pd.read_csv(RES / "A2_heights_by_lake.csv", dtype={"lake_id": str})
    pld = pd.read_csv(RES / "A_pld_lakes.csv", dtype={"lake_id": str})
    lakes = lakes.merge(pld[["lake_id", "p_lon", "p_lat"]], on="lake_id")
    lakes["key"] = lakes.tank_name.fillna("").map(skeleton)

    rows = []
    for _, r in sheet.iterrows():
        key = skeleton(r.Reservoir)
        lay = layer[layer.key == key]
        fsl = lay.FSL_m_MSL.dropna()
        fsl = fsl[fsl > 0]
        lk = lakes[(lakes.key == key) & (lakes.key != "")]
        how = "name"
        if lk.empty:  # fall back to the biggest SWOT lake within 2 km of the sheet's coordinates
            lat, lon = num(r.Latitude), num(r.Longitude)
            d = np.hypot((lakes.p_lon - lon) * 110.6, (lakes.p_lat - lat) * 110.6)
            lk = lakes[d <= 2].sort_values("ref_area_ha", ascending=False).head(1)
            how = "location"
        lk = lk.sort_values("ref_area_ha", ascending=False).head(1)
        rec = {"reservoir": r.Reservoir, "kind": r.kind, "gauge_date": r.Date,
               "FSD_ft": num(r.FSD_ft), "depth_ft": num(r["Water Depth_ft"]),
               "FSL_m_MSL": fsl.iloc[0] if len(fsl) else np.nan}
        rec["gauge_level_m"] = (rec["FSL_m_MSL"] - (rec["FSD_ft"] - rec["depth_ft"]) * 0.3048
                                if len(fsl) else np.nan)
        if not lk.empty:
            l = lk.iloc[0]
            use_good = isinstance(l.latest_good_time, str)
            rec.update({"lake_id": l.lake_id, "matched_by": how, "ref_area_ha": l.ref_area_ha,
                        "swot_kind": "good" if use_good else "usable",
                        "swot_time": (l.latest_good_time if use_good else l.latest_usable_time),
                        "swot_wse_m": (l.latest_good_wse if use_good else l.latest_usable_wse)})
            rec["offset_m"] = rec["swot_wse_m"] - rec["gauge_level_m"]
            try:
                g = datetime.strptime(r.Date, "%m/%d/%Y")
                s = datetime.fromisoformat(str(rec["swot_time"])[:10])
                rec["days_apart"] = (g - s).days
            except (TypeError, ValueError):
                rec["days_apart"] = np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    cols = ["reservoir", "kind", "lake_id", "matched_by", "ref_area_ha", "FSL_m_MSL", "FSD_ft", "depth_ft",
            "gauge_level_m", "gauge_date", "swot_kind", "swot_time", "swot_wse_m", "days_apart", "offset_m"]
    out = out[[c for c in cols if c in out.columns]].round(2)
    out.to_csv(RES / "A3_gauge_vs_swot.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    ok = out.dropna(subset=["offset_m"])
    if len(ok):
        print(f"\noffset SWOT - gauge: median {ok.offset_m.median():.2f} m, "
              f"spread (IQR) {ok.offset_m.quantile(.75) - ok.offset_m.quantile(.25):.2f} m, n={len(ok)}")


if __name__ == "__main__":
    main()
