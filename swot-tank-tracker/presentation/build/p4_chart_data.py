"""Chart data and the QR code for the deck.

data/levels.json   monthly mean of each big reservoir's best-quality SWOT heights, as metres above
                   that reservoir's lowest monthly mean (so all five share one axis); a month
                   without a best-quality reading is null and the chart line spans it
img/qr_explorer.png  QR code to the SWOT Lake and River Explorer
"""
import json
import sys
from pathlib import Path

import pandas as pd
import qrcode

ROOT = Path(__file__).resolve().parents[1]
FEAS = ROOT.parent / "feasibility-check"
sys.path.insert(0, str(FEAS / "work"))
import a2_summarize as a2  # noqa: E402

TANKS = [("4580013372", "Hurulu Wewa"), ("4580006192", "Mahakanadarawa"), ("4580013332", "Manankattiya"),
         ("4580051323", "Nachchaduwa"), ("4580009523", "Nuwara Wewa")]
EXPLORER = "https://mibrechb.github.io/swot-data-explorer/"


def main():
    months = pd.period_range("2023-08", "2026-10", freq="M")
    out = {"labels": [m.strftime("%b %Y") for m in months], "series": []}
    for lake_id, name in TANKS:
        d = a2.load(lake_id)
        d = d[d.good]
        monthly = d.groupby(d.time.dt.tz_localize(None).dt.to_period("M")).wse.mean().reindex(months)
        rel = (monthly - monthly.min()).round(2)
        out["series"].append({"name": name, "best": int(len(d)),
                              "range_m": round(float(rel.max()), 1),
                              "values": [None if pd.isna(v) else float(v) for v in rel]})
        print(f"{name}: {len(d)} best readings, {int(rel.isna().sum())} empty months, range {rel.max():.2f} m")
    (ROOT / "data" / "levels.json").write_text(json.dumps(out, indent=1))
    img = qrcode.make(EXPLORER, border=1, box_size=12)
    img.save(ROOT / "img" / "qr_explorer.png")
    print("saved levels.json and qr_explorer.png")


if __name__ == "__main__":
    main()
