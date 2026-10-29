"""Turn the department's four hidden storage tabs into one long daily table.

Each tab row is one reservoir on one day of the season; each season is a column.
Maha runs 1 Oct - 31 Mar, Yala runs 1 Apr - 30 Sep.
Writes storage_daily.csv: reservoir, kind, lat, lon, capacity_acft, date, storage_acft, lta_acft.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
TABS = Path(__file__).resolve().parents[1] / "data-sources" / "ground-data" / "downloads" / "sheet_tabs"
FILES = {("Major", "Maha"): "tab_673998835.csv", ("Major", "Yala"): "tab_1244158041.csv",
         ("Medium", "Maha"): "tab_155979206.csv", ("Medium", "Yala"): "tab_1831537855.csv"}


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip().replace({"": None, "-": None, "nan": None}),
                         errors="coerce")


rows = []
for (kind, season), name in FILES.items():
    t = pd.read_csv(TABS / name, dtype=str)
    t.columns = [c.strip() for c in t.columns]
    col = {c.lower(): c for c in t.columns}
    date_c = col.get("updated date") or col.get("update date")
    res_c, lat_c, lon_c = col["reservoir"], col["latitude"], col["longitude"]
    cap_c, lta_c = col["active storage (acft)"], col["long term avarage"]
    md = pd.to_datetime(t[date_c], format="%m/%d/%Y")
    for c in t.columns:
        m = re.match(r"ES_(\d\d)_(\d\d)_Acft", c) or re.match(r"EFFECTIVE STORAGE \(Acft\) (\d{4})", c)
        if not m:
            continue
        if season == "Maha":
            y1 = 2000 + int(m.group(1))
            years = md.dt.month.map(lambda mo: y1 if mo >= 10 else y1 + 1)
        else:
            years = pd.Series(int(m.group(1)), index=t.index)
        ok = ~((md.dt.month == 2) & (md.dt.day == 29))
        dates = pd.to_datetime(dict(year=years[ok], month=md.dt.month[ok], day=md.dt.day[ok]))
        rows.append(pd.DataFrame({
            "reservoir": t[res_c][ok].str.strip(), "kind": kind,
            "lat": num(t[lat_c][ok]), "lon": num(t[lon_c][ok]),
            "capacity_acft": num(t[cap_c][ok]), "date": dates,
            "storage_acft": num(t[c][ok]), "lta_acft": num(t[lta_c][ok])}))

d = pd.concat(rows, ignore_index=True)
d = d.dropna(subset=["storage_acft"]).sort_values(["kind", "reservoir", "date"])
d = d.drop_duplicates(["kind", "reservoir", "date"])
d.to_csv("storage_daily.csv", index=False)
per = d.groupby(["kind", "reservoir"]).agg(first=("date", "min"), last=("date", "max"), days=("date", "size"),
                                           cap=("capacity_acft", "first"))
print(len(d), "daily values;", per.reset_index().groupby("kind").size().to_dict(), "reservoirs")
print("date range:", d.date.min().date(), "to", d.date.max().date())
print("days per reservoir: median", int(per.days.median()), "| min", per.days.min(), "| max", per.days.max())
frac = d.storage_acft / d.capacity_acft
print("storage as share of capacity: 1%", round(frac.quantile(.01), 2), "| 50%", round(frac.median(), 2),
      "| 99%", round(frac.quantile(.99), 2), "| above 1.05:", int((frac > 1.05).sum()))
flat = d.groupby(["kind", "reservoir"]).storage_acft.apply(lambda s: (s.diff() == 0).mean())
print("share of days with no change, median over reservoirs:", round(flat.median(), 2),
      "| reservoirs with >80% unchanged days:", int((flat > .8).sum()))
print(d.groupby(["kind", d.date.dt.to_period("M")]).reservoir.nunique().unstack(0).iloc[::3].to_string())
