"""Daily rain (ERA5, via the free Open-Meteo archive) at each department reservoir, 2020-2026.
Writes rain_daily.csv: reservoir, kind, date, rain_mm."""
import sys
import time

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
URL = "https://archive-api.open-meteo.com/v1/archive"
d = pd.read_csv("storage_daily.csv")
res = d.groupby(["kind", "reservoir"])[["lat", "lon"]].first().reset_index()
bad = res[~(res.lat.between(5.8, 10) & res.lon.between(79.4, 82.1))]
print("reservoirs:", len(res), "| odd coordinates:", bad[["reservoir", "lat", "lon"]].to_dict("records"))
res = res.drop(bad.index)
out = []
for i in range(0, len(res), 20):
    part = res.iloc[i:i + 20]
    params = {"latitude": ",".join(f"{v:.4f}" for v in part.lat), "longitude": ",".join(f"{v:.4f}" for v in part.lon),
              "start_date": "2020-08-01", "end_date": "2026-10-27", "daily": "precipitation_sum",
              "timezone": "Asia/Colombo"}
    for attempt in range(4):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            break
        print("retry", r.status_code, r.text[:120]); time.sleep(20 * (attempt + 1))
    js = r.json()
    js = js if isinstance(js, list) else [js]
    for (_, row), j in zip(part.iterrows(), js):
        out.append(pd.DataFrame({"reservoir": row.reservoir, "kind": row.kind,
                                 "date": j["daily"]["time"], "rain_mm": j["daily"]["precipitation_sum"]}))
    print(f"{i + len(part)}/{len(res)}"); time.sleep(3)
rain = pd.concat(out, ignore_index=True)
rain.to_csv("rain_daily.csv", index=False)
print(len(rain), "rows; missing:", int(rain.rain_mm.isna().sum()), "| last date with rain value:",
      rain.dropna().date.max())
