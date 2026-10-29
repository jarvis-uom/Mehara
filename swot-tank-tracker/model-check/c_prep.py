"""Shared data prep for the Member C forecast pilot: a full daily panel per reservoir."""
import numpy as np
import pandas as pd

CUT = pd.Timestamp("2025-04-01")      # train on origins whose target is before this; test on origins after
LAST = pd.Timestamp("2026-10-27")     # the sheet was downloaded on 27 Oct 2026
HORIZONS = (7, 14, 28)


CUM = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def doy365(idx):
    """Day of the year on a 365-day calendar (29 Feb counts as 28 Feb)."""
    day = np.where((idx.month == 2) & (idx.day == 29), 28, idx.day)
    return CUM[idx.month - 1] + day


def climatology(frac, idx):
    """Mean storage share for each day of the year, from dates before CUT only, smoothed over +-15 days."""
    doy = doy365(idx)
    train = (idx < CUT) & frac.notna().values
    s = pd.Series(frac.values[train]).groupby(doy[train]).mean().reindex(range(1, 366))
    wrapped = pd.concat([s.iloc[-15:], s, s.iloc[:15]], ignore_index=True)
    smooth = wrapped.rolling(31, center=True, min_periods=5).mean().iloc[15:-15].to_numpy()
    return smooth[doy - 1]


def panel():
    d = pd.read_csv("storage_daily.csv", parse_dates=["date"])
    d = d[(d.date <= LAST) & (d.capacity_acft > 0)]
    rain = pd.read_csv("rain_daily.csv", parse_dates=["date"])
    out = []
    for (kind, name), g in d.groupby(["kind", "reservoir"]):
        cap = g.capacity_acft.iloc[0]
        g = g.set_index("date")[["storage_acft", "lta_acft"]].asfreq("D")
        idx = pd.date_range(g.index.min(), LAST)
        g = g.reindex(idx)
        p = pd.DataFrame(index=idx)
        p["frac"] = (g.storage_acft / cap).clip(0, 1.3)
        # seasonal average from the training years only (the sheet's own "long term average" is the mean of
        # every season in the sheet, test years included, so it would leak the answers)
        p["lta"] = climatology(p.frac, idx)
        r = rain[(rain.reservoir == name) & (rain.kind == kind)].set_index("date").rain_mm
        p["rain"] = r.reindex(idx).values
        p["reservoir"], p["kind"], p["cap"] = name, kind, cap
        out.append(p.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)


def features(p, h):
    """One row per reservoir and origin day t with storage known at t and at t+h."""
    rows = []
    for (kind, name), g in p.groupby(["kind", "reservoir"], sort=False):
        g = g.set_index("date")
        f, lta, rain = g.frac, g.lta, g.rain.fillna(0)
        x = pd.DataFrame(index=g.index)
        x["f0"] = f
        for k in (1, 3, 7, 14, 28):
            x[f"d{k}"] = f - f.shift(k)
        x["lta0"], x["lta_h"] = lta, lta.shift(-h)
        x["dlta"] = x.lta_h - x.lta0
        x["anom"] = f - lta
        doy = g.index.dayofyear.values
        x["doy_sin"], x["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        x["log_cap"], x["major"] = np.log10(g.cap.iloc[0]), int(kind == "Major")
        for k in (3, 7, 30, 90):
            x[f"rain{k}"] = rain.rolling(k, min_periods=1).sum()
        x["rain_next"] = rain[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)   # future rain: only for the upper-bound test
        x["y"] = f.shift(-h) - f
        x["reservoir"], x["kind"] = name, kind
        x = x[x.f0.notna() & x.y.notna()]
        rows.append(x.rename_axis("date").reset_index())
    t = pd.concat(rows, ignore_index=True)
    t["test"] = t.date >= CUT
    t = t[(t.date + pd.Timedelta(days=h) < CUT) | t.test]
    return t
