"""A2 step 2: summarise SWOT heights per lake and plot 10 example tanks.

Good point (the brief, after Kohn & Nielsen 2026): quality_f = 0, xovr_cal_q <= 1, dark_frac <= 0.1.
Usable point (relaxed): the same, but quality_f may also be 1 ("suspect").
Jitter: median height change between usable points at most 10 days apart. Real tank levels
change by centimetres to a few decimetres in that time, so a large jitter means noise.
Tank size: SWOT's own reference area (p_ref_area).
Writes results/A2_heights_by_lake.csv and results/A_swot_heights.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
WORK, RES = BASE / "work", BASE / "results"
NUM = ["wse", "wse_u", "area_total", "quality_f", "xovr_cal_q", "dark_frac", "partial_f", "xtrk_dist"]


def load(lake_id):
    path = WORK / "hydrocron" / f"{lake_id}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"lake_id": str})
    df = df[df.time_str != "no_data"].copy()
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df.wse > -1e10].drop_duplicates("time_str")
    df["time"] = pd.to_datetime(df.time_str, utc=True)
    base = df.xovr_cal_q.between(0, 1) & df.dark_frac.between(0, 0.1)
    df["good"] = base & (df.quality_f == 0)
    df["usable"] = base & df.quality_f.between(0, 1)
    return df.sort_values("time")


def rise_fall(s):
    return round(s.quantile(0.95) - s.quantile(0.05), 2) if len(s) >= 5 else None


def jitter(df):
    if len(df) < 5:
        return None
    dt = df.time.diff().dt.total_seconds() / 86400
    dw = df.wse.diff().abs()
    sel = dw[(dt > 0) & (dt <= 10)]
    return round(sel.median(), 2) if len(sel) >= 3 else None


def main():
    pld = pd.read_csv(RES / "A_pld_lakes.csv", dtype={"lake_id": str})
    pld["tank_name"] = pld.tank_name.fillna("")
    pld["size_class"] = pd.cut(pld.p_ref_area_ha, [0, 6.25, 25, 1e9],
                               labels=["<6.25 ha", "6.25-25 ha", ">=25 ha"], right=False)
    rows, series = [], {}
    for _, lake in pld.iterrows():
        df = load(lake.lake_id)
        series[lake.lake_id] = df
        good = df[df.good] if len(df) else df
        use = df[df.usable] if len(df) else df
        rows.append({
            "lake_id": lake.lake_id, "tank_name": lake.tank_name,
            "ref_area_ha": lake.p_ref_area_ha, "size_class": lake.size_class,
            "points": len(df), "good_points": len(good), "usable_points": len(use),
            "first": df.time.min().date() if len(df) else None,
            "last": df.time.max().date() if len(df) else None,
            "rise_fall_good_m": rise_fall(good.wse) if len(good) else None,
            "rise_fall_usable_m": rise_fall(use.wse) if len(use) else None,
            "jitter_usable_m": jitter(use) if len(use) else None,
            "median_wse_u_usable_m": round(use.wse_u.median(), 3) if len(use) else None,
            "latest_good_time": good.time.max().isoformat() if len(good) else None,
            "latest_good_wse": round(good.wse.iloc[-1], 3) if len(good) else None,
            "latest_usable_time": use.time.max().isoformat() if len(use) else None,
            "latest_usable_wse": round(use.wse.iloc[-1], 3) if len(use) else None,
        })
    out = pd.DataFrame(rows).sort_values("ref_area_ha", ascending=False)
    out.to_csv(RES / "A2_heights_by_lake.csv", index=False)

    def block(g, n_pts, col_pts, col_rf):
        ok = g[(g[col_pts] >= n_pts) & (g[col_rf] >= 1.0)]
        return {"n": len(g), "ok": len(ok), "share_ok": round(len(ok) / len(g), 3) if len(g) else None,
                "median_points": float(g[col_pts].median()) if len(g) else None}

    big = out[out.size_class == ">=25 ha"]
    mid = out[out.size_class == "6.25-25 ha"]
    small = out[out.size_class == "<6.25 ha"]
    strict_big = block(big, 25, "good_points", "rise_fall_good_m")
    relaxed_big = block(big, 25, "usable_points", "rise_fall_usable_m")
    stats = {
        "lakes": len(out), "with_any_points": int((out.points > 0).sum()),
        "all_points": int(out.points.sum()), "good_points": int(out.good_points.sum()),
        "usable_points": int(out.usable_points.sum()),
        "strict_ge25ha": strict_big, "relaxed_ge25ha": relaxed_big,
        "mid_strict_ge10good": int((mid.good_points >= 10).sum()),
        "mid_relaxed_ge10usable": int((mid.usable_points >= 10).sum()), "mid_n": len(mid),
        "small_relaxed_ge10usable": int((small.usable_points >= 10).sum()), "small_n": len(small),
        "jitter_median_m": {k: (float(g.jitter_usable_m.median()) if g.jitter_usable_m.notna().any() else None)
                            for k, g in (("ge25", big), ("6.25-25", mid), ("lt6.25", small))},
        "rise_fall_usable_median_m": {k: (float(g.rise_fall_usable_m.median()) if g.rise_fall_usable_m.notna().any() else None)
                                      for k, g in (("ge25", big), ("6.25-25", mid), ("lt6.25", small))},
        "date_range": [str(out["first"].dropna().min()), str(out["last"].dropna().max())],
        "big_table": big[["tank_name", "ref_area_ha", "points", "good_points", "usable_points",
                          "rise_fall_usable_m", "jitter_usable_m"]].to_dict("records"),
        "verdict_strict": ("PASS" if strict_big["share_ok"] >= 0.9 else
                           "RISK" if strict_big["share_ok"] >= 0.5 else "FAIL"),
        "verdict_relaxed": ("PASS" if relaxed_big["share_ok"] >= 0.9 else
                            "RISK" if relaxed_big["share_ok"] >= 0.5 else "FAIL"),
    }
    (WORK / "a2_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "big_table"}, indent=1, default=str))
    print(pd.DataFrame(stats["big_table"]).to_string(index=False))

    # Plot: the 5 largest tanks, and 5 village tanks spread across the usable-point counts
    largest = big.head(5)
    vil = mid[mid.usable_points > 0].sort_values("usable_points")
    village = pd.DataFrame([vil.iloc[int(q * (len(vil) - 1))] for q in (0.1, 0.3, 0.5, 0.7, 0.9)]) if len(vil) >= 5 else vil
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5), sharex=True)
    for row_axes, group, label in ((axes[0], largest, "Large"), (axes[1], village, "Village")):
        for ax, (_, lake) in zip(row_axes, group.iterrows()):
            df = series[lake.lake_id]
            ax.plot(df.time, df.wse, ".", color="0.8", ms=4, label="other")
            u = df[df.usable & ~df.good]
            ax.plot(u.time, u.wse, ".", color="tab:orange", ms=5, label="usable (quality 1)")
            g = df[df.good]
            ax.plot(g.time, g.wse, "o", color="tab:blue", ms=4, label="good (quality 0)")
            keep = df[df.usable]
            if len(keep) >= 5:
                lo, hi = keep.wse.quantile(0.02), keep.wse.quantile(0.98)
                pad = max(0.5, 0.15 * (hi - lo))
                ax.set_ylim(lo - pad, hi + pad)
            name = lake.tank_name or f"lake {lake.lake_id}"
            ax.set_title(f"{label}: {name}, {lake.ref_area_ha:g} ha\n"
                         f"{lake.good_points} good, {lake.usable_points} usable of {lake.points}", fontsize=9)
            ax.tick_params(labelsize=7)
    for ax in (axes[0][0], axes[1][0]):
        ax.set_ylabel("Water surface height (m)")
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("SWOT water heights, July 2023 to October 2026, study box east of Anuradhapura", fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "A_swot_heights.png", dpi=110)
    print("plot saved")


if __name__ == "__main__":
    main()
