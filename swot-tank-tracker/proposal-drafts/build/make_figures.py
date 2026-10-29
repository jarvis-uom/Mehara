"""Redraw the method pipeline and the project timeline in a plain academic style (SVG)."""
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "svg.fonttype": "path", "font.size": 10})
INK, GREY, LIGHT = "#000000", "#555555", "#ececec"


def pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    W, H = 40, 11

    def box(cx, cy, title, sub, fill="white", lw=0.9):
        ax.add_patch(Rectangle((cx - W / 2, cy - H / 2), W, H, facecolor=fill, edgecolor=INK, lw=lw))
        ax.text(cx, cy + 2.2, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(cx, cy - 2.4, sub, ha="center", va="center", fontsize=9, color=GREY)
        return cx, cy

    def arrow(points):
        xs, ys = zip(*points)
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=INK, lw=0.8)
        ax.annotate("", xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=9))

    L, R, C = 24, 76, 50
    box(L, 92, "Sentinel-1 and Sentinel-2", "Radar and optical images, every 5–12 days", fill=LIGHT)
    box(R, 92, "SWOT", "Water height per pass, 21-day repeat", fill=LIGHT)
    box(L, 72, "AI water mapping", "Open water, water under plants, land")
    box(R, 72, "SWOT height series", "Hydrocron API, poor passes removed")
    box(C, 50, "Height–area–volume curve", "Pairs SWOT height with AI area, per tank")
    box(C, 29, "Storage time series", "Each new water map becomes a volume")
    box(L, 8, "Validation", "Gauge readings and official figures")
    box(R, 8, "Dashboard", "Storage per tank, every 5–12 days", lw=1.6)
    arrow([(L, 92 - H / 2), (L, 72 + H / 2)])
    arrow([(R, 92 - H / 2), (R, 72 + H / 2)])
    arrow([(L + 10, 72 - H / 2), (L + 10, 61), (C - 8, 61), (C - 8, 50 + H / 2)])
    arrow([(R - 10, 72 - H / 2), (R - 10, 61), (C + 8, 61), (C + 8, 50 + H / 2)])
    arrow([(C, 50 - H / 2), (C, 29 + H / 2)])
    arrow([(L - 12, 72 - H / 2), (L - 12, 29), (C - W / 2, 29)])
    ax.text(L - 10.5, 40, "each new map", fontsize=9, color=GREY, style="italic")
    arrow([(C - 8, 29 - H / 2), (C - 8, 18), (L, 18), (L, 8 + H / 2)])
    arrow([(C + 8, 29 - H / 2), (C + 8, 18), (R, 18), (R, 8 + H / 2)])
    fig.savefig(OUT / "fig2_pipeline.svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def timeline():
    phases = [
        ("Proposal and ethics", date(2026, 10, 19), date(2026, 11, 30)),
        ("Tank name check", date(2026, 11, 16), date(2026, 12, 15)),
        ("Study area confirmed", date(2026, 12, 15), None),
        ("Labelled water masks", date(2026, 12, 1), date(2027, 2, 28)),
        ("Gauge readings on pass dates", date(2027, 1, 4), date(2027, 4, 30)),
        ("AI water mapping model", date(2027, 1, 4), date(2027, 3, 31)),
        ("SWOT heights and storage model", date(2027, 1, 18), date(2027, 4, 15)),
        ("Storage forecast model", date(2027, 1, 18), date(2027, 4, 30)),
        ("Storage series and validation", date(2027, 3, 15), date(2027, 5, 15)),
        ("Dashboard", date(2027, 4, 1), date(2027, 5, 31)),
        ("Thesis and paper", date(2027, 5, 1), date(2027, 6, 30)),
        ("Final submission", date(2027, 6, 30), None),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.subplots_adjust(left=0.47, right=0.985, top=0.88, bottom=0.03)
    n = len(phases)
    for i, (name, start, end) in enumerate(phases):
        y = n - 1 - i
        if end is None:
            ax.plot(mdates.date2num(start), y, marker="D", color=INK, ms=6)
            label = f"{name} ({start.day} {start.strftime('%b')})"
        else:
            s, e = mdates.date2num(start), mdates.date2num(end)
            ax.barh(y, e - s, left=s, height=0.5, color="#bdbdbd", edgecolor=INK, lw=0.6)
            label = f"{name} ({start.day} {start.strftime('%b')} – {end.day} {end.strftime('%b')})"
        ax.text(mdates.date2num(date(2026, 10, 1)) - 3, y, label, ha="right", va="center", fontsize=10,
                fontweight="bold" if end is None else "normal")
    ax.axvline(mdates.date2num(date(2026, 12, 15)), color="#333333", lw=0.8, ls="--", dashes=(4, 3))
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(mdates.date2num(date(2026, 10, 1)), mdates.date2num(date(2027, 7, 12)))
    months = [date(2026, m, 1) for m in (10, 11, 12)] + [date(2027, m, 1) for m in range(1, 7)]
    ax.set_xticks([mdates.date2num(m) for m in months])
    ax.set_xticklabels([m.strftime("%b") + (f"\n{m.year}" if m.month in (10, 1) else "\n") for m in months])
    ax.axvline(mdates.date2num(date(2027, 7, 1)), color="#d0d0d0", lw=0.5)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=9.5, length=0)
    ax.set_yticks([])
    ax.grid(axis="x", color="#d0d0d0", lw=0.5)
    for s in ("left", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["top"].set_color(INK)
    ax.spines["top"].set_linewidth(0.6)
    fig.savefig(OUT / "fig3_timeline.svg")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    timeline()
    print("figures saved")
