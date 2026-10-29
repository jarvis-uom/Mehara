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
    """The method chain: three inputs, Member A's water map, Member B's trust screen and storage model,
    the storage series, Member C's forecast, validation and the dashboard (same layout as the online doc)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.76))
    ax.set_xlim(0, 760)
    ax.set_ylim(672, 64)          # y grows downwards, like the online diagram
    ax.axis("off")

    def box(x, y, w, h, name, lines, fill="white", lw=0.9):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK, lw=lw))
        cx = x + w / 2
        ax.text(cx, y + 21, name, ha="center", va="center", fontsize=9.5, fontweight="bold")
        for k, line in enumerate(lines):
            ax.text(cx, y + 38 + 16 * k, line, ha="center", va="center", fontsize=8, color=GREY)

    def arrow(points):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=INK, lw=0.8)
        ax.annotate("", xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=9))

    def note(x, y, words):
        ax.text(x, y, words, ha="left", va="center", fontsize=8, color=GREY, style="italic")

    box(24, 72, 224, 56, "Sentinel-1 and Sentinel-2", ["Pictures every 5–12 days"], fill=LIGHT)
    box(268, 72, 224, 56, "117 gauged reservoirs", ["Daily storage since 2020"], fill=LIGHT)
    box(512, 72, 224, 56, "SWOT", ["Water height, 21-day repeat"], fill=LIGHT)
    box(24, 176, 224, 56, "A · AI water map", ["Open water, water under plants"])
    box(512, 176, 224, 56, "B · Trust screen", ["Keeps good “suspect” readings"])
    box(210, 280, 340, 56, "B · Storage model", ["Height + area → % full, for tanks with no records"])
    box(210, 384, 340, 56, "Storage time series", ["% full for every tank, every 5–12 days"])
    box(24, 488, 300, 72, "C · Forecast", ["2–4 weeks ahead, from rain and evaporation",
                                             "Trained on the 117 reservoirs' storage"])
    box(436, 488, 300, 72, "Validation", ["Gauges at the 5 big reservoirs", "and Kanhindawewa's drone survey"])
    box(210, 608, 340, 56, "Dashboard", ["% full, safe limits and forecast for every tank"], lw=1.6)
    arrow([(136, 128), (136, 176)])
    arrow([(624, 128), (624, 176)])
    arrow([(380, 128), (380, 280)])
    arrow([(136, 232), (136, 308), (210, 308)])
    arrow([(624, 232), (624, 308), (550, 308)])
    arrow([(380, 336), (380, 384)])
    arrow([(210, 412), (174, 412), (174, 488)])
    arrow([(550, 412), (586, 412), (586, 488)])
    arrow([(380, 440), (380, 608)])
    arrow([(174, 560), (174, 636), (210, 636)])
    note(388, 156, "trains the models")
    note(140, 290, "water area")
    note(562, 290, "height")
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
