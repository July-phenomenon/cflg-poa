# -*- coding: utf-8 -*-
"""The four manuscript figures + graphical abstract (authoritative generator).

This is the script that produced the PDFs shipped in results/figures/. It reads
the exact-run artefacts only (the enumeration JSON and the two CSVs written by
export_exact_figure_inputs.py).

Outputs (PDF + PNG 300 dpi) to:
    D:\电工杯opencode\paper_research\_check\figures\

Naming:
    fig1_poa21.pdf          -> Figure 1  (21-city lollipop)
    fig2_poa_dist.pdf       -> Figure 2  (enumeration histogram + KDE)
    fig3_channel.pdf        -> Figure 3  (three-panel channel decomposition)
    fig4_subsidy.pdf        -> Figure 4  (subsidy dumbbell)
    GraphicalAbstract.pdf   -> Graphical Abstract (donut + histogram)

Rules enforced:
    - No text boxes / annotations inside the manuscript figures.
    - Unified COLORS palette.
    - Arial font (fallback DejaVu Sans).
    - Violin jittered points are placed INSIDE the violin envelope.
"""
import json, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = os.environ.get(
    "CFLG_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
# Exact-benchmark outputs (2026-10-23 revision): the enumeration, the 21-city
# table and the subsidy table are regenerated from the exact single-MILP Phase-2
# runs, so the figures must read those files rather than the superseded
# fixed-point archive.  Both CSVs are (re)written by
# code/05_figures_tables/export_exact_figure_inputs.py.
PRM = os.path.join(ROOT, "results", "tables", "_exact_src")
NUM = os.path.join(ROOT, "results", "main")

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
COLORS = {
    "primary":    "#2E5A88",
    "secondary":  "#E8873A",
    "negative":   "#C0392B",
    "neutral":    "#7F7F7F",
    "light_fill": "#A9C5E0",
    "tie":        "#A6A6A6",
    "dark":       "#1A3A5A",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "axes.unicode_minus": False,
})

# Channel-decomposition constants (same as V1_figure_channel_decomposition.py)
W1, W2, W3, S3 = 0.2, 0.3, 0.5, 0.60


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    """Save both PDF and PNG 300 dpi."""
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(os.path.join(NUM, "results_enumeration_exact.json"), encoding="utf-8") as f:
    enum_raw = json.load(f)["results"]

enum_valid = [c for c in enum_raw if c["nash_F2"] > 0]
enum_poa = np.array([c["milp_F2"] / c["nash_F2"] for c in enum_valid])

# Channel decomposition arrays
f1n = np.array([c["nash_F1"] for c in enum_valid])
f1m = np.array([c["milp_F1"] for c in enum_valid])
f2n = np.array([c["nash_F2"] for c in enum_valid])
f2m = np.array([c["milp_F2"] for c in enum_valid])
s2n = np.array([c["nash_S2_weighted"] for c in enum_valid])
s2m = np.array([c["milp_S2_weighted"] for c in enum_valid])

dcov = f1m - f1n
dA2 = s2m - s2n
A1n = (f2n - W2 * s2n - W3 * S3 * f1n) / W1
A1m = (f2m - W2 * s2m - W3 * S3 * f1m) / W1
dA1 = A1m - A1n

ch_assign = W1 * dA1      # α ΔA1
ch_congest = W2 * dA2     # β ΔA2
ch_cov = W3 * S3 * dcov   # γ S3 Δcov
locked = dcov <= 1e-9

# Identity check (sanity)
resid = np.abs(ch_assign + ch_congest + ch_cov - (f2m - f2n))
print(f"Channel identity check: max residual = {resid.max():.3e}")

# 21-city data
c21 = read_csv(os.path.join(PRM, "Table5_2_21cities_PoA.csv"))
c21_city = [r["city"] for r in c21]
c21_poa = np.array([float(r["PoA"]) for r in c21])

# Subsidy data
sub = read_csv(os.path.join(PRM, "Table7_1_subsidy_comparison.csv"))
sub = [r for r in sub if r["city"] != "MEAN"]
sub_city = [r["city"] for r in sub]
sub_base = np.array([float(r["PoA_base"]) for r in sub])
sub_con = np.array([float(r["PoA_subsidized_consistent"]) for r in sub])


# ---------------------------------------------------------------------------
# Figure 1: 21-city sequential PoA (lollipop chart)
# ---------------------------------------------------------------------------
def fig1_poa21():
    order = np.argsort(c21_poa)
    cities = [c21_city[i] for i in order]
    vals = c21_poa[order]
    y = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    # horizontal stems from x=0.98 to value
    for i, v in enumerate(vals):
        ax.plot([0.98, v], [i, i], color=COLORS["neutral"], lw=0.6, zorder=1)

    colors = [COLORS["primary"] if v > 1.0 else COLORS["tie"] for v in vals]
    ax.scatter(vals, y, c=colors, s=40, zorder=2, edgecolor="white", linewidth=0.5)

    ax.axvline(1.0, color=COLORS["neutral"], lw=0.8, ls=":", zorder=0)
    mean_val = float(f"{np.mean(vals):.4f}")
    ax.axvline(mean_val, color=COLORS["neutral"], lw=1.0, ls="--", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(cities, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Sequential PoA", fontsize=10)
    ax.set_xlim(0.985, 1.095)

    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["primary"],
               markersize=8, label="PoA > 1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["tie"],
               markersize=8, label="PoA = 1"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=9, frameon=False)
    ax.set_title("Sequential PoA across the 21 real cities", fontsize=11, pad=10)

    fig.tight_layout()
    save(fig, "fig1_poa21")


# ---------------------------------------------------------------------------
# Figure 2: enumeration PoA distribution (histogram + KDE)
# ---------------------------------------------------------------------------
def fig2_poa_dist():
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    bins = 50
    range_ = (0.9, 1.7)
    counts, bin_edges = np.histogram(enum_poa, bins=bins, range=range_)
    ax.hist(enum_poa, bins=bins, range=range_, color=COLORS["primary"],
            edgecolor="white", linewidth=0.3, alpha=0.75)

    # KDE scaled to histogram counts
    kde = gaussian_kde(enum_poa)
    x_grid = np.linspace(range_[0], range_[1], 500)
    bin_width = (range_[1] - range_[0]) / bins
    kde_scaled = kde(x_grid) * len(enum_poa) * bin_width
    ax.plot(x_grid, kde_scaled, color=COLORS["primary"], lw=1.5, zorder=2)

    ax.axvline(1.0, color=COLORS["neutral"], lw=0.8, ls="--", label="PoA = 1")
    mean_val = float(f"{np.mean(enum_poa):.4f}")
    median_val = float(f"{np.median(enum_poa):.4f}")
    ax.axvline(mean_val, color=COLORS["negative"], lw=1.2, ls="--",
               label=f"Mean {mean_val}")
    ax.axvline(median_val, color=COLORS["secondary"], lw=1.2, ls="-.",
               label=f"Median {median_val}")

    ax.set_xlabel("Sequential PoA", fontsize=10)
    ax.set_ylabel("Configurations", fontsize=10)
    ax.set_xlim(range_)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title("Distribution of the sequential PoA", fontsize=11, pad=10)

    fig.tight_layout()
    save(fig, "fig2_poa_dist")


# ---------------------------------------------------------------------------
# Helpers for violin + box plots
# ---------------------------------------------------------------------------
def _horizontal_violin_with_points(ax, y_pos, values, color, violin_width=0.7,
                                   box_width=0.12, point_color=COLORS["neutral"],
                                   x_grid=None, xlim=(-0.04, 0.08), seed=42,
                                   max_points=400):
    """Symmetric horizontal violin (values on the x-axis) centred on y_pos,
    with a box plot at the centre and jittered points INSIDE the envelope."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)

    # KDE
    kde = gaussian_kde(values)
    if x_grid is None:
        x_grid = np.linspace(xlim[0], xlim[1], 400)
    density = kde(x_grid)
    max_d = density.max()
    if max_d == 0:
        max_d = 1.0

    # symmetric violin centred on y_pos (vertical half-thickness per x)
    half = (density / max_d) * (violin_width / 2.0)
    ax.fill_between(x_grid, y_pos - half, y_pos + half,
                    facecolor=color, edgecolor=color, alpha=0.50,
                    linewidth=0.8, zorder=1)

    # horizontal box plot at y_pos (q25-q75 along the value axis)
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    iqr = q75 - q25
    whisk_low = np.min(values[values >= q25 - 1.5 * iqr])
    whisk_high = np.max(values[values <= q75 + 1.5 * iqr])

    hb = box_width / 2.0
    ax.add_patch(plt.Rectangle((q25, y_pos - hb), q75 - q25, box_width,
                               facecolor=COLORS["dark"], edgecolor=COLORS["dark"],
                               alpha=0.85, zorder=3))
    ax.plot([q50, q50], [y_pos - hb, y_pos + hb], color="white", lw=1.8, zorder=4)
    ax.plot([whisk_low, q25], [y_pos, y_pos], color=COLORS["dark"], lw=1.2, zorder=2)
    ax.plot([q75, whisk_high], [y_pos, y_pos], color=COLORS["dark"], lw=1.2, zorder=2)
    ax.plot([whisk_low, whisk_low], [y_pos - hb * 0.7, y_pos + hb * 0.7],
            color=COLORS["dark"], lw=1.2, zorder=2)
    ax.plot([whisk_high, whisk_high], [y_pos - hb * 0.7, y_pos + hb * 0.7],
            color=COLORS["dark"], lw=1.2, zorder=2)

    # points inside the violin envelope (subsampled so the violin fill stays
    # visible under the point cloud):
    if len(values) > max_points:
        values = rng.choice(values, size=max_points, replace=False)
    # vertical offset limited by the local half-thickness at each value
    d_at_vals = np.interp(values, x_grid, density) / max_d * (violin_width / 2.0)
    y_off = rng.uniform(-1.0, 1.0, size=len(values)) * d_at_vals
    x_jit = values + rng.normal(0, 0.0004, size=len(values))

    ax.scatter(x_jit, y_pos + y_off, c=point_color, s=4, alpha=0.30,
               edgecolors="none", zorder=2, linewidths=0)


def _vertical_violin_box(ax, x_pos, values, color, violin_width=0.7,
                         box_width=0.12, y_grid=None, ylim=(0.9, 1.5)):
    """Symmetric vertical violin (values on the y-axis) centred on x_pos
    with an embedded box plot."""
    values = np.asarray(values)
    kde = gaussian_kde(values)
    if y_grid is None:
        y_grid = np.linspace(ylim[0], ylim[1], 400)
    density = kde(y_grid)
    max_d = density.max() if density.max() > 0 else 1.0

    half = (density / max_d) * (violin_width / 2.0)
    ax.fill_between(y_grid, x_pos - half, x_pos + half,
                    facecolor=color, edgecolor=color, alpha=0.50, linewidth=0.8, zorder=1)

    # vertical box plot at x_pos (q25-q75 along the value axis)
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    iqr = q75 - q25
    whisk_low = np.min(values[values >= q25 - 1.5 * iqr])
    whisk_high = np.max(values[values <= q75 + 1.5 * iqr])

    hb = box_width / 2
    ax.add_patch(plt.Rectangle((x_pos - hb, q25), box_width, q75 - q25,
                               facecolor=COLORS["dark"], edgecolor=COLORS["dark"],
                               alpha=0.85, zorder=3))
    ax.plot([x_pos - hb, x_pos + hb], [q50, q50], color="white", lw=1.8, zorder=4)
    ax.plot([x_pos, x_pos], [whisk_low, q25], color=COLORS["dark"], lw=1.2, zorder=2)
    ax.plot([x_pos, x_pos], [q75, whisk_high], color=COLORS["dark"], lw=1.2, zorder=2)
    ax.plot([x_pos - hb * 0.7, x_pos + hb * 0.7], [whisk_low, whisk_low],
            color=COLORS["dark"], lw=1.2, zorder=2)
    ax.plot([x_pos - hb * 0.7, x_pos + hb * 0.7], [whisk_high, whisk_high],
            color=COLORS["dark"], lw=1.2, zorder=2)


# ---------------------------------------------------------------------------
# Figure 3: channel decomposition (three panels)
# ---------------------------------------------------------------------------
def fig3_channel():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"wspace": 0.35})

    xlim_a = (-0.020, 0.120)

    # ---- Panel A: horizontal violins with in-violin points ----
    ax = axes[0]
    channels = [
        (ch_congest, r"Congestion $\beta \Delta A_2$", COLORS["negative"]),
        (ch_assign,  r"Assignment $\alpha \Delta A_1$", COLORS["light_fill"]),
        (ch_cov,     r"Coverage $\gamma S_3 \Delta \mathrm{cov}$", COLORS["primary"]),
    ]
    y_positions = [0, 1, 2]
    x_grid = np.linspace(xlim_a[0], xlim_a[1], 400)
    for (vals, label, color), yp in zip(channels, y_positions):
        _horizontal_violin_with_points(ax, yp, vals, color, violin_width=0.7,
                                       x_grid=x_grid, xlim=xlim_a,
                                       seed=20261012 + yp)

    ax.axvline(0, color=COLORS["neutral"], lw=0.8, zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([lab for _, lab, _ in channels], fontsize=9)
    ax.set_xlim(xlim_a)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("Contribution to $\\Delta F_2$", fontsize=10)
    ax.set_title("A. Channel contributions", fontsize=11, pad=10)

    # ---- Panel B: stacked median-share bars ----
    ax = axes[1]
    labels_x = ["Locked\n$n=1{,}604$",
                "Locked non-deg.\n$n=1{,}074$",
                "21 cities\n$n=18$"]
    s1 = [0.68, 0.29, 0.05]
    s2 = [0.32, 0.71, 0.70]
    sc = [0.00, 0.00, 0.20]
    x = np.arange(len(labels_x))
    w = 0.52

    ax.bar(x, s1, w, label="$S_1$ (distance)", color=COLORS["primary"])
    ax.bar(x, s2, w, bottom=s1, label="$S_2$ (congestion)", color=COLORS["secondary"])
    ax.bar(x, sc, w, bottom=np.array(s1) + np.array(s2),
           label="coverage", color=COLORS["negative"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylabel("Median share", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", fontsize=9, frameon=False,
              bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_title("B. Median shares by lock status", fontsize=11, pad=10)

    # ---- Panel C: layered PoA (vertical violins, no scatter) ----
    ax = axes[2]
    ylim_c = (0.9, 1.7)
    y_grid = np.linspace(ylim_c[0], ylim_c[1], 400)
    _vertical_violin_box(ax, 0, enum_poa[locked], COLORS["primary"],
                         violin_width=0.6, y_grid=y_grid, ylim=ylim_c)
    _vertical_violin_box(ax, 1, enum_poa[~locked], COLORS["secondary"],
                         violin_width=0.6, y_grid=y_grid, ylim=ylim_c)
    ax.axhline(1.0, color=COLORS["neutral"], lw=0.8, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Locked\n($n = 4{,}317$)", "Unlocked\n($n = 10{,}890$)"], fontsize=9)
    ax.set_ylabel("PoA", fontsize=10)
    ax.set_ylim(ylim_c)
    ax.set_title("C. Layered Price of Anarchy", fontsize=11, pad=10)

    fig.tight_layout()
    save(fig, "fig3_channel")


# ---------------------------------------------------------------------------
# Figure 4: subsidy before/after (dumbbell plot)
# ---------------------------------------------------------------------------
def fig4_subsidy():
    order = np.argsort(sub_base)
    cities = [sub_city[i] for i in order]
    b = sub_base[order]
    c = sub_con[order]
    y = np.arange(len(b))

    fig, ax = plt.subplots(figsize=(5.0, 5.5))
    for i in range(len(b)):
        ax.plot([b[i], c[i]], [i, i], color=COLORS["neutral"], lw=0.6, zorder=1)

    ax.scatter(b, y, color=COLORS["primary"], s=50, zorder=2,
               edgecolor="white", linewidth=0.5)
    ax.scatter(c, y, color=COLORS["secondary"], s=50, zorder=2,
               edgecolor="white", linewidth=0.5)

    ax.axvline(1.0, color=COLORS["neutral"], lw=0.8, zorder=0)
    mean_b = float(f"{np.mean(b):.4f}")
    mean_c = float(f"{np.mean(c):.4f}")
    ax.axvline(mean_b, color=COLORS["primary"], lw=1.0, ls="--", zorder=0)
    ax.axvline(mean_c, color=COLORS["secondary"], lw=1.0, ls="--", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(cities, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Measured PoA (consistent basis)", fontsize=10)
    ax.set_xlim(0.995, 1.18)

    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["primary"],
               markersize=8, label=f"Base {mean_b}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["secondary"],
               markersize=8, label=f"Subsidized {mean_c}"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=9, frameon=False)
    ax.set_title("Per-city PoA before and after the V2 subsidy", fontsize=11, pad=10)

    fig.tight_layout()
    save(fig, "fig4_subsidy")


# ---------------------------------------------------------------------------
# Graphical Abstract: donut + histogram
# ---------------------------------------------------------------------------
def ga():
    # 1200 x 1600 px at 300 dpi -> 4 x 5.333 in
    fig = plt.figure(figsize=(4.0, 5.333))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.25,
                          top=0.88, bottom=0.14)

    # Title block
    fig.text(0.5, 0.97, "Bounded Price of Anarchy in", ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945, "Capacitated Facility Location Games", ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.920, "with Congestion Feedback", ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.895, "Censored satisfaction bounds the PoA; coverage drives the gap",
             ha="center", va="top", fontsize=10, style="italic")

    # ---- Top panel: donut chart ----
    ax1 = fig.add_subplot(gs[0, 0])
    sectors = ["Coverage", "Assignment", "Congestion"]
    # Use absolute magnitudes for wedge sizes; congestion shown in negative color.
    values = [81.9, 34.2, abs(-14.9)]
    colors = [COLORS["primary"], COLORS["light_fill"], COLORS["negative"]]
    wedges, texts = ax1.pie(values, colors=colors, startangle=90, counterclock=False,
                            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=1.5))
    # Move labels/legend to the right, clear of the donut
    ax1.legend(wedges, sectors, loc="center left", bbox_to_anchor=(0.95, 0.5),
               frameon=False, fontsize=10)
    ax1.set_aspect("equal")

    # ---- Bottom panel: histogram + KDE ----
    ax2 = fig.add_subplot(gs[1, 0])
    bins = 50
    range_ = (0.9, 1.5)
    ax2.hist(enum_poa, bins=bins, range=range_, color=COLORS["primary"],
             edgecolor="white", linewidth=0.3, alpha=0.75)

    kde = gaussian_kde(enum_poa)
    x_grid = np.linspace(range_[0], range_[1], 500)
    bin_width = (range_[1] - range_[0]) / bins
    kde_scaled = kde(x_grid) * len(enum_poa) * bin_width
    ax2.plot(x_grid, kde_scaled, color=COLORS["primary"], lw=1.5)

    mean_val = float(f"{np.mean(enum_poa):.4f}")
    ax2.axvline(mean_val, color=COLORS["negative"], lw=1.2, ls="--")
    ax2.axvline(1.0, color=COLORS["neutral"], lw=0.8, ls="--")

    ax2.set_xlabel("Measured sequential PoA (≥)", fontsize=11)
    ax2.set_ylabel("Configurations", fontsize=11)
    ax2.set_xlim(range_)

    # Bottom text
    fig.text(0.5, 0.055,
             "21 cities (mean PoA 1.0420)   ·   15,207 configs (mean PoA 1.0610)",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 0.020,
             "Assignment channel: +69.3%  ·  Coverage: +43.1%  ·  Congestion: −12.6%",
             ha="center", va="top", fontsize=11, fontweight="bold")

    fig.savefig(os.path.join(FIG, "GraphicalAbstract.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "GraphicalAbstract.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1_poa21()
    fig2_poa_dist()
    fig3_channel()
    fig4_subsidy()
    ga()
    print("Figures written to", FIG)
    for f in sorted(os.listdir(FIG)):
        print(" ", f)
