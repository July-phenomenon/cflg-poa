# -*- coding: utf-8 -*-
"""Unified plotting style for every figure in the paper.

Rules (from the figure execution plan):
  * no text boxes / annotations inside manuscript figures,
  * one palette for all figures,
  * Arial when available (DejaVu Sans fallback),
  * PDF (vector) + PNG (300 dpi), never JPG.
"""
import os
import matplotlib
from matplotlib import font_manager

COLORS = {
    "primary":    "#2E5A88",   # main blue
    "secondary":  "#E8873A",   # secondary orange
    "negative":   "#C0392B",   # negative value red
    "neutral":    "#7F7F7F",   # neutral grey
    "light_fill": "#A9C5E0",   # light blue fill
    "tie":        "#A6A6A6",   # light grey (tie / artifact)
    "dark":       "#1F3B57",   # dark outline for box plots
}

LABEL_SIZE = 10
TICK_SIZE = 9
LEGEND_SIZE = 9
TITLE_SIZE = 11
DPI = 300

SINGLE_COL = (4.5, 3.5)
DOUBLE_COL = (7.5, 3.5)
THREE_PANEL = (15, 4)


def setup():
    """Install the house font and rcParams. Call once at import time."""
    matplotlib.use("Agg")
    family = "DejaVu Sans"
    for f in font_manager.fontManager.ttflist:
        if f.name.lower().startswith("arial"):
            family = "Arial"
            break
    matplotlib.rcParams.update({
        "font.family": family,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "pdf.fonttype": 42,       # embed TrueType so the PDF is self-contained
        "ps.fonttype": 42,
        "figure.dpi": DPI,
    })
    return family


def save(fig, path_without_ext):
    """Write PDF (vector) + PNG (300 dpi)."""
    os.makedirs(os.path.dirname(path_without_ext), exist_ok=True)
    fig.savefig(path_without_ext + ".pdf", bbox_inches="tight")
    fig.savefig(path_without_ext + ".png", bbox_inches="tight", dpi=DPI)
