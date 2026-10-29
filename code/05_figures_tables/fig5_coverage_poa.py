# -*- coding: utf-8 -*-
"""E3 / E10: coverage-PoA scatter and the network-size gradient (paper Figure 5).

Why this script exists
----------------------
Two claims in Section~5 are supported by regressions but were never shown as a
single picture:

  (i)  the measured PoA rises with the *model's own* coverage c^MILP
       (the coverage-PoA mechanism; r = 0.84 over the 21 cities).  Figure~2
       shows PoA against the *external* 2025 rate, not against c^MILP, so the
       internal relation was asserted but not displayed.
  (ii) the strongest structural predictor is network size, not geography:
       PoA rises with log10(n_sta) (standardised beta = +0.64, p = 0.0023).

Panel (a) plots (i) with points coloured by the coverage band of
``table_cov_bands.py`` so that the two deliverables share one classification.
Panel (b) plots (ii) and marks the median split used as the heterogeneity
summary in the appendix.

Inputs
------
``results/main/21cities_exact_locked.json``

Outputs
-------
``results/figures/fig5_coverage_poa.{pdf,png}``

Usage
-----
    python code/05_figures_tables/fig5_coverage_poa.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get(
    "CFLG_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
SRC = os.path.join(ROOT, "results", "main", "21cities_exact_locked.json")
OUT = os.path.join(ROOT, "results", "figures")

BAND_COLORS = ("#1f6fb2", "#e08a1e", "#b03a2e")   # low / medium / high
BAND_LABELS = ("Low coverage (<20%)", "Medium (20-35%)", "High ($\\geq$35%)")


def band_of(cov):
    return 0 if cov < 0.20 else (1 if cov < 0.35 else 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(SRC, encoding="utf-8"))["results"]
    cities = list(res)
    cov = np.array([res[c]["milp_cov"] for c in cities])
    poa = np.array([res[c]["PoA"] for c in cities])
    nsta = np.array([res[c]["n_stations"] for c in cities], dtype=float)
    bands = np.array([band_of(c) for c in cov])

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # ---------------- panel (a): PoA vs model coverage --------------------
    for b in range(3):
        m = bands == b
        ax[0].scatter(100 * cov[m], poa[m], s=46, c=BAND_COLORS[b],
                      edgecolor="k", linewidth=0.4, alpha=0.9, zorder=3,
                      label=f"{BAND_LABELS[b]} (n={int(m.sum())})")
    r_a, p_a = pearsonr(cov, poa)
    rho_a, _ = spearmanr(cov, poa)
    slope, intercept = np.polyfit(cov, poa, 1)
    xs = np.linspace(cov.min(), cov.max(), 60)
    ax[0].plot(100 * xs, intercept + slope * xs, "--", color="0.3", lw=1.3,
               zorder=2, label=f"OLS fit ($R^2$={r_a**2:.2f})")
    ax[0].set_xlabel("Planner coverage, $c^{\\mathrm{MILP}}$ (%)")
    ax[0].set_ylabel("Measured Price of Anarchy, PoA")
    ax[0].set_title(f"(a) PoA rises with model coverage\n"
                    f"$r$={r_a:.2f} ($p$={p_a:.1e}), $\\rho$={rho_a:.2f}",
                    fontsize=10)
    ax[0].legend(fontsize=7, loc="lower right", framealpha=0.92)
    ax[0].grid(alpha=0.25, lw=0.5)

    # ---------------- panel (b): PoA vs network size ----------------------
    logn = np.log10(nsta)
    r_b, p_b = pearsonr(logn, poa)
    slope_b, intercept_b = np.polyfit(logn, poa, 1)
    xs_b = np.linspace(logn.min(), logn.max(), 60)
    ax[1].scatter(logn, poa, s=46, c="#2c3e50", edgecolor="k", linewidth=0.4,
                  alpha=0.85, zorder=3)
    ax[1].plot(xs_b, intercept_b + slope_b * xs_b, "--", color="0.3", lw=1.3,
               zorder=2, label=f"OLS fit ($R^2$={r_b**2:.2f})")
    # Heterogeneity summary uses the same bottom-5 / top-5 definition as
    # exp1_regression.py (P1_1_regression_locked.json: small_scale_mean_PoA
    # / large_scale_mean_PoA), so the figure and the appendix quote one number.
    order = np.argsort(nsta)
    small, large = order[:5], order[-5:]
    ax[1].annotate(f"bottom-5 by $n_\\mathrm{{sta}}$: mean {poa[small].mean():.4f}\n"
                   f"top-5 by $n_\\mathrm{{sta}}$: mean {poa[large].mean():.4f}",
                   xy=(0.03, 0.95), xycoords="axes fraction", fontsize=7.5,
                   va="top", color="0.25")
    ax[1].set_xlabel("$\\log_{10}$ number of stations, $\\log_{10} n_{\\mathrm{sta}}$")
    ax[1].set_ylabel("Measured Price of Anarchy, PoA")
    ax[1].set_title(f"(b) PoA rises with network size\n"
                    f"$r$={r_b:.2f} ($p$={p_b:.4f})", fontsize=10)
    ax[1].legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax[1].grid(alpha=0.25, lw=0.5)

    # Manual label offsets keep the two low-coverage cities (Kunming, Dongguan)
    # and the two high-coverage ones (Chongqing, Beijing) from colliding.  The
    # offsets are panel-specific because the x-axis scales differ by an order
    # of magnitude.
    OFF_A = {"Kunming": (7, 6), "Dongguan": (6, -12), "Chongqing": (-48, 5),
             "Nanjing": (-40, 3), "Harbin": (4, 4)}
    OFF_B = {"Kunming": (7, 9), "Dongguan": (-10, 9), "Chongqing": (-50, 5),
             "Nanjing": (-38, 5), "Harbin": (4, 4)}
    for a, xkey, offs in ((ax[0], cov * 100, OFF_A), (ax[1], logn, OFF_B)):
        for c, xv in zip(cities, xkey):
            if c in offs:
                a.annotate(c, (xv, res[c]["PoA"]), fontsize=6.5,
                           xytext=offs[c], textcoords="offset points",
                           color="0.2")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig5_coverage_poa.{ext}"), dpi=300,
                    bbox_inches="tight")

    print(f"(a) PoA vs c^MILP : r={r_a:+.3f} (p={p_a:.2e})  rho={rho_a:+.3f}  "
          f"slope={slope:.3f} per unit coverage  R2={r_a**2:.3f}")
    print(f"(b) PoA vs log10 n_sta: r={r_b:+.3f} (p={p_b:.4f})  "
          f"R2={r_b**2:.3f}")
    print(f"heterogeneity (bottom-5 vs top-5 by n_sta): "
          f"{poa[small].mean():.4f} (n_sta<= "
          f"{[int(nsta[i]) for i in small]}) vs {poa[large].mean():.4f} "
          f"(n_sta>= {[int(nsta[i]) for i in large]})")
    print("[saved] results/figures/fig5_coverage_poa.{pdf,png}")


if __name__ == "__main__":
    main()
