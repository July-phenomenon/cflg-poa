# -*- coding: utf-8 -*-
"""E8 外部效度图：2 面板散点（模型覆盖/PoA vs Wang2026 真实护理型覆盖）+ 内部一致性检查。"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = os.environ.get("CFLG_ROOT", r"D:/电工杯opencode/复现包/cflg-poa-reproducibility")
OUT = os.path.join(ROOT, "results/figures")
os.makedirs(OUT, exist_ok=True)
m = pd.read_csv(os.path.join(ROOT, "results/e8_wang2026/closed_loop_merged.csv"))

# 内部一致性：PoA vs 模型自身覆盖
pr, pp = pearsonr(m.milp_cov, m.PoA)
sr, sp = spearmanr(m.milp_cov, m.PoA)
print(f"PoA vs model milp_cov: Pearson {pr:+.3f} (p={pp:.4f}) Spearman {sr:+.3f} (p={sp:.4f})")

# 离群点（|残差| 大者）简查
print("\n护理型覆盖率最高/最低的几个城：")
print(m.nlargest(3, "w26_Nursing_t60_k0_rate_pct")[["city","milp_cov","w26_Nursing_t60_k0_rate_pct"]].to_string(index=False))
print(m.nsmallest(3, "w26_Nursing_t60_k0_rate_pct")[["city","milp_cov","w26_Nursing_t60_k0_rate_pct"]].to_string(index=False))

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0))

# Panel A: model coverage vs real nursing coverage
x = m.milp_cov * 100
y = m.w26_Nursing_t60_k0_rate_pct
r1, p1 = pearsonr(x, y); s1, sp1 = spearmanr(x, y)
ax[0].scatter(x, y, s=42, c="#c0392b", edgecolor="k", linewidth=0.4, alpha=0.85, zorder=3)
b, a = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 50)
ax[0].plot(xx, a + b * xx, "--", color="0.35", lw=1.3, zorder=2)
ax[0].set_xlabel("Model coverage, $c^{MILP}$ (%)")
ax[0].set_ylabel("Real nursing-grade coverage, 2025 (%)")
ax[0].set_title(f"(a) Model vs real coverage\n$r$={r1:.2f} ($p$={p1:.3f}), $\\rho$={s1:.2f}", fontsize=10)
ax[0].grid(alpha=0.25, lw=0.5)

# Panel B: PoA vs real nursing coverage
x2 = m.PoA
r2, p2 = pearsonr(x2, y); s2, sp2 = spearmanr(x2, y)
ax[1].scatter(x2, y, s=42, c="#2c3e50", edgecolor="k", linewidth=0.4, alpha=0.85, zorder=3)
b2, a2 = np.polyfit(x2, y, 1); xx2 = np.linspace(x2.min(), x2.max(), 50)
ax[1].plot(xx2, a2 + b2 * xx2, "--", color="0.35", lw=1.3, zorder=2)
ax[1].set_xlabel("Measured Price of Anarchy, PoA")
ax[1].set_ylabel("Real nursing-grade coverage, 2025 (%)")
ax[1].set_title(f"(b) PoA vs real coverage\n$r$={r2:.2f} ($p$={p2:.3f}), $\\rho$={s2:.2f}", fontsize=10)
ax[1].grid(alpha=0.25, lw=0.5)

# 标注几个代表性城市
for _, row in m.iterrows():
    if row.city in ("Kunming", "Dongguan", "Nanjing", "Chongqing", "Beijing", "Hangzhou"):
        ax[0].annotate(row.city, (row.milp_cov*100, row.w26_Nursing_t60_k0_rate_pct),
                       fontsize=6.5, xytext=(3, 3), textcoords="offset points", color="0.2")
        ax[1].annotate(row.city, (row.PoA, row.w26_Nursing_t60_k0_rate_pct),
                       fontsize=6.5, xytext=(3, 3), textcoords="offset points", color="0.2")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"e8_external_validation.{ext}"), dpi=300, bbox_inches="tight")
print("\n[saved]", os.path.join(OUT, "e8_external_validation.{png,pdf}"))
