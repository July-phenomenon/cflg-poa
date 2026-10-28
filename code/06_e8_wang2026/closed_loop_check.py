# -*- coding: utf-8 -*-
"""E8 闭环检验：论文模型覆盖 vs Wang2026 真实覆盖（跨 21 城秩相关）。

自包含：直接从 Wang2026 的 city_level.csv 读取 *_accessible_rate，不依赖中间文件。
路径可用环境变量覆盖：CFLG_ROOT（复现包根）、W26_DIR（Wang2026 数据根）。

假说：
  H1  论文 milp_cov（容量约束覆盖）与 Wang2026 覆盖率正相关（方向性外部效度）。
  H2  PoA 与覆盖率正相关（覆盖越低，中央-居民差异越小，PoA→1）——论文机制。
输出：results/e8_wang2026/closed_loop_merged.csv, closed_loop_correlations.csv
"""
import json, os, re
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = os.environ.get("CFLG_ROOT", r"D:/电工杯opencode/复现包/cflg-poa-reproducibility")
W26 = os.environ.get("W26_DIR", r"D:/电工杯opencode/数据/Wang2026v2")
J = os.path.join(ROOT, "results/main/21cities_exact_locked.json")
OUT = os.path.join(ROOT, "results/e8_wang2026")
os.makedirs(OUT, exist_ok=True)


def _find_city_level():
    for p in (os.path.join(W26, "2_Administrative_Aggregation", "city_level.csv"),
              os.path.join(W26, "_inspect", "city_level.csv")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("city_level.csv not found under W26_DIR; extract it from the archive first")


def key(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s*(city|市)$", "", s)          # "Beijing City" -> "beijing"
    s = s.replace("'", "").replace("’", "")
    return "harbin" if s in ("haerbin", "harbin") else s


# --- paper side ---
d = json.load(open(J, encoding="utf-8"))
res = d["results"]
paper = pd.DataFrame([{
    "city": c, "milp_cov": v["milp_cov"], "nash_cov": v["nash_cov"],
    "PoA": v["PoA"], "milp_F2": v["milp_F2"], "nash_F2": v["nash_F2"],
    "n_stations": v["n_stations"], "cluster_r": v["cluster_r"],
} for c, v in res.items()])
paper["key"] = paper["city"].map(key)

# --- Wang2026 side（直接算覆盖率）---
cl = pd.read_csv(_find_city_level())
cl["key"] = cl["name_en"].map(key)
w = pd.DataFrame({
    "key": cl["key"],
    "w26_Total_t60_k0_rate_pct": cl["Total_t60_k0_accessible_rate"] * 100,
    "w26_Nursing_t60_k0_rate_pct": cl["Nursing_t60_k0_accessible_rate"] * 100,
    "w26_Total_t60_k3_rate_pct": cl["Total_t60_k3_accessible_rate"] * 100,
})

m = paper.merge(w, on="key", how="inner")
print("merged n =", len(m))


def rep(a, b, label):
    x = m[a].astype(float); y = m[b].astype(float)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    pr, pp = pearsonr(x, y); sr, sp = spearmanr(x, y)
    print(f"{label:<44s} Pearson r={pr:+.3f} (p={pp:.3f})  Spearman ρ={sr:+.3f} (p={sp:.3f})  n={len(x)}")
    return dict(comparison=label, n=len(x), pearson_r=round(pr, 3), pearson_p=round(pp, 3),
                spearman_rho=round(sr, 3), spearman_p=round(sp, 3))


print("\n=== 闭环相关检验 ===")
rows = [
    rep("milp_cov", "w26_Nursing_t60_k0_rate_pct", "模型覆盖 vs 真实护理型覆盖"),
    rep("milp_cov", "w26_Total_t60_k0_rate_pct",   "模型覆盖 vs 真实普通床位覆盖"),
    rep("nash_cov", "w26_Nursing_t60_k0_rate_pct", "均衡覆盖 vs 真实护理型覆盖"),
    rep("nash_cov", "w26_Total_t60_k0_rate_pct",   "均衡覆盖 vs 真实普通床位覆盖"),
    rep("PoA",      "w26_Nursing_t60_k0_rate_pct", "PoA vs 真实护理型覆盖"),
    rep("PoA",      "w26_Total_t60_k0_rate_pct",   "PoA vs 真实普通床位覆盖"),
    rep("PoA",      "milp_cov",                    "PoA vs 模型自身覆盖（内部一致性）"),
]

print("\n=== 论文侧覆盖统计（对照） ===")
print("milp_cov: min %.3f  median %.3f  mean %.3f  max %.3f" % (m.milp_cov.min(), m.milp_cov.median(), m.milp_cov.mean(), m.milp_cov.max()))
print("W26 Nursing rate: median %.2f%%   W26 Total rate: median %.2f%%" % (
    m.w26_Nursing_t60_k0_rate_pct.median(), m.w26_Total_t60_k0_rate_pct.median()))

m.to_csv(os.path.join(OUT, "closed_loop_merged.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "closed_loop_correlations.csv"), index=False, encoding="utf-8-sig")
print("\n[saved]", OUT)
