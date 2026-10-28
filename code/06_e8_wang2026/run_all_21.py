#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E8：用 Wang2026 1km 基线栅格 (Total_t60_k0) 按 21 城边界裁剪，重推三步法参数。
输出与 Han 旧参数并排的对比表。供"可比性审计"使用。

注意（重要）：
  - n_stations / cluster_r 两套数据同为"可达性热点"口径，可比；
  - A_built / n_communities 因 Wang2026 无"建成区"图层、栅格铺满全域，不可比
    （仅记录，不用于对照结论）。
"""
import os, sys, json
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from derive_params_generic import load_raster, derive  # noqa

W26 = os.environ.get("W26_DIR", r"D:/电工杯opencode/数据/Wang2026v2")
ROOT = os.environ.get("CFLG_ROOT", r"D:/电工杯opencode/复现包/cflg-poa-reproducibility")
RASTER = os.path.join(W26, "extracted", "Total_t60_k0.tif")
BDIR = os.path.join(W26, "boundaries")
HAN = os.path.join(ROOT, "data", "processed", "21city_parameters.csv")
OUTDIR = os.path.join(ROOT, "results", "e8_wang2026")
os.makedirs(OUTDIR, exist_ok=True)

CITIES = ["Beijing","Tianjin","Shenyang","Dalian","Harbin","Shanghai","Nanjing",
          "Hangzhou","Jinan","Qingdao","Zhengzhou","Wuhan","Changsha","Guangzhou",
          "Shenzhen","Foshan","Dongguan","Chongqing","Chengdu","Kunming","Xian"]

def load_poly(path):
    gj = json.load(open(path, encoding="utf-8"))
    poly = {"type": "MultiPolygon", "coordinates": []}
    for f in gj["features"]:
        g = f["geometry"]
        if g["type"] == "Polygon":
            poly["coordinates"].append(g["coordinates"])
        elif g["type"] == "MultiPolygon":
            poly["coordinates"].extend(g["coordinates"])
    return poly

recs = []
for c in CITIES:
    pj = os.path.join(BDIR, c + ".json")
    if not os.path.exists(pj):
        print(f"[MISS] {c} 无边界"); continue
    poly = load_poly(pj)
    data, valid, tr = load_raster(RASTER, poly, None)
    r = derive(data, valid, tr, high_pct=0.0, city=c)
    recs.append(r)
    print(f"{c:<11s} n_sta={r['n_stations']:4d} clu_r={r['cluster_r_m'] if r['cluster_r_m'] is None else round(r['cluster_r_m']):>6} "
          f"A_built={r['A_built_km2']:9.1f} n_com={r['n_communities']:5d} valid={r['n_px_valid']}")

w = pd.DataFrame(recs)
w.to_json(os.path.join(OUTDIR, "w26_params_21city.json"), orient="records", force_ascii=False, indent=2)
w.to_csv(os.path.join(OUTDIR, "w26_params_21city.csv"), index=False, encoding="utf-8-sig")

# 并排对比
def _key(s):
    s = str(s).lower().replace("'", "").replace("’", "")
    if s in ("haerbin", "harbin"):
        return "harbin"
    return s

han = pd.read_csv(HAN)
han["key"] = han["city"].map(_key)
w["key"] = w["city"].map(_key)
m = han.merge(w, on="key", how="inner", suffixes=("_han", "_w26"))
cmp = pd.DataFrame({
    "city": m["city_han"],
    "han_n_sta": m["n_stations_han"], "w26_n_sta": m["n_stations_w26"],
    "sta_ratio": (m["n_stations_w26"] / m["n_stations_han"]).round(2),
    "han_clu_r": m["cluster_r_m_han"].round(0), "w26_clu_r": m["cluster_r_m_w26"].round(0),
    "clu_ratio": (m["cluster_r_m_w26"] / m["cluster_r_m_han"]).round(2),
    "han_n_com": m["n_communities_han"], "w26_n_com_raw": m["n_communities_w26"],
    "han_A_built": m["A_built_km2_han"], "w26_A_raw": m["A_built_km2_w26"],
})
cmp.to_csv(os.path.join(OUTDIR, "compare_21city_w26_vs_han.csv"), index=False, encoding="utf-8-sig")
print("\n=== 对比表 ===")
print(cmp.to_string(index=False))
print("\n=== 汇总 ===")
print("n_sta 比值(W26/Han): min %.2f median %.2f max %.2f" % (cmp.sta_ratio.min(), cmp.sta_ratio.median(), cmp.sta_ratio.max()))
print("clu_r 比值(W26/Han): min %.2f median %.2f max %.2f" % (cmp.clu_ratio.min(), cmp.clu_ratio.median(), cmp.clu_ratio.max()))
print("n_sta 相关: %.3f" % np.corrcoef(cmp.han_n_sta, cmp.w26_n_sta)[0, 1])
print("clu_r 相关: %.3f" % np.corrcoef(cmp.han_clu_r, cmp.w26_clu_r)[0, 1])
print("\n[saved]", OUTDIR)
