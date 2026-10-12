#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数推导 v3 —— 从 Han et al. (2026) 可达性栅格直接测量 21 城实验参数
====================================================================
执行文档：《从 Han et al. (2026) 栅格直接推导 21 城实验参数》v1.0

核心原则（数据为王）：
  1. 不引入任何来源不明的经验公式（严禁 R = 313*exp(-0.022*p) 之类）
  2. 所有参数从栅格像素级数据按明确步骤测量
  3. 步骤透明可复现：相同栅格 + 相同代码 = 相同结果
  4. 参数可溯源到 Han et al. (2026) 数据集
  5. 接受任何结果（即使 21 城参数差异小）也如实报告

数据源
------
Han, X., Wang, Y., Wei, Z. & Jiang, H. High-Resolution dataset on elderly care
facility accessibility and inequality in 21 Chinese cities (2020).
Scientific Data 13, 606 (2026). DOI: 10.1038/s41597-026-07014-8

  - 栅格: $CFLG_RASTER_DIR/Accessibility/Accessibility/{City}_Accessibility.tif
  - 统计: $CFLG_RASTER_DIR/City_Summary_Statistics.csv

（$CFLG_RASTER_DIR 由环境变量给出，默认 D:/zenodo/17010522；栅格本身不在本包内，
 见 data/external/DATA_SOURCES.md。）

三步法
------
Step 1  建成区提取 + 高可达性阈值反推（阈值由 CSV 的 High-Accessibility % 分位数确定）
Step 2  高斯平滑 + 局部峰值检测（设施聚类中心）
Step 3  参数推导：n_stations = 峰值数；cluster_r = 峰值最近邻距离中位数/2；
        n_communities = 建成区面积 / 7.07 km²（民政部小型设施服务半径 3km 对应空间单元）

相对执行文档伪代码的两处技术修正（均已实测，理由见注释）
-------------------------------------------------------
【修正 1】像元尺寸按实际 3 弧秒而非假定 100m
    栅格 CRS 为 EPSG:4326，像元 0.0008333333° = 3 arc-sec。
    纬向 0.0008333° × 111320 m/° = 92.77 m；经向需乘 cos(lat)。
    论文称 "100m" 为近似表述；本脚本采用实测值以保证面积/距离不系统性偏大。

【修正 2】建成区保留 0 值像元（不施加 data > 0）
    实测：哈尔滨有效像元中 36.2% 取值为 0。Ga2SFCA 可达性为 0 表示
    “有人口但搜索半径内无设施”，正是 CSV 中 “Low-Accessibility Zones” 的本体。
    若按 data > 0 过滤，将系统性剔除低可达性区域（哈尔滨 CSV 该类占 79.49%），
    导致高可达性占比的分母失真。故建成区定义为「栅格有值（有限值）的像元」。

输出
----
  <城市统计>/实城实验参数v3.json   主输出（供实验脚本读取）
  <城市统计>/实城实验参数v3.csv    同名表格（供人工核对）
"""

import os
import sys
import json
import glob

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

# ==========================================================================
# 常量（每一个都有明确来源，无“魔数”）
# ==========================================================================
BASE = os.environ.get("CFLG_RASTER_DIR", r"D:/zenodo/17010522")
ACC_DIR = os.path.join(BASE, 'Accessibility', 'Accessibility')   # 实测为嵌套结构
CSV_PATH = os.path.join(BASE, 'City_Summary_Statistics.csv')

OUT_DIR = os.path.join(os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))), "data", "processed")
OUT_JSON = os.path.join(OUT_DIR, "21city_parameters.json")
OUT_CSV = os.path.join(OUT_DIR, "21city_parameters.csv")

# 地球常数：纬度 1° 对应的大圆弧长（m），WGS84 平均
M_PER_DEG_LAT = 111320.0

# 平滑尺度 σ：论文 Methods 的中型设施服务半径 l0 = 5 km，
# 高斯衰减 G = exp(-(l/l0)^2)（Ga2SFCA），在 l = l0/3 处权重 ≈ exp(-1/9) ≈ 0.895。
# 执行文档取 σ = l0/3 = 1.667 km。此处沿用该定义（唯一有出处的主观选择）。
SIGMA_KM = 5.0 / 3.0

# 社区单元面积：民政部小型设施服务半径 l0 = 3 km（论文 Methods 引述），
# 对应一个基本服务单元的等效面积 π·(l0/2)² = π·1.5² = 7.0686 km²
COMMUNITY_UNIT_KM2 = np.pi * 1.5 ** 2

# 非极大值抑制：邻域边长取 2σ（执行文档定义），保证峰值间距 ≥ 2σ
NMS_FACTOR = 2.0


def lat_of_row(transform, nrows):
    """返回每一行中心像元的纬度（度）。transform 为 rasterio Affine。"""
    # 行 r 中心的 y 坐标（投影/地理坐标）：transform.f + (r + 0.5) * transform.e
    rows = np.arange(nrows, dtype=np.float64)
    # EPSG:4326 下 y 即纬度；e 为负（北在上）
    return transform.f + (rows + 0.5) * transform.e


def derive_city_params(city_name, raster_path, high_pct, verbose=True):
    """按三步法从单个城市的可达性栅格推导实验参数。"""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        mask = src.read_masks(1) > 0
        transform = src.transform
        nodata = src.nodata
        res_x, res_y = abs(transform.a), abs(transform.e)

    # ---------- Step 1.1 建成区 ----------
    # 有效像元 = 掩膜有效 且 数值有限（NoData 实测为 -inf 或 -3.4e38）
    valid = mask & np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        valid &= (data != nodata)
    n_px_valid = int(valid.sum())
    if n_px_valid == 0:
        raise RuntimeError(f'{city_name}: 无有效像元')

    # 裁到有效区域的包围盒，显著降低大栅格（如重庆 5829×4784）的计算量
    ys, xs = np.where(valid)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    data_c = data[y0:y1, x0:x1]
    valid_c = valid[y0:y1, x0:x1]

    # 每像元的实际面积（m²）：纬向固定，经向随纬度收缩
    lats = lat_of_row(transform, data.shape[0])[y0:y1]
    m_per_px_y = res_y * M_PER_DEG_LAT
    m_per_px_x = res_x * M_PER_DEG_LAT * np.cos(np.deg2rad(lats))
    px_area_m2 = (m_per_px_x[:, None] * np.ones((1, data_c.shape[1]))) * m_per_px_y

    A_built_m2 = float(px_area_m2[valid_c].sum())
    A_built_km2 = A_built_m2 / 1e6

    vals = data_c[valid_c]
    n_px_zero = int((vals == 0).sum())

    # ---------- Step 1.2 反推高可达性阈值 ----------
    # 使得 P(accessibility > τ_high) = high_pct（占建成区面积比例）
    tau_high = float(np.percentile(vals, 100.0 - high_pct))
    high_mask_c = valid_c & (data_c > tau_high)
    A_high_km2 = float(px_area_m2[high_mask_c].sum()) / 1e6

    # ---------- Step 2.1 高斯平滑 ----------
    # 像元边长取纬向实际米数（≈92.77 m），σ 由 1.667 km 换算
    sigma_px = SIGMA_KM * 1000.0 / m_per_px_y
    filled = np.where(valid_c, data_c, 0.0).astype(np.float32)
    smoothed = gaussian_filter(filled, sigma=sigma_px, mode='constant', cval=0.0)

    # ---------- Step 2.2 局部峰值检测（非极大值抑制） ----------
    neighborhood = max(3, int(round(NMS_FACTOR * sigma_px)))
    local_max = maximum_filter(smoothed, size=neighborhood, mode='constant', cval=-np.inf)
    cand = (smoothed >= local_max) & valid_c & (smoothed > 0)

    # 再做一次贪心抑制，强制峰值间最小间距 = neighborhood 像元
    # （消除平坦区并列极大值造成的伪峰）
    py, px = np.where(cand)
    if len(py) > 0:
        svals = smoothed[py, px]
        order = np.argsort(-svals)
        tree = cKDTree(np.column_stack([py, px]))
        min_sep = float(neighborhood)
        keep = []
        accepted = []
        for idx in order:
            p = (py[idx], px[idx])
            if accepted:
                d, _ = cKDTree(np.array(accepted)).query(np.array([p]), k=1)
                if float(d[0]) < min_sep:
                    continue
            accepted.append(p)
            keep.append(idx)
        keep = np.array(keep, dtype=int)
        py, px = py[keep], px[keep]
    n_peaks = int(len(py))

    # ---------- Step 3 参数推导 ----------
    # 峰值像元 → 米制坐标（用于距离计算）
    if n_peaks > 0:
        peak_y_m = (py + y0 + 0.5) * m_per_px_y
        peak_x_m = (px + x0 + 0.5) * m_per_px_x[py]
        coords = np.column_stack([peak_x_m, peak_y_m])
    else:
        coords = np.zeros((0, 2))

    if n_peaks >= 2:
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        nn_dists = dists[:, 1]
        cluster_r_m = float(np.median(nn_dists) / 2.0)
        nn_median_m = float(np.median(nn_dists))
    else:
        cluster_r_m = None
        nn_median_m = None

    n_communities = int(round(A_built_km2 / COMMUNITY_UNIT_KM2))
    n_communities = max(1, n_communities)

    # 供审计的中间量
    share_zero = float(n_px_zero / n_px_valid) if n_px_valid else None
    share_high_derived = float(A_high_km2 / A_built_km2) if A_built_km2 > 0 else None

    rec = {
        'city': city_name,
        'n_communities': n_communities,
        'cluster_r_m': cluster_r_m,
        'n_stations': n_peaks,
        'A_built_km2': round(A_built_km2, 2),
        'A_high_km2': round(A_high_km2, 2),
        'high_pct_csv': high_pct,
        'n_peaks': n_peaks,
        'nn_median_m': round(nn_median_m, 1) if nn_median_m is not None else None,
        'tau_high': round(tau_high, 6),
        'n_px_valid': n_px_valid,
        'n_px_zero': n_px_zero,
        'share_zero': round(share_zero, 4) if share_zero is not None else None,
        'share_high_derived': round(share_high_derived, 4) if share_high_derived is not None else None,
        'sigma_px': round(float(sigma_px), 2),
        'nms_neighborhood_px': neighborhood,
        'px_size_m': round(float(m_per_px_y), 2),
        'raster': os.path.basename(raster_path),
    }

    if verbose:
        print(f"  {city_name:<11s} A_built={A_built_km2:8.1f}km²  n_comm={n_communities:5d}  "
              f"peaks={n_peaks:4d}  cluster_r={cluster_r_m if cluster_r_m is None else round(cluster_r_m):>6}m  "
              f"high%={high_pct:5.2f}")
    return rec


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- CSV：城市名 + 高可达性占比 ----
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['City'])
    col_high = 'Areal Proportion of High-Accessibility Zones (%)'
    col_low = 'Areal Proportion of Low-Accessibility Zones (%)'
    col_gdp = 'Per Capita GDP (yuan), 2020'
    high_map = {str(r['City']).strip(): float(str(r[col_high]).rstrip('%')) for _, r in df.iterrows()}
    low_map = {str(r['City']).strip(): float(str(r[col_low]).rstrip('%')) for _, r in df.iterrows()}
    gdp_map = {str(r['City']).strip(): float(r[col_gdp]) for _, r in df.iterrows()}

    # ---- 栅格：以实际文件为准，保证与 CSV 交叉核对 ----
    raster_files = sorted(glob.glob(os.path.join(ACC_DIR, '*_Accessibility.tif')))
    raster_cities = [os.path.basename(f).replace('_Accessibility.tif', '') for f in raster_files]
    print(f"[INFO] 栅格城市数 = {len(raster_cities)}，CSV 城市数 = {len(high_map)}")
    missing_csv = [c for c in raster_cities if c not in high_map]
    missing_ras = [c for c in high_map if c not in raster_cities]
    print(f"[INFO] 有栅格无CSV: {missing_csv or '无'}")
    print(f"[INFO] 有CSV无栅格: {missing_ras or '无'}")

    only = sys.argv[1:] if len(sys.argv) > 1 else None
    records = []
    for city, path in zip(raster_cities, raster_files):
        if only and city not in only:
            continue
        if city not in high_map:
            print(f"[WARN] 跳过 {city}：CSV 无统计值")
            continue
        rec = derive_city_params(city, path, high_map[city])
        rec['low_pct_csv'] = low_map.get(city)
        rec['gdp_per_capita'] = gdp_map.get(city)
        records.append(rec)

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    pd.DataFrame(records).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n[DONE] 已处理 {len(records)} 城")
    print(f"       主输出: {OUT_JSON}")
    print(f"       表格  : {OUT_CSV}")

    # 汇总范围检查（执行文档 Step 3 S3.1）
    d = pd.DataFrame(records)
    print("\n=== 参数范围检查 ===")
    for col, lo, hi in [('n_communities', 50, 2000), ('cluster_r_m', 200, 2000),
                        ('n_stations', 10, 500), ('A_built_km2', 100, 3000)]:
        s = d[col].astype(float)
        flag = 'OK'
        if s.min() < lo or s.max() > hi:
            flag = 'OUT-OF-RANGE'
        print(f"  {col:<14s} min={s.min():>10.1f}  median={s.median():>10.1f}  max={s.max():>10.1f}   "
              f"[期望 {lo}~{hi}] {flag}")


if __name__ == '__main__':
    main()
