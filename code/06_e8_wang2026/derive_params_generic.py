#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用参数提取器（E8 用）—— 与 derive_21city_params.py 完全同一套三步法
=====================================================================
目的：把「Han 100m 逐城栅格」上的三步法，原样应用到「Wang2026 全国 1km 栅格」，
      以回答"换 2025 数据后参数/结论是否稳健"。

与 derive_21city_params.py 的差异（仅输入侧，算法不变）：
  1. --raster   单张栅格（Han 是逐城已裁；Wang 是全国大图，需配 --polygon 裁到城市）
  2. --polygon  可选 GeoJSON 多边形（WGS84/EPSG:4326），按 bbox+within 掩膜
  3. --resample-deg  可选：把栅格重采样到指定像元度数（如 0.0008333≈3弧秒）
                  ——用于把 1km 数据上采样到与 Han 同像元，保证 σ/邻域在"物理尺度"一致
  4. --high-pct 仅用于审计 A_high，不影响 n_stations/cluster_r/n_communities

自检基准（Han Beijing）：
  n_stations=76, n_communities=345, cluster_r≈3686.19, A_built=2437.68, n_px_valid=369648

用法：
  python derive_params_generic.py --raster <tif> --high-pct 3.09 --out <prefix>
  python derive_params_generic.py --raster <tif> --polygon <geojson> --resample-deg 0.0008333333 \
                                  --high-pct 0 --out <prefix> --json-list out.json
"""
import argparse, json, os, sys
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

M_PER_DEG_LAT = 111320.0
SIGMA_KM = 5.0 / 3.0                     # 与 derive_21city_params.py 一致
COMMUNITY_UNIT_KM2 = np.pi * 1.5 ** 2    # 7.0686 km²
NMS_FACTOR = 2.0


def lat_of_row(transform, nrows):
    rows = np.arange(nrows, dtype=np.float64)
    return transform.f + (rows + 0.5) * transform.e


def load_raster(path, polygon=None, resample_deg=None):
    """读栅格 → (data, valid_mask, transform)。可选按多边形裁剪、可选重采样。"""
    with rasterio.open(path) as src:
        if polygon is not None:
            shapes = [polygon]
            out_img, out_transform = rio_mask(src, shapes, crop=True, filled=False, all_touched=False)
            data = out_img[0].filled(np.nan)
            # filled mask
            valid = ~out_img[0].mask if np.ma.is_masked(out_img[0]) else np.isfinite(data)
            transform = out_transform
            nodata = src.nodata
        else:
            data = src.read(1).astype(np.float64)
            m = src.read_masks(1) > 0
            valid = m & np.isfinite(data)
            transform = src.transform
            nodata = src.nodata
        # 统一：-inf / nodata / 非有限 → 无效
        valid = valid & np.isfinite(data)
        if nodata is not None and np.isfinite(nodata):
            valid &= (data != nodata)

        # Wang2026 的 NoData 常为 -9999（Readme 明示）
        valid &= (data != -9999)

    # 可选重采样到目标像元度数（保持物理平滑尺度一致）
    if resample_deg is not None and abs(abs(transform.a) - resample_deg) > 1e-9:
        src_res = abs(transform.a)
        scale = resample_deg / src_res          # >1 表示放大到更细像元
        new_w = int(round(data.shape[1] * scale))
        new_h = int(round(data.shape[0] * scale))
        new_transform = rasterio.Affine(resample_deg, 0, transform.c,
                                        0, -resample_deg, transform.f)
        new_data = np.full((new_h, new_w), np.nan, dtype=np.float64)
        # 用 0 填无效？不——重采样时把无效当 nan 传播：先转成 float 用 bilinear
        src_arr = np.where(valid, data, np.nan)
        reproject(
            source=src_arr, destination=new_data,
            src_transform=transform, src_crs="EPSG:4326",
            dst_transform=new_transform, dst_crs="EPSG:4326",
            resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan,
        )
        data = new_data
        valid = np.isfinite(data)
        transform = new_transform

    return data, valid, transform


def derive(data, valid, transform, high_pct=0.0, city="city"):
    if valid.sum() == 0:
        raise RuntimeError(f"{city}: 无有效像元")
    ys, xs = np.where(valid)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    data_c = data[y0:y1, x0:x1]
    valid_c = valid[y0:y1, x0:x1]

    res_x, res_y = abs(transform.a), abs(transform.e)
    lats = lat_of_row(transform, data.shape[0])[y0:y1]
    m_per_px_y = res_y * M_PER_DEG_LAT
    m_per_px_x = res_x * M_PER_DEG_LAT * np.cos(np.deg2rad(lats))
    px_area_m2 = (m_per_px_x[:, None] * np.ones((1, data_c.shape[1]))) * m_per_px_y

    A_built_km2 = float(px_area_m2[valid_c].sum()) / 1e6
    vals = data_c[valid_c]
    n_px_valid = int(valid_c.sum())
    n_px_zero = int((vals == 0).sum())

    if high_pct and high_pct > 0:
        tau_high = float(np.percentile(vals, 100.0 - high_pct))
        high_mask_c = valid_c & (data_c > tau_high)
        A_high_km2 = float(px_area_m2[high_mask_c].sum()) / 1e6
    else:
        tau_high, A_high_km2 = None, None

    sigma_px = SIGMA_KM * 1000.0 / m_per_px_y
    filled = np.where(valid_c, data_c, 0.0).astype(np.float32)
    smoothed = gaussian_filter(filled, sigma=sigma_px, mode='constant', cval=0.0)

    neighborhood = max(3, int(round(NMS_FACTOR * sigma_px)))
    local_max = maximum_filter(smoothed, size=neighborhood, mode='constant', cval=-np.inf)
    cand = (smoothed >= local_max) & valid_c & (smoothed > 0)

    py, px = np.where(cand)
    if len(py) > 0:
        svals = smoothed[py, px]
        order = np.argsort(-svals)
        min_sep = float(neighborhood)
        accepted, keep = [], []
        for idx in order:
            p = (int(py[idx]), int(px[idx]))
            if accepted:
                d, _ = cKDTree(np.array(accepted)).query(np.array([p]), k=1)
                if float(d[0]) < min_sep:
                    continue
            accepted.append(p); keep.append(idx)
        keep = np.array(keep, dtype=int)
        py, px = py[keep], px[keep]
    n_peaks = int(len(py))

    if n_peaks > 0:
        peak_y_m = (py + y0 + 0.5) * m_per_px_y
        peak_x_m = (px + x0 + 0.5) * m_per_px_x[py]
        coords = np.column_stack([peak_x_m, peak_y_m])
    else:
        coords = np.zeros((0, 2))

    if n_peaks >= 2:
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        nn_median_m = float(np.median(dists[:, 1]))
        cluster_r_m = float(nn_median_m / 2.0)
    else:
        nn_median_m = cluster_r_m = None

    n_communities = max(1, int(round(A_built_km2 / COMMUNITY_UNIT_KM2)))

    return {
        'city': city,
        'n_communities': n_communities,
        'cluster_r_m': cluster_r_m,
        'n_stations': n_peaks,
        'A_built_km2': round(A_built_km2, 2),
        'A_high_km2': round(A_high_km2, 2) if A_high_km2 is not None else None,
        'high_pct': high_pct,
        'n_px_valid': n_px_valid,
        'n_px_zero': n_px_zero,
        'share_zero': round(n_px_zero / n_px_valid, 4) if n_px_valid else None,
        'sigma_px': round(float(sigma_px), 3),
        'nms_neighborhood_px': neighborhood,
        'px_size_m': round(float(m_per_px_y), 2),
        'nn_median_m': round(nn_median_m, 1) if nn_median_m is not None else None,
        'tau_high': round(tau_high, 6) if tau_high is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raster', required=True)
    ap.add_argument('--polygon', default=None, help='GeoJSON 文件（EPSG:4326）')
    ap.add_argument('--resample-deg', type=float, default=None)
    ap.add_argument('--high-pct', type=float, default=0.0)
    ap.add_argument('--city', default=None)
    ap.add_argument('--out', default=None, help='输出 JSON 前缀（不含扩展）')
    ARGS = ap.parse_args()

    poly = None
    if ARGS.polygon:
        gj = json.load(open(ARGS.polygon, encoding='utf-8'))
        feats = gj['features'] if gj.get('type') == 'FeatureCollection' else [gj]
        # 合并为单个 geometry（MultiPolygon）
        geoms = [f['geometry'] for f in feats]
        poly = {"type": "MultiPolygon", "coordinates": []}
        for g in geoms:
            if g['type'] == 'Polygon':
                poly["coordinates"].append(g['coordinates'])
            elif g['type'] == 'MultiPolygon':
                poly["coordinates"].extend(g['coordinates'])

    city = ARGS.city or os.path.splitext(os.path.basename(ARGS.raster))[0]
    data, valid, transform = load_raster(ARGS.raster, poly, ARGS.resample_deg)
    rec = derive(data, valid, transform, high_pct=ARGS.high_pct, city=city)
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    if ARGS.out:
        with open(ARGS.out + '.json', 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print('[saved]', ARGS.out + '.json')


if __name__ == '__main__':
    main()
