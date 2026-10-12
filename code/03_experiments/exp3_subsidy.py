# -*- coding: utf-8 -*-
"""
T4: 21 城 V2 补贴前后 PoA 对比实验
===================================
【重要说明 / 与执行文档的偏差】
执行文档 T4 假设可“复用 补贴机制设计实验.py”并传 --enable-subsidy/--output，
但该脚本实为 10 小区（A-J）代表配置的机制设计实验，无 CLI 参数、不跑 21 城、
也不输出 PoA。因此本脚本按 T4 的**真实研究意图**重新实现：
  在 21 城精确 MILP 框架内，对去中心化 Nash 施加 V2 补贴（按站 S3 boost），
  以 PoA = MILP_F2 / Nash_F2 衡量补贴能否降低无效率（逼近 1）。

V2 机制（与 补贴机制设计实验.py::compute_subsidy_s3_v2 一致）：
  1) 先算基线 Nash，得到每站服务的小区集合；
  2) 每站 service_gap = max(0, 潜在需求 - 实际服务需求)，潜在需求=该站 S1>0 范围内
     所有小区需求之和；
  3) S3 boost = min(0.25, 2 * S3_BOOST_FACTOR * gap_ratio)，gap_ratio=gap/容量；
  4) 用 boosted S3 重算 Nash。

口径：成本 18/32/45（milp_common）、预算 36 万/站、floor(2/3) 大站、seed_for 稳定几何。
"""
import numpy as np
import json
import os
import warnings
import argparse
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_core_solver")))
from core_solver import (seed_for, run_nash, run_milp_exact, run_milp_exact_with_s3,
                         run_nash_with_s3,
                         BUILD_COST_DICT, BASELINE_S3, s1_fn, s2_fn,
                         W1, W2, W3, S1_T, S2_T)

_AP = argparse.ArgumentParser(description='21城 V2 补贴前后 PoA 对比 (T4)')
_AP.add_argument('--budget-per-station', type=float, default=36.0,
                 help='建设预算系数（万元/站），默认 36 与已定案基准一致')
_AP.add_argument('--boost-factor', type=float, default=0.08,
                 help='V2 S3 boost 系数（赛题机制设计原值 0.08）')
_AP.add_argument('--output', type=str,
                 default=os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data"),
                 help='输出 JSON 路径')
ARGS = _AP.parse_args()
BUDGET_PER_STATION = ARGS.budget_per_station
S3_BOOST_FACTOR = ARGS.boost_factor
CAPACITY = np.array([0, 1000., 2000., 3000.])

PARAMS_V3 = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    "data", "processed", "21city_parameters.json"))

def load_city_profiles(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        recs = json.load(f)
    profiles = []
    for r in recs:
        if r.get('cluster_r_m') is None or int(r.get('n_stations', 0)) < 1:
            continue
        profiles.append((r['city'], int(r['n_communities']),
                         float(r['cluster_r_m']), int(r['n_stations'])))
    return profiles

CITY_PROFILES = load_city_profiles(PARAMS_V3)


def capacity_gini(nl, nm, ns):
    caps = np.sort(np.array([3000] * nl + [2000] * nm + [1000] * ns, dtype=float))
    n = len(caps)
    if n == 0:
        return 0.0
    cum = caps.sum()
    return (2 * np.sum((np.arange(1, n + 1)) * caps) / (n * cum) - (n + 1) / n)


print("=" * 70)
print("T4: 21-CITY V2 SUBSIDY BEFORE/AFTER PoA COMPARISON")
print(f"budget/station={BUDGET_PER_STATION}  boost_factor={S3_BOOST_FACTOR}  cities={len(CITY_PROFILES)}")
print("=" * 70)

all_results = {}
base_poas, sub_poas = [], []
nash_base_f2, nash_sub_f2, milp_f2_all = [], [], []
milp_wins_base = milp_wins_sub = 0
nash_wins_base = nash_wins_sub = 0

for city_name, n, cluster_r, n_sta in CITY_PROFILES:
    rng = np.random.RandomState(seed_for(city_name))
    extent = cluster_r * 3.0
    comm_locs = rng.uniform(-extent / 2, extent / 2, size=(n, 2))
    fac_locs = rng.uniform(-cluster_r * 0.8, cluster_r * 0.8, size=(n_sta, 2))

    dist_matrix = np.zeros((n_sta, n))
    for i in range(n_sta):
        for j in range(n):
            dx = comm_locs[j, 0] - fac_locs[i, 0]
            dy = comm_locs[j, 1] - fac_locs[i, 1]
            dist_matrix[i, j] = np.sqrt(dx ** 2 + dy ** 2)

    within_1000 = np.sum(dist_matrix <= 1000.0, axis=0)
    connectivity = float(np.mean(within_1000) / n_sta)

    S1m = np.zeros((n_sta, n))
    for i in range(n_sta):
        for j in range(n):
            S1m[i, j] = s1_fn(dist_matrix[i, j])

    pop = rng.randint(400, 1000, size=n).astype(float)

    # 基准规模分配（规则 A：floor(2/3) 大站，与已定案一致）
    BUDGET_WAN = BUDGET_PER_STATION * n_sta
    n_large = max(1, int(n_sta * 2 / 3))
    n_small = n_sta - n_large
    total_cost = n_large * BUILD_COST_DICT[3] + n_small * BUILD_COST_DICT[1]
    while total_cost > BUDGET_WAN and n_large > 1:
        n_large -= 1
        n_small += 1
        total_cost = n_large * BUILD_COST_DICT[3] + n_small * BUILD_COST_DICT[1]
    sizes = [3] * n_large + [1] * n_small
    n_sta_real = len(sizes)
    sta_caps = np.array([CAPACITY[s] for s in sizes])

    target_total_demand = sta_caps.sum() * 1.15
    raw_demand = rng.uniform(0.7, 1.3, size=n)
    demand = np.round(raw_demand / raw_demand.sum() * target_total_demand).astype(np.int64)

    # 系统最优（集中式 MILP，不补贴——规划者解）
    milp_F2, milp_F1, _ = run_milp_exact(n, pop, demand, S1m, sta_caps)

    # 基线 Nash（S3=0.60）
    base_F2, base_F1, base_cov, base_choices = run_nash_with_s3(
        n, pop, demand, S1m, sta_caps, return_choices=True)

    # V2 补贴：每站 S3 boost
    s3_boost = np.zeros(n_sta_real)
    for si in range(n_sta_real):
        served = [cj for cj, a in base_choices.items() if a == si]
        potential = [cj for cj in range(n) if S1m[si, cj] > 0]
        potential_demand = sum(demand[cj] for cj in potential)
        actual_demand = sum(demand[cj] for cj in served)
        gap = max(0.0, potential_demand - actual_demand)
        gap_ratio = gap / sta_caps[si] if sta_caps[si] > 0 else 0.0
        s3_boost[si] = min(0.25, S3_BOOST_FACTOR * 2.0 * gap_ratio)
    s3_array = BASELINE_S3 + s3_boost

    # 补贴后 Nash
    sub_F2, sub_F1, sub_cov, _ = run_nash_with_s3(
        n, pop, demand, S1m, sta_caps, s3_array=s3_array, return_choices=True)

    # 口径一致：补贴机制下的集中式最优（同一 boosted S3），用于公平比较
    milp_F2_boosted, _, _ = run_milp_exact_with_s3(
        n, pop, demand, S1m, sta_caps, s3_array=s3_array)

    poa_base = milp_F2 / base_F2 if base_F2 > 1e-9 else None
    # 方法1：未补贴 MILP 作分母（混合机制，可能出现 <1）
    poa_sub_mixed = milp_F2 / sub_F2 if sub_F2 > 1e-9 else None
    # 方法2（口径一致）：补贴后 MILP 作分母
    poa_sub_consistent = milp_F2_boosted / sub_F2 if (sub_F2 > 1e-9 and milp_F2_boosted > 0) else None

    if poa_base is not None:
        base_poas.append(poa_base)
        if milp_F2 - base_F2 > 0.001:
            milp_wins_base += 1
        elif milp_F2 - base_F2 < -0.001:
            nash_wins_base += 1
    if poa_sub_consistent is not None:
        sub_poas.append(poa_sub_consistent)
        if milp_F2_boosted - sub_F2 > 0.001:
            milp_wins_sub += 1
        elif milp_F2_boosted - sub_F2 < -0.001:
            nash_wins_sub += 1

    nash_base_f2.append(base_F2)
    nash_sub_f2.append(sub_F2)
    milp_f2_all.append(milp_F2)

    all_results[city_name] = {
        'n': n, 'cluster_r': round(cluster_r, 1), 'n_stations': n_sta_real,
        'connectivity': round(connectivity, 4),
        'n_large': int(n_large), 'n_medium': 0, 'n_small': int(n_small),
        'large_ratio': round(n_large / n_sta_real, 4),
        'capacity_gini': round(capacity_gini(n_large, 0, n_small), 4),
        's3_boost_mean': round(float(np.mean(s3_boost)), 4),
        's3_boost_max': round(float(np.max(s3_boost)), 4),
        'milp_F2': round(milp_F2, 4),
        'milp_F2_boosted': round(milp_F2_boosted, 4),
        'nash_F2_base': round(base_F2, 4),
        'nash_F2_sub': round(sub_F2, 4),
        'nash_F2_improve_pct': round((sub_F2 - base_F2) / base_F2 * 100, 2) if base_F2 > 0 else 0,
        'PoA_base': round(poa_base, 4) if poa_base else None,
        'PoA_sub_mixed': round(poa_sub_mixed, 4) if poa_sub_mixed else None,
        'PoA_sub_consistent': round(poa_sub_consistent, 4) if poa_sub_consistent else None,
        'PoA_reduction_consistent': round(poa_base - poa_sub_consistent, 4)
            if (poa_base and poa_sub_consistent) else None,
    }
    print(f"{city_name}: Nash_base={base_F2:.4f} Nash_sub={sub_F2:.4f} "
          f"MILP={milp_F2:.4f} MILP_boosted={milp_F2_boosted:.4f} | "
          f"PoA_base={poa_base:.4f} PoA_sub_consistent={poa_sub_consistent:.4f} "
          f"(boost_mean={np.mean(s3_boost):.3f})")

print("\n" + "=" * 70)
print("SUMMARY (T4)")
print("=" * 70)
import statistics as st
mean_poa_base = st.mean(base_poas)
mean_poa_sub = st.mean(sub_poas)  # 口径一致：补贴后 MILP / 补贴后 Nash
mixed_vals = [v['PoA_sub_mixed'] for v in all_results.values() if v.get('PoA_sub_mixed')]
mean_poa_sub_mixed = st.mean(mixed_vals) if mixed_vals else None
mean_loss_base = (1 - 1 / mean_poa_base) * 100
mean_loss_sub = (1 - 1 / mean_poa_sub) * 100
mean_loss_sub_mixed = (1 - 1 / mean_poa_sub_mixed) * 100 if mean_poa_sub_mixed else None
mean_nash_imp = st.mean([(s - b) / b * 100 for s, b in zip(nash_sub_f2, nash_base_f2) if b > 0])
print(f"Mean PoA (no subsidy):          {mean_poa_base:.4f}  (Loss {mean_loss_base:.2f}%)")
print(f"Mean PoA (V2 subsidy, consistent): {mean_poa_sub:.4f}  (Loss {mean_loss_sub:.2f}%)")
print(f"Mean PoA (V2 subsidy, mixed denom): {mean_poa_sub_mixed:.4f}  (Loss {mean_loss_sub_mixed:.2f}%)  [透明参考]")
print(f"Mean PoA reduction (consistent): {mean_poa_base - mean_poa_sub:.4f}")
print(f"Mean Nash F2 improvement:        {mean_nash_imp:+.2f}%")
print(f"MILP wins  no-subsidy / subsidy: {milp_wins_base}/21  /  {milp_wins_sub}/21")
print(f"Nash wins  no-subsidy / subsidy: {nash_wins_base}/21  /  {nash_wins_sub}/21")

out_path = ARGS.output
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        'experiment': 'T4 21城 V2 补贴前后 PoA 对比',
        'note': '原 补贴机制设计实验.py 为10小区实验，本脚本在21城框架重做；V2 机制与之一致',
        'budget_per_station_wan': BUDGET_PER_STATION,
        's3_boost_factor': S3_BOOST_FACTOR,
        'cost_wan': {str(k): v for k, v in BUILD_COST_DICT.items()},
        'summary': {
            'mean_PoA_base': round(mean_poa_base, 4),
            'mean_PoA_sub_consistent': round(mean_poa_sub, 4),
            'mean_PoA_sub_mixed': round(mean_poa_sub_mixed, 4) if mean_poa_sub_mixed else None,
            'mean_Loss_base_pct': round(mean_loss_base, 2),
            'mean_Loss_sub_consistent_pct': round(mean_loss_sub, 2),
            'mean_Loss_sub_mixed_pct': round(mean_loss_sub_mixed, 2) if mean_loss_sub_mixed else None,
            'mean_PoA_reduction_consistent': round(mean_poa_base - mean_poa_sub, 4),
            'mean_nash_F2_improve_pct': round(mean_nash_imp, 2),
            'milp_wins_base': milp_wins_base,
            'milp_wins_sub': milp_wins_sub,
        },
        'results': all_results,
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")

# CSV
csv_path = os.path.join(os.path.dirname(out_path), '补贴前后PoA对比.csv')
import csv


def main():
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as cf:
        w = csv.writer(cf)
        w.writerow(['city', 'n_stations', 'connectivity', 'large_ratio', 'capacity_gini',
                    'milp_F2', 'milp_F2_boosted', 'nash_F2_base', 'nash_F2_sub', 'nash_F2_improve_pct',
                    'PoA_base', 'PoA_sub_mixed', 'PoA_sub_consistent', 'PoA_reduction_consistent', 's3_boost_mean'])
        for c, v in all_results.items():
            w.writerow([c, v['n_stations'], v['connectivity'], v['large_ratio'], v['capacity_gini'],
                        v['milp_F2'], v['milp_F2_boosted'], v['nash_F2_base'], v['nash_F2_sub'], v['nash_F2_improve_pct'],
                        v['PoA_base'], v['PoA_sub_mixed'], v['PoA_sub_consistent'], v['PoA_reduction_consistent'], v['s3_boost_mean']])
    print(f"Saved CSV to {csv_path}")


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
