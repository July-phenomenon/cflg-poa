"""
Real-City Validation with EXACT Two-Phase MILP (M2 fix)
Replace greedy heuristic with precise MILP solver.
"""
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')
import pulp
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_core_solver")))
from core_solver import (seed_for, run_nash, BUILD_COST_DICT,  # [T4 fix] 稳定种子 + 统一 Nash（与鲁棒性测试一致）, BUILD_COST_DICT
                         run_nash_locked, milp_phase1_coverage)  # [T1] 两阶段序贯 Nash（锁定覆盖）

import argparse


def main():
    _AP = argparse.ArgumentParser(description='21城精确MILP vs 去中心化Nash（参数来自 v3 栅格推导）')
    _AP.add_argument('--budget-per-station', type=float, default=12.0,
                     help='建设预算系数（万元/站）。12=方案X(120万/10候选位置)；30=选项1(120万/约4已建站)')
    _AP.add_argument('--output', type=str, default=None, help='输出 JSON 路径（默认覆盖标准结果文件）')
    _AP.add_argument('--exclude', type=str, default='',
                     help='逗号分隔的城市名，用于敏感性检验（例：Shenzhen）')
    _AP.add_argument('--sizing-rule', type=str, choices=['A', 'B', 'C'], default='A',
                     help='站点规模分配规则 A=成本最优(2/3大+1/3小) B=均衡发展(三档均衡) C=大站优先(上限50%)')
    _AP.add_argument('--nash-mode', type=str, choices=['classic', 'locked'], default='classic',
                     help='Nash 口径：classic=经典单阶段CFLG(可丢弃未覆盖小区，旧口径)；'
                          'locked=两阶段序贯(Phase1锁定覆盖集，Nash 不得丢弃，与定理1同设定)')
    ARGS = _AP.parse_args()
    BUDGET_PER_STATION = ARGS.budget_per_station        # 万元/站
    EXCLUDE = set([c.strip() for c in ARGS.exclude.split(',') if c.strip()])


    W1, W2, W3 = 0.2, 0.3, 0.5
    BASELINE_S3 = 0.60
    S1_T = [(300, 1.0), (500, 0.9), (650, 0.75), (1000, 0.6)]
    S2_T = [(0.6, 1.0), (0.75, 0.93), (0.85, 0.85), (0.95, 0.72), (1.0, 0.6)]
    CAPACITY = np.array([0, 1000., 2000., 3000.])  # BUG-03+04 Fix: Define CAPACITY array

    def s1_fn(d):
        for u, v in S1_T:
            if d <= u: return v
        return 0.0

    def s2_fn(u):
        for u2, v in S2_T:
            if u <= u2: return v
        return S2_T[-1][1]

    def load_city_profiles(json_path):
        """动态读取城市参数（禁止硬编码，保证可追溯、可复现）。

        来源链：Han et al. (2026) 可达性栅格
                -> 参数推导v3_栅格直接测量.py
                -> 实城实验参数v3.json
          n_communities = 建成区面积 / 7.07 km2（民政部小型设施 3km 服务半径对应空间单元）
          cluster_r     = 可达性峰值最近邻距离中位数 / 2   （定义 A，经实测验证）
          n_stations    = 可达性栅格局部峰值数
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            recs = json.load(f)
        profiles = []
        for r in recs:
            if r.get('cluster_r_m') is None or int(r.get('n_stations', 0)) < 1:
                print(f"  [跳过] {r.get('city')}: 参数不完整")
                continue
            profiles.append((r['city'], int(r['n_communities']),
                             float(r['cluster_r_m']), int(r['n_stations'])))
        return profiles


    PARAMS_V3 = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..',
        "data", "processed", "21city_parameters.json"))
    CITY_PROFILES_ALL = load_city_profiles(PARAMS_V3)


    # run_nash 现从 milp_common 导入（与鲁棒性测试共用同一实现，保证可复现一致）


    def allocate_stations(rule, n_sta, BUDGET_WAN, COST):
        """按规模分配规则计算 (n_large, n_medium, n_small)，保证总成本 <= BUDGET_WAN。

        成本 COST = {1:18, 2:32, 3:45}（赛题附件3 真值，万元）。
        规则 A（成本最优）：n_large=floor(2/3·n_sta), n_medium=0, n_small=剩余
            —— 单位成本容量最高，即已定案基准（36万/站 下 ≈0.19-0.22 gini）
        规则 B（均衡发展）：n_large=floor(n_sta/3)，剩余预算尽量配中站
            —— 三档并存，gini 最低（~0.10-0.15）
        规则 C（大站优先）：n_large=ceil(n_sta/2)，剩余预算尽量配小站
            —— 大/小双峰，gini 最高（~0.22-0.28）
        预算护栏：若超预算，优先 大→小（保中站），其次 中→小。
        """
        cL, cM, cS = int(COST[3]), int(COST[2]), int(COST[1])
        B = int(round(BUDGET_WAN))  # 预算转为整数（36·n_sta 恒为整数）
        if rule == 'A':
            n_large = int(max(1, n_sta * 2 / 3)); n_medium = 0; n_small = n_sta - n_large
        elif rule == 'B':
            n_large = max(0, int(n_sta // 3))
            rem = B - cL * n_large
            n_medium = int(min(rem // cM, n_sta - n_large)) if (n_sta - n_large) > 0 else 0
            n_small = n_sta - n_large - n_medium
        elif rule == 'C':
            n_large = min(n_sta, int((n_sta + 1) // 2))  # ceil(n_sta/2)
            rem = B - cL * n_large
            n_small = int(min(rem // cS, n_sta - n_large)) if (n_sta - n_large) > 0 else 0
            n_medium = n_sta - n_large - n_small
        else:
            raise ValueError(f"unknown sizing rule {rule}")
        n_large, n_medium, n_small = int(n_large), int(n_medium), int(n_small)
        cost = cL * n_large + cM * n_medium + cS * n_small
        while cost > BUDGET_WAN + 1e-6:
            if n_large > 0:
                n_large -= 1; n_small += 1
            elif n_medium > 0:
                n_medium -= 1; n_small += 1
            else:
                break
            cost = cL * n_large + cM * n_medium + cS * n_small
        return n_large, n_medium, n_small


    def run_milp_exact(n, pop, demand, S1m, sta_caps, return_assignment=False):
        """Exact two-phase MILP (M2 fix: replace greedy with exact solver).

        [V3] return_assignment=True 时额外返回 (best_w, best_S2)：最优分配矩阵与对应
        站点 S2 数组，用于通道分解（F2 = W1*A1 + W2*A2 + W3*S3*cov，S3=0.60 常量）。
        """
        n_sta = len(sta_caps)
        total_pop = pop.sum()

        # Phase 1: Maximize F1 (coverage)
        prob1 = pulp.LpProblem("Phase1", pulp.LpMaximize)
        w1 = {}
        for si in range(n_sta):
            for cj in range(n):
                if S1m[si, cj] > 0:
                    w1[(si, cj)] = pulp.LpVariable(f"w1_{si}_{cj}", cat='Binary')

        if not w1:
            return (0.0, 0.0, set(), None, None) if return_assignment else (0.0, 0.0, set())

        prob1 += pulp.lpSum([pop[cj] * w1[(si, cj)] for (si, cj) in w1])

        for cj in range(n):
            vfs = [w1[(si, cj)] for si in range(n_sta) if (si, cj) in w1]
            if vfs:
                prob1 += pulp.lpSum(vfs) <= 1

        for si in range(n_sta):
            vfs = [(cj, w1[(si, cj)]) for cj in range(n) if (si, cj) in w1]
            prob1 += pulp.lpSum([demand[cj] * var for cj, var in vfs]) <= sta_caps[si]

        prob1.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob1.status] != 'Optimal':
            return (0.0, 0.0, set(), None, None) if return_assignment else (0.0, 0.0, set())

        w1m = np.zeros((n_sta, n))
        for (si, cj), var in w1.items():
            w1m[si, cj] = round(pulp.value(var))

        cov1 = np.array([np.any(w1m[:, cj] > 0.5) for cj in range(n)])
        F1_max = pop[cov1].sum() / total_pop

        # Phase 2: Iterative MILP for F2, maintaining F1 >= F1_max
        S2_cur = np.ones(n_sta)
        best_F2 = -1
        best_w = None
        best_S2 = None  # [V3] 与 best_F2 对应的站点 S2 数组

        for it in range(10):
            prob2 = pulp.LpProblem(f"Phase2_{it}", pulp.LpMaximize)
            w2 = {}
            for si in range(n_sta):
                for cj in range(n):
                    if S1m[si, cj] > 0:
                        w2[(si, cj)] = pulp.LpVariable(f"w2_{si}_{cj}", cat='Binary')

            if not w2:
                break

            # Objective: max F2
            prob2 += pulp.lpSum([
                pop[cj] * (W1 * S1m[si, cj] + W2 * S2_cur[si] + W3 * BASELINE_S3) * w2[(si, cj)]
                for (si, cj) in w2
            ])

            # Constraints
            for cj in range(n):
                vfs = [w2[(si, cj)] for si in range(n_sta) if (si, cj) in w2]
                if vfs:
                    prob2 += pulp.lpSum(vfs) <= 1

            for si in range(n_sta):
                vfs = [(cj, w2[(si, cj)]) for cj in range(n) if (si, cj) in w2]
                prob2 += pulp.lpSum([demand[cj] * var for cj, var in vfs]) <= sta_caps[si]

            # F1 constraint: maintain coverage from Phase 1
            prob2 += pulp.lpSum([pop[cj] * w2[(si, cj)] for (si, cj) in w2]) >= F1_max * total_pop - 1

            prob2.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob2.status] != 'Optimal':
                break

            wm = np.zeros((n_sta, n))
            for (si, cj), var in w2.items():
                wm[si, cj] = round(pulp.value(var))

            loads = np.array([sum(demand[cj] * wm[si, cj] for cj in range(n)) for si in range(n_sta)])
            utils = loads / sta_caps
            S2_new = np.array([s2_fn(u) for u in utils])

            F2_actual = sum(
                pop[cj] * (W1 * S1m[si, cj] + W2 * S2_new[si] + W3 * BASELINE_S3) * wm[si, cj]
                for cj in range(n) for si in range(n_sta)
            ) / total_pop

            if F2_actual > best_F2:
                best_F2 = F2_actual
                best_w = wm.copy()
                best_S2 = S2_new.copy()  # [V3]

            if np.max(np.abs(S2_new - S2_cur)) < 0.01:
                break
            S2_cur = S2_new

        if best_w is None:
            return (0.0, 0.0, set(), None, None) if return_assignment else (0.0, 0.0, set())

        milp_cov = set()
        for cj in range(n):
            for si in range(n_sta):
                if best_w[si, cj] > 0.5:
                    milp_cov.add(cj)
                    break
        milp_F1 = pop[list(milp_cov)].sum() / total_pop

        if return_assignment:  # [V3]
            return best_F2, milp_F1, milp_cov, best_w, best_S2
        return best_F2, milp_F1, milp_cov


    def _channel_decomp(choices, s2_arr, pop, S1m):
        """[V3] 由分配计算通道分量：F2 = W1*A1 + W2*A2 + W3*S3*cov（S3=0.60 常量）。

        A1 = 人口加权指派质量（S1 均值）；A2 = 人口加权利用率满意度；cov = 覆盖率。
        F2_check 用于恒等式自检（应与求解器返回的 F2 一致到浮点残差）。
        """
        if not choices:
            return {'A1': 0.0, 'A2': 0.0, 'cov': 0.0, 'F2_check': 0.0}
        tp = pop.sum()
        cov = sum(pop[cj] for cj in choices) / tp
        A1 = sum(pop[cj] * S1m[si, cj] for cj, si in choices.items()) / tp
        A2 = sum(pop[cj] * s2_arr[si] for cj, si in choices.items()) / tp
        F2c = W1 * A1 + W2 * A2 + W3 * BASELINE_S3 * cov
        return {'A1': A1, 'A2': A2, 'cov': cov, 'F2_check': F2c}


    print("=" * 70)
    print("REAL-CITY VALIDATION WITH EXACT MILP (M2 fix)")
    print("Data source: Han et al. (2026), Scientific Data")
    print("=" * 70)

    all_results = {}
    all_gaps = []

    CITY_PROFILES = [p for p in CITY_PROFILES_ALL if p[0] not in EXCLUDE]
    print(f'[CFG] 预算系数 = {BUDGET_PER_STATION} 万元/站 | 城市数 = {len(CITY_PROFILES)} | 排除 = {sorted(EXCLUDE) or "无"}')
    for city_name, n, cluster_r, n_sta in CITY_PROFILES:
        rng = np.random.RandomState(seed_for(city_name))

        extent = cluster_r * 3.0
        comm_locs = rng.uniform(-extent/2, extent/2, size=(n, 2))
        fac_locs = rng.uniform(-cluster_r*0.8, cluster_r*0.8, size=(n_sta, 2))

        dist_matrix = np.zeros((n_sta, n))
        for i in range(n_sta):
            for j in range(n):
                dx = comm_locs[j, 0] - fac_locs[i, 0]
                dy = comm_locs[j, 1] - fac_locs[i, 1]
                dist_matrix[i, j] = np.sqrt(dx**2 + dy**2)

        # 连通度指标（与差距同源，保证一致性）：平均每个小区在 1000m 内可达站点数 / 总站点数
        within_1000 = np.sum(dist_matrix <= 1000.0, axis=0)
        connectivity = float(np.mean(within_1000) / n_sta)

        S1m = np.zeros((n_sta, n))
        for i in range(n_sta):
            for j in range(n):
                S1m[i, j] = s1_fn(dist_matrix[i, j])

        pop = rng.randint(400, 1000, size=n).astype(float)

        # ============================================================
        # 站点规模异质化（支持 --sizing-rule A/B/C，预算统一 BUDGET_PER_STATION 万/站）
        # ============================================================
        BUDGET_WAN = BUDGET_PER_STATION * n_sta   # 按站点数线性缩放（原 120 万为 10 小区街道实例常数）
        # BUILD_COST_DICT 现由 milp_common 从赛题附件3 读取（18/32/45），不再硬编码

        n_large, n_medium, n_small = allocate_stations(
            ARGS.sizing_rule, n_sta, BUDGET_WAN, BUILD_COST_DICT)
        total_cost = (BUILD_COST_DICT[3] * n_large + BUILD_COST_DICT[2] * n_medium
                      + BUILD_COST_DICT[1] * n_small)

        sizes = [3] * n_large + [2] * n_medium + [1] * n_small
        n_sta = len(sizes)  # 保持 n_sta 不变（异质分配后数量相同）
        sta_caps = np.array([CAPACITY[s] for s in sizes])

        print(f"  [站点规模 rule={ARGS.sizing_rule}] n_sta={n_sta}, L/M/S={n_large}/{n_medium}/{n_small}, 总成本={total_cost}万, 总容量={sta_caps.sum()}")

        # BUG-03: Scale demand to ~115% utilization
        target_total_demand = sta_caps.sum() * 1.15
        raw_demand = rng.uniform(0.7, 1.3, size=n)
        demand = np.round(raw_demand / raw_demand.sum() * target_total_demand).astype(np.int64)

        # Run Nash（口径由 --nash-mode 决定）
        n_rep = None
        nash_ch = None
        if ARGS.nash_mode == 'locked':
            # 两阶段序贯：先解 Phase1 锁定覆盖集，再在其上跑去中心化博弈
            _F1_phase1, lock_cov = milp_phase1_coverage(n, pop, demand, S1m, sta_caps)
            nash_F2, nash_F1, nash_cov, n_rep, nash_choices, nash_S2_arr = run_nash_locked(
                n, pop, demand, S1m, sta_caps, lock_cov, return_details=True)  # [V3]
            nash_ch = _channel_decomp(nash_choices, nash_S2_arr, pop, S1m)
        else:
            nash_F2, nash_F1, nash_cov = run_nash(n, pop, demand, S1m, sta_caps)

        # Run EXACT MILP (M2 fix)；[V3] return_assignment → 通道分解
        milp_F2, milp_F1, milp_cov, milp_w, milp_S2_arr = run_milp_exact(
            n, pop, demand, S1m, sta_caps, return_assignment=True)
        milp_choices = {cj: si for si in range(len(sta_caps))
                        for cj in range(n) if milp_w is not None and milp_w[si, cj] > 0.5}
        milp_ch = _channel_decomp(milp_choices, milp_S2_arr, pop, S1m)

        # [V3] 通道分解字段（classic 口径下 nash_ch=None → 记 None）
        if nash_ch is not None:
            dA1 = milp_ch['A1'] - nash_ch['A1']
            dA2 = milp_ch['A2'] - nash_ch['A2']
            dcov = milp_ch['cov'] - nash_ch['cov']
            ch_s1, ch_s2, ch_cov = W1 * dA1, W2 * dA2, W3 * BASELINE_S3 * dcov
            dF2 = milp_F2 - nash_F2
            ch_fields = {
                'nash_A1': round(nash_ch['A1'], 6), 'milp_A1': round(milp_ch['A1'], 6),
                'nash_S2_weighted': round(nash_ch['A2'], 6), 'milp_S2_weighted': round(milp_ch['A2'], 6),
                'delta_S2': round(dA2, 6),
                'nash_cov': round(nash_ch['cov'], 6), 'milp_cov': round(milp_ch['cov'], 6),
                'delta_cov': round(dcov, 6),
                'ch_S1': round(ch_s1, 6), 'ch_S2': round(ch_s2, 6), 'ch_cov': round(ch_cov, 6),
                'identity_resid': round(abs(dF2 - (ch_s1 + ch_s2 + ch_cov)), 9),
            }
        else:
            ch_fields = {'nash_A1': None, 'milp_A1': None, 'nash_S2_weighted': None,
                         'milp_S2_weighted': None, 'delta_S2': None, 'nash_cov': None,
                         'milp_cov': None, 'delta_cov': None, 'ch_S1': None,
                         'ch_S2': None, 'ch_cov': None, 'identity_resid': None}

        gap = nash_F2 - milp_F2
        all_gaps.append(gap)
        all_results[city_name] = {
            'n': n, 'cluster_r': round(cluster_r, 1), 'n_stations': n_sta,
            'budget_wan': round(float(total_cost), 1),
            'n_large': int(n_large), 'n_medium': int(n_medium), 'n_small': int(n_small),
            'large_ratio': round(n_large / n_sta, 4) if n_sta > 0 else None,
            'small_ratio': round(n_small / n_sta, 4) if n_sta > 0 else None,
            'connectivity': round(connectivity, 4),
            'nash_F1': round(nash_F1, 4), 'nash_F2': round(nash_F2, 4),
            'milp_F1': round(milp_F1, 4), 'milp_F2': round(milp_F2, 4),
            'gap': round(gap, 4),
            'gap_pct': round(gap/nash_F2*100, 1) if nash_F2 > 0 else 0,
            'PoA': round(milp_F2/nash_F2, 4) if nash_F2 > 0 else None,
            'Loss_pct': round((1 - nash_F2/milp_F2)*100, 1) if milp_F2 > 0 else None,
            'method': 'exact_MILP',  # Mark as exact MILP
            'nash_mode': ARGS.nash_mode,          # [T1] 记录口径，避免与旧单阶段结果混淆
            'n_repaired': n_rep,                  # [T1] locked 模式下的兜底修复次数（0=自然可行）
            **ch_fields,                          # [V3] 通道分解字段
        }
        print(f"{city_name}: Nash F2={nash_F2:.4f}, MILP F2={milp_F2:.4f}, Gap={gap:+.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    positive = sum(1 for g in all_gaps if g > 0.001)
    negative = sum(1 for g in all_gaps if g < -0.001)
    near_zero = sum(1 for g in all_gaps if abs(g) <= 0.001)
    print(f"Nash > MILP:  {positive}/{len(all_gaps)} ({positive/len(all_gaps)*100:.0f}%)")
    print(f"MILP > Nash:  {negative}/{len(all_gaps)} ({negative/len(all_gaps)*100:.0f}%)")
    print(f"Near-zero:    {near_zero}/{len(all_gaps)} ({near_zero/len(all_gaps)*100:.0f}%)")
    print(f"Mean gap:     {np.mean(all_gaps):+.4f}")
    print(f"Median gap:   {np.median(all_gaps):+.4f}")

    poas = [r['PoA'] for r in all_results.values() if r.get('PoA')]
    if poas:
        print(f"Mean PoA:     {np.mean(poas):.4f}")
        print(f"Max PoA:      {np.max(poas):.4f}")
        print(f"Mean Loss:    {(1 - 1/np.mean(poas))*100:.2f}%")

    # Save results
    out_path = ARGS.output or os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': 'Han et al. (2026), Scientific Data',
            'experiment': '21城市精确MILP验证 (M2 fix)',
            'method': 'exact_two_phase_MILP',
            'nash_mode': ARGS.nash_mode,
            'nash_mode_note': ('locked=两阶段序贯(Phase1锁定覆盖,Nash不得丢弃) [T1 新增, 与定理1同设定]'
                               if ARGS.nash_mode == 'locked' else
                               'classic=经典单阶段CFLG(Nash可丢弃未覆盖小区) [旧口径, 仅存档对比]'),
            'params_source': 'Han et al. (2026) 栅格 -> 参数推导v3 -> 实城实验参数v3.json',
            'budget_per_station_wan': BUDGET_PER_STATION,
            'excluded_cities': sorted(EXCLUDE),
            'n_cities': len(all_results),
            'summary': {
                'nash_wins': positive,
                'milp_wins': negative,
                'near_zero': near_zero,
                'mean_gap': round(float(np.mean(all_gaps)), 4),
            },
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
