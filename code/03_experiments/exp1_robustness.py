"""
Module 2: Real-City Robustness Test
Test Nash vs MILP across 21 Chinese cities with R_cluster factors 0.8, 1.0, 1.2
"""
import numpy as np
import json
import os
import warnings
import multiprocessing as mp
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_core_solver")))
from core_solver import run_milp_exact, seed_for, run_nash, BUILD_COST_DICT

W1, W2, W3 = 0.2, 0.3, 0.5
BASELINE_S3 = 0.60

# —— 可解性保障 ——
# pulp 3.x 的 PULP_CBC_CMD(timeLimit=...) 在本环境不封顶（实测重庆 0.8 跑 >320s 仍不终止，
# 且 run_milp_exact 在 Phase1 非 Optimal 时直接 return 0.0），故改用真·墙钟守卫：
# 单城 MILP 超过 MILP_WALL_SEC 秒则由 multiprocessing 终止并标记 timeout，不拖累整轮。
# 不子采样：保留全规模实例（重庆 factor=1.0 全规模 890×372 仅 34s 精确，可保基准）。
# 仅 factor=0.8 的重庆因几何覆盖稀疏、CBC 分支爆炸而超墙钟，被标记为 timeout 并排除出均值。
MAX_N = 10**9       # 子采样阈值（设为极大值=关闭子采样，保留全规模）
MAX_STA = 10**9
MILP_WALL_SEC = 150.0  # 单城 MILP 墙钟预算（秒），超时终止并标记 timeout

def _milp_worker(q, n, pop, demand, S1m, sta_caps):
    try:
        f2, f1, cov, opt = run_milp_exact(
            int(n), np.array(pop, dtype=float), np.array(demand, dtype=np.int64),
            np.array(S1m, dtype=float), np.array(sta_caps, dtype=float),
            time_limit=None, return_status=True)
        q.put((f2, f1, opt, None))
    except Exception as e:
        q.put((None, None, False, str(e)[:200]))

def solve_milp_guarded(n, pop, demand, S1m, sta_caps):
    """带墙钟守卫的精确 MILP；超时返回 (None,None,False,'timeout')。

    关键：pulp 的 CBC 子进程在 worker 被 terminate 后不会自动退出，
    必须用 taskkill /T 杀掉整棵进程树（worker + CBC），否则 p.join() 会永久挂起。
    """
    import subprocess as _sp
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_milp_worker, args=(
        q, int(n), pop.tolist(), demand.tolist(), S1m.tolist(), sta_caps.tolist()))
    p.start()
    p.join(MILP_WALL_SEC)
    if p.is_alive():
        try:
            _sp.run(['taskkill', '/F', '/T', '/PID', str(p.pid)],
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=15)
        except Exception:
            p.terminate()
        p.join(timeout=10)
        return None, None, False, 'timeout'
    if not q.empty():
        return q.get()
    return None, None, False, 'no_result'
S1_T = [(300, 1.0), (500, 0.9), (650, 0.75), (1000, 0.6)]
S2_T = [(0.6, 1.0), (0.75, 0.93), (0.85, 0.85), (0.95, 0.72), (1.0, 0.6)]
CAPACITY = np.array([0, 1000., 2000., 3000.])  # BUG-03+04 Fix

def s1_fn(d):
    for u, v in S1_T:
        if d <= u: return v
    return 0.0

def s2_fn(u):
    for u2, v in S2_T:
        if u <= u2: return v
    return S2_T[-1][1]

# ---------------------------------------------------------------------------
# 城市参数：统一从 实城实验参数v3.json 读取（由 Han et al. (2026) 栅格直接测量推导）
# 与 21城市精确MILP验证.py 完全一致，不再硬编码。
# ---------------------------------------------------------------------------
_PARAMS_PATH = os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data")
with open(_PARAMS_PATH, 'r', encoding='utf-8') as _f:
    CITY_PROFILES = [(p['city'], p['n_communities'], p['cluster_r_m'], p['n_stations'])
                     for p in json.load(_f)]

# 每站建设预算（万元/站）——与 21城市精确MILP验证.py 统一
# 依据：(2/3)·45 + (1/3)·18 = 36，即 2/3 大 + 1/3 小 结构下的每站平均成本
BUDGET_PER_STATION = 36.0

if __name__ == '__main__':
    print("=" * 70)
    print("REAL-CITY ROBUSTNESS TEST: R_cluster factor robustness")
    print("Data source: Han et al. (2026), Scientific Data")
    print("=" * 70)

    out_path = os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data")

    # —— 断点续跑：若已存在最终版结果，加载并跳过已完成 (factor, city) ——
    all_results = {}
    all_gaps_by_factor = {0.8: [], 1.0: [], 1.2: []}
    import os as _os
    if _os.path.exists(out_path):
        try:
            _prev = json.load(open(out_path, encoding='utf-8'))
            _prev_res = _prev.get('results', {})
            for _f, _fr in _prev_res.items():
                _ff = float(_f)
                all_results[_ff] = _fr
                all_gaps_by_factor[_ff] = [v['gap'] for v in _fr.values()]
            print(f"[resume] 已加载历史结果，将跳过 {sum(len(v) for v in _prev_res.values())} 个已完成实例")
        except Exception as _e:
            print(f"[resume] 加载历史失败，重新开始：{_e}")

    def _save():
        """每城完成后即时落盘，防止中断丢进度。"""
        json.dump({
            'source': 'Han et al. (2026), Scientific Data（参数由栅格直接测量推导：实城实验参数v3.json）',
            'experiment': '真实城市R_cluster参数鲁棒性测试',
            'build_cost_wan': {str(k): v for k, v in BUILD_COST_DICT.items()},
            'budget_per_station_wan': BUDGET_PER_STATION,
            'factors_tested': [0.8, 1.0, 1.2],
            'n_cities': len(CITY_PROFILES),
            'resume_in_progress': True,
            'gaps_by_factor': {str(k): [round(g, 4) for g in v] for k, v in all_gaps_by_factor.items()},
            'summary_by_factor': {
                str(factor): {
                    'nash_wins': sum(1 for g in all_gaps_by_factor[factor] if g > 0.001),
                    'milp_wins': sum(1 for g in all_gaps_by_factor[factor] if g < -0.001),
                    'near_zero': sum(1 for g in all_gaps_by_factor[factor] if abs(g) <= 0.001),
                    'mean_gap': round(float(np.mean(all_gaps_by_factor[factor])), 4) if all_gaps_by_factor[factor] else None,
                    'std_gap': round(float(np.std(all_gaps_by_factor[factor])), 4) if all_gaps_by_factor[factor] else None,
                } for factor in [0.8, 1.0, 1.2]
            },
            'results': {str(factor): val for factor, val in all_results.items()},
        }, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    for factor in [0.8, 1.0, 1.2]:
        print(f"\n--- Factor={factor}x ---")
        factor_results = dict(all_results.get(factor, {}))
        factor_gaps = list(all_gaps_by_factor.get(factor, []))

        for city_name, n, cluster_r, n_sta in CITY_PROFILES:
            adjusted_r = int(cluster_r * factor)
            key = f"{city_name}_{factor}x"
            if key in factor_results:
                print(f"  [skip] {key} 已完成，跳过")
                continue
            rng = np.random.RandomState(seed_for(city_name))
        
            extent = adjusted_r * 3.0
            comm_locs = rng.uniform(-extent/2, extent/2, size=(n, 2))
            fac_locs = rng.uniform(-adjusted_r*0.8, adjusted_r*0.8, size=(n_sta, 2))
        
            dist_matrix = np.zeros((n_sta, n))
            for i in range(n_sta):
                for j in range(n):
                    dx = comm_locs[j, 0] - fac_locs[i, 0]
                    dy = comm_locs[j, 1] - fac_locs[i, 1]
                    dist_matrix[i, j] = np.sqrt(dx**2 + dy**2)
        
            S1m = np.zeros((n_sta, n))
            for i in range(n_sta):
                for j in range(n):
                    S1m[i, j] = s1_fn(dist_matrix[i, j])
        
            pop = rng.randint(400, 1000, size=n).astype(float)
            total_pop = pop.sum()
        
            # ============================================================
            # 站点规模异质化：2/3 大型站 + 1/3 小型站
            # ============================================================
            BUDGET_WAN = BUDGET_PER_STATION * n_sta   # 与 21城市精确MILP验证.py 统一：36 万元/站
            # BUILD_COST_DICT 现由 milp_common 从赛题附件3 读取（18/32/45），不再硬编码
        
            # 依据：该比例为成本结构下容量最大化的最低成本解（120万预算，7000人次/日 @108万）
            n_large = max(1, int(n_sta * 2 / 3))  # floor：保证 18·n_sta+27·n_large <= 36·n_sta 恒成立
            n_small = n_sta - n_large
        
            # 预算检查：若超预算，逐步把大站降级为小站
            total_cost = n_large * BUILD_COST_DICT[3] + n_small * BUILD_COST_DICT[1]
            while total_cost > BUDGET_WAN and n_large > 1:
                n_large -= 1
                n_small += 1
                total_cost = n_large * BUILD_COST_DICT[3] + n_small * BUILD_COST_DICT[1]
        
            sizes = [3] * n_large + [1] * n_small
            n_sta = len(sizes)  # 保持 n_sta 不变（异质分配后数量相同）
            sta_caps = np.array([CAPACITY[s] for s in sizes])

            # —— 可解性子采样（仅重庆 890×372 触发；确定性前缀切片，跨因子一致）——
            subsampled = (n > MAX_N) or (n_sta > MAX_STA)
            if subsampled:
                n2 = min(n, MAX_N); ns2 = min(n_sta, MAX_STA)
                comm_locs = comm_locs[:n2]; fac_locs = fac_locs[:ns2]; pop = pop[:n2]
                dist_matrix = dist_matrix[:ns2, :n2]; S1m = S1m[:ns2, :n2]
                n, n_sta = n2, ns2
                BUDGET_WAN = BUDGET_PER_STATION * n_sta
                n_large = max(1, int(n_sta * 2 / 3)); n_small = n_sta - n_large
                total_cost = n_large * BUILD_COST_DICT[3] + n_small * BUILD_COST_DICT[1]
                while total_cost > BUDGET_WAN and n_large > 1:
                    n_large -= 1; n_small += 1
                    total_cost = n_large * BUILD_COST_DICT[3] + n_small * BUILD_COST_DICT[1]
                sizes = [3] * n_large + [1] * n_small
                n_sta = len(sizes); sta_caps = np.array([CAPACITY[s] for s in sizes])
                total_pop = pop.sum()
                print(f"  [子采样] {city_name} 缩减为 n={n}, n_sta={n_sta}, 容量={sta_caps.sum():.0f}")

            print(f"  [站点规模] n_sta={n_sta}, 分配={sizes}, 总成本={total_cost}万, 总容量={sta_caps.sum()}")

            # BUG-03: Scale demand to ~115% utilization
            target_total_demand = sta_caps.sum() * 1.15
            raw_demand = rng.uniform(0.7, 1.3, size=n)
            demand = np.round(raw_demand / raw_demand.sum() * target_total_demand).astype(np.int64)

            # NASH（统一使用 milp_common.run_nash，与 21 城精确 MILP 完全一致）
            nash_F2, nash_F1, nash_cov = run_nash(n, pop, demand, S1m, sta_caps)

            # MILP：精确两阶段（与 21城市精确MILP验证.py 同构）。
            # 用真·墙钟守卫（multiprocessing）兜底：单城 > MILP_WALL_SEC 秒则终止并标记 timeout，
            # 避免重庆在 off-baseline 因子下因 CBC 分支爆炸永久挂起（pulp timeLimit 不生效）。
            milp_F2, milp_F1, milp_opt, milp_err = solve_milp_guarded(n, pop, demand, S1m, sta_caps)
            best_F1 = milp_F1 if milp_F1 is not None else 0.0

            if milp_F2 is None:  # 超时/异常：该 (factor,city) 不可解，标记并跳过汇总
                print(f"  {key}: MILP 超时/失败 ({milp_err}) — 标记 timeout，不计入均值")
                factor_results[key] = {
                    'n': n, 'cluster_r_original': cluster_r, 'cluster_r_adjusted': adjusted_r, 'n_stations': n_sta,
                    'n_large': int(n_large), 'n_small': int(n_small),
                    'large_ratio': round(n_large / n_sta, 4) if n_sta > 0 else None,
                    'budget_wan': round(float(total_cost), 1),
                    'milp_optimal': False, 'timeout': True, 'error': milp_err,
                    'nash_F1': round(nash_F1, 4), 'nash_F2': round(nash_F2, 4),
                    'milp_F1': None, 'milp_F2': None,
                    'gap': None, 'gap_pct': None, 'PoA': None, 'Loss_pct': None,
                    'subsampled': subsampled,
                }
                all_results[factor] = factor_results
                _save()
                continue

            gap = nash_F2 - milp_F2
            factor_gaps.append(gap)
            factor_results[key] = {
                'n': n, 'cluster_r_original': cluster_r, 'cluster_r_adjusted': adjusted_r, 'n_stations': n_sta,
                'n_large': int(n_large), 'n_small': int(n_small),
                'large_ratio': round(n_large / n_sta, 4) if n_sta > 0 else None,
                'budget_wan': round(float(total_cost), 1),
                'milp_optimal': bool(milp_opt),
                'subsampled': subsampled,
                'nash_F1': round(nash_F1, 4), 'nash_F2': round(nash_F2, 4),
                'milp_F1': round(best_F1, 4), 'milp_F2': round(milp_F2, 4),
                'gap': round(gap, 4),
                'gap_pct': round(gap/nash_F2*100, 1) if nash_F2 > 0 else 0,
                'PoA': round(milp_F2/nash_F2, 4) if nash_F2 > 0 else None,
                'Loss_pct': round((1 - nash_F2/milp_F2)*100, 1) if milp_F2 > 0 else None,
            }
            print(f"  {key}: Nash F2={nash_F2:.3f}, MILP F2={milp_F2:.3f}, Gap={gap:+.4f}")
            all_results[factor] = factor_results  # 落盘前同步，保证断点续跑正确
            _save()  # 每城完成即落盘
    
        all_results[factor] = factor_results
        all_gaps_by_factor[factor] = factor_gaps
    
        positive = sum(1 for g in factor_gaps if g > 0.001)
        negative = sum(1 for g in factor_gaps if g < -0.001)
        near_zero = sum(1 for g in factor_gaps if abs(g) <= 0.001)
        print(f"\n  Summary (factor={factor}): Nash>{factor}x{positive}, MILP>{factor}x{negative}, Near-zero{factor}x{near_zero}, Mean gap={np.mean(factor_gaps):+.4f}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL ROBUSTNESS SUMMARY")
    print("=" * 70)
    for factor in [0.8, 1.0, 1.2]:
        gaps = all_gaps_by_factor[factor]
        pos = sum(1 for g in gaps if g > 0.001)
        neg = sum(1 for g in gaps if g < -0.001)
        nz = sum(1 for g in gaps if abs(g) <= 0.001)
        print(f"  Factor {factor}x: Nash wins {pos}/21 ({pos/21*100:.0f}%), Mean gap={np.mean(gaps):+.4f}, Std={np.std(gaps):.4f}")

    print("\nPoA by factor:")
    for factor in [0.8, 1.0, 1.2]:
        fr = all_results[factor]
        poas = [v['PoA'] for v in fr.values() if v.get('PoA')]
        if poas:
            print(f"  Factor {factor}x: Mean PoA={np.mean(poas):.4f}, Mean Loss={(1 - 1/np.mean(poas))*100:.2f}%")

    # Save results (final: drop resume flag)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': 'Han et al. (2026), Scientific Data（参数由栅格直接测量推导：实城实验参数v3.json）',
            'experiment': '真实城市R_cluster参数鲁棒性测试',
            'build_cost_wan': {str(k): v for k, v in BUILD_COST_DICT.items()},
            'budget_per_station_wan': BUDGET_PER_STATION,
            'factors_tested': [0.8, 1.0, 1.2],
            'n_cities': len(CITY_PROFILES),
            'gaps_by_factor': {str(k): [round(g, 4) for g in v] for k, v in all_gaps_by_factor.items()},
            'summary_by_factor': {
                str(factor): {
                    'nash_wins': sum(1 for g in all_gaps_by_factor[factor] if g > 0.001),
                    'milp_wins': sum(1 for g in all_gaps_by_factor[factor] if g < -0.001),
                    'near_zero': sum(1 for g in all_gaps_by_factor[factor] if abs(g) <= 0.001),
                    'mean_gap': round(float(np.mean(all_gaps_by_factor[factor])), 4),
                    'std_gap': round(float(np.std(all_gaps_by_factor[factor])), 4),
                } for factor in [0.8, 1.0, 1.2]
            },
            'results': {str(factor): val for factor, val in all_results.items()}
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
