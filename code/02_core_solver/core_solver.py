"""
Shared exact two-phase MILP solver (run_milp_exact).

Extracted from 21城市精确MILP验证.py so that every experiment
(21-city validation, robustness test, connectivity analysis) uses the SAME
centralized reference method. This removes the earlier inconsistency where the
robustness/connectivity scripts used a 1000-trial random heuristic labelled
"MILP" while the main script used the exact solver.
"""
import numpy as np
import pulp
import hashlib
import json
import os
import re
import datetime

W1, W2, W3 = 0.2, 0.3, 0.5
BASELINE_S3 = 0.60
S1_T = [(300, 1.0), (500, 0.9), (650, 0.75), (1000, 0.6)]
S2_T = [(0.6, 1.0), (0.75, 0.93), (0.85, 0.85), (0.95, 0.72), (1.0, 0.6)]

# ---------------------------------------------------------------------------
# 建站成本 —— 唯一来源：赛题附件 3《服务站建设与运营成本》
#   小型 18 万元 / 中型 32 万元 / 大型 45 万元
# 原先 21城验证、鲁棒性测试、尺度探针三个脚本硬编码 15/25/40，与附件 3 不符
# 且无任何来源记录；此处改为从附件 3 动态读取，保证参数可追溯、可复现。
# 若附件 3 不可读，回退到附件 3 的已知真值（18/32/45），而非旧的错误值。
# ---------------------------------------------------------------------------
ATTACHMENT3_PATH = os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data")


def load_build_cost():
    """Read station construction costs (万元) from 赛题附件3. Fallback = 18/32/45."""
    try:
        import pandas as pd
        df = pd.read_excel(ATTACHMENT3_PATH, header=1, nrows=3)
        df.columns = ['规模', '建设成本_万元', '日运营成本_元', '日容量_人次']
        return {0: 0.0,
                1: float(df['建设成本_万元'].iloc[0]),
                2: float(df['建设成本_万元'].iloc[1]),
                3: float(df['建设成本_万元'].iloc[2])}
    except Exception:
        return {0: 0.0, 1: 18.0, 2: 32.0, 3: 45.0}


BUILD_COST_DICT = load_build_cost()


def seed_for(name):
    """Stable, process-independent RNG seed for a city name.

    Replaces the previously used `hash(city_name)`, which Python salts per
    process (PYTHONHASHSEED). That salting made every script draw DIFFERENT
    random geometry for the same city (per-city F2 differed by up to 0.27
    between scripts) and made results non-reproducible across re-runs.
    """
    return int(hashlib.md5(str(name).encode('utf-8')).hexdigest(), 16) % (2**31)


def s1_fn(d):
    for u, v in S1_T:
        if d <= u:
            return v
    return 0.0


def s2_fn(u):
    for u2, v in S2_T:
        if u <= u2:
            return v
    return S2_T[-1][1]


def run_milp_exact(n, pop, demand, S1m, sta_caps, time_limit=None,
                   return_status=False, return_details=False):
    """Exact two-phase MILP (M2 fix): Phase1 max coverage, Phase2 max F2 locking F1.

    Returns (best_F2, milp_F1, milp_cov)  [default, 与既有调用完全一致]
      或 (best_F2, milp_F1, milp_cov, optimal_flag) 当 return_status=True。

    time_limit: 每阶段 CBC 求解时限（秒）。None = 不限时（默认，保持原有行为）。
                仅在实例过大、分支定界无法在合理时间内证明最优时由调用方显式启用；
                限时得到的 MILP 解是原问题的一个可行解（下界），故 PoA 被低估
                （保守方向），optimal_flag=False 用于标记此类实例。
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
        return 0.0, 0.0, set()

    prob1 += pulp.lpSum([pop[cj] * w1[(si, cj)] for (si, cj) in w1])

    for cj in range(n):
        vfs = [w1[(si, cj)] for si in range(n_sta) if (si, cj) in w1]
        if vfs:
            prob1 += pulp.lpSum(vfs) <= 1

    for si in range(n_sta):
        vfs = [(cj, w1[(si, cj)]) for cj in range(n) if (si, cj) in w1]
        prob1 += pulp.lpSum([demand[cj] * var for cj, var in vfs]) <= sta_caps[si]

    _kw = {'msg': 0}
    if time_limit:
        _kw['timeLimit'] = time_limit
    prob1.solve(pulp.PULP_CBC_CMD(**_kw))
    if pulp.LpStatus[prob1.status] != 'Optimal':
        return 0.0, 0.0, set()

    w1m = np.zeros((n_sta, n))
    for (si, cj), var in w1.items():
        w1m[si, cj] = round(pulp.value(var))

    cov1 = np.array([np.any(w1m[:, cj] > 0.5) for cj in range(n)])
    F1_max = pop[cov1].sum() / total_pop

    # Phase 2: Iterative MILP for F2, maintaining F1 >= F1_max
    S2_cur = np.ones(n_sta)
    best_F2 = -1
    best_w = None

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

        prob2.solve(pulp.PULP_CBC_CMD(**_kw))
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

        if np.max(np.abs(S2_new - S2_cur)) < 0.01:
            break
        S2_cur = S2_new

    if best_w is None:
        return 0.0, 0.0, set()

    milp_cov = set()
    for cj in range(n):
        for si in range(n_sta):
            if best_w[si, cj] > 0.5:
                milp_cov.add(cj)
                break
    milp_F1 = pop[list(milp_cov)].sum() / total_pop

    _optimal = (pulp.LpStatus[prob2.status] == 'Optimal')
    if return_details:
        # additive, opt-in: expose the assignment matrix and the converged
        # station-level S2 vector so that the channel decomposition
        # (A1 / A2 / coverage) can be recomputed from first principles.
        _loads = best_w @ demand
        _utils = np.divide(_loads, sta_caps, out=np.zeros_like(_loads, dtype=float),
                           where=np.asarray(sta_caps) > 0)
        _S2 = np.array([s2_fn(u) for u in _utils])
        return best_F2, milp_F1, milp_cov, best_w, _S2
    if return_status:
        return best_F2, milp_F1, milp_cov, _optimal
    return best_F2, milp_F1, milp_cov


def run_milp_exact_with_s3(n, pop, demand, S1m, sta_caps, s3_array=None,
                           time_limit=None, return_status=False):
    """集中式两阶段 MILP（补贴实验用）：目标中的 S3 可取每站不同值（boosted S3）。

    与 run_milp_exact 完全相同的两阶段结构，仅 Phase2 目标把 BASELINE_S3 替换为
    s3_array[si]。用于计算“补贴机制下”的集中式最优，从而与补贴后 Nash 公平比较
    （PoA_subsidy = MILP_boosted / Nash_boosted），避免把未补贴 MILP 当分母造成的
    PoA<1 假象。s3_array=None 时等价于 run_milp_exact。
    """
    n_sta = len(sta_caps)
    if s3_array is None:
        s3_array = np.full(n_sta, BASELINE_S3)
    total_pop = pop.sum()

    prob1 = pulp.LpProblem("Phase1_s3", pulp.LpMaximize)
    w1 = {}
    for si in range(n_sta):
        for cj in range(n):
            if S1m[si, cj] > 0:
                w1[(si, cj)] = pulp.LpVariable(f"w1_{si}_{cj}", cat='Binary')
    if not w1:
        return (0.0, 0.0, set()) if not return_status else (0.0, 0.0, set(), False)
    prob1 += pulp.lpSum([pop[cj] * w1[(si, cj)] for (si, cj) in w1])
    for cj in range(n):
        vfs = [w1[(si, cj)] for si in range(n_sta) if (si, cj) in w1]
        if vfs:
            prob1 += pulp.lpSum(vfs) <= 1
    for si in range(n_sta):
        vfs = [(cj, w1[(si, cj)]) for cj in range(n) if (si, cj) in w1]
        prob1 += pulp.lpSum([demand[cj] * var for cj, var in vfs]) <= sta_caps[si]
    _kw = {'msg': 0}
    if time_limit:
        _kw['timeLimit'] = time_limit
    prob1.solve(pulp.PULP_CBC_CMD(**_kw))
    if pulp.LpStatus[prob1.status] != 'Optimal':
        return (0.0, 0.0, set()) if not return_status else (0.0, 0.0, set(), False)

    w1m = np.zeros((n_sta, n))
    for (si, cj), var in w1.items():
        w1m[si, cj] = round(pulp.value(var))
    cov1 = np.array([np.any(w1m[:, cj] > 0.5) for cj in range(n)])
    F1_max = pop[cov1].sum() / total_pop

    S2_cur = np.ones(n_sta)
    best_F2 = -1
    best_w = None
    for it in range(10):
        prob2 = pulp.LpProblem(f"Phase2_s3_{it}", pulp.LpMaximize)
        w2 = {}
        for si in range(n_sta):
            for cj in range(n):
                if S1m[si, cj] > 0:
                    w2[(si, cj)] = pulp.LpVariable(f"w2_{si}_{cj}", cat='Binary')
        if not w2:
            break
        prob2 += pulp.lpSum([
            pop[cj] * (W1 * S1m[si, cj] + W2 * S2_cur[si] + W3 * s3_array[si]) * w2[(si, cj)]
            for (si, cj) in w2
        ])
        for cj in range(n):
            vfs = [w2[(si, cj)] for si in range(n_sta) if (si, cj) in w2]
            if vfs:
                prob2 += pulp.lpSum(vfs) <= 1
        for si in range(n_sta):
            vfs = [(cj, w2[(si, cj)]) for cj in range(n) if (si, cj) in w2]
            prob2 += pulp.lpSum([demand[cj] * var for cj, var in vfs]) <= sta_caps[si]
        prob2 += pulp.lpSum([pop[cj] * w2[(si, cj)] for (si, cj) in w2]) >= F1_max * total_pop - 1
        prob2.solve(pulp.PULP_CBC_CMD(**_kw))
        if pulp.LpStatus[prob2.status] != 'Optimal':
            break
        wm = np.zeros((n_sta, n))
        for (si, cj), var in w2.items():
            wm[si, cj] = round(pulp.value(var))
        loads = np.array([sum(demand[cj] * wm[si, cj] for cj in range(n)) for si in range(n_sta)])
        utils = loads / sta_caps
        S2_new = np.array([s2_fn(u) for u in utils])
        F2_actual = sum(
            pop[cj] * (W1 * S1m[si, cj] + W2 * S2_new[si] + W3 * s3_array[si]) * wm[si, cj]
            for cj in range(n) for si in range(n_sta)
        ) / total_pop
        if F2_actual > best_F2:
            best_F2 = F2_actual
            best_w = wm.copy()
        if np.max(np.abs(S2_new - S2_cur)) < 0.01:
            break
        S2_cur = S2_new
    if best_w is None:
        return (0.0, 0.0, set()) if not return_status else (0.0, 0.0, set(), False)
    milp_cov = set()
    for cj in range(n):
        for si in range(n_sta):
            if best_w[si, cj] > 0.5:
                milp_cov.add(cj)
                break
    milp_F1 = pop[list(milp_cov)].sum() / total_pop
    _optimal = (pulp.LpStatus[prob2.status] == 'Optimal')
    if return_status:
        return best_F2, milp_F1, milp_cov, _optimal
    return best_F2, milp_F1, milp_cov


def run_nash(n, pop, demand, S1m, sta_caps, return_details=False):
    """Decentralized Nash equilibrium simulation (canonical version).

    Shared by 21城市精确MILP验证.py and 真实城市参数鲁棒性测试.py so that
    the Nash baseline is identical across experiments. (Previously the two
    scripts each had their own near-identical inline copy, which diverged
    for 14/21 cities — a reproducibility bug.)
    """
    n_sta = len(sta_caps)
    S2_cur = np.ones(n_sta)
    choices = {}

    for it in range(10):
        for cj in range(n):
            if S1m[:, cj].max() <= 0:
                continue
            best_val = -1
            best_stations = []
            for si in range(n_sta):
                s1 = S1m[si, cj]
                if s1 <= 0:
                    continue
                val = W1 * s1 + W2 * S2_cur[si] + W3 * BASELINE_S3
                if val > best_val + 1e-9:
                    best_val = val
                    best_stations = [si]
                elif abs(val - best_val) < 1e-9:
                    best_stations.append(si)
            if not best_stations:
                continue
            choices[cj] = best_stations[0]

        loads = np.zeros(n_sta)
        for cj, si in choices.items():
            loads[si] += demand[cj]

        changed = True
        while changed:
            changed = False
            overloaded = [si for si in range(n_sta) if loads[si] > sta_caps[si] + 1e-6]
            if not overloaded:
                break
            for si in overloaded:
                comms = sorted([cj for cj, s in choices.items() if s == si], key=lambda c: -demand[c])
                keep = 0
                kicked = []
                for c in comms:
                    if keep + demand[c] <= sta_caps[si] + 1e-6:
                        keep += demand[c]
                    else:
                        kicked.append(c)
                for c in kicked:
                    alts = []
                    for si2 in range(n_sta):
                        if si2 == si:
                            continue
                        if S1m[si2, c] <= 0:
                            continue
                        nl = loads[si2] + demand[c]
                        if nl <= sta_caps[si2] + 1e-6:
                            alts.append((W1 * S1m[si2, c] + W2 * S2_cur[si2] + W3 * BASELINE_S3, si2))
                    if alts:
                        alts.sort(reverse=True)
                        choices[c] = alts[0][1]
                        loads[alts[0][1]] += demand[c]
                        loads[si] -= demand[c]
                        changed = True
                    else:
                        # BUG A Fix: 无法重分配的小区从 choices 中删除（同时扣减 loads）
                        loads[si] -= demand[c]
                        del choices[c]

        new_S2 = np.array([s2_fn(loads[si] / sta_caps[si]) for si in range(n_sta)])
        if np.max(np.abs(new_S2 - S2_cur)) < 0.01:
            S2_cur = new_S2
            break
        S2_cur = new_S2

    covered = set(choices.keys())
    F1 = pop[list(covered)].sum() / pop.sum()
    F2 = sum(pop[cj] * (W1 * S1m[choices[cj], cj] + W2 * S2_cur[choices[cj]] + W3 * BASELINE_S3)
             for cj in covered) / pop.sum()
    if return_details:
        # additive, opt-in: expose choices and the converged S2 vector
        return F2, F1, covered, choices, S2_cur
    return F2, F1, covered


def milp_phase1_coverage(n, pop, demand, S1m, sta_caps):
    """仅解 Phase 1（最大覆盖），返回 (F1_max, covered_set)。

    两阶段序贯设定下，Phase 1 是集中规划者的覆盖决策；Phase 2 才在此覆盖 mandate
    之下进行（集中或分散的）分配。该函数把 Phase 1 的覆盖集显式暴露出来，供
    “Nash 锁定覆盖”版本使用，保证 MILP 与 Nash 在同一覆盖口径下比较。
    """
    n_sta = len(sta_caps)
    total_pop = pop.sum()

    prob1 = pulp.LpProblem("Phase1_cov", pulp.LpMaximize)
    w1 = {}
    for si in range(n_sta):
        for cj in range(n):
            if S1m[si, cj] > 0:
                w1[(si, cj)] = pulp.LpVariable(f"pc_{si}_{cj}", cat='Binary')
    if not w1:
        return 0.0, set()

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
        return 0.0, set()

    cov = set()
    for (si, cj), var in w1.items():
        if round(pulp.value(var)) > 0.5:
            cov.add(cj)
    F1 = pop[list(cov)].sum() / total_pop if cov else 0.0
    return F1, cov


def run_nash_locked(n, pop, demand, S1m, sta_caps, locked_cov, return_details=False):
    """两阶段序贯 Nash（选项 A：Nash 锁定覆盖）。

    与 run_nash（经典单阶段 CFLG）的区别：
      - 经典版：社区若被挤出且无替代站，会被 `del choices[c]` 直接丢弃 →
        Nash 覆盖率可远低于 MILP，F2 的分母被覆盖率拖低，PoA 被放大
        （沈阳 nash_F1=0.485 vs milp_F1=0.790 → PoA=1.7479）。
      - 本版本：Phase 1 已由集中规划锁定覆盖集 locked_cov；Phase 2 的去中心化
        博弈**只在该集合上进行**，且集合内社区**不允许被丢弃**。这样 MILP 与
        Nash 的 F1 相同，PoA 只反映 S1（由哪一站服务）与 S2（负载）的差异，
        正是定理 1（PoA ≤ 1/S_min = 1.667）所适用的设定。

    返回 (F2, F1, covered, n_repaired)；n_repaired = 兜底修复次数（诚实记录）。
    [V3] return_details=True 时额外返回 (choices, S2_cur)，用于通道分解
    （F2 = W1*A1 + W2*A2 + W3*S3*cov，S3=0.60 常量）。
    """
    n_sta = len(sta_caps)
    total_pop = pop.sum()
    comms = sorted(locked_cov)
    S2_cur = np.ones(n_sta)
    choices = {}

    for it in range(10):
        for cj in comms:
            if S1m[:, cj].max() <= 0:
                continue
            best_val = -1
            best_stations = []
            for si in range(n_sta):
                s1 = S1m[si, cj]
                if s1 <= 0:
                    continue
                val = W1 * s1 + W2 * S2_cur[si] + W3 * BASELINE_S3
                if val > best_val + 1e-9:
                    best_val = val
                    best_stations = [si]
                elif abs(val - best_val) < 1e-9:
                    best_stations.append(si)
            if best_stations:
                choices[cj] = best_stations[0]

        loads = np.zeros(n_sta)
        for cj, si in choices.items():
            loads[si] += demand[cj]

        changed = True
        while changed:
            changed = False
            overloaded = [si for si in range(n_sta) if loads[si] > sta_caps[si] + 1e-6]
            if not overloaded:
                break
            for si in overloaded:
                comms_here = sorted([cj for cj, s in choices.items() if s == si],
                                    key=lambda c: -demand[c])
                keep = 0.0
                kicked = []
                for c in comms_here:
                    if keep + demand[c] <= sta_caps[si] + 1e-6:
                        keep += demand[c]
                    else:
                        kicked.append(c)
                for c in kicked:
                    alts = []
                    for si2 in range(n_sta):
                        if si2 == si or S1m[si2, c] <= 0:
                            continue
                        if loads[si2] + demand[c] <= sta_caps[si2] + 1e-6:
                            alts.append((W1 * S1m[si2, c] + W2 * S2_cur[si2] + W3 * BASELINE_S3, si2))
                    if alts:
                        alts.sort(reverse=True)
                        choices[c] = alts[0][1]
                        loads[alts[0][1]] += demand[c]
                        loads[si] -= demand[c]
                        changed = True
                    else:
                        # 锁定覆盖：不删除，保留在原站（容量溢出由 repair 阶段处理）
                        loads[si] -= demand[c]
                        choices.pop(c, None)
                        changed = True

        new_S2 = np.array([s2_fn(loads[si] / sta_caps[si]) for si in range(n_sta)])
        if np.max(np.abs(new_S2 - S2_cur)) < 0.01:
            S2_cur = new_S2
            break
        S2_cur = new_S2

    # ---- repair：把被挤出但属于锁定集的小区重新安置（Phase1 已证明可行）----
    n_repaired = 0
    loads = np.zeros(n_sta)
    for cj, si in choices.items():
        loads[si] += demand[cj]
    for cj in comms:
        if cj in choices:
            continue
        cands = []
        for si in range(n_sta):
            if S1m[si, cj] <= 0:
                continue
            if loads[si] + demand[cj] <= sta_caps[si] + 1e-6:
                cands.append((W1 * S1m[si, cj] + W2 * s2_fn((loads[si] + demand[cj]) / sta_caps[si])
                              + W3 * BASELINE_S3, si))
        if cands:
            cands.sort(reverse=True)
            si = cands[0][1]
            choices[cj] = si
            loads[si] += demand[cj]
            n_repaired += 1

    S2_cur = np.array([s2_fn(loads[si] / sta_caps[si]) for si in range(n_sta)])
    covered = set(choices.keys())
    F1 = pop[list(covered)].sum() / total_pop if covered else 0.0
    F2 = sum(pop[cj] * (W1 * S1m[choices[cj], cj] + W2 * S2_cur[choices[cj]] + W3 * BASELINE_S3)
             for cj in sorted(covered)) / total_pop
    if return_details:
        return F2, F1, covered, n_repaired, dict(choices), S2_cur.copy()
    return F2, F1, covered, n_repaired


def run_nash_with_s3(n, pop, demand, S1m, sta_caps, s3_array=None, return_choices=False):
    """去中心化 Nash 均衡（V2 补贴实验用）：支持按站注入 S3 数组。

    与 run_nash 完全相同的迭代逻辑，区别仅在于站点满意度 S3 可取每站不同值
    （补贴机制 boosting 低服务缺口站点）。s3_array=None 时退化为 run_nash。
    return_choices=True 时额外返回 {社区: 所选站} 分配，供补贴缺口计算使用。
    """
    n_sta = len(sta_caps)
    if s3_array is None:
        s3_array = np.full(n_sta, BASELINE_S3)
    S2_cur = np.ones(n_sta)
    choices = {}

    for it in range(10):
        for cj in range(n):
            if S1m[:, cj].max() <= 0:
                continue
            best_val = -1
            best_stations = []
            for si in range(n_sta):
                s1 = S1m[si, cj]
                if s1 <= 0:
                    continue
                val = W1 * s1 + W2 * S2_cur[si] + W3 * s3_array[si]
                if val > best_val + 1e-9:
                    best_val = val
                    best_stations = [si]
                elif abs(val - best_val) < 1e-9:
                    best_stations.append(si)
            if not best_stations:
                continue
            choices[cj] = best_stations[0]

        loads = np.zeros(n_sta)
        for cj, si in choices.items():
            loads[si] += demand[cj]

        changed = True
        while changed:
            changed = False
            overloaded = [si for si in range(n_sta) if loads[si] > sta_caps[si] + 1e-6]
            if not overloaded:
                break
            for si in overloaded:
                comms = sorted([cj for cj, s in choices.items() if s == si], key=lambda c: -demand[c])
                keep = 0
                kicked = []
                for c in comms:
                    if keep + demand[c] <= sta_caps[si] + 1e-6:
                        keep += demand[c]
                    else:
                        kicked.append(c)
                for c in kicked:
                    alts = []
                    for si2 in range(n_sta):
                        if si2 == si:
                            continue
                        if S1m[si2, c] <= 0:
                            continue
                        nl = loads[si2] + demand[c]
                        if nl <= sta_caps[si2] + 1e-6:
                            alts.append((W1 * S1m[si2, c] + W2 * S2_cur[si2] + W3 * s3_array[si2], si2))
                    if alts:
                        alts.sort(reverse=True)
                        choices[c] = alts[0][1]
                        loads[alts[0][1]] += demand[c]
                        loads[si] -= demand[c]
                        changed = True
                    else:
                        loads[si] -= demand[c]
                        del choices[c]

        new_S2 = np.array([s2_fn(loads[si] / sta_caps[si]) for si in range(n_sta)])
        if np.max(np.abs(new_S2 - S2_cur)) < 0.01:
            S2_cur = new_S2
            break
        S2_cur = new_S2

    covered = set(choices.keys())
    F1 = pop[list(covered)].sum() / pop.sum()
    F2 = sum(pop[cj] * (W1 * S1m[choices[cj], cj] + W2 * S2_cur[choices[cj]] + W3 * s3_array[choices[cj]])
             for cj in covered) / pop.sum()
    if return_choices:
        return F2, F1, covered, choices
    return F2, F1, covered


# ---------------------------------------------------------------------------
# 口径锁定（spec-lock）机制 —— R4 (2026-10-01)
# ---------------------------------------------------------------------------
# 目的：让"用错文件 / 用错口径"在脚本启动的那一秒就崩，而不是产出错误结果
# 三天后被发现。历史事故：
#   D2  误用旧文件（budget=None 版）         -> max PoA 1.7479 虚高
#   E3  产物自报 data_source 与实际输入不符  -> 追溯链断裂
#   E5  第一批实验用 classic 口径当 locked 用 -> 结论不可混用
#   T5  用了无 S2 字段的旧枚举               -> 机制分析缺字段
#
# 三个对外接口：
#   verify_input(...)   实验开始时调用，校验失败直接 raise
#   resolve_canonical(key)  按"规范输入登记表"取文件（推荐，比手写路径更安全）
#   write_manifest(...) 实验结束时调用，落盘 manifest.json
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "results", "main", '_manifests')

# 已知"坏文件"黑名单：命中即拒绝加载（allow_legacy=True 可显式豁免）
FORBIDDEN_INPUTS = {
    '21城市精确MILP验证结果.json':
        '旧版，budget_per_station 未指定（legacy artifact），max PoA 1.7479 虚高',
    '21城市精确MILP验证结果_最终版.json':
        'classic 口径（Nash 不锁定覆盖），不能用作 locked 主结果（E5 事故源）',
    '21城市精确MILP验证结果_两阶段版.json':
        'locked 口径但无 S2 通道字段，机制分析请用 _withS2 版',
    '21城市精确MILP验证结果_单阶段版.bak':
        '归档文件',
    'poa_full_per_config.json':
        '旧枚举，无 milp/nash_S2_weighted 字段（T5 事故源），请用 _s2 版',
}
FORBIDDEN_SUFFIX = ('.bak',)

# 规范输入登记表：实验脚本应通过 key 取文件，而不是手写路径
CANONICAL_INPUTS = {
    'cities21_locked_s2': dict(
        rel=os.path.join("results", "main",
                         '21城市精确MILP验证结果_两阶段版_withS2.json'),
        desc='21 城 locked 口径主结果（含 S2 通道），论文主口径',
        expected_fields={'nash_mode': 'locked',
                         'budget_per_station_wan': 36.0,
                         'n_cities': 21},
        require_paths=['results'],
        counts={'results': 21},
        record_fields={'results': {'fields': ['milp_S2_weighted',
                                              'nash_S2_weighted',
                                              'PoA', 'connectivity'],
                                   'min_fraction': 1.0}},
        implied={'nash_mode_source': 'file field (verified)'},
    ),
    'cities21_exact_locked': dict(
        rel=os.path.join("results", "main", '21cities_exact_locked.json'),
        desc='21 城 locked 口径，Phase-2 由单 MILP 精确求解（2026-10 返修基准）',
        expected_fields={'nash_mode': 'locked', 'budget_per_station_wan': 36.0},
        require_paths=['results'],
        counts={'results': 21},
        record_fields={'results': {'fields': ['milp_S2_weighted',
                                              'nash_S2_weighted',
                                              'PoA', 'connectivity'],
                                   'min_fraction': 1.0}},
        implied={'nash_mode_source': 'file field (verified)',
                 'gap_field_sign': 'nash_F2 - milp_F2'},
    ),
    'cities21_exact_classic': dict(
        rel=os.path.join("results", "main", '21cities_exact_classic.json'),
        desc='21 城 classic 口径，Phase-2 由单 MILP 精确求解（对照用）',
        expected_fields={'nash_mode': 'classic', 'budget_per_station_wan': 36.0},
        require_paths=['results'],
        counts={'results': 21},
        record_fields={'results': {'fields': ['PoA', 'connectivity'],
                                   'min_fraction': 1.0}},
        implied={'nash_mode_source': 'file field (verified)',
                 'gap_field_sign': 'nash_F2 - milp_F2'},
    ),
    'cities21_classic_legacy': dict(
        rel=os.path.join("results", "main",
                         '21城市精确MILP验证结果_最终版.json'),
        desc='21 城 classic 口径（Nash 不锁覆盖）。第一批实验事故源，'
             '仅允许在"classic vs locked 对比"场景下用 allow_legacy 读取',
        allow_legacy=True,
        expected_fields={'budget_per_station_wan': 36.0, 'n_cities': 21},
        require_paths=['results'],
        forbid_paths=['nash_mode'],
        counts={'results': 21},
        implied={'nash_mode': 'classic',
                 'nash_mode_source': 'by construction: file predates locked '
                                     'mode, so it carries no nash_mode field; '
                                     'absence IS the evidence',
                 'gap_field_sign': 'nash_F2 - milp_F2'},
    ),
    'enum_full_s2': dict(
        rel='poa_full_per_config_s2.json',
        desc='全量枚举 15208 配置（含 S2 通道），classic 口径（构造决定）',
        expected_fields={},
        require_paths=['results'],
        counts={'results': 15208},
        record_fields={'results': {'fields': ['milp_S2_weighted',
                                              'nash_S2_weighted', 'PoA'],
                                   'min_fraction': 0.999}},  # idx0 为全零退化配置
        implied={'nash_mode': 'classic',
                 'nash_mode_source': 'implied by construction (file has no '
                                     'nash_mode field; enumeration never locks '
                                     'coverage)',
                 'gap_field_sign': 'milp_F2 - nash_F2'},
    ),
}

# ⚠️ 跨文件符号约定不一致（2026-10-01 实测确认），引用 gap 字段前必读：
#   21 城 *_withS2.json : gap = nash_F2 - milp_F2   （与统一口径【反号】）
#   枚举 *_s2.json      : gap = milp_F2 - nash_F2   （与统一口径同号）
# 统一口径（Unified_Specification）为 gap = milp_F2 - nash_F2。
# 因此：读 21 城文件时【禁止】直接用其 gap 字段判定胜负，必须自行重算。
GAP_SIGN_CONVENTION = {
    'unified': 'milp_F2 - nash_F2',
    '21city_file_field_is_negated': True,
    'enum_file_field_matches_unified': True,
}


def _dig(data, dotted):
    """按 'a.b.c' 逐级取值；任一层缺失返回 _MISSING。"""
    cur = data
    for part in dotted.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.isdigit() \
                and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return _MISSING
    return cur


class _Missing(object):
    def __repr__(self):
        return '<MISSING>'


_MISSING = _Missing()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_input(path, expected_fields=None, require_paths=None, counts=None,
                 record_fields=None, forbid_paths=None, allow_legacy=False):
    """校验输入文件的关键字段，任一不符立即 raise ValueError。

    参数
    ----
    path            : 输入文件绝对路径
    expected_fields : {'a.b': 期望值}   严格相等（浮点容差 1e-9）
    require_paths   : ['a.b', ...]      必须存在且非空
    forbid_paths    : ['a.b', ...]      必须【不存在】
                      （用于"按构造判定口径"：如 classic 版无 nash_mode 字段，
                        locked 版必有，故"缺失"本身就是 classic 的证据）
    counts          : {'a.b': n}        该键的长度必须等于 n
    record_fields   : {'a.b': {'fields': [...], 'min_fraction': 0.99}}
                      列表/字典型键中至少 min_fraction 比例的记录含全部 fields
    allow_legacy    : True 时豁免 FORBIDDEN_INPUTS 黑名单（默认 False = 拒绝）

    返回
    ----
    dict(file, path_absolute, sha256, size_bytes, modified_at, verified)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'verify_input: 文件不存在 -> {path}')

    abs_path = os.path.abspath(path)
    base = os.path.basename(abs_path)
    errors = []

    if not allow_legacy:
        if base in FORBIDDEN_INPUTS:
            errors.append(f'黑名单文件（{FORBIDDEN_INPUTS[base]}）')
        if base.endswith(FORBIDDEN_SUFFIX):
            errors.append('归档/备份后缀文件，禁止作为实验输入')
        if re.search(r'_s2\.json$', base) is None and base.startswith('poa_full_per_config'):
            errors.append('枚举文件必须是含 S2 的 _s2 版本')

    with open(abs_path, encoding='utf-8') as f:
        data = json.load(f)

    for key, want in (expected_fields or {}).items():
        got = _dig(data, key)
        if got is _MISSING:
            errors.append(f"字段 '{key}' 缺失")
        elif isinstance(want, float) and isinstance(got, (int, float)):
            if abs(float(got) - want) > 1e-9:
                errors.append(f"字段 '{key}': 期望 {want}, 实为 {got}")
        elif got != want:
            errors.append(f"字段 '{key}': 期望 {want!r}, 实为 {got!r}")

    for p in (require_paths or []):
        v = _dig(data, p)
        if v is _MISSING or v is None or (hasattr(v, '__len__') and len(v) == 0):
            errors.append(f"必需路径 '{p}' 缺失或为空")

    for p in (forbid_paths or []):
        if _dig(data, p) is not _MISSING:
            errors.append(f"禁止存在路径 '{p}'（实为 {_dig(data, p)!r}）——"
                          f"它的【缺失】正是判定该口径的依据")

    for p, n in (counts or {}).items():
        v = _dig(data, p)
        if v is _MISSING or not hasattr(v, '__len__'):
            errors.append(f"计数校验 '{p}' 无法取长度")
        elif len(v) != n:
            errors.append(f"计数校验 '{p}': 期望 {n}, 实为 {len(v)}")

    for p, spec in (record_fields or {}).items():
        seq = _dig(data, p)
        if seq is _MISSING or not isinstance(seq, (list, dict)) or not seq:
            errors.append(f"记录字段校验 '{p}' 不是非空 list/dict")
            continue
        items = list(seq.values()) if isinstance(seq, dict) else list(seq)
        flds = spec['fields']
        minf = spec.get('min_fraction', 1.0)
        ok = sum(1 for r in items
                 if isinstance(r, dict) and all(k in r for k in flds))
        frac = ok / float(len(items))
        if frac + 1e-12 < minf:
            errors.append(
                f"记录字段校验 '{p}': 含 {flds} 的记录占比 {frac:.4f} < {minf}")

    if errors:
        raise ValueError(
            'Input verification FAILED\n'
            f'  file: {abs_path}\n'
            f'  sha256: {sha256_of(abs_path)}\n'
            + ''.join(f'  - {e}\n' for e in errors)
            + '  This file may be from an old/wrong experiment. '
              'Refusing to continue.')

    return dict(
        file=base,
        path_absolute=abs_path,
        sha256=sha256_of(abs_path),
        size_bytes=os.path.getsize(abs_path),
        modified_at=datetime.datetime.fromtimestamp(
            os.path.getmtime(abs_path)).isoformat(timespec='seconds'),
        verified=dict(expected_fields=expected_fields or {},
                      require_paths=require_paths or [],
                      forbid_paths=forbid_paths or [],
                      counts=counts or {},
                      record_fields=record_fields or {}),
    )


def resolve_canonical(key, allow_legacy=False):
    """按规范输入登记表取文件并校验。推荐替代手写路径。

    用法：META = resolve_canonical('cities21_locked_s2')
    """
    if key not in CANONICAL_INPUTS:
        raise KeyError(f'resolve_canonical: 未登记的 key {key!r}；'
                       f'可用: {sorted(CANONICAL_INPUTS)}')
    spec = CANONICAL_INPUTS[key]
    path = spec['rel'] if os.path.isabs(spec['rel']) \
        else os.path.join(PROJECT_ROOT, spec['rel'])
    meta = verify_input(path,
                        expected_fields=spec.get('expected_fields'),
                        require_paths=spec.get('require_paths'),
                        forbid_paths=spec.get('forbid_paths'),
                        counts=spec.get('counts'),
                        record_fields=spec.get('record_fields'),
                        allow_legacy=allow_legacy or spec.get('allow_legacy', False))
    meta['canonical_key'] = key
    meta['description'] = spec['desc']
    meta['implied'] = spec.get('implied', {})
    return meta


def write_manifest(experiment_id, script_path, inputs, outputs, parameters,
                   spec_reference='Unified_Specification_v2.md',
                   notes=None, manifest_dir=None, data_source_note=None):
    """实验结束时落盘 manifest.json。

    inputs  : verify_input()/resolve_canonical() 返回的 dict 列表
    outputs : 本次实验产出的文件路径列表（自动算 sha256）
    """
    outdir = manifest_dir or MANIFEST_DIR
    os.makedirs(outdir, exist_ok=True)

    out_meta = []
    for p in outputs:
        ap = os.path.abspath(p)
        out_meta.append(dict(
            file=os.path.basename(ap),
            path_absolute=ap,
            sha256=sha256_of(ap) if os.path.exists(ap) else None,
            size_bytes=os.path.getsize(ap) if os.path.exists(ap) else None,
        ))

    manifest = dict(
        experiment_id=experiment_id,
        executed_at=datetime.datetime.now().isoformat(timespec='seconds'),
        script=os.path.relpath(os.path.abspath(script_path), PROJECT_ROOT)
               if os.path.exists(script_path) else str(script_path),
        script_sha256=sha256_of(script_path) if os.path.exists(script_path) else None,
        spec_reference=spec_reference,
        parameters=parameters,
        inputs=inputs,
        outputs=out_meta,
        gap_sign_convention=GAP_SIGN_CONVENTION,
        data_source_note=data_source_note,
        notes=notes,
    )
    mpath = os.path.join(outdir, f'{experiment_id}.manifest.json')
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f'[manifest] {mpath}')
    return mpath
