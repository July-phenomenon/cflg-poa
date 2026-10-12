#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
尺度可行性探针：用 v3 栅格推导参数跑单城，实测
  (a) 精确两阶段 MILP 的求解耗时
  (b) 覆盖率 F1 / 满意度 F2 是否落在有语义的区间
  (c) 预算约束是否被静默违反
用于判断「模型尺度」是否需要调整。不产生论文结果。
"""
import os
import json
import time
import sys

import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_core_solver")))
from core_solver import seed_for, run_nash, run_milp_exact, s1_fn, BUILD_COST_DICT, BUILD_COST_DICT

CAPACITY = np.array([0, 1000., 2000., 3000.])
BUDGET_WAN = 120
# BUILD_COST_DICT 现由 milp_common 从赛题附件3 读取（18/32/45），不再硬编码

PARAMS = os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data")


def build_and_run(city, n, cluster_r, n_sta, timeout_hint=None):
    rng = np.random.RandomState(seed_for(city))
    extent = cluster_r * 3.0
    comm_locs = rng.uniform(-extent / 2, extent / 2, size=(n, 2))
    fac_locs = rng.uniform(-cluster_r * 0.8, cluster_r * 0.8, size=(n_sta, 2))

    dist = np.sqrt(((comm_locs[None, :, :] - fac_locs[:, None, :]) ** 2).sum(-1))
    S1m = np.vectorize(s1_fn)(dist)

    # 可达性诊断：多少小区至少有一个 1000m 内站点
    reachable = (S1m > 0).any(axis=0).mean()

    pop = rng.randint(400, 1000, size=n).astype(float)

    n_large = max(1, int(round(n_sta * 2 / 3)))
    n_small = n_sta - n_large
    total_cost = n_large * BUILD_COST_DICT[3] + n_small * BUILD_COST_DICT[1]
    while total_cost > BUDGET_WAN and n_large > 1:
        n_large -= 1
        n_small += 1
        total_cost = n_large * BUILD_COST_DICT[3] + n_small * BUILD_COST_DICT[1]
    sizes = [3] * n_large + [1] * n_small
    sta_caps = np.array([CAPACITY[s] for s in sizes])

    target_total_demand = sta_caps.sum() * 1.15
    raw_demand = rng.uniform(0.7, 1.3, size=n)
    demand = np.round(raw_demand / raw_demand.sum() * target_total_demand).astype(np.int64)

    t0 = time.time()
    nash_F2, nash_F1, _ = run_nash(n, pop, demand, S1m, sta_caps)
    t_nash = time.time() - t0

    t1 = time.time()
    milp_F2, milp_F1, _ = run_milp_exact(n, pop, demand, S1m, sta_caps)
    t_milp = time.time() - t1

    print(f"{city:<11s} n={n:4d} n_sta={n_sta:4d} 变量={n*n_sta:7d} | "
          f"Nash {t_nash:6.2f}s MILP {t_milp:8.2f}s | "
          f"可达小区比={reachable:5.1%} | "
          f"F1 Nash={nash_F1:.3f} MILP={milp_F1:.3f} | "
          f"F2 Nash={nash_F2:.4f} MILP={milp_F2:.4f} | PoA={milp_F2/nash_F2 if nash_F2>0 else float('nan'):.4f} | "
          f"预算={total_cost}万(限{BUDGET_WAN},{'违规' if total_cost>BUDGET_WAN else 'OK'})")
    sys.stdout.flush()
    return dict(city=city, n=n, n_sta=n_sta, nvars=n * n_sta, t_nash=t_nash, t_milp=t_milp,
                reachable=reachable, nash_F1=nash_F1, milp_F1=milp_F1, nash_F2=nash_F2,
                milp_F2=milp_F2, cost=total_cost)


if __name__ == '__main__':
    recs = json.load(open(PARAMS, encoding='utf-8'))
    by = {r['city']: r for r in recs}

    # 支持 --defB：改用「高可达性区域等效半径」定义 cluster_r = sqrt(A_high/(pi*n_peaks))
    args = sys.argv[1:]
    use_b = '--defB' in args
    args = [a for a in args if a != '--defB']
    if use_b:
        print('[MODE] cluster_r 采用定义B（高可达性区域等效半径）')
    targets = args or ['Dongguan', 'Nanjing', 'Beijing']
    out = []
    for c in targets:
        r = by[c]
        cr = r['cluster_r_m']
        if use_b:
            cr = float(np.sqrt(r['A_high_km2'] / (np.pi * r['n_peaks'])) * 1000)
        try:
            out.append(build_and_run(c, r['n_communities'], cr, r['n_stations']))
        except Exception as e:
            print(f"{c}: 失败 {type(e).__name__}: {e}")
    with open('scale_probe_result.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
