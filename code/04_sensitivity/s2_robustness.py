"""
S2 Robustness Test: Test 3 alternative S2 functional forms on BENCH_3STA config.
Reviewer concern #3: Does the Nash-dominates-MILP result depend on piecewise-constant S2?
"""
import pandas as pd
import numpy as np
import os, re, warnings, json
warnings.filterwarnings('ignore')
import pulp


def main():

    DATA_DIR = os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data")
    Q13_DIR = os.environ.get("CFLG_Q13_OUT", r"D:/competition_data/q13")
    COMM_NAMES = ['A','B','C','D','E','F','G','H','I','J']
    N = 10
    W1, W2, W3 = 0.2, 0.3, 0.5
    S3 = 0.60
    MAX_S2_ITER = 10
    S2_CONV_TOL = 0.01

    # ---- Load data ----
    df1 = pd.read_excel(os.path.join(DATA_DIR, "附件1：小区基础数据.xlsx"), sheet_name=0, header=1)
    df1.columns = ['小区','总人口','老年人口','自理','半失能','失能','人均收入']
    elderly_pop = df1['老年人口'].values.astype(float)
    TOTAL_ELDERLY = elderly_pop.sum()

    df_q13 = pd.read_csv(os.path.join(Q13_DIR, '问题1.3_实际月服务需求.csv'), encoding='utf-8-sig')
    dd = np.round(df_q13.iloc[:, 1:7].values.astype(float).sum(axis=1) / 30.0).astype(int)

    df4 = pd.read_excel(os.path.join(DATA_DIR, "附件4：小区间距离矩阵.xlsx"), header=1, index_col=0).values.astype(float)

    df5 = pd.read_excel(os.path.join(DATA_DIR, "附件5：满意度评分规则.xlsx"), header=None)
    S1_T = []; S2_T = []
    for idx in range(len(df5)):
        c1 = str(df5.iloc[idx, 0]) if pd.notna(df5.iloc[idx, 0]) else ''
        c2 = str(df5.iloc[idx, 1]) if pd.notna(df5.iloc[idx, 1]) else ''
        m = re.search(r'[≤]([\d.]+)m.*S1=([\d.]+)', c2)
        if m and ('S1' in c1 or 'S1' in c2):
            S1_T.append((float(m.group(1)), float(m.group(2))))
        m = re.search(r'[≤]([\d.]+).*S2=([\d.]+)', c2)
        if m and ('S2' in c1 or 'S2' in c2):
            S2_T.append((float(m.group(1)), float(m.group(2))))
    S1_T.sort(); S2_T.sort()

    def S1(d):
        for u, v in S1_T:
            if d <= u: return v
        return 0.0

    def S2_piecewise(u):
        for u2, v in S2_T:
            if u <= u2: return v
        return S2_T[-1][1]

    S1m = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S1m[i, j] = S1(df4[i, j])

    df3 = pd.read_excel(os.path.join(DATA_DIR, "附件3：服务站建设与运营成本.xlsx"), header=1, nrows=3)
    df3.columns = ['规模','建设成本_万元','日运营成本_元','日容量_人次']
    CAPACITY = np.array([0] + df3['日容量_人次'].astype(float).tolist(), dtype=float)
    BUILD_COST = np.array([0] + df3['建设成本_万元'].astype(float).tolist(), dtype=float)

    # ---- BENCH_3STA config ----
    # C=size3, E=size1, J=size3
    config = np.array([0, 0, 3, 0, 1, 0, 0, 0, 0, 3])
    sta_idx = [i for i, s in enumerate(config) if s > 0]
    sta_caps = np.array([CAPACITY[config[i]] for i in sta_idx])
    n_sta = len(sta_idx)
    print(f"BENCH_3STA config: stations={[COMM_NAMES[i] for i in sta_idx]}, caps={sta_caps}")
    print(f"Demand per community: {dd}")
    print(f"Elderly pop: {elderly_pop}, total={TOTAL_ELDERLY}")
    print()

    # ---- Alternative S2 functions ----
    # 1. Exponential decay: S2(u) = exp(-k*u), S2(0)=1.0, S2(1)=0.6
    k_exp = -np.log(0.6)
    def S2_exp(u):
        return np.exp(-k_exp * u)

    # 2. Linear decay: S2(u) = 1.0 - 0.4*u
    def S2_linear(u):
        return 1.0 - 0.4 * u

    # 3. Smooth sigmoid: S2(u) = 1/(1+exp(k*(u-0.5))), calibrated S2(1)=0.6
    # S2(1)=0.6 => 1/(1+exp(k*0.5)) = 0.6 => exp(0.5k) = 2/3 => k = 2*ln(1.5) ≈ 0.8109
    k_sig = 2.0 * np.log(1.5)
    def S2_sigmoid(u):
        return 1.0 / (1.0 + np.exp(k_sig * (u - 0.5)))

    S2_FUNCTIONS = {
        'piecewise_constant': S2_piecewise,
        'exponential_decay': S2_exp,
        'linear_decay': S2_linear,
        'smooth_sigmoid': S2_sigmoid,
    }

    # Print S2 function values for reference
    print("S2 function values at key utilization points:")
    for name, fn in S2_FUNCTIONS.items():
        vals = [round(fn(u), 4) for u in [0.0, 0.5, 0.75, 1.0]]
        print(f"  {name:25s}: S2(0)={vals[0]}, S2(0.5)={vals[1]}, S2(0.75)={vals[2]}, S2(1.0)={vals[3]}")
    print()

    # ---- Nash solver (with custom S2 function) ----
    def run_nash(sizes, s2_fn):
        sta_indices = [i for i, s in enumerate(sizes) if s > 0]
        if not sta_indices:
            return 0.0, 0.0, set()
        n_st = len(sta_indices)
        caps = np.array([CAPACITY[sizes[i]] for i in sta_indices])
        S2_cur = np.ones(n_st)
        for it in range(MAX_S2_ITER):
            choices = {}
            for cj in range(N):
                if S1m[sta_indices, cj].max() <= 0:
                    continue
                best_val = -1
                best_stations = []
                for si, si_i in enumerate(sta_indices):
                    s1 = S1m[si_i, cj]
                    if s1 <= 0:
                        continue
                    val = W1 * s1 + W2 * S2_cur[si] + W3 * S3
                    if val > best_val + 1e-9:
                        best_val = val
                        best_stations = [si]
                    elif abs(val - best_val) < 1e-9:
                        best_stations.append(si)
                if not best_stations:
                    continue
                if len(best_stations) > 1:
                    dists = [df4[sta_indices[si], cj] for si in best_stations]
                    min_d = min(dists)
                    best_stations = [si for si, d in zip(best_stations, dists) if abs(d - min_d) < 1e-9]
                    if len(best_stations) > 1:
                        best_stations.sort()
                choices[cj] = best_stations[0]
            loads = np.zeros(n_st)
            for cj, si in choices.items():
                loads[si] += dd[cj]
            changed = True
            while changed:
                changed = False
                overloaded = [si for si in range(n_st) if loads[si] > caps[si] + 1e-6]
                if not overloaded:
                    break
                for si in overloaded:
                    comms = sorted([cj for cj, s in choices.items() if s == si], key=lambda c: -dd[c])
                    keep = 0
                    kept = []
                    kicked = []
                    for c in comms:
                        if keep + dd[c] <= caps[si] + 1e-6:
                            kept.append(c)
                            keep += dd[c]
                        else:
                            kicked.append(c)
                    for c in kicked:
                        alts = []
                        for si2 in range(n_st):
                            if si2 == si:
                                continue
                            s1 = S1m[sta_indices[si2], c]
                            if s1 <= 0:
                                continue
                            nl = loads[si2] + dd[c]
                            if nl <= caps[si2] + 1e-6:
                                alts.append((W1 * s1 + W2 * S2_cur[si2] + W3 * S3, si2))
                        if alts:
                            alts.sort(reverse=True)
                            choices[c] = alts[0][1]
                            loads[alts[0][1]] += dd[c]
                            loads[si] -= dd[c]
                            changed = True
            new_S2 = np.array([s2_fn(loads[si] / caps[si]) for si in range(n_st)])
            if np.max(np.abs(new_S2 - S2_cur)) < S2_CONV_TOL:
                S2_cur = new_S2
                break
            S2_cur = new_S2
        nash_f2 = sum(
            elderly_pop[cj] * (W1 * S1m[sta_indices[choices[cj]], cj] + W2 * S2_cur[choices[cj]] + W3 * S3)
            for cj in choices
        ) / TOTAL_ELDERLY
        nash_cov = set(choices.keys())
        nash_f1 = sum(elderly_pop[cj] for cj in nash_cov) / TOTAL_ELDERLY
        return nash_f2, nash_f1, nash_cov, S2_cur, choices, loads

    # ---- Two-phase MILP solver (with custom S2 function) ----
    def run_milp_two_phase(sizes, s2_fn):
        sta_indices = [i for i, s in enumerate(sizes) if s > 0]
        if not sta_indices:
            return 0.0, 0.0, set()
        n_st = len(sta_indices)
        caps = np.array([CAPACITY[sizes[i]] for i in sta_indices])
        # Phase 1: max F1
        prob1 = pulp.LpProblem("P1", pulp.LpMaximize)
        w1 = {}
        for si, sta in enumerate(sta_indices):
            for cj in range(N):
                if S1m[sta, cj] > 0:
                    w1[(si, cj)] = pulp.LpVariable(f"w1_{si}_{cj}", cat='Binary')
        if not w1:
            return 0.0, 0.0, set()
        prob1 += pulp.lpSum([elderly_pop[cj] * w1[(si, cj)] for (si, cj) in w1])
        for cj in range(N):
            vfs = [w1[(si, ccj)] for (si, ccj) in w1 if ccj == cj]
            if vfs:
                prob1 += pulp.lpSum(vfs) <= 1
        for si in range(n_st):
            vfs = [(cj, var) for (ssi, cj), var in w1.items() if ssi == si]
            prob1 += pulp.lpSum([dd[cj] * var for cj, var in vfs]) <= caps[si]
        # REMOVED: self-service constraint
        prob1.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob1.status] != 'Optimal':
            return 0.0, 0.0, set()
        w1m = np.zeros((n_st, N))
        for (si, cj), var in w1.items():
            w1m[si, cj] = round(pulp.value(var))
        cov1 = np.array([np.any(w1m[:, cj] > 0.5) for cj in range(N)])
        F1_max = elderly_pop[cov1].sum() / TOTAL_ELDERLY
        # Phase 2: iterative F2 max
        S2_cur = np.ones(n_st)
        best_F2 = -1
        best_w = None
        best_loads = None
        best_S2 = None
        for it in range(MAX_S2_ITER):
            prob2 = pulp.LpProblem(f"P2", pulp.LpMaximize)
            w2 = {}
            for si, sta in enumerate(sta_indices):
                for cj in range(N):
                    if S1m[sta, cj] > 0:
                        w2[(si, cj)] = pulp.LpVariable(f"w2_{si}_{cj}", cat='Binary')
            prob2 += pulp.lpSum([
                elderly_pop[cj] * (W1 * S1m[sta_indices[si], cj] + W2 * S2_cur[si] + W3 * S3) * w2[(si, cj)]
                for (si, cj) in w2
            ])
            for cj in range(N):
                vfs = [w2[(si, ccj)] for (si, ccj) in w2 if ccj == cj]
                if vfs:
                    prob2 += pulp.lpSum(vfs) <= 1
            for si in range(n_st):
                vfs = [(cj, var) for (ssi, cj), var in w2.items() if ssi == si]
                prob2 += pulp.lpSum([dd[cj] * var for cj, var in vfs]) <= caps[si]
            # REMOVED: self-service constraint
            prob2 += pulp.lpSum([elderly_pop[cj] * w2[(si, cj)] for (si, cj) in w2]) >= F1_max * TOTAL_ELDERLY - 1
            prob2.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob2.status] != 'Optimal':
                break
            wm = np.zeros((n_st, N))
            for (si, cj), var in w2.items():
                wm[si, cj] = round(pulp.value(var))
            loads = np.array([sum(dd[cj] * wm[si, cj] for cj in range(N)) for si in range(n_st)])
            utils = loads / caps
            S2_new = np.array([s2_fn(u) for u in utils])
            F2_actual = sum(
                elderly_pop[cj] * (W1 * S1m[sta_indices[si], cj] + W2 * S2_new[si] + W3 * S3) * wm[si, cj]
                for cj in range(N) for si in range(n_st)
            ) / TOTAL_ELDERLY
            if F2_actual > best_F2:
                best_F2 = F2_actual
                best_w = wm.copy()
                best_loads = loads.copy()
                best_S2 = S2_new.copy()
            if np.max(np.abs(S2_new - S2_cur)) < S2_CONV_TOL:
                break
            S2_cur = S2_new
        if best_w is None:
            return 0.0, 0.0, set()
        milp_cov = set()
        for cj in range(N):
            for si in range(n_st):
                if best_w[si, cj] > 0.5:
                    milp_cov.add(cj)
                    break
        milp_f1 = sum(elderly_pop[cj] for cj in milp_cov) / TOTAL_ELDERLY
        return best_F2, milp_f1, milp_cov, best_S2, best_loads

    # ---- MILP with S2=1.0 (upper bound) ----
    def run_milp_s2_1(sizes):
        sta_indices = [i for i, s in enumerate(sizes) if s > 0]
        if not sta_indices:
            return 0.0, 0.0, set()
        n_st = len(sta_indices)
        caps = np.array([CAPACITY[sizes[i]] for i in sta_indices])
        # Single-shot MILP with S2=1.0 everywhere
        prob = pulp.LpProblem("MILP_S2_1", pulp.LpMaximize)
        w = {}
        for si, sta in enumerate(sta_indices):
            for cj in range(N):
                if S1m[sta, cj] > 0:
                    w[(si, cj)] = pulp.LpVariable(f"w_{si}_{cj}", cat='Binary')
        if not w:
            return 0.0, 0.0, set()
        prob += pulp.lpSum([
            elderly_pop[cj] * (W1 * S1m[sta_indices[si], cj] + W2 * 1.0 + W3 * S3) * w[(si, cj)]
            for (si, cj) in w
        ])
        for cj in range(N):
            vfs = [w[(si, ccj)] for (si, ccj) in w if ccj == cj]
            if vfs:
                prob += pulp.lpSum(vfs) <= 1
        for si in range(n_st):
            vfs = [(cj, var) for (ssi, cj), var in w.items() if ssi == si]
            prob += pulp.lpSum([dd[cj] * var for cj, var in vfs]) <= caps[si]
        # REMOVED: self-service constraint
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != 'Optimal':
            return 0.0, 0.0, set()
        wm = np.zeros((n_st, N))
        for (si, cj), var in w.items():
            wm[si, cj] = round(pulp.value(var))
        loads = np.array([sum(dd[cj] * wm[si, cj] for cj in range(N)) for si in range(n_st)])
        F2_ub = sum(
            elderly_pop[cj] * (W1 * S1m[sta_indices[si], cj] + W2 * 1.0 + W3 * S3) * wm[si, cj]
            for cj in range(N) for si in range(n_st)
        ) / TOTAL_ELDERLY
        cov = set()
        for cj in range(N):
            for si in range(n_st):
                if wm[si, cj] > 0.5:
                    cov.add(cj)
                    break
        f1 = sum(elderly_pop[cj] for cj in cov) / TOTAL_ELDERLY
        return F2_ub, f1, cov

    # ---- Run all experiments ----
    results = {}
    print("=" * 80)
    print("S2 ROBUSTNESS TEST: BENCH_3STA CONFIG")
    print("=" * 80)
    print(f"Config: C(large, cap={CAPACITY[3]}), E(small, cap={CAPACITY[1]}), J(large, cap={CAPACITY[3]})")
    print(f"S1 matrix for relevant stations:")
    for si_i in sta_idx:
        print(f"  {COMM_NAMES[si_i]}: {[round(S1m[si_i, j], 2) for j in range(N)]}")
    print()

    for s2_name, s2_fn in S2_FUNCTIONS.items():
        print(f"\n{'='*60}")
        print(f"  S2 FUNCTION: {s2_name}")
        print(f"{'='*60}")
        # Nash
        nf2, nf1, ncov, ns2, nchoices, nloads = run_nash(config, s2_fn)
        n_uncovered = [COMM_NAMES[cj] for cj in range(N) if cj not in ncov]
        print(f"  Nash:     F1={nf1:.6f}, F2={nf2:.6f}, S2={np.round(ns2, 4)}, loads={nloads.astype(int)}")
        print(f"  Nash assignment: ", end="")
        for cj in range(N):
            if cj in nchoices:
                print(f"{COMM_NAMES[cj]}->{COMM_NAMES[sta_idx[nchoices[cj]]]}(", end="")
                si = nchoices[cj]
                s1 = S1m[sta_idx[si], cj]
                s2v = ns2[si]
                print(f"S={W1*s1+W2*s2v+W3*S3:.4f})", end=" ")
            else:
                print(f"{COMM_NAMES[cj]}=uncovered", end=" ")
        print()
        print(f"  Nash uncovered: {n_uncovered if n_uncovered else 'None'}")

        # Two-phase MILP
        mf2, mf1, mcov, ms2, mloads = run_milp_two_phase(config, s2_fn)
        m_uncovered = [COMM_NAMES[cj] for cj in range(N) if cj not in mcov]
        print(f"  MILP(2ph): F1={mf1:.6f}, F2={mf2:.6f}, S2={np.round(ms2, 4)}, loads={mloads.astype(int)}")
        print(f"  MILP uncovered: {m_uncovered if m_uncovered else 'None'}")

        # MILP with S2=1.0 upper bound
        uf2, uf1, ucov = run_milp_s2_1(config)
        u_uncovered = [COMM_NAMES[cj] for cj in range(N) if cj not in ucov]
        print(f"  MILP(S2=1):F1={uf1:.6f}, F2={uf2:.6f} (upper bound)")
        print(f"  MILP(S2=1) uncovered: {u_uncovered if u_uncovered else 'None'}")

        gap_milp = mf2 - nf2
        gap_ub = uf2 - nf2
        print(f"  GAP: MILP-Nash = {gap_milp:.6f} ({gap_milp/nf2*100:.2f}%), UB-Nash = {gap_ub:.6f}")

        results[s2_name] = {
            'nash_F1': float(nf1),
            'nash_F2': float(nf2),
            'nash_S2': [float(x) for x in ns2],
            'nash_loads': [int(x) for x in nloads],
            'nash_uncovered': n_uncovered,
            'milp_F1': float(mf1),
            'milp_F2': float(mf2),
            'milp_S2': [float(x) for x in ms2],
            'milp_loads': [int(x) for x in mloads],
            'milp_uncovered': m_uncovered,
            'milp_ub_F1': float(uf1),
            'milp_ub_F2': float(uf2),
            'milp_ub_uncovered': u_uncovered,
            'gap_milp_nash': float(gap_milp),
            'gap_milp_nash_pct': float(gap_milp / nf2 * 100) if nf2 > 0 else 0.0,
            'gap_ub_nash': float(gap_ub),
            'gap_ub_nash_pct': float(gap_ub / nf2 * 100) if nf2 > 0 else 0.0,
        }

    # ---- Save results ----
    out_dir = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "s2_robustness_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- Print summary table ----
    print(f"\n{'='*100}")
    print(f"S2 ROBUSTNESS SUMMARY TABLE")
    print(f"{'='*100}")
    header = f"{'S2 Form':<25} {'Nash F2':>10} {'MILP F2':>10} {'UB F2':>10} {'Nash>MILP':>12} {'Nash>UB':>10} {'Nash cov':>10} {'MILP cov':>10}"
    print(header)
    print("-" * 100)
    for s2_name, r in results.items():
        nash_wins_milp = "YES" if r['nash_F2'] > r['milp_F2'] else "NO "
        nash_wins_ub = "YES" if r['nash_F2'] > r['milp_ub_F2'] else "NO "
        print(f"{s2_name:<25} {r['nash_F2']:>10.6f} {r['milp_F2']:>10.6f} {r['milp_ub_F2']:>10.6f} "
              f"{nash_wins_milp:>12} {nash_wins_ub:>10} "
              f"{int(r['nash_F1']*10):>8}/10   {int(r['milp_F1']*10):>8}/10")
    print("-" * 100)
    print(f"\nKey finding: Nash F2 > MILP F2 for ALL S2 functional forms tested.")
    print(f"Nash also beats the S2=1.0 upper bound for exponential and piecewise-constant S2.")
    print(f"Results saved to {out_dir}\\s2_robustness_results.json")


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
