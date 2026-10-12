"""
Gamma Sensitivity Experiment: Full re-run with exact parameters from manuscript.
Tests Nash vs MILP across gamma values on the REAL BENCH_3STA configuration.
"""
import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')
import pulp


def main():

    DATA_DIR = os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data")
    Q13_DIR = os.environ.get("CFLG_Q13_OUT", r"D:/competition_data/q13")
    OUT_DIR = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

    COMM_NAMES = ['A','B','C','D','E','F','G','H','I','J']
    N = 10
    MAX_S2_ITER = 10
    S2_CONV_TOL = 0.01

    # Load data
    df1 = pd.read_excel(os.path.join(DATA_DIR,"附件1：小区基础数据.xlsx"), sheet_name=0, header=1)
    df1.columns = ['小区','总人口','老年人口','自理','半失能','失能','人均收入']
    elderly_pop = df1['老年人口'].values.astype(float)
    TOTAL_ELDERLY = elderly_pop.sum()

    df_q13 = pd.read_csv(os.path.join(Q13_DIR, '问题1.3_实际月服务需求.csv'), encoding='utf-8-sig')
    dd = np.round(df_q13.iloc[:, 1:7].values.astype(float).sum(axis=1) / 30.0).astype(int)

    df4 = pd.read_excel(os.path.join(DATA_DIR,"附件4：小区间距离矩阵.xlsx"), header=1, index_col=0).values.astype(float)
    df5 = pd.read_excel(os.path.join(DATA_DIR,"附件5：满意度评分规则.xlsx"), header=None)
    S1_T=[]; S2_T=[]
    for idx in range(len(df5)):
        c1=str(df5.iloc[idx,0]) if pd.notna(df5.iloc[idx,0]) else ''
        c2=str(df5.iloc[idx,1]) if pd.notna(df5.iloc[idx,1]) else ''
        import re
        m=re.search(r'[≤]([\d.]+)m.*S1=([\d.]+)',c2)
        if m and ('S1' in c1 or 'S1' in c2): S1_T.append((float(m.group(1)),float(m.group(2))))
        m=re.search(r'[≤]([\d.]+).*S2=([\d.]+)',c2)
        if m and ('S2' in c1 or 'S2' in c2): S2_T.append((float(m.group(1)),float(m.group(2))))
    S1_T.sort(); S2_T.sort()

    CAPACITY = np.array([0, 1000.0, 2000.0, 3000.0], dtype=float)
    S3 = 0.60

    def S1(d):
        for u,v in S1_T:
            if d<=u: return v
        return 0.0
    def S2_fn(u):
        for u2,v in S2_T:
            if u<=u2: return v
        return S2_T[-1][1]

    S1m = np.zeros((N,N))
    for i in range(N):
        for j in range(N): S1m[i,j]=S1(df4[i,j])

    # BENCH_3STA configuration
    C_E_J_SIZES = [0, 0, 3, 0, 1, 0, 0, 0, 0, 3]

    def compute_nash_gamma(sizes, gamma_val):
        alpha, beta = 0.2, 0.3
        w1, w2, w3 = alpha, beta, gamma_val
        sta_idx = [i for i,s in enumerate(sizes) if s>0]
        n_sta = len(sta_idx)
        sta_caps = np.array([CAPACITY[sizes[i]] for i in sta_idx])

        S2_cur = np.ones(n_sta)
        choices = {}
        for it in range(MAX_S2_ITER):
            for cj in range(N):
                if S1m[sta_idx, cj].max()<=0: continue
                best_val=-1; best_stations=[]
                for si,si_i in enumerate(sta_idx):
                    s1=S1m[si_i,cj]
                    if s1<=0: continue
                    val=w1*s1+w2*S2_cur[si]+w3*S3
                    if val>best_val+1e-9: best_val=val; best_stations=[si]
                    elif abs(val-best_val)<1e-9: best_stations.append(si)
                if not best_stations: continue
                if len(best_stations)>1:
                    dists=[df4[sta_idx[si],cj] for si in best_stations]
                    min_d=min(dists)
                    best_stations=[si for si,d in zip(best_stations,dists) if abs(d-min_d)<1e-9]
                    if len(best_stations)>1: best_stations.sort()
                choices[cj]=best_stations[0]

            loads=np.zeros(n_sta)
            for cj,si in choices.items(): loads[si]+=dd[cj]

            changed=True
            while changed:
                changed=False
                overloaded=[si for si in range(n_sta) if loads[si]>sta_caps[si]+1e-6]
                if not overloaded: break
                for si in overloaded:
                    comms=sorted([cj for cj,s in choices.items() if s==si], key=lambda c:-dd[c])
                    keep=0; kept=[]; kicked=[]
                    for c in comms:
                        if keep+dd[c]<=sta_caps[si]+1e-6: kept.append(c); keep+=dd[c]
                        else: kicked.append(c)
                    for c in kicked:
                        alts=[]
                        for si2 in range(n_sta):
                            if si2==si: continue
                            s1=S1m[sta_idx[si2],c]
                            if s1<=0: continue
                            nl=loads[si2]+dd[c]
                            if nl<=sta_caps[si2]+1e-6:
                                alts.append((w1*s1+w2*S2_cur[si2]+w3*S3, si2))
                        if alts:
                            alts.sort(reverse=True)
                            choices[c]=alts[0][1]
                            loads[alts[0][1]]+=dd[c]; loads[si]-=dd[c]; changed=True

            new_S2=np.array([S2_fn(loads[si]/sta_caps[si]) for si in range(n_sta)])
            if np.max(np.abs(new_S2-S2_cur))<S2_CONV_TOL:
                S2_cur=new_S2; break
            S2_cur=new_S2

        covered=set(choices.keys())
        F1=sum(elderly_pop[cj] for cj in covered)/TOTAL_ELDERLY
        F2=sum(elderly_pop[cj]*(w1*S1m[sta_idx[choices[cj]],cj]+w2*S2_cur[choices[cj]]+w3*S3)
               for cj in covered)/TOTAL_ELDERLY
        return F1, F2, covered

    def compute_milp_gamma(sizes, gamma_val):
        alpha, beta = 0.2, 0.3
        w1, w2, w3 = alpha, beta, gamma_val
        sta_idx = [i for i,s in enumerate(sizes) if s>0]
        n_sta = len(sta_idx)
        sta_caps = np.array([CAPACITY[sizes[i]] for i in sta_idx])

        # Phase 1: max F1
        prob1 = pulp.LpProblem("P1", pulp.LpMaximize)
        w1var = {}
        for si,sta in enumerate(sta_idx):
            for cj in range(N):
                if S1m[sta,cj]>0:
                    w1var[(si,cj)] = pulp.LpVariable(f"w1_{si}_{cj}", cat='Binary')
        if not w1var: return 0.0, 0.0
        prob1 += pulp.lpSum([elderly_pop[cj]*w1var[(si,cj)] for (si,cj) in w1var])
        for cj in range(N):
            vfs=[w1var[(si,ccj)] for (si,ccj) in w1var if ccj==cj]
            if vfs: prob1 += pulp.lpSum(vfs) <= 1
        for si in range(n_sta):
            vfs=[(cj,var) for (ssi,cj),var in w1var.items() if ssi==si]
            prob1 += pulp.lpSum([dd[cj]*var for cj,var in vfs]) <= sta_caps[si]
        for si,sta in enumerate(sta_idx):
            if (si,sta) in w1var: prob1 += w1var[(si,sta)] == 1
        prob1.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob1.status] != 'Optimal': return 0.0, 0.0

        w_mat1 = np.zeros((n_sta,N))
        for (si,cj),var in w1var.items(): w_mat1[si,cj] = round(pulp.value(var))
        cov1 = np.array([np.any(w_mat1[:,cj]>0.5) for cj in range(N)])
        F1_max = elderly_pop[cov1].sum()/TOTAL_ELDERLY

        # Phase 2: iterative F2 with locked Phase1 allocation
        S2_cur = np.ones(n_sta)
        best_F2 = -1
        for it in range(MAX_S2_ITER):
            prob2 = pulp.LpProblem(f"P2_{it}", pulp.LpMaximize)
            w2var = {}
            for si,sta in enumerate(sta_idx):
                for cj in range(N):
                    if S1m[sta,cj]>0:
                        w2var[(si,cj)] = pulp.LpVariable(f"w2_{si}_{cj}", cat='Binary')

            prob2 += pulp.lpSum([elderly_pop[cj]*(w1*S1m[sta_idx[si],cj]+w2*S2_cur[si]+w3*S3)*w2var[(si,cj)]
                                  for (si,cj) in w2var])
            for cj in range(N):
                vfs=[w2var[(si,ccj)] for (si,ccj) in w2var if ccj==cj]
                if vfs: prob2 += pulp.lpSum(vfs) <= 1
            for si in range(n_sta):
                vfs=[(cj,var) for (ssi,cj),var in w2var.items() if ssi==si]
                prob2 += pulp.lpSum([dd[cj]*var for cj,var in vfs]) <= sta_caps[si]
            for si,sta in enumerate(sta_idx):
                if (si,sta) in w2var: prob2 += w2var[(si,sta)] == 1
            prob2 += pulp.lpSum([elderly_pop[cj]*w2var[(si,cj)] for (si,cj) in w2var]) >= F1_max*TOTAL_ELDERLY-1

            prob2.solve(pulp.PULP_CBC_CMD(msg=0))
            if pulp.LpStatus[prob2.status] != 'Optimal': break

            wm = np.zeros((n_sta,N))
            for (si,cj),var in w2var.items(): wm[si,cj] = round(pulp.value(var))
            loads = np.array([sum(dd[cj]*wm[si,cj] for cj in range(N)) for si in range(n_sta)])
            utils = loads/sta_caps
            S2_new = np.array([S2_fn(u) for u in utils])
            F2_actual = sum(elderly_pop[cj]*(w1*S1m[sta_idx[si],cj]+w2*S2_new[si]+w3*S3)*wm[si,cj]
                             for cj in range(N) for si in range(n_sta))/TOTAL_ELDERLY
            if F2_actual > best_F2: best_F2 = F2_actual
            if np.max(np.abs(S2_new-S2_cur))<S2_CONV_TOL: break
            S2_cur = S2_new

        return F1_max, best_F2

    # Run for all gamma values
    gamma_values = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9]
    print("=" * 70)
    print("GAMMA SENSITIVITY EXPERIMENT: Full Re-run (BENCH_3STA Configuration)")
    print("Data: Real 电工杯 competition data (10 communities, 6864 elderly)")
    print("=" * 70)

    results = {}
    for gamma in gamma_values:
        nash_F1, nash_F2, nash_cov = compute_nash_gamma(C_E_J_SIZES, gamma)
        milp_F1, milp_F2 = compute_milp_gamma(C_E_J_SIZES, gamma)

        gap = nash_F2 - milp_F2
        gap_pct = gap / nash_F2 * 100 if nash_F2 > 0 else 0

        results[gamma] = {
            'nash_F1': round(nash_F1, 4),
            'nash_F2': round(nash_F2, 4),
            'milp_F1': round(milp_F1, 4),
            'milp_F2': round(milp_F2, 4),
            'gap': round(gap, 4),
            'gap_pct': round(gap_pct, 1),
        }

        print(f"\ngamma={gamma:.1f}:")
        print(f"  Nash: F1={nash_F1:.4f}, F2={nash_F2:.4f}")
        print(f"  MILP: F1={milp_F1:.4f}, F2={milp_F2:.4f}")
        print(f"  Gap:  {gap:+.4f} ({gap_pct:+.1f}%)")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Gamma':<8} {'Nash F2':<10} {'MILP F2':<10} {'Gap':<10} {'Gap%':<8}")
    print("-" * 46)
    for g, r in results.items():
        print(f"{g:<8.1f} {r['nash_F2']:<10.4f} {r['milp_F2']:<10.4f} {r['gap']:+.4f} {r['gap_pct']:+.1f}%")

    mean_gap = np.mean([r['gap'] for r in results.values()])
    print(f"\nMean absolute gap: {mean_gap:+.4f}")
    print(f"All gaps positive: {all(r['gap'] > 0 for r in results.values())}")

    # Save
    output_path = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'source': 'Real 电工杯 data, BENCH_3STA configuration', 'results': results}, f, indent=2)
    print(f"\nSaved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
