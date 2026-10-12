"""
CRITICAL EXPERIMENT: Joint MILP vs Nash vs Two-Phase MILP
Test on 500 sampled configs to determine paper's positioning.
"""
import pandas as pd, numpy as np, os, re, warnings, random
warnings.filterwarnings('ignore')
import pulp

DATA_DIR = os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data")
Q13_DIR = os.environ.get("CFLG_Q13_OUT", r"D:/competition_data/q13")
COMM_NAMES = ['A','B','C','D','E','F','G','H','I','J']
N = 10; W1,W2,W3 = 0.2,0.3,0.5; S3 = 0.60
MAX_S2_ITER = 10; S2_CONV_TOL = 0.01

df1 = pd.read_excel(os.path.join(DATA_DIR,"附件1：小区基础数据.xlsx"), sheet_name=0, header=1)
df1.columns = ['小区','总人口','老年人口','自理','半失能','失能','人均收入']
elderly_pop = df1['老年人口'].values.astype(float); TOTAL_ELDERLY = elderly_pop.sum()
df_q13 = pd.read_csv(os.path.join(Q13_DIR,'问题1.3_实际月服务需求.csv'), encoding='utf-8-sig')
dd = np.round(df_q13.iloc[:,1:7].values.astype(float).sum(axis=1)/30.0).astype(int)
df4 = pd.read_excel(os.path.join(DATA_DIR,"附件4：小区间距离矩阵.xlsx"), header=1, index_col=0).values.astype(float)
df5 = pd.read_excel(os.path.join(DATA_DIR,"附件5：满意度评分规则.xlsx"), header=None)
S1_T=[]; S2_T=[]
for idx in range(len(df5)):
    c1=str(df5.iloc[idx,0]) if pd.notna(df5.iloc[idx,0]) else ''
    c2=str(df5.iloc[idx,1]) if pd.notna(df5.iloc[idx,1]) else ''
    m=re.search(r'[≤]([\d.]+)m.*S1=([\d.]+)',c2)
    if m and ('S1' in c1 or 'S1' in c2): S1_T.append((float(m.group(1)),float(m.group(2))))
    m=re.search(r'[≤]([\d.]+).*S2=([\d.]+)',c2)
    if m and ('S2' in c1 or 'S2' in c2): S2_T.append((float(m.group(1)),float(m.group(2))))
S1_T.sort(); S2_T.sort()
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
df3 = pd.read_excel(os.path.join(DATA_DIR,"附件3：服务站建设与运营成本.xlsx"), header=1, nrows=3)
df3.columns=['规模','建设成本_万元','日运营成本_元','日容量_人次']
CAPACITY = np.array([0]+df3['日容量_人次'].astype(float).tolist(),dtype=float)
BUILD_COST = np.array([0]+df3['建设成本_万元'].astype(float).tolist(),dtype=float)

def run_nash(sizes):
    sta_idx=[i for i,s in enumerate(sizes) if s>0]
    if not sta_idx: return 0,0,set()
    n_sta=len(sta_idx); sta_caps=np.array([CAPACITY[sizes[i]] for i in sta_idx])
    S2_cur=np.ones(n_sta); choices={}
    for it in range(MAX_S2_ITER):
        for cj in range(N):
            if S1m[sta_idx,cj].max()<=0: continue
            best_val=-1; best_stations=[]
            for si,si_i in enumerate(sta_idx):
                s1=S1m[si_i,cj]
                if s1<=0: continue
                val=W1*s1+W2*S2_cur[si]+W3*S3
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
                        if nl<=sta_caps[si2]+1e-6: alts.append((W1*s1+W2*S2_cur[si2]+W3*S3,si2))
                    if alts:
                        alts.sort(reverse=True); choices[c]=alts[0][1]
                        loads[alts[0][1]]+=dd[c]; loads[si]-=dd[c]; changed=True
        new_S2=np.array([S2_fn(loads[si]/sta_caps[si]) for si in range(n_sta)])
        if np.max(np.abs(new_S2-S2_cur))<S2_CONV_TOL: S2_cur=new_S2; break
        S2_cur=new_S2
    nash_f2=sum(elderly_pop[cj]*(W1*S1m[sta_idx[choices[cj]],cj]+W2*S2_cur[choices[cj]]+W3*S3) for cj in choices)/TOTAL_ELDERLY
    nash_cov=set(choices.keys())
    nash_f1=sum(elderly_pop[cj] for cj in nash_cov)/TOTAL_ELDERLY
    return nash_f2, nash_f1, nash_cov

def run_milp_two_phase(sizes):
    sta_idx=[i for i,s in enumerate(sizes) if s>0]
    if not sta_idx: return 0,0,set()
    n_sta=len(sta_idx); sta_caps=np.array([CAPACITY[sizes[i]] for i in sta_idx])
    prob1=pulp.LpProblem("P1",pulp.LpMaximize)
    w1={}
    for si,sta in enumerate(sta_idx):
        for cj in range(N):
            if S1m[sta,cj]>0: w1[(si,cj)]=pulp.LpVariable(f"w1_{si}_{cj}",cat='Binary')
    if not w1: return 0,0,set()
    prob1+=pulp.lpSum([elderly_pop[cj]*w1[(si,cj)] for (si,cj) in w1])
    for cj in range(N):
        vfs=[w1[(si,ccj)] for (si,ccj) in w1 if ccj==cj]
        if vfs: prob1+=pulp.lpSum(vfs)<=1
    for si in range(n_sta):
        vfs=[(cj,var) for (ssi,cj),var in w1.items() if ssi==si]
        prob1+=pulp.lpSum([dd[cj]*var for cj,var in vfs])<=sta_caps[si]
    # 修复：删除强制自服务约束，允许求解器自由分配
    prob1.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob1.status]!='Optimal': return 0,0,set()
    w1m=np.zeros((n_sta,N))
    for (si,cj),var in w1.items(): w1m[si,cj]=round(pulp.value(var))
    cov1=np.array([np.any(w1m[:,cj]>0.5) for cj in range(N)])
    F1_max=elderly_pop[cov1].sum()/TOTAL_ELDERLY
    S2_cur=np.ones(n_sta); best_F2=-1; best_w=None
    for it in range(MAX_S2_ITER):
        prob2=pulp.LpProblem(f"P2",pulp.LpMaximize)
        w2={}
        for si,sta in enumerate(sta_idx):
            for cj in range(N):
                if S1m[sta,cj]>0: w2[(si,cj)]=pulp.LpVariable(f"w2_{si}_{cj}",cat='Binary')
        prob2+=pulp.lpSum([elderly_pop[cj]*(W1*S1m[sta_idx[si],cj]+W2*S2_cur[si]+W3*S3)*w2[(si,cj)] for (si,cj) in w2])
        for cj in range(N):
            vfs=[w2[(si,ccj)] for (si,ccj) in w2 if ccj==cj]
            if vfs: prob2+=pulp.lpSum(vfs)<=1
        for si in range(n_sta):
            vfs=[(cj,var) for (ssi,cj),var in w2.items() if ssi==si]
            prob2+=pulp.lpSum([dd[cj]*var for cj,var in vfs])<=sta_caps[si]
        # 修复：删除强制自服务约束
        prob2+=pulp.lpSum([elderly_pop[cj]*w2[(si,cj)] for (si,cj) in w2])>=F1_max*TOTAL_ELDERLY-1
        prob2.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob2.status]!='Optimal': break
        wm=np.zeros((n_sta,N))
        for (si,cj),var in w2.items(): wm[si,cj]=round(pulp.value(var))
        loads=np.array([sum(dd[cj]*wm[si,cj] for cj in range(N)) for si in range(n_sta)])
        utils=loads/sta_caps
        S2_new=np.array([S2_fn(u) for u in utils])
        F2_actual=sum(elderly_pop[cj]*(W1*S1m[sta_idx[si],cj]+W2*S2_new[si]+W3*S3)*wm[si,cj] for cj in range(N) for si in range(n_sta))/TOTAL_ELDERLY
        if F2_actual>best_F2: best_F2=F2_actual; best_w=wm.copy()
        if np.max(np.abs(S2_new-S2_cur))<S2_CONV_TOL: break
        S2_cur=S2_new
    if best_w is None: return 0,0,set()
    milp_cov=set()
    for cj in range(N):
        for si in range(n_sta):
            if best_w[si,cj]>0.5: milp_cov.add(cj); break
    milp_f1=sum(elderly_pop[cj] for cj in milp_cov)/TOTAL_ELDERLY
    return best_F2, milp_f1, milp_cov

def run_milp_joint_weighted(sizes, alpha=0.5):
    """Joint MILP: max alpha*F1 + (1-alpha)*F2 simultaneously"""
    sta_idx=[i for i,s in enumerate(sizes) if s>0]
    if not sta_idx: return 0,0,set()
    n_sta=len(sta_idx); sta_caps=np.array([CAPACITY[sizes[i]] for i in sta_idx])
    
    # We need to handle S2 feedback. Use iterative approach but with JOINT objective.
    # In each iteration: max alpha*F1 + (1-alpha)*F2 with CURRENT S2 values
    # Then update S2 from new loads. Track best across iterations.
    
    S2_cur=np.ones(n_sta); best_F2=-1; best_F1=0; best_w=None; best_cov=None
    
    for it in range(MAX_S2_ITER):
        prob=pulp.LpProblem(f"Joint_{it}",pulp.LpMaximize)
        w={}
        for si,sta in enumerate(sta_idx):
            for cj in range(N):
                if S1m[sta,cj]>0: w[(si,cj)]=pulp.LpVariable(f"wj_{si}_{cj}",cat='Binary')
        if not w: continue
        
        # Joint objective: alpha * (coverage) + (1-alpha) * (satisfaction)
        # Coverage: sum of pop_j * w_ij for all covered j
        # Satisfaction: sum of pop_j * S_ij * w_ij
        obj_terms = []
        for cj in range(N):
            for si,sta in enumerate(sta_idx):
                if S1m[sta,cj]>0:
                    s1=S1m[sta_idx[si],cj]
                    s2=S2_cur[si]
                    S_j=W1*s1+W2*s2+W3*S3
                    # Contribution to F1: elderly_pop[cj] * w (if assigned)
                    # Contribution to F2: elderly_pop[cj] * S_j * w
                    # Joint: alpha * elderly_pop[cj] * w + (1-alpha) * elderly_pop[cj] * S_j * w
                    # = elderly_pop[cj] * w * (alpha + (1-alpha)*S_j)
                    coeff = elderly_pop[cj] * (alpha + (1-alpha)*S_j)
                    obj_terms.append(coeff * w[(si,cj)])
        
        prob += pulp.lpSum(obj_terms)
        
        for cj in range(N):
            vfs=[w[(si,ccj)] for (si,ccj) in w if ccj==cj]
            if vfs: prob+=pulp.lpSum(vfs)<=1
        for si in range(n_sta):
            vfs=[(cj,var) for (ssi,cj),var in w.items() if ssi==si]
            prob+=pulp.lpSum([dd[cj]*var for cj,var in vfs])<=sta_caps[si]
        # 修复：删除强制自服务约束
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status]!='Optimal': continue
        
        wm=np.zeros((n_sta,N))
        for (si,cj),var in w.items(): wm[si,cj]=round(pulp.value(var))
        loads=np.array([sum(dd[cj]*wm[si,cj] for cj in range(N)) for si in range(n_sta)])
        utils=loads/sta_caps
        S2_new=np.array([S2_fn(u) for u in utils])
        
        # Compute actual F1 and F2 with NEW S2
        cov=np.array([np.any(wm[:,cj]>0.5) for cj in range(N)])
        F1_actual=elderly_pop[cov].sum()/TOTAL_ELDERLY
        F2_actual=sum(elderly_pop[cj]*(W1*S1m[sta_idx[si],cj]+W2*S2_new[si]+W3*S3)*wm[si,cj] for cj in range(N) for si in range(n_sta))/TOTAL_ELDERLY
        
        if F2_actual>best_F2:
            best_F2=F2_actual; best_F1=F1_actual; best_w=wm.copy()
            cov_set=set()
            for cj in range(N):
                for si in range(n_sta):
                    if wm[si,cj]>0.5: cov_set.add(cj); break
            best_cov=cov_set
        
        if np.max(np.abs(S2_new-S2_cur))<S2_CONV_TOL: break
        S2_cur=S2_new
    
    if best_w is None: return 0,0,set()
    return best_F2, best_F1, best_cov

# Enumerate configs
configs=[]
for val in range(4**N):
    tmp=val; sizes=[0]*N; cost=0; ok=True
    for i in range(N):
        s=tmp%4; tmp//=4; sizes[i]=s; cost+=BUILD_COST[s]
        if cost>120: ok=False; break
    if ok: configs.append(sizes)

random.seed(42); random.shuffle(configs)
sample=configs[:500]

print("="*80)
print("CRITICAL EXPERIMENT: Joint MILP vs Nash vs Two-Phase MILP")
print("="*80)
print(f"Testing {len(sample)} configs...")

nash_win=0; joint_win=0; two_phase_win=0; ties=0
results=[]

for cfg in sample:
    nf2,nf1,_=run_nash(cfg)
    mf2,mf1,_=run_milp_two_phase(cfg)
    jf2,jf1,_=run_milp_joint_weighted(cfg, alpha=0.5)
    
    if nf2<=0 and mf2<=0 and jf2<=0: continue
    
    results.append({
        'config': cfg,
        'nash_F2': nf2, 'nash_F1': nf1,
        'milp_F2': mf2, 'milp_F1': mf1,
        'joint_F2': jf2, 'joint_F1': jf1
    })
    
    # Determine winner
    vals = {'nash': nf2, 'milp': mf2, 'joint': jf2}
    best_key = max(vals, key=vals.get)
    if vals[best_key] > max(v for k,v in vals.items() if k!=best_key) + 0.001:
        if best_key == 'nash': nash_win += 1
        elif best_key == 'milp': two_phase_win += 1
        else: joint_win += 1
    else:
        ties += 1

total = nash_win + joint_win + two_phase_win + ties
print(f"\n{'='*80}")
print(f"RESULTS ({len(results)} valid configs)")
print(f"{'='*80}")
print(f"Nash wins:     {nash_win} ({100*nash_win/total:.1f}%)")
print(f"Two-Phase MILP wins: {two_phase_win} ({100*two_phase_win/total:.1f}%)")
print(f"Joint MILP wins:     {joint_win} ({100*joint_win/total:.1f}%)")
print(f"Ties:              {ties}")

# Detailed comparison
print(f"\n{'='*80}")
print(f"DETAILED COMPARISON")
print(f"{'='*80}")
print(f"{'Config':<25} {'Nash_F2':>8} {'2Ph_F2':>8} {'Joint_F2':>8} {'Winner':>10}")
print("-"*80)

for r in results[:30]:
    cfg = r['config']
    cfg_str = '+'.join(f"{COMM_NAMES[i]}({cfg[i]})" for i in range(N) if cfg[i]>0)
    nf2=r['nash_F2']; mf2=r['milp_F2']; jf2=r['joint_F2']
    vals = {'Nash': nf2, '2Ph': mf2, 'Joint': jf2}
    winner = max(vals, key=vals.get)
    print(f"{cfg_str:<25} {nf2:>8.4f} {mf2:>8.4f} {jf2:>8.4f} {winner:>10}")

# Save
out = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
import json
with open(os.path.join(out,"joint_milp_experiment.json"),'w',encoding='utf-8') as f:
    json.dump({
        'total': total,
        'nash_wins': nash_win,
        'two_phase_wins': two_phase_win,
        'joint_wins': joint_win,
        'ties': ties,
        'results': results[:100]  # First 100 detailed
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out}\\joint_milp_experiment.json")
