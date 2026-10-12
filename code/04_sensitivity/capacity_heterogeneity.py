"""
Scaling Experiment v2: Decouple scale and density
Two versions:
- Version A: Fixed area (density increases with n)
- Version B: Fixed density (area increases with n)
"""
import numpy as np, pandas as pd, os, re, warnings, json, random
warnings.filterwarnings('ignore')
import pulp

# ========== Constants (same as original) ==========
W1, W2, W3 = 0.2, 0.3, 0.5
S3 = 0.60
MAX_S2_ITER = 5
S2_CONV_TOL = 0.01
SEED = 2024

# Station capacity and build cost (from 附件3)
CAPACITY = np.array([0.0, 1000.0, 2000.0, 3000.0])   # small/med/large
BUILD_COST = np.array([0.0, 15.0, 25.0, 40.0])        # 万元

# S1 distance thresholds (from 附件5)
S1_T = [(300, 1.0), (500, 0.9), (650, 0.75), (1000, 0.6)]
# S2 utilization thresholds (from 附件5)
S2_T = [(0.6, 1.0), (0.75, 0.93), (0.85, 0.85), (0.95, 0.72), (1.0, 0.6)]

OUTPUT_DIR = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


# ========== Satisfaction functions ==========
def S1_fn(d):
    for u, v in S1_T:
        if d <= u:
            return v
    return 0.0

def S2_fn(u):
    for u2, v in S2_T:
        if u <= u2:
            return v
    return S2_T[-1][1]


# ========== Instance generation (NEW: density-controlled) ==========
def generate_instance_v2(n, rng, mode='fixed_area', area_km2=9.0, density_per_km2=10.0):
    """Generate synthetic instance with controlled density.
    
    mode='fixed_area': fixed 3km x 3km area, density increases with n
    mode='fixed_density': fixed density (e.g., 10 communities/km^2), area grows with n
    """
    if mode == 'fixed_area':
        # Fixed 3km x 3km area
        side_m = np.sqrt(area_km2) * 1000  # 3000m
        comm_locs = rng.uniform(0, side_m, size=(n, 2))
    elif mode == 'fixed_density':
        # Fixed density: area = n / density_per_km2 * 1000^2
        area_m2 = (n / density_per_km2) * 1e6
        side_m = np.sqrt(area_m2)
        comm_locs = rng.uniform(0, side_m, size=(n, 2))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = comm_locs[i, 0] - comm_locs[j, 0]
            dy = comm_locs[i, 1] - comm_locs[j, 1]
            dist[i, j] = np.sqrt(dx**2 + dy**2)
    
    # Random elderly populations and demands
    elderly_pop = rng.integers(400, 2001, size=n).astype(float)
    total_elderly = elderly_pop.sum()
    dd = rng.integers(20, 201, size=n).astype(int)
    
    # Precompute S1 matrix
    S1m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S1m[i, j] = S1_fn(dist[i, j])
    
    return {
        'n': n,
        'dist': dist,
        'elderly_pop': elderly_pop,
        'total_elderly': total_elderly,
        'dd': dd,
        'S1m': S1m,
        'mode': mode,
        'area_km2': area_m2 / 1e6 if mode == 'fixed_density' else area_km2,
    }


# ========== Generate all feasible configs under budget ==========
def gen_feasible_configs(n, budget, rng, max_configs=500):
    """Randomly sample feasible station-size configurations under budget."""
    configs = []
    attempts = 0
    while len(configs) < max_configs and attempts < max_configs * 50:
        attempts += 1
        sizes = rng.integers(0, 4, size=n)
        cost = sum(BUILD_COST[s] for s in sizes)
        if cost <= budget and cost > 0:
            configs.append(sizes.tolist())
    return configs


# ========== Nash Equilibrium solver ==========
def run_nash(sizes, inst):
    n = inst['n']
    S1m = inst['S1m']
    dd = inst['dd']
    elderly_pop = inst['elderly_pop']
    total_elderly = inst['total_elderly']
    dist = inst['dist']

    sta_idx = [i for i, s in enumerate(sizes) if s > 0]
    if not sta_idx:
        return 0.0, 0.0, set()
    n_sta = len(sta_idx)
    sta_caps = np.array([CAPACITY[sizes[i]] for i in sta_idx])

    S2_cur = np.ones(n_sta)
    choices = {}

    for it in range(MAX_S2_ITER):
        for cj in range(n):
            max_s1 = S1m[sta_idx, cj].max()
            if max_s1 <= 0:
                continue
            best_val = -1
            best_stations = []
            for si, si_i in enumerate(sta_idx):
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
                dists = [dist[sta_idx[si], cj] for si in best_stations]
                min_d = min(dists)
                best_stations = [si for si, d in zip(best_stations, dists) if abs(d - min_d) < 1e-9]
                if len(best_stations) > 1:
                    best_stations.sort()
            choices[cj] = best_stations[0]

        loads = np.zeros(n_sta)
        for cj, si in choices.items():
            loads[si] += dd[cj]

        changed = True
        while changed:
            changed = False
            overloaded = [si for si in range(n_sta) if loads[si] > sta_caps[si] + 1e-6]
            if not overloaded:
                break
            for si in overloaded:
                comms = sorted([cj for cj, s in choices.items() if s == si], key=lambda c: -dd[c])
                keep = 0
                kept = []
                kicked = []
                for c in comms:
                    if keep + dd[c] <= sta_caps[si] + 1e-6:
                        kept.append(c)
                        keep += dd[c]
                    else:
                        kicked.append(c)
                for c in kicked:
                    alts = []
                    for si2 in range(n_sta):
                        if si2 == si:
                            continue
                        s1 = S1m[sta_idx[si2], c]
                        if s1 <= 0:
                            continue
                        nl = loads[si2] + dd[c]
                        if nl <= sta_caps[si2] + 1e-6:
                            alts.append((W1 * s1 + W2 * S2_cur[si2] + W3 * S3, si2))
                    if alts:
                        alts.sort(reverse=True)
                        choices[c] = alts[0][1]
                        loads[alts[0][1]] += dd[c]
                        loads[si] -= dd[c]
                        changed = True

        new_S2 = np.array([S2_fn(loads[si] / sta_caps[si]) for si in range(n_sta)])
        if np.max(np.abs(new_S2 - S2_cur)) < S2_CONV_TOL:
            S2_cur = new_S2
            break
        S2_cur = new_S2

    covered = set(choices.keys())
    nash_F1 = elderly_pop[list(covered)].sum() / total_elderly
    nash_F2 = sum(
        elderly_pop[cj] * (W1 * S1m[sta_idx[choices[cj]], cj] + W2 * S2_cur[choices[cj]] + W3 * S3)
        for cj in covered
    ) / total_elderly
    return nash_F2, nash_F1, covered


# ========== Two-phase MILP solver (with self-service constraint removed) ==========
def run_milp_two_phase(sizes, inst):
    n = inst['n']
    S1m = inst['S1m']
    dd = inst['dd']
    elderly_pop = inst['elderly_pop']
    total_elderly = inst['total_elderly']

    sta_idx = [i for i, s in enumerate(sizes) if s > 0]
    if not sta_idx:
        return 0.0, 0.0, set()
    n_sta = len(sta_idx)
    sta_caps = np.array([CAPACITY[sizes[i]] for i in sta_idx])

    # Phase 1: Maximize F1 (coverage)
    prob1 = pulp.LpProblem("P1", pulp.LpMaximize)
    w1 = {}
    for si, sta in enumerate(sta_idx):
        for cj in range(n):
            if S1m[sta, cj] > 0:
                w1[(si, cj)] = pulp.LpVariable(f"w1_{si}_{cj}", cat='Binary')
    if not w1:
        return 0.0, 0.0, set()
    prob1 += pulp.lpSum([elderly_pop[cj] * w1[(si, cj)] for (si, cj) in w1])
    for cj in range(n):
        vfs = [w1[(si, ccj)] for (si, ccj) in w1 if ccj == cj]
        if vfs:
            prob1 += pulp.lpSum(vfs) <= 1
    for si in range(n_sta):
        vfs = [(cj, var) for (ssi, cj), var in w1.items() if ssi == si]
        prob1 += pulp.lpSum([dd[cj] * var for cj, var in vfs]) <= sta_caps[si]
    # REMOVED: self-service constraint w1[(si, sta)] == 1
    prob1.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob1.status] != 'Optimal':
        return 0.0, 0.0, set()
    w1m = np.zeros((n_sta, n))
    for (si, cj), var in w1.items():
        w1m[si, cj] = round(pulp.value(var))
    cov1 = np.array([np.any(w1m[:, cj] > 0.5) for cj in range(n)])
    F1_max = elderly_pop[cov1].sum() / total_elderly

    # Phase 2: Iterative MILP for F2
    S2_cur = np.ones(n_sta)
    best_F2 = -1
    best_w = None
    for it in range(MAX_S2_ITER):
        prob2 = pulp.LpProblem(f"P2_{it}", pulp.LpMaximize)
        w2 = {}
        for si, sta in enumerate(sta_idx):
            for cj in range(n):
                if S1m[sta, cj] > 0:
                    w2[(si, cj)] = pulp.LpVariable(f"w2_{si}_{cj}", cat='Binary')
        if not w2:
            break
        prob2 += pulp.lpSum([
            elderly_pop[cj] * (W1 * S1m[sta_idx[si], cj] + W2 * S2_cur[si] + W3 * S3) * w2[(si, cj)]
            for (si, cj) in w2
        ])
        for cj in range(n):
            vfs = [w2[(si, ccj)] for (si, ccj) in w2 if ccj == cj]
            if vfs:
                prob2 += pulp.lpSum(vfs) <= 1
        for si in range(n_sta):
            vfs = [(cj, var) for (ssi, cj), var in w2.items() if ssi == si]
            prob2 += pulp.lpSum([dd[cj] * var for cj, var in vfs]) <= sta_caps[si]
        # REMOVED: self-service constraint w2[(si, sta)] == 1
        prob2 += pulp.lpSum([elderly_pop[cj] * w2[(si, cj)] for (si, cj) in w2]) >= F1_max * total_elderly - 1
        prob2.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob2.status] != 'Optimal':
            break
        wm = np.zeros((n_sta, n))
        for (si, cj), var in w2.items():
            wm[si, cj] = round(pulp.value(var))
        loads = np.array([sum(dd[cj] * wm[si, cj] for cj in range(n)) for si in range(n_sta)])
        utils = loads / sta_caps
        S2_new = np.array([S2_fn(u) for u in utils])
        F2_actual = sum(
            elderly_pop[cj] * (W1 * S1m[sta_idx[si], cj] + W2 * S2_new[si] + W3 * S3) * wm[si, cj]
            for cj in range(n) for si in range(n_sta)
        ) / total_elderly
        if F2_actual > best_F2:
            best_F2 = F2_actual
            best_w = wm.copy()
        if np.max(np.abs(S2_new - S2_cur)) < S2_CONV_TOL:
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
    milp_f1 = elderly_pop[list(milp_cov)].sum() / total_elderly
    return best_F2, milp_f1, milp_cov


# ========== Main scaling experiment ==========
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Run both versions
    versions = [
        ('fixed_area', {'area_km2': 9.0}),      # Version A: fixed 3km x 3km
        ('fixed_density', {'density_per_km2': 10.0}),  # Version B: 10 communities/km^2
    ]
    
    all_results = {}
    
    for version_name, version_params in versions:
        print(f"\n{'='*70}")
        print(f"VERSION {version_name.upper()}")
        print(f"{'='*70}")
        
        version_data = []
        
        for n in [10, 20, 30, 50]:
            print(f"\n--- n = {n} ---")
            
            nash_win_count = 0
            milp_win_count = 0
            tie_count = 0
            all_gaps = []
            valid_count = 0
            instance_data = []
            
            for inst_idx in range(5):
                rng = np.random.default_rng(SEED + hash(version_name) % (2**31) + n * 100 + inst_idx)
                inst = generate_instance_v2(n, rng, mode=version_name, **version_params)
                budget = n * 30
                
                feasible = gen_feasible_configs(n, budget, rng, max_configs=200)
                
                nash_w, milp_w, ties = 0, 0, 0
                inst_gaps = []
                inst_valid = 0
                
                for cfg in feasible[:100]:  # limit per instance
                    nf2, nf1, _ = run_nash(cfg, inst)
                    mf2, mf1, _ = run_milp_two_phase(cfg, inst)
                    
                    if nf2 <= 0 and mf2 <= 0:
                        continue
                    
                    inst_valid += 1
                    gap = nf2 - mf2
                    inst_gaps.append(gap)
                    all_gaps.append(gap)
                    
                    if mf2 > nf2 + 0.001:
                        milp_w += 1
                    elif nf2 > mf2 + 0.001:
                        nash_w += 1
                    else:
                        ties += 1
                
                instance_data.append({
                    'instance': inst_idx + 1,
                    'nash_wins': nash_w,
                    'milp_wins': milp_w,
                    'ties': ties,
                    'valid': inst_valid,
                    'mean_gap': float(np.mean(inst_gaps)) if inst_gaps else 0,
                })
                
                print(f"  Inst {inst_idx+1}: Nash={nash_w}, MILP={milp_w}, Ties={ties}, Valid={inst_valid}")
                
                # BUG-02 FIX:累加到总计数器
                nash_win_count += nash_w
                milp_win_count += milp_w
                tie_count += ties
                valid_count += inst_valid
            
            avg_nash_wr = 100 * nash_win_count / max(valid_count, 1)
            avg_milp_wr = 100 * milp_win_count / max(valid_count, 1)
            
            version_data.append({
                'n': n,
                'nash_win_rate_pct': round(avg_nash_wr, 1),
                'milp_win_rate_pct': round(avg_milp_wr, 1),
                'tie_rate_pct': round(100 * tie_count / max(valid_count, 1), 1),
                'mean_gap': round(float(np.mean(all_gaps)), 6) if all_gaps else 0,
                'std_gap': round(float(np.std(all_gaps)), 6) if all_gaps else 0,
                'instances': instance_data,
            })
            
            print(f"  Summary: Nash={avg_nash_wr:.1f}%, MILP={avg_milp_wr:.1f}%, Ties={100-avg_nash_wr-avg_milp_wr:.1f}%, Mean gap={np.mean(all_gaps):+.4f}" if all_gaps else "  Summary: No valid comparisons")
        
        all_results[version_name] = {
            'params': version_params,
            'results_by_n': version_data,
        }
    
    # Save results
    out_path = os.path.join(OUTPUT_DIR, "scaling_experiment_density_vs_scale.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"Results saved to {out_path}")
    print(f"{'='*70}")
    
    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'n':>4} | {'Nash% (fixed_area)':>16} | {'MILP% (fixed_area)':>16} | {'Nash% (fixed_density)':>18} | {'MILP% (fixed_density)':>18}")
    print("-" * 90)
    for n in [10, 20, 30, 50]:
        fa = next((r for r in all_results['fixed_area']['results_by_n'] if r['n'] == n), None)
        fd = next((r for r in all_results['fixed_density']['results_by_n'] if r['n'] == n), None)
        if fa and fd:
            print(f"{n:>4} | {fa['nash_win_rate_pct']:>15.1f}% | {fa['milp_win_rate_pct']:>15.1f}% | {fd['nash_win_rate_pct']:>17.1f}% | {fd['milp_win_rate_pct']:>17.1f}%")
    
    return all_results


if __name__ == '__main__':
    main()
