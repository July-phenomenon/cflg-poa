# Reproduction checklist

Tick as you go. "Exact" = the published digits are reproduced; "Approx" =
reproduced to the accuracy stated in `docs/REPRODUCIBILITY.md`.

## Environment

- [ ] Environment created (`environment.yml` or `requirements.txt`)
- [ ] `pulp.PULP_CBC_CMD(msg=0).available()` returns a path (CBC present, cross-check only)
- [ ] `pulp.HiGHS_CMD().available()` returns a path (HiGHS present, the exact solver)
- [ ] `python tests/test_solver.py` passes
- [ ] `python tests/test_satisfaction.py` passes

## Design

- [ ] `python code/02_core_solver/instance_generator.py` prints `15207` — **exact**

## §5 — 21 real cities

- [ ] `exp1_21cities.py --exact-phase2 --nash-mode locked --budget-per-station 36` finished
- [ ] mean PoA = 1.0376 (locked, median 1.0357) — **exact**
- [ ] max PoA = 1.0791 (Chongqing) — **exact**
- [ ] MILP strictly better in 18/21, tie in 3/21, Nash better in 0/21 — **exact**
- [ ] 0 violations of the 1.667 bound — **exact**
- [ ] Table 2 (21-city parameters) — **exact**
- [ ] Table 3 (21-city PoA) — **exact**
- [ ] Table 4 (regression): connectivity β = +2.1523 (p = 0.123), log(n_sta)
      β = +0.0246 (p = 0.0023), R² = 0.7025; LOO 21/21 — **exact**
- [ ] Figure 1 regenerated

## §6 — full enumeration

- [ ] `exp2_enumeration_exact.py` finished (or `--limit` smoke test accepted)
- [ ] 15,207 configurations — **exact**
- [ ] Table 5: mean 1.0610 / median 1.0372 / max 1.6404 — **exact**
      (73.98 % MILP strictly better, 17.84 % tied, 8.18 % Nash better)
- [ ] Table 1: locked layer n = 4,317, mean 1.0092, max 1.1560; unlocked
      n = 10,890, mean 1.0816, max 1.6404 — **exact**
- [ ] Table 8: assignment +69.3 %, coverage +43.1 %, congestion −12.6 %
      (median shares over the 12,494 non-tie configurations) — **exact**
- [ ] Channel signs: assignment > 0, coverage > 0, congestion < 0 — **exact**
- [ ] Channel identity residual ≤ 1e-15 — **exact**
- [ ] Figure 2 and Figure 3 regenerated

## §7 — subsidy

- [ ] `exp3_subsidy.py --exact-phase2` finished
- [ ] decentralized welfare gain = 5.97 % — **exact**
- [ ] mean PoA 1.0503 → 1.0656 — **exact**
- [ ] no city's gap closed — **exact**
- [ ] Table 6 and Figure 4 regenerated
- [ ] Table 9 (full per-city subsidy table)

## Appendix

- [ ] Table 7 (three specifications) — **exact**
- [ ] Budget sensitivity: PoA range over 25/30/36/40 万元 ≤ 0.011 — **exact**
- [ ] Capacity heterogeneity: PoA increasing in capacity gini — **exact**
- [ ] γ sensitivity and S₂ robustness — **exact**
- [ ] Cluster-radius factor: mean PoA 1.0688 (0.8) / 1.0376 (1.0) / 1.0267 (1.2),
      band 0.0421, worst city 1.1764, 0 violations of 1.667 — **exact**
- [ ] Shenzhen exclusion: mean moves 1.0376 → 1.0395 — **exact**
- [ ] 21-city instance/Nash/Phase-1 bit-reproducible; Phase-2 MILP wobbles
      ±2e-4 in a city's PoA (HiGHS multi-threaded, thread count not pinnable
      in this environment) — see REPRODUCIBILITY §4.4
- [ ] Coverage-lock shortfall: 13/21 cities, |Δcov| ∈ [0.0029, 0.0190] — **exact**

## Deliverables

- [ ] `results/figures/fig1_poa21.pdf` … `fig4_subsidy.pdf`, `GraphicalAbstract.pdf`
- [ ] `results/tables/*.csv`
- [ ] `results/logs/exp1_run.log`, `exp2_run.log`, `exp3_run.log`
- [ ] `python tests/verify_paper_numbers.py` → ALL CHECKS PASSED (109 numbers)

## Packaging hygiene

- [ ] No `__pycache__/` directories
- [ ] No competition specification documents
- [ ] No archived enumeration result files
- [ ] No internal audit or communication files
- [ ] No hardcoded absolute paths (`grep -rn "D:\\\\" code/` returns nothing)
