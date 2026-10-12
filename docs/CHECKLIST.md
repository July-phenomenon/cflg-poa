# Reproduction checklist

Tick as you go. "Exact" = the published digits are reproduced; "Approx" =
reproduced to the accuracy stated in `docs/REPRODUCIBILITY.md`.

## Environment

- [ ] Environment created (`environment.yml` or `requirements.txt`)
- [ ] `pulp.PULP_CBC_CMD(msg=0).available()` returns a path (CBC present)
- [ ] `python tests/test_solver.py` passes
- [ ] `python tests/test_satisfaction.py` passes

## Design

- [ ] `python code/02_core_solver/instance_generator.py` prints `15207` — **exact**

## §5 — 21 real cities

- [ ] `exp1_21cities.py` finished
- [ ] mean PoA = 1.0267 (locked) — **exact**
- [ ] max PoA = 1.0663 (Chongqing) — **exact**
- [ ] 0 violations of the 1.667 bound — **exact**
- [ ] Table 2 (21-city parameters) — **exact**
- [ ] Table 3 (21-city PoA) — **exact**
- [ ] Table 4 (regression): connectivity β = +1.3396, log(n_sta) β = +0.0183,
      R² = 0.516 — **exact**
- [ ] Figure 1 regenerated

## §6 — full enumeration

- [ ] `exp2_enumeration.py` finished (or `--limit` smoke test accepted)
- [ ] 15,207 configurations — **exact**
- [ ] Table 5: mean 1.0398 / median 1.0336 / max 1.4398 — **approx** (see
      REPRODUCIBILITY §4.2; median and the PoA > 1 share match closely)
- [ ] Table 1: locked layer n = 1,208, mean 1.0092; unlocked n = 13,999,
      mean 1.0424 — ordering and layer sizes **exact**, means **approx**
- [ ] Table 8: coverage +81.9 %, assignment +34.2 %, congestion −14.9 %
      (median shares) — **approx**
- [ ] Channel signs: coverage > 0, assignment > 0, congestion < 0 — **exact**
- [ ] Channel identity residual ≤ 1e-15 — **exact**
- [ ] Figure 2 and Figure 3 regenerated

## §7 — subsidy

- [ ] `exp3_subsidy.py` finished
- [ ] decentralized welfare gain = 5.97 % — **exact**
- [ ] mean PoA 1.0394 → 1.0497 — **exact**
- [ ] no city's gap closed — **exact**
- [ ] Table 6 and Figure 4 regenerated
- [ ] Table 9 (full per-city subsidy table)

## Appendix

- [ ] Table 7 (three specifications) — **exact**
- [ ] Budget sensitivity: PoA range over 25/30/36/40 万元 ≤ 0.011 — **exact**
- [ ] Capacity heterogeneity: PoA increasing in capacity gini — **exact**
- [ ] γ sensitivity and S₂ robustness — **exact**

## Deliverables

- [ ] `results/figures/figure1.pdf` … `figure4.pdf`, `GraphicalAbstract.pdf`
- [ ] `results/tables/*.csv`
- [ ] `results/logs/exp1_run.log`, `exp2_run.log`, `exp3_run.log`

## Packaging hygiene

- [ ] No `__pycache__/` directories
- [ ] No competition specification documents
- [ ] No archived enumeration result files
- [ ] No internal audit or communication files
- [ ] No hardcoded absolute paths (`grep -rn "D:\\\\" code/` returns nothing)
