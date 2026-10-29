# Enumeration regression ladder (audit trail)

source: `results_enumeration_exact.json`, n = 15207, locked share = 28.4%

| model | R2 | adj. R2 | coefficients (p) |
|---|---|---|---|
| M1_connectivity | 0.0069 | 0.0068 | const = 0.9948 (0), connectivity = 0.1655 (1.26e-24) |
| M2_locked | 0.1944 | 0.1944 | const = 1.082 (0), locked = -0.07236 (0) |
| M3_locked_connectivity | 0.2002 | 0.2000 | const = 1.021 (0), locked = -0.07215 (0), connectivity = 0.1512 (1.82e-25) |
| M4_locked_connectivity_lognsta | 0.2040 | 0.2038 | const = 0.991 (0), locked = -0.06962 (0), connectivity = 0.1517 (9.76e-26), log n_sta = 0.02083 (1.57e-17) |
