# -*- coding: utf-8 -*-
"""T2 补充：按“覆盖是否锁定”分层，给出条件收紧界，并与 21 城两阶段版互证。

分层依据：Δcov = milp_F1 - nash_F1
  - 锁定层：Δcov ≈ 0（21 城 two-phase locked 对应的设定）
  - 未锁定层：Δcov > 0（15208 枚举的默认设定）
输出：delta_S2_conditional_bound.json
"""
import json, os, numpy as np


def main():

    DATA = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    OUT = os.path.join(DATA, "results", "main")
    W1, W2, W3, S3 = 0.2, 0.3, 0.5, 0.60

    # 2026-10 revision: read the exact single-MILP enumeration (the legacy
    # <root>/poa_full_per_config_s2.json archive is no longer shipped, so this
    # script previously could not run at all).  delta_S2_eff is reconstructed as
    # the difference of the population-weighted S2 the exact run reports.
    exact = os.path.join(OUT, "results_enumeration_exact.json")
    if not os.path.exists(exact):
        raise SystemExit(
            "missing %s -- run code/03_experiments/exp2_enumeration_exact.py first"
            % os.path.relpath(exact, DATA))
    res = json.load(open(exact, encoding="utf-8"))["results"]
    for r in res:
        r["delta_S2_eff"] = r["milp_S2_weighted"] - r["nash_S2_weighted"]

    rows = [r for r in res if r.get('delta_S2_eff') is not None and r.get('PoA') is not None]
    poa = np.array([r['PoA'] for r in rows])
    nf2 = np.array([r.get('nash_F2') or 0.0 for r in rows])
    mf2 = np.array([r.get('milp_F2') or 0.0 for r in rows])
    dcov = np.array([(r.get('milp_F1') or 0.0) - (r.get('nash_F1') or 0.0) for r in rows])
    dA2 = np.array([r['delta_S2_eff'] for r in rows])
    dF2 = mf2 - nf2
    dA1 = (dF2 - W2 * dA2 - W3 * S3 * dcov) / W1
    ch1 = W1 * dA1
    ch2 = W2 * dA2
    chc = W3 * S3 * dcov

    TOL = 1e-9
    locked = dcov <= TOL
    unlocked = dcov > TOL


    def block(mask, name):
        p = poa[mask]
        if len(p) == 0:
            return {'name': name, 'n': 0}
        nf = np.where(nf2[mask] > 0, nf2[mask], np.nan)
        q = lambda a: float(np.nanpercentile(a, 99))
        r1 = ch1[mask] / nf
        r2 = ch2[mask] / nf
        rc = chc[mask] / nf
        B = 1 + q(r1) + q(r2) + q(rc)
        viol = int(np.nansum(p > B))
        return {
            'name': name, 'n': int(len(p)),
            'share_of_sample': round(float(len(p) / len(poa)), 4),
            'PoA_mean': round(float(p.mean()), 4),
            'PoA_median': round(float(np.median(p)), 4),
            'PoA_p95': round(float(np.percentile(p, 95)), 4),
            'PoA_p99': round(float(np.percentile(p, 99)), 4),
            'PoA_max': round(float(p.max()), 4),
            'PoA_min': round(float(p.min()), 4),
            'q99_r_S1': round(q(r1), 6), 'q99_r_S2': round(q(r2), 6), 'q99_r_cov': round(q(rc), 6),
            'conditional_bound': round(B, 4),
            'viol': viol, 'rate': round(viol / len(p), 4),
            'required_bound_for_99pct_coverage': round(float(np.percentile(p, 99)), 4),
            'required_bound_for_100pct_coverage': round(float(p.max()), 4),
        }


    def _share(mask, ch):
        den = np.where(np.abs(dF2[mask]) > 1e-9, dF2[mask], np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            v = ch[mask] / den
        v = v[~np.isnan(v)]
        return round(float(np.nanmedian(v)), 4) if len(v) else None


    out = {
        'definition': 'Δcov = milp_F1 - nash_F1；锁定层 Δcov<=1e-9',
        'overall': block(np.ones(len(poa), dtype=bool), 'all'),
        'coverage_locked': block(locked, 'Δcov = 0'),
        'coverage_unlocked': block(unlocked, 'Δcov > 0'),
        'channel_share_by_layer': {
            'Δcov = 0': {'S1_median': _share(locked, ch1), 'S2_median': _share(locked, ch2)},
            'Δcov > 0': {'S1_median': _share(unlocked, ch1), 'S2_median': _share(unlocked, ch2),
                         'cov_median': _share(unlocked, chc)},
        },
        'cross_check_21cities_locked': {
            'note': '21 城两阶段（Nash 锁定覆盖）实测：mean 1.0267，max 1.0663',
            'source': '21城市精确MILP验证结果_两阶段版.json',
        },
    }

    json.dump(out, open(os.path.join(OUT, 'delta_S2_conditional_bound.json'), 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
