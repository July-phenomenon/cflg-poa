# -*- coding: utf-8 -*-
"""
V4+V6: 论文素材表格 + Figure 5.1 / Figure 7.1
产出（02_论文草稿与中间版本/Paper_Materials/）：
  Table5_1_21cities_params.csv
  Table5_2_21cities_PoA.csv
  Table6_1_enum_stats.csv
  Table6_2_layered_stats.csv
  Table7_1_subsidy_comparison.csv
  TableA_1_three_specs.csv
  figures/Figure5_1_21city_PoA_bar.{svg,pdf,png}
  figures/Figure7_1_subsidy_dumbbell.{svg,pdf,png}
所有数字直接从权威数据文件计算，不做任何修饰。
"""
import json, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():

    BASE = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    NUM = os.path.join(BASE, "results", "main")
    OUT = os.path.join(BASE, "results", "tables")
    FIG = os.path.join(BASE, "results", "figures")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Pre-flight: every table is computed from an experiment output file.  Those
    # outputs are deliberately NOT redistributed (they are derived from the
    # competition specification - see data/external/DATA_SOURCES.md), so they must
    # be regenerated first.  Fail early with actionable instructions instead of a
    # bare FileNotFoundError.
    # ---------------------------------------------------------------------------
    _REQUIRED = [
        '21城市精确MILP验证结果_两阶段版.json',   # exp1_21cities.py
        'delta_S2_conditional_bound.json',        # exp2_layered_bounds.py
        '21城市_补贴后.json',                     # exp3_subsidy.py
        'poa_full_per_config_s2.json',            # exp2_enumeration.py
    ]
    _missing = [f for f in _REQUIRED if not os.path.exists(os.path.join(NUM, f))]
    if _missing:
        raise SystemExit(
            "results/main/ is missing the following experiment outputs:\n"
            + "\n".join("  - " + f for f in _missing)
            + "\n\nRegenerate them first, e.g.\n"
              "    python code/03_experiments/exp1_21cities.py\n"
              "    python code/03_experiments/exp2_enumeration.py --workers 8\n"
              "    python code/03_experiments/exp2_layered_bounds.py\n"
              "    python code/03_experiments/exp3_subsidy.py\n"
              "See docs/REPRODUCIBILITY.md for the full ordering.")

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.unicode_minus': False,
        'axes.labelsize': 9.5, 'axes.titlesize': 10.5, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.dpi': 150,
    })

    def wcsv(path, header, rows):
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print('saved', path)

    # ============ Table 5.1 / 5.2（21 城 locked 口径） ============
    d21 = json.load(open(os.path.join(NUM, '21城市精确MILP验证结果_两阶段版.json'), encoding='utf-8'))
    res21 = d21['results']
    cities = sorted(res21.keys())

    rows51, rows52 = [], []
    for c in cities:
        v = res21[c]
        rows51.append([c, v['n'], v['cluster_r'], v['n_stations'],
                       v['n_large'], v['n_medium'], v['n_small'],
                       v['budget_wan'], v['connectivity']])
        rows52.append([c, v['nash_F1'], v['nash_F2'], v['milp_F1'], v['milp_F2'],
                       v['PoA'], v['Loss_pct'], v.get('n_repaired')])
    poas21 = np.array([res21[c]['PoA'] for c in cities])
    rows51.append(['MEAN', '-', '-', round(float(np.mean([res21[c]['n_stations'] for c in cities])), 1),
                   '-', '-', '-', '-', '-'])
    wcsv(os.path.join(OUT, 'Table5_1_21cities_params.csv'),
         ['city', 'n_communities', 'cluster_r_km', 'n_stations', 'n_large', 'n_medium',
          'n_small', 'budget_wan', 'connectivity'], rows51)
    wcsv(os.path.join(OUT, 'Table5_2_21cities_PoA.csv'),
         ['city', 'nash_F1', 'nash_F2', 'milp_F1', 'milp_F2', 'PoA', 'Loss_pct', 'n_repaired'], rows52)
    print('21城 locked: mean=%.4f max=%.4f min=%.4f' % (poas21.mean(), poas21.max(), poas21.min()))

    # ---- Figure 5.1: 21 城 PoA 排序条形图 ----
    order = sorted(cities, key=lambda c: res21[c]['PoA'])
    vals = [res21[c]['PoA'] for c in order]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    cols = ['#2b6cb0' if v <= 1.0 else '#dd6b20' for v in vals]
    ax.barh(range(len(order)), vals, color=cols, alpha=0.85, height=0.68)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.5)
    ax.axvline(1.0, color='#4a5568', lw=1.0)
    ax.axvline(poas21.mean(), color='#2b6cb0', lw=1.2, ls='--')
    ax.annotate(f'mean {poas21.mean():.4f}', xy=(poas21.mean(), len(order) - 0.2),
                fontsize=7.5, color='#2b6cb0', rotation=90, va='top', ha='right',
                xytext=(-3, 0), textcoords='offset points')
    for i, v in enumerate(vals):
        ax.annotate(f'{v:.4f}', xy=(v, i), xytext=(3, 0), textcoords='offset points',
                    va='center', fontsize=6.5, color='#2d3748')
    ax.set_xlabel(r'PoA $= F_2^{\mathrm{MILP}}/F_2^{\mathrm{Nash}}$   (two-phase locked, 36 wan/station)')
    ax.set_title('Sequential PoA across 21 real cities (coverage locked)', fontweight='bold')
    ax.set_xlim(0.90, 1.13)
    ax.grid(axis='x', alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'Figure5_1_21city_PoA_bar.{ext}'), bbox_inches='tight')
    print('saved Figure5_1')

    # ============ Table 6.1 / 6.2（15208 枚举） ============
    den = json.load(open(os.path.join(BASE, 'poa_full_per_config_s2.json'), encoding='utf-8'))
    rows_all = [x for x in den['results'] if x.get('PoA')]
    poa = np.array([x['PoA'] for x in rows_all])
    dcov = np.array([x['milp_F1'] - x['nash_F1'] for x in rows_all])
    locked = dcov <= 1e-9
    pct_milp_wins = float((poa > 1.0 + 1e-9).mean() * 100)
    pct_equal = float((np.abs(poa - 1.0) <= 1e-9).mean() * 100)
    pct_nash_wins = float((poa < 1.0 - 1e-9).mean() * 100)
    wcsv(os.path.join(OUT, 'Table6_1_enum_stats.csv'), ['statistic', 'value'], [
        ['n_configurations', len(poa)],
        ['mean_PoA', round(float(poa.mean()), 4)],
        ['median_PoA', round(float(np.median(poa)), 4)],
        ['p95_PoA', round(float(np.percentile(poa, 95)), 4)],
        ['p99_PoA', round(float(np.percentile(poa, 99)), 4)],
        ['max_PoA', round(float(poa.max()), 4)],
        ['min_PoA', round(float(poa.min()), 4)],
        ['pct_MILP_strictly_better', round(pct_milp_wins, 2)],
        ['pct_equal', round(pct_equal, 2)],
        ['pct_Nash_strictly_better', round(pct_nash_wins, 2)],
    ])
    print('枚举: mean=%.4f p99=%.4f max=%.4f | MILP> %.2f%% = %.2f%% Nash> %.2f%%' %
          (poa.mean(), np.percentile(poa, 99), poa.max(), pct_milp_wins, pct_equal, pct_nash_wins))

    cb = json.load(open(os.path.join(NUM, 'delta_S2_conditional_bound.json'), encoding='utf-8'))
    rows62 = []
    for key, lab in [('overall', 'all'), ('coverage_locked', 'coverage locked (dcov=0)'),
                     ('coverage_unlocked', 'coverage unlocked (dcov>0)')]:
        b = cb[key]
        rows62.append([lab, b['n'], round(b['share_of_sample'] * 100, 1), b['PoA_mean'],
                       b['PoA_median'], b['PoA_p95'], b['PoA_p99'], b['PoA_max'], b['PoA_min'],
                       b['conditional_bound'], b['viol'], b['rate']])
    wcsv(os.path.join(OUT, 'Table6_2_layered_stats.csv'),
         ['layer', 'n', 'share_pct', 'mean_PoA', 'median_PoA', 'p95_PoA', 'p99_PoA',
          'max_PoA', 'min_PoA', 'empirical_conditional_bound', 'violations', 'violation_rate'], rows62)

    # ============ Table 7.1（补贴） ============
    dsu = json.load(open(os.path.join(NUM, '21城市_补贴后.json'), encoding='utf-8'))
    rsu = dsu['results']
    rows71 = []
    for c in sorted(rsu.keys()):
        v = rsu[c]
        rows71.append([c, v['n_stations'], round(v['nash_F2_base'], 4), round(v['nash_F2_sub'], 4),
                       round(v['nash_F2_improve_pct'], 2), round(v['PoA_base'], 4),
                       round(v['PoA_sub_consistent'], 4), round(v['PoA_sub_mixed'], 4),
                       round(v['PoA_reduction_consistent'], 4)])
    s = dsu['summary']
    rows71.append(['MEAN', '-', '-', '-',
                   round(s['mean_nash_F2_improve_pct'], 2), round(s['mean_PoA_base'], 4),
                   round(s['mean_PoA_sub_consistent'], 4), round(s['mean_PoA_sub_mixed'], 4),
                   round(s['mean_PoA_reduction_consistent'], 4)])
    wcsv(os.path.join(OUT, 'Table7_1_subsidy_comparison.csv'),
         ['city', 'n_stations', 'nash_F2_base', 'nash_F2_subsidized', 'nash_F2_improve_pct',
          'PoA_base', 'PoA_subsidized_consistent', 'PoA_subsidized_mixed',
          'PoA_reduction_consistent'], rows71)

    # ---- Figure 7.1: 补贴前后 PoA 哑铃图 ----
    cs = sorted(rsu.keys(), key=lambda c: rsu[c]['PoA_base'])
    pb = [rsu[c]['PoA_base'] for c in cs]
    ps = [rsu[c]['PoA_sub_consistent'] for c in cs]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    y = np.arange(len(cs))
    for i in range(len(cs)):
        ax.plot([pb[i], ps[i]], [i, i], color='#a0aec0', lw=1.2, zorder=1)
    ax.scatter(pb, y, s=26, c='#2b6cb0', zorder=2, label='baseline PoA')
    ax.scatter(ps, y, s=26, c='#dd6b20', zorder=2, label='subsidized PoA (consistent)')
    ax.set_yticks(y)
    ax.set_yticklabels(cs, fontsize=7.5)
    ax.axvline(1.0, color='#4a5568', lw=1.0)
    ax.axvline(s['mean_PoA_base'], color='#2b6cb0', lw=1.1, ls='--', alpha=0.8)
    ax.axvline(s['mean_PoA_sub_consistent'], color='#dd6b20', lw=1.1, ls='--', alpha=0.8)
    ax.annotate(f"mean base {s['mean_PoA_base']:.4f}", xy=(s['mean_PoA_base'], len(cs) - 0.2),
                fontsize=7, color='#2b6cb0', rotation=90, va='top', ha='right',
                xytext=(-3, 0), textcoords='offset points')
    ax.annotate(f"mean sub {s['mean_PoA_sub_consistent']:.4f}", xy=(s['mean_PoA_sub_consistent'], len(cs) - 0.2),
                fontsize=7, color='#dd6b20', rotation=90, va='top', ha='left',
                xytext=(3, 0), textcoords='offset points')
    ax.set_xlabel(r'PoA (21 cities, 36 wan/station)')
    ax.set_title('V2 subsidy does not close the efficiency gap\n'
                 r'(Nash welfare +5.97%, but PoA $1.0394 \rightarrow 1.0497$)',
                 fontweight='bold')
    ax.legend(loc='lower right', frameon=False)
    ax.grid(axis='x', alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'Figure7_1_subsidy_dumbbell.{ext}'), bbox_inches='tight')
    print('saved Figure7_1')

    # ============ Table A.1（三口径对照） ============
    specs = [
        ('Legacy artifact (stale file, budget unspecified)', None, 'classic (unlocked)',
         '21城市精确MILP验证结果_单阶段版.bak'),
        ('Canonical single-stage', 36.0, 'classic (unlocked)', '21城市精确MILP验证结果_最终版.json'),
        ('Two-phase coverage-locked (paper specification)', 36.0, 'locked', '21城市精确MILP验证结果_两阶段版.json'),
    ]
    rowsA1 = []
    for lab, bud, mode, fn in specs:
        dd = json.load(open(os.path.join(NUM, fn), encoding='utf-8'))
        rr = dd['results']
        pp = [v['PoA'] for v in rr.values() if v.get('PoA')]
        rowsA1.append([lab, bud if bud is not None else 'unspecified', mode, len(pp),
                       round(float(np.mean(pp)), 4), round(float(np.median(pp)), 4),
                       round(float(np.max(pp)), 4), sum(1 for p in pp if p > 1.667),
                       sum(1 for p in pp if p > 1.2), fn])
    wcsv(os.path.join(OUT, 'TableA_1_three_specs.csv'),
         ['specification', 'budget_wan_per_station', 'nash_mode', 'n', 'mean_PoA', 'median_PoA',
          'max_PoA', 'viol_1.667', 'viol_1.20', 'source_file'], rowsA1)
    print('V4+V6 表格与图全部生成完毕')


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
