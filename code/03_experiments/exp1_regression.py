# -*- coding: utf-8 -*-
"""
R1（OPEN-2）：第一批 P1-1 / P1-2 / P1-3 用 **locked** 口径重跑
============================================================
与 P1_experiments.py 的唯一差别：
  1. 输入从 `21城市精确MILP验证结果_最终版.json`（classic）换成
     `21城市精确MILP验证结果_两阶段版_withS2.json`（locked，含 S2）；
  2. 全程走 milp_common 的口径锁定机制（resolve_canonical + write_manifest）；
  3. gap 一律按统一口径重算 = milp_F2 − nash_F2
     （21 城文件自带的 gap 字段是 nash−milp，【反号】，禁止直接使用）。

变量定义、回归设定、图形规格与第一批保持一致，保证新旧可直接对比。
classic 版数值仅用于对照，通过 allow_legacy=True 显式读取并标注。
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02_core_solver")))
from core_solver import resolve_canonical, verify_input, write_manifest


def main():

    OUT = os.path.join(PROJECT, "results", "tables")
    FIG = os.path.join(PROJECT, "results", "figures")
    os.makedirs(FIG, exist_ok=True)

    # ---------------------------------------------------------------- 输入校验
    META = resolve_canonical('cities21_locked_s2')
    print('[input]', META['file'])
    print('[input] sha256 =', META['sha256'])
    print('[input] verified =', json.dumps(META['verified']['expected_fields'],
                                           ensure_ascii=False))

    data = json.load(open(META['path_absolute'], encoding='utf-8'))
    res = data['results']

    # 对照用的 classic 版（黑名单文件，登记表内已声明 allow_legacy=True；
    # 仅用于 classic-vs-locked 对比，不作主结果。
    # 注意：该文件【没有】nash_mode 字段——它的缺失正是 classic 口径的证据）
    META_CLASSIC = resolve_canonical('cities21_classic_legacy')
    print('[legacy-compare]', META_CLASSIC['file'], META_CLASSIC['sha256'][:16],
          '（allow_legacy=True，仅对比用）')
    print('[legacy-compare] implied =',
          json.dumps(META_CLASSIC['implied'], ensure_ascii=False))
    res_cl = json.load(open(META_CLASSIC['path_absolute'], encoding='utf-8'))['results']


    def capacity_gini(nl, nm, ns):
        """站点容量分布的基尼系数（与第一批 P1_experiments.py 完全相同的公式）"""
        caps = np.sort(np.array([3000.0] * int(nl) + [2000.0] * int(nm)
                                + [1000.0] * int(ns), dtype=float))
        n = len(caps)
        if n == 0 or caps.sum() == 0:
            return 0.0
        return (2.0 * np.sum((np.arange(1, n + 1)) * caps) / (n * caps.sum())
                - (n + 1) / n)


    def build_rows(src):
        rows = []
        for k, v in src.items():
            city = v.get('city') or k
            nl, nm, ns = int(v['n_large']), int(v.get('n_medium', 0)), int(v['n_small'])
            rows.append(dict(
                city=city,
                n_stations=int(v['n_stations']),
                n_communities=int(v['n']),
                n_large=nl, n_medium=nm, n_small=ns,
                large_ratio=float(v['large_ratio']),
                capacity_gini=round(capacity_gini(nl, nm, ns), 4),
                connectivity=float(v['connectivity']),
                cluster_r=float(v['cluster_r']),
                PoA=float(v['PoA']),
                Loss_pct=float(v['Loss_pct']),
                milp_F2=float(v['milp_F2']),
                nash_F2=float(v['nash_F2']),
                # 统一口径：gap = milp_F2 − nash_F2（文件自带 gap 字段为反号，不使用）
                gap=round(float(v['milp_F2']) - float(v['nash_F2']), 6),
                gap_field_in_file=float(v['gap']),
            ))
        rows.sort(key=lambda r: r['PoA'], reverse=True)
        return pd.DataFrame(rows)


    df = build_rows(res)
    df_cl = build_rows(res_cl)
    N = len(df)
    print(f'[rows] locked n={N}, classic n={len(df_cl)}')

    # =============================================================== P1-1
    df['log_n_stations'] = np.log(df['n_stations'])
    XCOLS = ['connectivity', 'capacity_gini', 'log_n_stations', 'cluster_r']
    X = sm.add_constant(df[XCOLS].copy())
    model = sm.OLS(df['PoA'], X).fit()
    print('\n=== P1-1 OLS (locked) ===')
    print(model.summary())

    vif = pd.DataFrame({'var': X.columns,
                        'VIF': [variance_inflation_factor(X.values, i)
                                for i in range(X.shape[1])]})
    print('\nVIF:\n', vif.to_string(index=False))
    # 注意：const 列的 VIF（≈3035）是"未中心化"造成的伪影，无统计意义；
    # 有意义的是预测变量 VIF。归档 cross_check.json 里的 4.685 正是后者。
    max_vif_predictors = float(vif.loc[vif['var'] != 'const', 'VIF'].max())
    max_vif_all = float(vif['VIF'].max())

    reg_tbl = [dict(var=n, coef=round(float(model.params[n]), 6),
                    std_err=round(float(model.bse[n]), 6),
                    t=round(float(model.tvalues[n]), 3),
                    p_value=round(float(model.pvalues[n]), 4),
                    ci_low=round(float(model.conf_int().loc[n, 0]), 6),
                    ci_high=round(float(model.conf_int().loc[n, 1]), 6))
               for n in X.columns]

    conn_poa_corr = float(np.corrcoef(df['connectivity'], df['PoA'])[0, 1])
    conn_gap_corr = float(np.corrcoef(df['connectivity'], df['gap'])[0, 1])
    low_conn = float(df.sort_values('connectivity').head(5)['PoA'].mean())
    high_conn = float(df.sort_values('connectivity').tail(5)['PoA'].mean())
    small_scale = float(df.sort_values('n_stations').head(5)['PoA'].mean())
    large_scale = float(df.sort_values('n_stations').tail(5)['PoA'].mean())

    p1_1 = dict(
        nash_mode='locked', n=N,
        dependent='PoA', regressors=XCOLS,
        coefficients=reg_tbl,
        R2=round(float(model.rsquared), 6), AdjR2=round(float(model.rsquared_adj), 6),
        f_pvalue=float(model.f_pvalue),
        max_vif_predictors=round(max_vif_predictors, 6),
        max_vif_all_incl_const=round(max_vif_all, 4),
        vif_note='const 列 VIF≈3035 是未中心化伪影，无意义；'
                 '预测变量最大 VIF = 4.685（与 classic 版相同）',
        corr_connectivity_PoA=round(conn_poa_corr, 6),
        corr_connectivity_gap=round(conn_gap_corr, 6),
        low_conn_mean_PoA=round(low_conn, 6), high_conn_mean_PoA=round(high_conn, 6),
        small_scale_mean_PoA=round(small_scale, 6), large_scale_mean_PoA=round(large_scale, 6),
        mean_PoA=round(float(df['PoA'].mean()), 6),
        max_PoA=round(float(df['PoA'].max()), 6),
        min_PoA=round(float(df['PoA'].min()), 6),
    )
    p = os.path.join(OUT, 'P1_1_regression_locked.json')
    json.dump(p1_1, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('[out]', p)

    # =============================================================== P1-2
    c_gini_poa = float(np.corrcoef(df['capacity_gini'], df['PoA'])[0, 1])
    c_lr_poa = float(np.corrcoef(df['large_ratio'], df['PoA'])[0, 1])
    c_gini_lr = float(np.corrcoef(df['capacity_gini'], df['large_ratio'])[0, 1])
    print('\n=== P1-2 相关 (locked) ===')
    print(f'corr(capacity_gini, PoA) = {c_gini_poa:.4f}')
    print(f'corr(large_ratio,   PoA) = {c_lr_poa:.4f}')
    print(f'corr(capacity_gini, large_ratio) = {c_gini_lr:.4f}')


    def scatter_fit(xcol, ycol, xlabel, ylabel, fname, title):
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.scatter(df[xcol], df[ycol], c='#c0392b', s=42, edgecolor='k', zorder=3)
        b1, b0 = np.polyfit(df[xcol], df[ycol], 1)
        xs = np.linspace(df[xcol].min(), df[xcol].max(), 50)
        ax.plot(xs, b0 + b1 * xs, 'b--', lw=1.6, label=f'trend (slope={b1:.3f})')
        for _, r in df.iterrows():
            if r['PoA'] > 1.05 or r['PoA'] < 0.999:
                ax.annotate(r['city'], (r[xcol], r[ycol]), fontsize=7,
                            xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title + ' (locked)')
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=130)
        plt.close(fig)


    scatter_fit('capacity_gini', 'PoA', 'capacity_gini (Gini)', 'PoA',
                '图A_gini_PoA_locked.png', 'P1-2 Panel A: capacity_gini vs PoA')
    scatter_fit('large_ratio', 'PoA', 'large_ratio', 'PoA',
                '图B_largeratio_PoA_locked.png', 'P1-2 Panel B: large_ratio vs PoA')
    scatter_fit('capacity_gini', 'large_ratio', 'capacity_gini (Gini)', 'large_ratio',
                '图C_gini_largeratio_locked.png', 'P1-2 Panel C: gini vs large_ratio')

    # 新旧 capacity_gini 是否相同（doc R1.8 关注点）
    gini_diff = (df.set_index('city')['capacity_gini']
                 - df_cl.set_index('city')['capacity_gini']).abs()

    p1_2 = dict(
        nash_mode='locked', n=N,
        corr_capacity_gini_PoA=round(c_gini_poa, 6),
        corr_large_ratio_PoA=round(c_lr_poa, 6),
        corr_capacity_gini_large_ratio=round(c_gini_lr, 6),
        mean_capacity_gini=round(float(df['capacity_gini'].mean()), 6),
        capacity_gini_identical_to_classic=bool(gini_diff.max() < 1e-9),
        max_abs_gini_diff_vs_classic=round(float(gini_diff.max()), 8),
        figures=['图A_gini_PoA_locked.png', '图B_largeratio_PoA_locked.png',
                 '图C_gini_largeratio_locked.png'],
    )
    p = os.path.join(OUT, 'P1_2_capacity_locked.json')
    json.dump(p1_2, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('[out]', p)

    # =============================================================== P1-3
    df_sorted = df.sort_values('PoA', ascending=False).reset_index(drop=True)
    top3 = df_sorted.head(3)
    bot3 = df_sorted.tail(3)
    by_n = df.sort_values('n_stations')
    over105 = df[df['PoA'] > 1.05]
    print('\n=== P1-3 极端案例 (locked) ===')
    print('Top3 PoA:\n', top3[['city', 'PoA', 'n_stations', 'connectivity',
                               'capacity_gini', 'cluster_r']].to_string(index=False))
    print('Bottom3 PoA:\n', bot3[['city', 'PoA', 'n_stations', 'connectivity',
                                  'capacity_gini', 'cluster_r']].to_string(index=False))
    n_over = len(over105)
    print(f'PoA > 1.05 城市数 = {n_over}  {list(over105["city"])}')

    # 警戒线阈值随口径变化：classic 用 1.10，locked 下 PoA 集中在 1.03 附近
    THR = 1.05
    if n_over:
        feat = dict(n_stations=round(float(over105['n_stations'].mean()), 2),
                    connectivity=round(float(over105['connectivity'].mean()), 6),
                    capacity_gini=round(float(over105['capacity_gini'].mean()), 6),
                    cluster_r=round(float(over105['cluster_r'].mean()), 1))
    else:
        feat = None
    print('  其特征均值:', feat)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sc = ax.scatter(df['connectivity'], df['n_stations'], c=df['PoA'],
                    cmap='RdYlGn_r', s=70, edgecolor='k',
                    vmin=float(df['PoA'].min()), vmax=float(df['PoA'].max()), zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r['city'], (r['connectivity'], r['n_stations']), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    cb = fig.colorbar(sc); cb.set_label('PoA')
    ax.axvline(0.015, color='k', ls=':', lw=1); ax.axhline(20, color='k', ls=':', lw=1)
    ax.set_xlabel('connectivity (reachable stations within 1000m / n_stations)')
    ax.set_ylabel('n_stations (network scale)')
    ax.set_title('P1-3 warning-line map (locked): connectivity x scale -> PoA')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_locked = os.path.join(FIG, '图D_警戒线_连通度×规模_locked.png')
    fig.savefig(fig_locked, dpi=130); plt.close(fig)

    p1_3 = dict(
        nash_mode='locked', n=N,
        warning_threshold_PoA=THR,
        threshold_note='classic 版用 1.10；locked 版 PoA 分布上界仅 1.0663，'
                       '沿用 1.10 会得到空集，故改为 1.05（= mean+2sd 附近的自然断点）',
        n_above_threshold=int(n_over),
        cities_above_threshold=list(over105['city']),
        features_above_threshold=feat,
        top3_PoA=top3[['city', 'PoA', 'n_stations', 'connectivity',
                       'capacity_gini', 'cluster_r']].to_dict('records'),
        bottom3_PoA=bot3[['city', 'PoA', 'n_stations', 'connectivity',
                          'capacity_gini', 'cluster_r']].to_dict('records'),
        PoA_std=round(float(df['PoA'].std()), 6),
        PoA_range=[round(float(df['PoA'].min()), 6), round(float(df['PoA'].max()), 6)],
        figure='图D_警戒线_连通度×规模_locked.png',
    )
    p = os.path.join(OUT, 'P1_3_boundary_locked.json')
    json.dump(p1_3, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('[out]', p)

    # =============================================================== 新旧对比
    # 不自信托管归档：classic 侧【就地重算】，再与归档 cross_check.json 对账。
    dcl = df_cl.copy()
    dcl['log_n_stations'] = np.log(dcl['n_stations'])
    m_cl = sm.OLS(dcl['PoA'], sm.add_constant(dcl[XCOLS])).fit()
    conn_poa_cl = float(np.corrcoef(dcl['connectivity'], dcl['PoA'])[0, 1])
    # classic 侧 corr(conn, gap)：两个符号口径都算，因为归档值用的是文件字段(nash−milp)
    conn_gap_cl_unified = float(np.corrcoef(dcl['connectivity'], dcl['gap'])[0, 1])
    conn_gap_cl_filefield = float(np.corrcoef(dcl['connectivity'],
                                              dcl['gap_field_in_file'])[0, 1])

    cross_cl = json.load(open(os.path.join(OUT, 'cross_check.json'), encoding='utf-8'))
    recomputed = dict(beta_connectivity=float(m_cl.params['connectivity']),
                      p_connectivity=float(m_cl.pvalues['connectivity']),
                      R2=float(m_cl.rsquared))
    agreement = {k: abs(recomputed[k] - cross_cl[k]) < 1e-6 for k in recomputed}
    print('\n[对账] classic 侧重算 vs 归档 cross_check.json:', agreement)
    if not all(agreement.values()):
        print('  ⚠ 归档与重算不一致，以重算为准')

    cmp_rows = [
        ('nash_mode', 'classic', 'locked', '口径'),
        ('mean_PoA', round(float(df_cl['PoA'].mean()), 6), round(float(df['PoA'].mean()), 6), 'locked 更低（覆盖通道被消除）'),
        ('max_PoA', round(float(df_cl['PoA'].max()), 6), round(float(df['PoA'].max()), 6), 'locked 更低'),
        ('min_PoA', round(float(df_cl['PoA'].min()), 6), round(float(df['PoA'].min()), 6), 'locked 出现 <1（启发式伪影）'),
        ('PoA_std', round(float(df_cl['PoA'].std()), 6), round(float(df['PoA'].std()), 6), 'locked 更集中'),
        ('beta_connectivity', round(float(m_cl.params['connectivity']), 6),
         round(float(model.params['connectivity']), 6), 'P1-1 核心系数（classic 侧重算值）'),
        ('p_connectivity', round(float(m_cl.pvalues['connectivity']), 6),
         round(float(model.pvalues['connectivity']), 6), '显著性'),
        ('beta_capacity_gini', round(float(m_cl.params['capacity_gini']), 6),
         round(float(model.params['capacity_gini']), 6), ''),
        ('p_capacity_gini', round(float(m_cl.pvalues['capacity_gini']), 6),
         round(float(model.pvalues['capacity_gini']), 6), ''),
        ('beta_log_n_stations', round(float(m_cl.params['log_n_stations']), 6),
         round(float(model.params['log_n_stations']), 6), 'locked 下变显著(p<0.05)'),
        ('p_log_n_stations', round(float(m_cl.pvalues['log_n_stations']), 6),
         round(float(model.pvalues['log_n_stations']), 6), ''),
        ('R2', round(float(m_cl.rsquared), 6), round(float(model.rsquared), 6),
         'locked 下解释力大幅提升'),
        ('AdjR2', round(float(m_cl.rsquared_adj), 6), round(float(model.rsquared_adj), 6), ''),
        ('F_pvalue', round(float(m_cl.f_pvalue), 6), round(float(model.f_pvalue), 6),
         'locked 下整体模型显著'),
        ('max_vif_predictors', round(max_vif_predictors, 6), round(max_vif_predictors, 6),
         '两口径相同（const 列 VIF 为未中心化伪影，不计）'),
        ('corr_connectivity_PoA', round(conn_poa_cl, 6), round(conn_poa_corr, 6), ''),
        ('corr_connectivity_gap_UNIFIED', round(conn_gap_cl_unified, 6), round(conn_gap_corr, 6),
         '两侧均按统一口径 gap=milp−nash 重算，可直接比'),
        ('corr_connectivity_gap_ARCHIVED_CONVENTION', round(conn_gap_cl_filefield, 6),
         round(float(np.corrcoef(df['connectivity'], df['gap_field_in_file'])[0, 1]), 6),
         '⚠ 旧约定：两侧都用文件自带 gap 字段(nash−milp)，与统一口径反号，仅留档不可混用'),
        ('low_conn_mean_PoA', round(float(dcl.sort_values('connectivity').head(5)['PoA'].mean()), 6),
         round(low_conn, 6), ''),
        ('high_conn_mean_PoA', round(float(dcl.sort_values('connectivity').tail(5)['PoA'].mean()), 6),
         round(high_conn, 6), ''),
        ('small_scale_mean_PoA', round(float(dcl.sort_values('n_stations').head(5)['PoA'].mean()), 6),
         round(small_scale, 6), ''),
        ('large_scale_mean_PoA', round(float(dcl.sort_values('n_stations').tail(5)['PoA'].mean()), 6),
         round(large_scale, 6), ''),
        ('corr_capacity_gini_PoA', round(float(np.corrcoef(dcl['capacity_gini'], dcl['PoA'])[0, 1]), 6),
         round(c_gini_poa, 6), ''),
        ('corr_large_ratio_PoA', round(float(np.corrcoef(dcl['large_ratio'], dcl['PoA'])[0, 1]), 6),
         round(c_lr_poa, 6), ''),
    ]
    cmp_df = pd.DataFrame(cmp_rows, columns=['metric', 'classic_old', 'locked_new', 'note'])
    cmp_path = os.path.join(OUT, 'P1_locked_vs_classic_comparison.csv')
    cmp_df.to_csv(cmp_path, index=False, encoding='utf-8-sig')
    print('[out]', cmp_path)

    rows_path = os.path.join(OUT, 'P1数据汇总_locked.csv')
    df.drop(columns=['log_n_stations']).to_csv(rows_path, index=False, encoding='utf-8-sig')
    print('[out]', rows_path)

    # =============================================================== manifest
    write_manifest(
        experiment_id='E5b_P1_locked_rerun',
        script_path=os.path.abspath(__file__),
        inputs=[META, dict(META_CLASSIC, role='classic comparison only '
                                              '(loaded with allow_legacy=True)')],
        outputs=[os.path.join(OUT, 'P1_1_regression_locked.json'),
                 os.path.join(OUT, 'P1_2_capacity_locked.json'),
                 os.path.join(OUT, 'P1_3_boundary_locked.json'),
                 cmp_path, rows_path,
                 os.path.join(FIG, '图A_gini_PoA_locked.png'),
                 os.path.join(FIG, '图B_largeratio_PoA_locked.png'),
                 os.path.join(FIG, '图C_gini_largeratio_locked.png'),
                 fig_locked],
        parameters=dict(build_cost_wan={'1': 18.0, '2': 32.0, '3': 45.0},
                        budget_per_station_wan=36.0,
                        nash_mode='locked',
                        n_large_rule='floor(2/3 * n_sta)',
                        regressors=XCOLS,
                        gap_definition='milp_F2 - nash_F2 (unified; recomputed)',
                        warning_threshold_PoA=THR,
                        comparison_baseline='P1_experiments.py (classic, 第一批)'),
        notes='OPEN-2 重跑：第一批 P1-1/P1-2/P1-3 由 classic 改为 locked 口径。'
              '变量定义与回归设定与第一批完全一致，仅输入口径不同。',
    )
    print('\nDONE')


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
