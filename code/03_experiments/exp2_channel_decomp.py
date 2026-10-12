# -*- coding: utf-8 -*-
"""
V1: 通道分解主图（论文 Figure X）
Panel A: 三通道贡献箱线图   W1*dA1 / W2*dA2 / W3*S3*dcov
Panel B: 通道中位占比分层堆积柱图（S2 为负，画在零线下方）
Panel C: Δcov vs PoA 散点图（蓝=覆盖锁定 Δcov=0，橙=未锁 Δcov>0）

口径：W1=0.2, W2=0.3, W3=0.5, S3=0.60（枚举设定，S3 为常量标量）
恒等式：F2 = W1*A1 + W2*A2 + W3*S3*cov  →  dA1 由残差反解（已自检，最大偏差 2.1e-6）
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():

    BASE = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    DATA = os.path.join(BASE, 'poa_full_per_config_s2.json')
    OUTDIR = os.path.join(BASE, "results", "figures")
    os.makedirs(OUTDIR, exist_ok=True)

    W1, W2, W3, S3 = 0.2, 0.3, 0.5, 0.60

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.unicode_minus': False,
        'axes.labelsize': 9.5,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8,
        'figure.dpi': 150,
    })

    # ---------- 载入 ----------
    d = json.load(open(DATA, encoding='utf-8'))
    rows = [x for x in d['results'] if x.get('nash_S2_weighted') is not None and x.get('PoA')]
    print(f'有效样本 n = {len(rows)}')

    poa = np.array([x['PoA'] for x in rows])
    f1n = np.array([x['nash_F1'] for x in rows])
    f1m = np.array([x['milp_F1'] for x in rows])
    f2n = np.array([x['nash_F2'] for x in rows])
    f2m = np.array([x['milp_F2'] for x in rows])
    s2n = np.array([x['nash_S2_weighted'] for x in rows])
    s2m = np.array([x['milp_S2_weighted'] for x in rows])

    dcov = f1m - f1n                      # 覆盖通道（原始）
    dA2 = s2m - s2n                       # 利用率通道（原始，MILP - Nash）
    # 反解 A1：A1 = (F2 - W2*A2 - W3*S3*cov)/W1
    A1n = (f2n - W2 * s2n - W3 * S3 * f1n) / W1
    A1m = (f2m - W2 * s2m - W3 * S3 * f1m) / W1
    dA1 = A1m - A1n

    ch1 = W1 * dA1          # 指派质量通道
    ch2 = W2 * dA2          # 拥堵/利用率通道
    chc = W3 * S3 * dcov    # 覆盖通道
    dF2 = f2m - f2n

    # 恒等式自检（每配置：三通道之和应等于 ΔF2）
    resid = np.abs(ch1 + ch2 + chc - dF2)
    print(f'恒等式自检：max|Σ通道 − ΔF2| = {resid.max():.3e}')

    locked = dcov <= 1e-9
    print(f'覆盖锁定层 n={locked.sum()} (Δcov=0)，未锁层 n={(~locked).sum()}')


    def _share(mask, ch):
        den = np.where(np.abs(dF2[mask]) > 1e-9, dF2[mask], np.nan)
        v = ch[mask] / den
        v = v[~np.isnan(v)]
        return float(np.median(v))


    shares = {
        'Δcov = 0': {'S1': _share(locked, ch1), 'S2': _share(locked, ch2), 'cov': _share(locked, chc)},
        'Δcov > 0': {'S1': _share(~locked, ch1), 'S2': _share(~locked, ch2), 'cov': _share(~locked, chc)},
        'All': {'S1': _share(np.ones(len(poa), bool), ch1),
                'S2': _share(np.ones(len(poa), bool), ch2),
                'cov': _share(np.ones(len(poa), bool), chc)},
    }
    print('通道中位占比:', json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in shares.items()},
                                      ensure_ascii=False))

    # ---------- 画布 ----------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.15))
    COL_S1, COL_S2, COL_COV = '#2b6cb0', '#c53030', '#2f855a'
    COL_LOCK, COL_UNLOCK = '#2b6cb0', '#dd6b20'

    # ===== Panel A: 箱线图 =====
    ax = axes[0]
    data = [ch1, ch2, chc]
    bp = ax.boxplot(data, vert=True, widths=0.55, whis=[5, 95], showfliers=False,
                    patch_artist=True, medianprops=dict(color='black', lw=1.4),
                    boxprops=dict(lw=0.9), whiskerprops=dict(lw=0.9), capprops=dict(lw=0.9))
    for patch, col in zip(bp['boxes'], [COL_S1, COL_S2, COL_COV]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax.axhline(0, color='#4a5568', lw=1.0, ls='-')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([r'$\alpha\,\Delta A_1$' + '\n(assignment)',
                        r'$\beta\,\Delta A_2$' + '\n(congestion)',
                        r'$\gamma S_3\,\Delta cov$' + '\n(coverage)'])
    ax.set_ylabel('Contribution to  ' + r'$F_2^{\mathrm{MILP}}-F_2^{\mathrm{Nash}}$')
    ax.set_title('A. Channel contributions', loc='left', fontweight='bold')
    for i, ch in enumerate(data, start=1):
        if i < 3:
            ax.annotate(f'med {np.median(ch):+.4f}', xy=(i, np.median(ch)),
                        xytext=(i + 0.30, np.median(ch)), fontsize=7.5, va='center', color='#2d3748')
        else:  # 最右侧箱体：标签放左边，避免裁剪
            ax.annotate(f'med {np.median(ch):+.4f}', xy=(i, np.median(ch)),
                        xytext=(i - 0.33, np.median(ch)), fontsize=7.5, va='center',
                        ha='right', color='#2d3748')
    ax.annotate('congestion channel\nis NEGATIVE on average',
                xy=(2, np.median(ch2)), xytext=(1.05, -0.030), fontsize=7.5, color=COL_S2,
                arrowprops=dict(arrowstyle='->', color=COL_S2, lw=0.9))
    ax.set_ylim(-0.038, 0.075)
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # ===== Panel B: 分层占比（发散堆积柱）=====
    ax = axes[1]
    groups = ['Δcov = 0', 'Δcov > 0', 'All']
    xpos = np.arange(len(groups))
    w = 0.52
    for i, g in enumerate(groups):
        s1, s2, cv = shares[g]['S1'], shares[g]['S2'], shares[g]['cov']
        # 正向：coverage 在下，S1 在上
        ax.bar(xpos[i], max(cv, 0), w, color=COL_COV, alpha=0.85, label='coverage' if i == 0 else None)
        ax.bar(xpos[i], max(s1, 0), w, bottom=max(cv, 0), color=COL_S1, alpha=0.85,
               label='S1 (assignment)' if i == 0 else None)
        # 负向：S2
        if s2 < 0:
            ax.bar(xpos[i], s2, w, color=COL_S2, alpha=0.85, label='S2 (congestion)' if i == 0 else None)
            ax.annotate(f'{s2:.2f}', xy=(xpos[i], s2), xytext=(0, -11),
                        textcoords='offset points', ha='center', fontsize=7.5, color=COL_S2)
        ax.annotate(f'{s1:.2f}', xy=(xpos[i], max(cv, 0) + max(s1, 0) / 2), ha='center', va='center',
                    fontsize=7.5, color='white', fontweight='bold')
        if cv > 0:
            ax.annotate(f'{cv:.2f}', xy=(xpos[i], max(cv, 0) / 2), ha='center', va='center',
                        fontsize=7.5, color='white', fontweight='bold')
    ax.axhline(0, color='#4a5568', lw=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(['coverage locked\n' + r'($\Delta cov=0$)', 'coverage unlocked\n' + r'($\Delta cov>0$)', 'all'])
    ax.set_ylabel('Median share of  ' + r'$\Delta F_2$')
    ax.set_title('B. Channel shares by coverage-lock status', loc='left', fontweight='bold')
    ax.set_ylim(-0.32, 1.32)
    ax.legend(loc='lower left', frameon=False, fontsize=7.5)
    ax.annotate('residual gap is\n100% assignment quality',
                xy=(0.03, 1.02), xytext=(0.28, 1.21), fontsize=7.5, color=COL_S1,
                arrowprops=dict(arrowstyle='->', color=COL_S1, lw=0.9))
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # ===== Panel C: Δcov vs PoA =====
    ax = axes[2]
    rng = np.random.default_rng(20260930)
    jit = rng.uniform(-0.0025, 0.0025, size=locked.sum())
    ax.scatter(dcov[~locked], poa[~locked], s=3.5, c=COL_UNLOCK, alpha=0.28,
               edgecolors='none', label=rf'unlocked ($\Delta cov>0$, n={(~locked).sum()})')
    ax.scatter(dcov[locked] + jit, poa[locked], s=6, c=COL_LOCK, alpha=0.65,
               edgecolors='none', label=rf'locked ($\Delta cov=0$, n={locked.sum()})')
    ax.axhline(1.0, color='#718096', lw=0.9, ls=':')
    ax.axhline(1.0576, color=COL_LOCK, lw=1.3, ls='--')
    ax.axhline(1.2339, color=COL_UNLOCK, lw=1.3, ls='--')
    ax.annotate('locked-layer bound 1.0576', xy=(0.16, 1.0576), xytext=(0.155, 1.075),
                fontsize=7.5, color=COL_LOCK)
    ax.annotate('full-sample bound 1.2339', xy=(0.16, 1.2339), xytext=(0.155, 1.252),
                fontsize=7.5, color=COL_UNLOCK)
    ax.set_xlabel(r'$\Delta cov \;=\; F_1^{\mathrm{MILP}}-F_1^{\mathrm{Nash}}$')
    ax.set_ylabel('PoA  ' + r'$=\;F_2^{\mathrm{MILP}}/F_2^{\mathrm{Nash}}$')
    ax.set_title('C. PoA vs. coverage gap', loc='left', fontweight='bold')
    ax.set_xlim(-0.008, 0.278)
    ax.set_ylim(0.90, 1.47)
    ax.legend(loc='lower right', frameon=False, fontsize=7.5, markerscale=2.2)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    fig.suptitle('Channel decomposition of the efficiency gap  '
                 r'($n = 15{,}207$ enumerated configurations)',
                 fontsize=11.5, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    for ext in ('svg', 'pdf', 'png'):
        p = os.path.join(OUTDIR, f'Figure_channel_decomposition.{ext}')
        fig.savefig(p, bbox_inches='tight')
        print('saved', p, os.path.getsize(p), 'bytes')

    # ---------- 供素材复用的数值 ----------
    stats = {
        'n': int(len(rows)),
        'identity_max_residual': float(resid.max()),
        'channel_contribution': {
            'W1_dA1': {k: round(float(v), 5) for k, v in zip(
                ['mean', 'median', 'p05', 'p95', 'max', 'min'],
                [ch1.mean(), np.median(ch1), np.percentile(ch1, 5), np.percentile(ch1, 95), ch1.max(), ch1.min()])},
            'W2_dA2': {k: round(float(v), 5) for k, v in zip(
                ['mean', 'median', 'p05', 'p95', 'max', 'min'],
                [ch2.mean(), np.median(ch2), np.percentile(ch2, 5), np.percentile(ch2, 95), ch2.max(), ch2.min()])},
            'W3_S3_dcov': {k: round(float(v), 5) for k, v in zip(
                ['mean', 'median', 'p05', 'p95', 'max', 'min'],
                [chc.mean(), np.median(chc), np.percentile(chc, 5), np.percentile(chc, 95), chc.max(), chc.min()])},
        },
        'channel_share_by_layer': {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in shares.items()},
        'layer_n': {'locked': int(locked.sum()), 'unlocked': int((~locked).sum())},
    }
    with open(os.path.join(OUTDIR, 'Figure_channel_decomposition_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print('saved stats json')


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
