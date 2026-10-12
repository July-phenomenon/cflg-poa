"""
连通度 vs 差距（PoA）关系分析 —— 纯读取版本（BUG-05 修复）

此前该脚本用自己的 RNG 重新生成几何并重新跑 MILP，而 Python 的 hash()
种子在每个进程里随机加盐，导致“同一城市”在不同脚本里拿到不同几何，
连通度与差距不是在同一实例上算的，相关系数不可信（人为虚假相关）。

本版本不再重新生成几何，也不重新求解：直接读取
`21城市精确MILP验证结果_最终版.json`（该脚本已经在同一几何实例上同时算了
connectivity 与 gap / PoA），从而连通度与差距来自完全一致的数据，
相关系数才可信。
"""
import json
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():

    DATA_PATH = os.environ.get("CFLG_COMPETITION_DIR", r"D:/competition_data")
    OUT_JSON = os.path.join(os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))), "results", "main", "connectivity_regression.json")
    FIG_DIR = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    FIG_PATH = os.path.join(FIG_DIR, "results", "figures", "connectivity_gap_scatter.png")

    print("=" * 70)
    print("连通度 vs 差距（PoA）关系分析 —— 读取 21 城精确 MILP 结果")
    print("=" * 70)

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']  # city-keyed dict
    rows = []
    for city, r in results.items():
        rows.append({
            'city': city,
            'n_communities': r.get('n'),
            'cluster_r': r.get('cluster_r'),
            'n_stations': r.get('n_stations'),
            'connectivity': r.get('connectivity'),
            'nash_F2': r.get('nash_F2'),
            'milp_F2': r.get('milp_F2'),
            'gap': r.get('gap'),
            'gap_pct': r.get('gap_pct'),
            'PoA': r.get('PoA'),
            'Loss_pct': r.get('Loss_pct'),
        })

    print(f"Loaded {len(rows)} cities from 21-city JSON.")

    conn = np.array([d['connectivity'] for d in rows], dtype=float)
    gap = np.array([d['gap'] for d in rows], dtype=float)
    poa = np.array([d['PoA'] for d in rows if d['PoA'] is not None], dtype=float)
    loss = np.array([d['Loss_pct'] for d in rows if d['Loss_pct'] is not None], dtype=float)

    def safe_corr(a, b):
        if len(a) >= 3 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
            return float(np.corrcoef(a, b)[0, 1])
        return None

    corr_conn_gap = safe_corr(conn, gap)
    corr_conn_poa = safe_corr(conn, poa) if len(poa) == len(conn) else None
    corr_conn_loss = safe_corr(conn, loss) if len(loss) == len(conn) else None

    nash_wins = sum(1 for d in rows if d['gap'] is not None and d['gap'] > 0.001)

    summary = {
        'n_cities': len(rows),
        'mean_connectivity': round(float(np.mean(conn)), 4),
        'mean_gap': round(float(np.mean(gap)), 4),
        'mean_PoA': round(float(np.mean(poa)), 4) if len(poa) else None,
        'mean_Loss_pct': round(float(np.mean(loss)), 2) if len(loss) else None,
        'correlation_connectivity_gap': round(corr_conn_gap, 4) if corr_conn_gap is not None else None,
        'correlation_connectivity_PoA': round(corr_conn_poa, 4) if corr_conn_poa is not None else None,
        'correlation_connectivity_Loss': round(corr_conn_loss, 4) if corr_conn_loss is not None else None,
        'nash_win_rate_pct': round(nash_wins / len(rows) * 100, 1),
        'note': 'connectivity 与 gap/PoA 均取自同一几何实例（21城市精确MILP验证.py 输出），无跨进程 RNG 不一致问题',
    }

    out = {
        'experiment': '连通度与差距关系分析 (reader)',
        'connectivity_formula': 'avg over communities of (#stations within 1000m) / n_stations',
        # [2026-09-30 审计修正] 原硬编码为 '21城市精确MILP验证结果.json'，与实际读取的
        # DATA_PATH(_最终版.json) 不符，导致追溯链断裂（D2 误用旧文件的同类缺陷）。
        # 改为由实际输入路径自动派生，杜绝将来再次脱节。
        'data_source': os.path.basename(DATA_PATH),
        'data_source_full_path': DATA_PATH,
        'nash_mode': 'classic',  # 该输入为 classic 口径；论文主口径为 locked
        'summary': summary,
        'data': rows,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_JSON}")

    # ---- 散点图 ----
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(conn, loss, c=gap, cmap='coolwarm', s=60, edgecolor='k', linewidth=0.5)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Connectivity (avg #stations within 1000m / n_stations)')
    ax.set_ylabel('Efficiency loss vs centralized optimum (%)  [= (1-1/PoA)x100]')
    ax.set_title('Connectivity vs PoA efficiency loss (21 cities)')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Gap = Nash F2 - MILP F2')
    if corr_conn_loss is not None:
        ax.text(0.05, 0.95, f"Pearson r(connectivity, loss) = {corr_conn_loss:.3f}",
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved figure to {FIG_PATH}")

    print("\n=== Scatter data (sorted by connectivity) ===")
    print(f"{'City':<12}{'Conn':<10}{'Gap':<10}{'PoA':<8}{'Loss%':<8}")
    for d in sorted(rows, key=lambda x: (x['connectivity'] if x['connectivity'] is not None else -1)):
        print(f"{d['city']:<12}{str(d['connectivity']):<10}{str(d['gap']):<10}"
              f"{str(d['PoA']):<8}{str(d['Loss_pct']):<8}")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
