# -*- coding: utf-8 -*-
"""T2：合并 8 个分片的枚举结果为全量 poa_full_per_config_s2.json。

用法：python T2_merge_shards.py [--check]   (--check 只报告各片完成度，不合并)
"""
import json, os, sys, glob


def main():

    DATA = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    OUT = os.path.join(DATA, "results", "main", "poa_full_per_config_s2.json")
    N_TOTAL = 15208
    SHARDS = 8

    def load_shard(i):
        """优先读最终 json，否则退化为读 checkpoint（部分进度）。返回 (rows, src, complete)"""
        p = os.path.join(DATA, f'poa_full_per_config_s2_shard{i}.json')
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding='utf-8'))
                return d['results'], 'final', True
            except Exception as e:
                print(f'  shard{i}: 最终文件读取失败 {e}')
        cp = os.path.join(DATA, f'gap_cp_shard{i}.json')
        if os.path.exists(cp):
            try:
                d = json.load(open(cp, encoding='utf-8'))
                rows = d.get('results', [])
                return rows, f'checkpoint@{d.get("start_cfg")}', False
            except Exception as e:
                print(f'  shard{i}: checkpoint 读取失败 {e}')
        return [], 'none', False


    check_only = '--check' in sys.argv
    partial = '--partial' in sys.argv
    all_rows, ok = [], True
    for i in range(SHARDS):
        rows, src, complete = load_shard(i)
        if not rows:
            print(f'  shard{i}: 未产出')
            ok = False
            continue
        has_s2 = sum(1 for r in rows if r.get('delta_S2_eff') is not None)
        print(f'  shard{i}: {len(rows)} 条，含 S2 字段 {has_s2} 条 [{src}]')
        if has_s2 < len(rows):
            ok = False
        if not complete:
            ok = False
        all_rows.extend(rows)

    print(f'合计 {len(all_rows)} / {N_TOTAL}')
    if check_only:
        print('状态:', 'OK' if (ok and len(all_rows) >= N_TOTAL) else '未完成')
        sys.exit(0 if ok else 1)

    # --partial：允许用已完成的进度产出带 _partial 后缀的文件，供提前分析
    if partial:
        seen, merged = set(), []
        for r in all_rows:
            k = tuple(r['config'])
            if k in seen:
                continue
            seen.add(k)
            merged.append(r)
        pout = OUT.replace('.json', '_partial.json')
        json.dump({'results': merged, 'note': f'partial merge of {len(merged)} configs (in-progress shards)'},
                  open(pout, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f'[partial] 已写出 {pout}（{len(merged)} 条，非全量，仅供提前探查）')
        sys.exit(0)

    if len(all_rows) < N_TOTAL:
        print('[中止] 尚未跑完，未合并（避免产出不完整文件）')
        sys.exit(1)

    # 按 config 去重 + 保持枚举顺序（config 为长度10的元组列表）
    seen, merged = set(), []
    for r in all_rows:
        k = tuple(r['config'])
        if k in seen:
            continue
        seen.add(k)
        merged.append(r)
    print(f'去重后 {len(merged)} 条（原 {len(all_rows)}）')

    json.dump({'results': merged}, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'已写出 {OUT}')

    # 清理分片 checkpoint（保留分片 json 以便复核）
    for i in range(SHARDS):
        cp = os.path.join(DATA, f'gap_cp_shard{i}.json')
        if os.path.exists(cp):
            os.remove(cp)
    print('已清理分片 checkpoint')


# ---------------------------------------------------------------------------
# Entry point.  The body above used to run at import time; it now lives in
# main() so that the file can also be imported without side effects.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
