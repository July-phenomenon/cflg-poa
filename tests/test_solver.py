# -*- coding: utf-8 -*-
"""R4 口径锁定机制自检：正向通过 + 负向必须崩溃。"""
import os, sys, json

_ROOT = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))
_SOLVER = os.path.join(_ROOT, "code", "02_core_solver")
if _SOLVER not in sys.path:
    sys.path.insert(0, _SOLVER)
from core_solver import (resolve_canonical, verify_input, write_manifest,
                         CANONICAL_INPUTS, FORBIDDEN_INPUTS)

BASE = os.environ.get("CFLG_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
ok = fail = 0


def check(name, fn, should_pass):
    global ok, fail
    try:
        fn()
        passed = True
        detail = ''
    except FileNotFoundError as e:
        # Canonical result files are intentionally not redistributed
        # (see data/external/DATA_SOURCES.md, section 2.2).  A missing input is
        # an environment condition, not a test failure.
        print('  SKIP ' + name + ' (canonical input not present)')
        return
    except Exception as e:
        passed = False
        detail = str(e).splitlines()[0:2]
    good = (passed == should_pass)
    print(('  PASS ' if good else '  FAIL ') + name + ('' if good else f' -> {detail}'))
    if good:
        ok += 1
    else:
        fail += 1


print('=== 1. 正向：规范输入登记表两个 key 均通过 ===')
for key in CANONICAL_INPUTS:
    def _f(k=key):
        m = resolve_canonical(k)
        print(f'      {k}: sha256={m["sha256"][:16]}... n_fields_verified=ok')
    check(f'resolve_canonical({key})', _f, True)

print('=== 2. 负向：黑名单文件必须被拒绝 ===')
for bad in FORBIDDEN_INPUTS:
    p = os.path.join(BASE, "results", "main", bad)
    if not os.path.exists(p):
        p = os.path.join(BASE, bad)
    if not os.path.exists(p):
        print(f'  SKIP {bad}（文件不存在，无法测）')
        continue
    check(f'reject {bad}',
          lambda p=p: verify_input(p, {}), False)

print('=== 3. 负向：字段不符必须崩溃 ===')
good_path = os.path.join(BASE, "results", "main",
                         '21城市精确MILP验证结果_两阶段版_withS2.json')
check('nash_mode 期望 locked、实为 locked -> 通过',
      lambda: verify_input(good_path, {'nash_mode': 'locked'}), True)
check('nash_mode 期望 classic -> 必须崩溃',
      lambda: verify_input(good_path, {'nash_mode': 'classic'}), False)
check('budget 期望 25.0 -> 必须崩溃',
      lambda: verify_input(good_path, {'budget_per_station_wan': 25.0}), False)
check('counts results=20 -> 必须崩溃',
      lambda: verify_input(good_path, counts={'results': 20}), False)
check('record_fields 要求不存在的字段 -> 必须崩溃',
      lambda: verify_input(good_path, record_fields={
          'results': {'fields': ['no_such_field'], 'min_fraction': 1.0}}), False)

print('=== 4. 负向：枚举用无 S2 旧文件必须被拒绝 ===')
old_enum = os.path.join(BASE, 'poa_full_per_config.json')
check('reject poa_full_per_config.json (无 S2)',
      lambda: verify_input(old_enum, counts={'results': 15208}), False)
check('同文件 allow_legacy=True 可豁免',
      lambda: verify_input(old_enum, counts={'results': 15208}, allow_legacy=True), True)

print('=== 5. write_manifest 落盘 ===')
# Needs an archived canonical result file, which is intentionally not
# redistributed (see data/external/DATA_SOURCES.md, section 2.2).  Skip with a
# clear message in a fresh checkout instead of failing.
try:
    m = resolve_canonical('cities21_locked_s2')
    mp = write_manifest('R4_selftest', __file__, [m], [__file__],
                        parameters={'selftest': True},
                        notes='R4 机制自检产生的示例 manifest')
    check('manifest 文件已写出且可解析',
          lambda: json.load(open(mp, encoding='utf-8')), True)
except FileNotFoundError as _e:
    print('  SKIP write_manifest: canonical result file not present '
          '(archived outputs are not redistributed) ->', _e)

print(f'\n=== 自检结果: PASS={ok}  FAIL={fail} ===')
sys.exit(1 if fail else 0)
