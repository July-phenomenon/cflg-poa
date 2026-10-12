# -*- coding: utf-8 -*-
"""Unit tests for the satisfaction functions S1 (distance) and S2 (utilisation).

Both are step functions taken verbatim from the competition specification and
recorded in code/utils/constants.py.  These tests pin the ladder values and the
behaviour at the breakpoints, which is where an off-by-one would silently
change every reported PoA.

    python tests/test_satisfaction.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SOLVER = os.path.join(_ROOT, "code", "02_core_solver")
if _SOLVER not in sys.path:
    sys.path.insert(0, _SOLVER)

from core_solver import s1_fn, s2_fn, S1_T, S2_T      # noqa: E402

ok = fail = 0


def eq(name, got, want, tol=1e-12):
    global ok, fail
    good = abs(got - want) <= tol
    print(("  PASS " if good else "  FAIL ") + name +
          ("" if good else f"  got={got!r} want={want!r}"))
    if good:
        ok += 1
    else:
        fail += 1


def le(name, a, b):
    """assert a <= b"""
    global ok, fail
    good = a <= b + 1e-12
    print(("  PASS " if good else "  FAIL ") + name +
          ("" if good else f"  {a!r} > {b!r}"))
    if good:
        ok += 1
    else:
        fail += 1


print("=== S1: distance -> satisfaction (step function, endpoints inclusive) ===")
# ladder: (300, 1.0), (500, 0.9), (650, 0.75), (1000, 0.6)
eq("S1(0)", s1_fn(0.0), 1.0)
eq("S1(300) == 1.0   (breakpoint is inclusive)", s1_fn(300.0), 1.0)
eq("S1(300.001) == 0.9", s1_fn(300.001), 0.9)
eq("S1(500) == 0.9", s1_fn(500.0), 0.9)
eq("S1(650) == 0.75", s1_fn(650.0), 0.75)
eq("S1(1000) == 0.6", s1_fn(1000.0), 0.6)
eq("S1(1000.001) == 0  (out of reach)", s1_fn(1000.001), 0.0)
eq("S1(50000) == 0", s1_fn(50000.0), 0.0)

print("=== S1: monotone non-increasing in distance ===")
prev = s1_fn(0.0)
for d in range(0, 1300, 25):
    le(f"S1({d}) <= previous", s1_fn(float(d)), prev)
    prev = s1_fn(float(d))

print("=== S2: utilisation -> satisfaction (step function) ===")
# ladder: (0.6, 1.0), (0.75, 0.93), (0.85, 0.85), (0.95, 0.72), (1.0, 0.6)
eq("S2(0) == 1.0  (empty station is perfectly satisfying)", s2_fn(0.0), 1.0)
eq("S2(0.6) == 1.0", s2_fn(0.6), 1.0)
eq("S2(0.75) == 0.93", s2_fn(0.75), 0.93)
eq("S2(0.85) == 0.85", s2_fn(0.85), 0.85)
eq("S2(0.95) == 0.72", s2_fn(0.95), 0.72)
eq("S2(1.0) == 0.6", s2_fn(1.0), 0.6)
eq("S2(1.5) == 0.6  (overload floors at the last step)", s2_fn(1.5), 0.6)
eq("S2(100) == 0.6", s2_fn(100.0), 0.6)

print("=== S2: monotone non-increasing in utilisation ===")
prev = s2_fn(0.0)
for i in range(0, 121, 5):
    u = i / 100.0
    le(f"S2({u:.2f}) <= previous", s2_fn(u), prev)
    prev = s2_fn(u)

print("=== ladders match code/utils/constants.py ===")
eq("len(S1_T)", float(len(S1_T)), 4.0)
eq("len(S2_T)", float(len(S2_T)), 5.0)
eq("S1_T first step", S1_T[0][1], 1.0)
eq("S2_T last step", S2_T[-1][1], 0.6)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
