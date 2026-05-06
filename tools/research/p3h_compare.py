#!/usr/bin/env python3
"""P3h dry-run analysis: 7GT baseline vs diverged-reset.

Stop-gate criteria (any-of):
  - 24-36s ERLE_inst median improves >= +2 dB
  - 24-36s ERLE_win improves >= +1 dB
  - 24-36s far-active filter_state in {refined_usable} improves >=+10 pp
  - reset fires at all (existence proof — followed by recovery)
"""
import csv
import math
from collections import Counter

def load(p):
    return list(csv.DictReader(open(p)))

base = load('/tmp/p3h_baseline_7gt.csv')
p3h  = load('/tmp/p3h_on_7gt.csv')

def slice_t(rows, lo, hi, far_only=True):
    out = [r for r in rows if lo <= float(r['time_s']) < hi]
    if far_only:
        out = [r for r in out if float(r['far_act']) > 0.3]
    return out

def med(rs, key):
    vals = sorted(float(r[key]) for r in rs)
    return vals[len(vals)//2] if vals else float('nan')

def state_dist(rs):
    return Counter(r['filter_state'] for r in rs)

print('=' * 70)
print('P3h dry-run: 7GTxyT_doubletalk  baseline vs diverged_reset_enabled')
print('=' * 70)

# Reset firings
reset_fires = [(float(r['time_s']), int(r['p3h_reset_count']))
               for r in p3h if int(r['p3h_reset_fired'])]
print(f'\nReset firings: {len(reset_fires)}')
for t, c in reset_fires:
    print(f'  t={t:.2f}s  reset_count={c}')

# 24-36s comparison
for lo, hi in [(8.0, 12.0), (12.0, 24.0), (24.0, 36.0)]:
    print(f'\n--- {lo:.0f}-{hi:.0f}s far-active ---')
    b = slice_t(base, lo, hi)
    p = slice_t(p3h,  lo, hi)
    print(f'  frames: base={len(b)}  p3h={len(p)}')
    for k in ['erle_inst_db', 'erle_win_db', 'main_err_ratio',
             'p3f_shadow_advantage']:
        bv, pv = med(b, k), med(p, k)
        print(f'  {k:>22}: base={bv:+.2f}  p3h={pv:+.2f}  Δ={pv-bv:+.2f}')
    print(f'  state dist base: {dict(state_dist(b))}')
    print(f'  state dist p3h : {dict(state_dist(p))}')

# Whole-trace once_converged_pct
b_oc = sum(int(r['once_conv']) for r in base)
p_oc = sum(int(r['once_conv']) for r in p3h)
print(f'\nonce_converged frames: base={b_oc}/{len(base)} p3h={p_oc}/{len(p3h)}')
