#!/usr/bin/env python3
"""P3g Phase 0 dry-run audit. Compare linear vs render residual per-state
on 5 trace cases."""
import csv
import math
from collections import defaultdict

CASES = [
    ('7GT_DT', '/tmp/p3f_7GTxyTksSUqCnP5y0ILG4A_doubletalk.csv'),
    ('FS_static_0Kjz', '/tmp/p3f_0KjzXA3g20qsd8zmSekADw_farend_singletalk.csv'),
    ('DT_static_0I0X', '/tmp/p3f_0I0XMl3M0ECO0U1N0cJvpg_doubletalk.csv'),
    ('FS_movement_0I0X', '/tmp/p3f_0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement.csv'),
    ('DT_movement_49II', '/tmp/p3f_49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement.csv'),
]

def db(x):
    return 10.0 * math.log10(max(x, 1e-12))

print('=' * 90)
print('P3g Phase 0 — residual-source dry-run audit')
print('Per-state median residual PSD (linear vs render-blended), '
      'far-active frames only.')
print('Render column shown only on frames where using_render=1 '
      '(otherwise NaN).')
print('=' * 90)

for name, path in CASES:
    rows = list(csv.DictReader(open(path)))
    by_state = defaultdict(list)
    for r in rows:
        if float(r['far_act']) < 0.3:
            continue
        st = r['filter_state']
        lin = float(r['residual_psd_linear'])
        ren = float(r['residual_psd_render'])
        ur = int(r['using_render'])
        by_state[st].append((lin, ren, ur))
    print(f'\n[{name}]')
    print(f'  state            n  | median lin_dB | median ren_dB (when ur=1) | render_dominance(dB)')
    print(f'  {"-"*88}')
    for st in ['idle', 'startup', 'coarse_learning', 'refined_usable',
               'suspicious_dt', 'diverged']:
        items = by_state.get(st, [])
        if not items:
            continue
        ur_items = [(l, r) for l, r, ur in items if ur]
        lins_all = sorted(l for l, _, _ in items)
        if not lins_all:
            continue
        lin_med = lins_all[len(lins_all)//2]
        if ur_items:
            ren_sorted = sorted(r for _, r in ur_items)
            lin_when_ur = sorted(l for l, _ in ur_items)
            ren_med = ren_sorted[len(ren_sorted)//2]
            lin_ur_med = lin_when_ur[len(lin_when_ur)//2]
            dom_db = db(ren_med) - db(lin_ur_med)
            ur_pct = 100 * len(ur_items) / len(items)
            print(f'  {st:<15} {len(items):4d}  | {db(lin_med):+8.1f}     | '
                  f'{db(ren_med):+8.1f} (ur={ur_pct:.0f}%)        | {dom_db:+6.1f}')
        else:
            print(f'  {st:<15} {len(items):4d}  | {db(lin_med):+8.1f}     | '
                  f'  (no using_render frames)        | --')

# Critical questions per user spec:
# 1) On 7GT 24-36s NE-evidence frames: is render >> linear (over-suppression
#    that hurts NE preservation)?
# 2) On FS_static / FS_movement frames where usable_linear=True (state==
#    refined_usable): is linear ≈ render (so soft blend wouldn't leak)?
print('\n' + '=' * 90)
print('Targeted comparison')
print('=' * 90)

def ne_evidence(r):
    return float(r['dt_energy']) > 0.3 or float(r['dt_shadow']) > 0.5

# 7GT 24-36s NE frames
rows = list(csv.DictReader(open('/tmp/p3f_7GTxyTksSUqCnP5y0ILG4A_doubletalk.csv')))
seven_ne = [r for r in rows
            if 24.0 <= float(r['time_s']) < 36.0
            and float(r['far_act']) > 0.3 and ne_evidence(r)]
ren = [r for r in seven_ne if int(r['using_render'])]
print(f'\n[7GT 24-36s NE-evidence] total {len(seven_ne)}, using_render {len(ren)}')
if ren:
    lin_med = sorted([float(r['residual_psd_linear']) for r in ren])[len(ren)//2]
    ren_med = sorted([float(r['residual_psd_render']) for r in ren])[len(ren)//2]
    print(f'  linear median: {db(lin_med):+.1f} dB')
    print(f'  render median: {db(ren_med):+.1f} dB')
    print(f'  render dominance: {db(ren_med)-db(lin_med):+.1f} dB '
          f'(>0 = render OVER-suppresses, hint that linear may preserve NE)')

# FS_static refined_usable frames where usable_linear=True
for name, path in [('FS_static_0Kjz',
                    '/tmp/p3f_0KjzXA3g20qsd8zmSekADw_farend_singletalk.csv'),
                   ('FS_movement_0I0X',
                    '/tmp/p3f_0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement.csv')]:
    rows = list(csv.DictReader(open(path)))
    ul = [r for r in rows
          if int(r['usable_linear']) and float(r['far_act']) > 0.3]
    ul_ren = [r for r in ul if int(r['using_render'])]
    print(f'\n[{name} usable_linear=True frames] total {len(ul)}, '
          f'using_render(actual) {len(ul_ren)}')
    if ul_ren:
        lin_med = sorted([float(r['residual_psd_linear'])
                          for r in ul_ren])[len(ul_ren)//2]
        ren_med = sorted([float(r['residual_psd_render'])
                          for r in ul_ren])[len(ul_ren)//2]
        print(f'  linear median: {db(lin_med):+.1f} dB')
        print(f'  render median: {db(ren_med):+.1f} dB')
        print(f'  render dominance: {db(ren_med)-db(lin_med):+.1f} dB '
              f'(>0 = swap to linear would suppress LESS — risk of FS leak)')
