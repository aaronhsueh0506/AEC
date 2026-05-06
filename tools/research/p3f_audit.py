#!/usr/bin/env python3
"""P3f Phase 2 invariant audit. Reads 5 trace CSVs and reports state
distributions + invariant pass/fail."""
import csv
import sys
from collections import Counter

CASES = [
    ('7GT_DT', '/tmp/p3f_7GTxyTksSUqCnP5y0ILG4A_doubletalk.csv'),
    ('FS_static_0Kjz', '/tmp/p3f_0KjzXA3g20qsd8zmSekADw_farend_singletalk.csv'),
    ('DT_static_0I0X', '/tmp/p3f_0I0XMl3M0ECO0U1N0cJvpg_doubletalk.csv'),
    ('FS_movement_0I0X', '/tmp/p3f_0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement.csv'),
    ('DT_movement_49II', '/tmp/p3f_49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement.csv'),
]

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def in_window(rows, t_lo, t_hi):
    return [r for r in rows if t_lo <= float(r['time_s']) < t_hi]

def state_dist(rows):
    return Counter(r['filter_state'] for r in rows)

def pct(c, n):
    return {k: f'{v}/{n} ({100*v/n:.0f}%)' for k, v in c.items()}

print('=' * 80)
for name, path in CASES:
    try:
        rows = load(path)
    except FileNotFoundError:
        print(f'{name}: missing CSV {path}')
        continue
    n = len(rows)
    active = [r for r in rows if float(r['far_act']) > 0.3]
    print(f'\n[{name}]  total={n}  far_active={len(active)}')
    print(f'  all-frame dist:    {dict(state_dist(rows))}')
    print(f'  far-active dist:   {dict(state_dist(active))}')

# 7GT-specific invariants (user's spec)
print('\n' + '=' * 80)
print('7GT invariants:')
rows = load('/tmp/p3f_7GTxyTksSUqCnP5y0ILG4A_doubletalk.csv')
def ne_evidence(r):
    return float(r['dt_energy']) > 0.3 or float(r['dt_shadow']) > 0.5
for label, lo, hi, expect, ne_only in [
    ('post-delay 4.6-8s', 4.6, 8.0, {'coarse_learning'}, False),
    ('post-delay 8-12s (refined or suspicious = filter matured)', 8.0, 12.0,
     {'refined_usable', 'suspicious_dt'}, False),
    ('NE contam 24-36s (NE-evidence frames)', 24.0, 36.0,
     {'suspicious_dt', 'diverged'}, True),
]:
    win = in_window(rows, lo, hi)
    win_act = [r for r in win if float(r['far_act']) > 0.3]
    if ne_only:
        win_act = [r for r in win_act if ne_evidence(r)]
    d = state_dist(win_act)
    n_active = len(win_act)
    if n_active == 0:
        print(f'  {label}: no qualifying frames, skip')
        continue
    expect_count = sum(d.get(s, 0) for s in expect)
    pass_pct = 100 * expect_count / n_active
    flag = 'PASS' if pass_pct >= 50 else 'FAIL'
    print(f'  {label}: expect {expect}, got {dict(d)} → {flag} ({pass_pct:.0f}% of {n_active})')

# FS invariant: early frames (0-3s after delay-solid) should NOT be suspicious_dt
print('\nFS invariants (no suspicious_dt during pure FS):')
for name, path in [('FS_static_0Kjz', '/tmp/p3f_0KjzXA3g20qsd8zmSekADw_farend_singletalk.csv'),
                    ('FS_movement_0I0X', '/tmp/p3f_0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement.csv')]:
    rows = load(path)
    active = [r for r in rows if float(r['far_act']) > 0.3]
    susp = sum(1 for r in active if r['filter_state'] == 'suspicious_dt')
    div = sum(1 for r in active if r['filter_state'] == 'diverged')
    flag_susp = 'PASS' if susp / max(len(active), 1) < 0.05 else 'FAIL'
    print(f'  {name}: suspicious_dt {susp}/{len(active)} ({100*susp/max(len(active),1):.1f}%) → {flag_susp}')
    print(f'           diverged    {div}/{len(active)} ({100*div/max(len(active),1):.1f}%)')

# DT invariant: on NE-evidence frames, suspicious_dt + diverged should
# dominate over refined_usable
print('\nDT invariants (NE-evidence frames only):')
for name, path in [('DT_static_0I0X', '/tmp/p3f_0I0XMl3M0ECO0U1N0cJvpg_doubletalk.csv'),
                    ('DT_movement_49II', '/tmp/p3f_49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement.csv')]:
    rows = load(path)
    active = [r for r in rows if float(r['far_act']) > 0.3]
    ne = [r for r in active if ne_evidence(r)]
    if not ne:
        print(f'  {name}: no NE-evidence frames')
        continue
    d = state_dist(ne)
    refined = d.get('refined_usable', 0)
    susp = d.get('suspicious_dt', 0)
    div = d.get('diverged', 0)
    coarse = d.get('coarse_learning', 0)
    flagged = susp + div
    flag = 'PASS' if flagged >= len(ne) * 0.30 else 'FAIL'
    print(f'  {name}: NE-evidence frames {len(ne)}/{len(active)}; '
          f'flagged(susp+div)={flagged} ({100*flagged/len(ne):.0f}%) → {flag}')
    print(f'           dist: {dict(d)}')
