#!/usr/bin/env python3
"""P3c summary stats from 800-case time-to-first-solid CSV."""
import csv
import statistics
from collections import defaultdict, Counter

PATH = '/tmp/p3c_fp40_summary.csv'
rows = list(csv.DictReader(open(PATH)))
print(f'Total cases: {len(rows)}')

# Overall stats
ttfs = [float(r['time_to_first_solid_s']) for r in rows
        if r['ever_solid'].lower() == 'true']
not_solid = [r for r in rows if r['ever_solid'].lower() != 'true']
print(f'\nEver solid: {len(ttfs)} ({100*len(ttfs)/len(rows):.1f}%)')
print(f'Never solid: {len(not_solid)} ({100*len(not_solid)/len(rows):.1f}%)')

if ttfs:
    print(f'\nTime-to-first-solid distribution (s):')
    print(f'  median: {statistics.median(ttfs):.2f}')
    print(f'  mean:   {statistics.mean(ttfs):.2f}')
    print(f'  p25:    {statistics.quantiles(ttfs, n=4)[0]:.2f}')
    print(f'  p75:    {statistics.quantiles(ttfs, n=4)[2]:.2f}')
    print(f'  p90:    {statistics.quantiles(ttfs, n=10)[8]:.2f}')
    print(f'  p95:    {statistics.quantiles(ttfs, n=20)[18]:.2f}')
    print(f'  p99:    {statistics.quantiles(ttfs, n=100)[98]:.2f}')
    print(f'  max:    {max(ttfs):.2f}')

# Histogram bins
bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 999]
counts = [0] * (len(bins) - 1)
for v in ttfs:
    for i in range(len(bins) - 1):
        if bins[i] <= v < bins[i+1]:
            counts[i] += 1
            break
print(f'\nHistogram of time-to-first-solid:')
for i in range(len(bins) - 1):
    pct = 100 * counts[i] / len(rows)
    bar = '#' * int(pct * 1.5)
    print(f'  [{bins[i]:>4.1f}, {bins[i+1]:>4.1f}) s : {counts[i]:4d} ({pct:5.1f}%) {bar}')
print(f'  never_solid       : {len(not_solid):4d} ({100*len(not_solid)/len(rows):5.1f}%)')

# Per-scenario breakdown
print(f'\nPer-scenario:')
sc_groups = defaultdict(list)
sc_ns = defaultdict(int)
for r in rows:
    if r['ever_solid'].lower() == 'true':
        sc_groups[r['scenario']].append(float(r['time_to_first_solid_s']))
    else:
        sc_ns[r['scenario']] += 1
for sc in sorted(set(r['scenario'] for r in rows)):
    vs = sc_groups[sc]
    n_total = len(vs) + sc_ns[sc]
    if vs:
        print(f'  {sc:<14}  n={n_total:3d}  median={statistics.median(vs):5.2f}s  '
              f'p90={statistics.quantiles(vs, n=10)[8]:5.2f}s  '
              f'max={max(vs):5.2f}s  never_solid={sc_ns[sc]}')
    else:
        print(f'  {sc:<14}  n={n_total:3d}  no solid')

# Top-20 worst cases
print(f'\nTop 20 worst (longest time_to_first_solid, or never_solid):')
ranked = sorted(rows, key=lambda r:
                (-1, 0) if r['ever_solid'].lower() != 'true'
                else (1, -float(r['time_to_first_solid_s'])))
for r in ranked[:20]:
    if r['ever_solid'].lower() == 'true':
        print(f'  {r["stem"]:<60} {r["scenario"]:<13} '
              f'ttfs={float(r["time_to_first_solid_s"]):5.2f}s  '
              f'delay={r["delay_at_first_solid_ms"]}ms  par_max={r["max_par_observed"]}')
    else:
        print(f'  {r["stem"]:<60} {r["scenario"]:<13} '
              f'NEVER_SOLID  par_max={r["max_par_observed"]}')

# 7GT specifically
seven = [r for r in rows if '7GTxyT' in r['stem']]
print(f'\n7GT cases:')
for r in seven:
    print(f'  {r["stem"]:<60} ttfs={r["time_to_first_solid_s"]}s  '
          f'delay={r["delay_at_first_solid_ms"]}ms  par_max={r["max_par_observed"]}')
