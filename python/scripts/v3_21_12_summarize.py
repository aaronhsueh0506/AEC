#!/usr/bin/env python3
"""Parse v3_21_12_abcd_trace output and produce aggregate stats:
- Per-variant mean refined-divergence rate
- Per-variant mean nores_lf/mic ratio across cases
- Win counts (which variant has best nores_lf / best div_strong per case)
"""
from __future__ import annotations
import sys, re
from pathlib import Path

if len(sys.argv) < 2:
    print('usage: v3_21_12_summarize.py <trace_output.txt>')
    sys.exit(2)
text = Path(sys.argv[1]).read_text()

case_re = re.compile(r'^=== (\S+) ===\s*$', re.MULTILINE)
row_re = re.compile(
    r'^(A|B|C|D)\s+\|\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s*\|'
    r'\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\|'
    r'\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)', re.MULTILINE)

cases = case_re.findall(text)
rows = row_re.findall(text)
# Group rows into per-case blocks of 4
assert len(rows) == 4 * len(cases), f'mismatched rows={len(rows)} cases={len(cases)}'
data = {}
for i, c in enumerate(cases):
    block = rows[i*4:(i+1)*4]
    data[c] = {}
    for r in block:
        var, hops, div, div_s, mu, herr, x2, e2i, eps, nl, nm, nh = r
        data[c][var] = dict(hops=int(hops), div=float(div), div_s=float(div_s),
                             mu_lf=float(mu), H_lf=float(herr), X2_lf=float(x2),
                             e2_inst_lf=float(e2i), e_psd_lf=float(eps),
                             nores_lf=float(nl), nores_mf=float(nm), nores_hf=float(nh))

# bucket cases by name pattern
def bucket(name: str) -> str:
    if 'farend_singletalk' in name:
        return 'FS' + ('_mvmt' if 'with_movement' in name else '_static')
    if 'nearend_singletalk' in name:
        return 'NE'
    if 'doubletalk' in name:
        return 'DT' + ('_mvmt' if 'with_movement' in name else '_static')
    return 'OTHER'

# Per-variant aggregates
print(f'\n=== Per-variant aggregates across {len(cases)} cases ===')
print(f"{'var':<3} | {'mean_div%':>9} {'mean_div_strong%':>16} | {'mean_nores_lf':>14} {'mean_nores_mf':>14} {'mean_nores_hf':>14}")
for var in ('A','B','C','D'):
    divs = [data[c][var]['div'] for c in cases]
    divs_s = [data[c][var]['div_s'] for c in cases]
    nlf = [data[c][var]['nores_lf'] for c in cases]
    nmf = [data[c][var]['nores_mf'] for c in cases]
    nhf = [data[c][var]['nores_hf'] for c in cases]
    print(f"{var:<3} | {sum(divs)/len(divs):>8.2f}% {sum(divs_s)/len(divs_s):>15.2f}% | "
          f"{sum(nlf)/len(nlf):>14.4f} {sum(nmf)/len(nmf):>14.4f} {sum(nhf)/len(nhf):>14.4f}")

# Per-bucket aggregates
print(f'\n=== Per-bucket nores_lf (lower = less artifact) ===')
buckets = sorted({bucket(c) for c in cases})
print(f"{'bucket':<10} | " + " ".join(f'{v:>9}' for v in 'ABCD'))
for bk in buckets:
    bcases = [c for c in cases if bucket(c) == bk]
    if not bcases: continue
    row = []
    for var in 'ABCD':
        m = sum(data[c][var]['nores_lf'] for c in bcases) / len(bcases)
        row.append(f'{m:>9.4f}')
    print(f"{bk:<10} ({len(bcases):>2}) | " + " ".join(row))

print(f'\n=== Per-bucket div_strong% (lower = more stable refined) ===')
for bk in buckets:
    bcases = [c for c in cases if bucket(c) == bk]
    if not bcases: continue
    row = []
    for var in 'ABCD':
        m = sum(data[c][var]['div_s'] for c in bcases) / len(bcases)
        row.append(f'{m:>8.2f}%')
    print(f"{bk:<10} ({len(bcases):>2}) | " + " ".join(row))

# Win-loss vs A (per-case)
print(f'\n=== Per-case wins vs A baseline (lower div_strong / lower nores_lf preferred) ===')
print(f"{'variant':<7} | {'div_strong improves':>21} {'nores_lf improves':>20}")
for var in ('B','C','D'):
    div_improve = sum(1 for c in cases if data[c][var]['div_s'] < data[c]['A']['div_s'])
    div_worsens = sum(1 for c in cases if data[c][var]['div_s'] > data[c]['A']['div_s'])
    nl_improve = sum(1 for c in cases if data[c][var]['nores_lf'] < data[c]['A']['nores_lf'])
    nl_worsens = sum(1 for c in cases if data[c][var]['nores_lf'] > data[c]['A']['nores_lf'])
    print(f"{var:<7} | {f'{div_improve}/{div_worsens} (vs A)':>21} {f'{nl_improve}/{nl_worsens} (vs A)':>20}")

# D vs B head-to-head
print(f'\n=== D vs B head-to-head (D > B = D better) ===')
d_div_better = sum(1 for c in cases if data[c]['D']['div_s'] < data[c]['B']['div_s'])
b_div_better = sum(1 for c in cases if data[c]['B']['div_s'] < data[c]['D']['div_s'])
d_nlf_better = sum(1 for c in cases if data[c]['D']['nores_lf'] < data[c]['B']['nores_lf'])
b_nlf_better = sum(1 for c in cases if data[c]['B']['nores_lf'] < data[c]['D']['nores_lf'])
print(f"  div_strong: D wins {d_div_better}, B wins {b_div_better}, tie {len(cases) - d_div_better - b_div_better}")
print(f"  nores_lf:   D wins {d_nlf_better}, B wins {b_nlf_better}, tie {len(cases) - d_nlf_better - b_nlf_better}")
