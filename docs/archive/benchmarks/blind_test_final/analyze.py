"""Blind test final consolidated analysis: 4 combos on 800-case aec_challenge_blind."""
import re, gzip, os
import numpy as np, soundfile as sf

REPO = '/Users/mingyu/Desktop/novatek/SE/AEC'
LOGS = {
    '(0,0) baseline': ('gz', f'{REPO}/docs/benchmarks/b16_stage_1d/baseline_aecmos.log.gz'),
    '(1,0) B-16':     ('gz', f'{REPO}/docs/benchmarks/b16_stage_1d/b16on_aecmos.log.gz'),
    '(0,1) Phase 2':  ('gz', f'{REPO}/docs/benchmarks/phase2_stage_1d/p2_aecmos_01.log.gz'),
    '(1,1) stacked':  ('gz', f'{REPO}/docs/benchmarks/phase2_stage_1d/p2_aecmos_11.log.gz'),
}
DIRS = {
    '(0,0) baseline': f'{REPO}/python/baselines/b16_baseline',
    '(1,0) B-16':     f'{REPO}/python/baselines/b16_on',
    '(0,1) Phase 2':  f'{REPO}/python/baselines/p2_b16_0_p2_1',
    '(1,1) stacked':  f'{REPO}/python/baselines/p2_b16_1_p2_1',
}

def read_log(meta):
    k, p = meta
    if k == 'gz':
        with gzip.open(p, 'rt') as f: return f.read()
    return open(p).read()

def parse_mean(text, scn, section):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    lines = text.split('\n')
    i = next((j for j,l in enumerate(lines) if scn in l and 'AECMOS' in l), None)
    if i is None: return None
    while i < len(lines) and section not in lines[i]: i += 1
    while i < len(lines) and not lines[i].strip().startswith('MEAN'): i += 1
    if i >= len(lines): return None
    parts = lines[i].split()
    try: return float(parts[1])
    except: return None

def parse_cases(text, scn):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    lines = text.split('\n')
    i = next((j for j,l in enumerate(lines) if scn in l and 'AECMOS' in l), None)
    if i is None: return []
    while i < len(lines) and '--- echo_mos ---' not in lines[i]: i += 1
    i += 1
    while i < len(lines):
        s = lines[i].strip()
        if s and not s.startswith('Case') and not s.startswith('---') and not s.startswith('='): break
        i += 1
    out = []
    while i < len(lines):
        s = lines[i].strip()
        if not s: i += 1; continue
        if s.startswith('MEAN') or s.startswith('---') or s.startswith('='): break
        parts = s.split()
        try: out.append(float(parts[1]))
        except: pass
        i += 1
    return out

def list_cases(scn):
    d = f'{REPO}/wav/aec_challenge_blind/{scn}'
    mics = sorted(f for f in os.listdir(d) if f.endswith('_mic.wav'))
    cases = [m.replace('_mic.wav','') for m in mics]
    return cases, ['_with_movement' in c for c in cases]

def rms_db(x):
    p = float(np.mean(x.astype(np.float64)**2) + 1e-20)
    return 10.0 * np.log10(p)

texts = {t: read_log(m) for t, m in LOGS.items()}

print("=" * 85)
print("BLIND TEST 800-case AECMOS — 4 flag combinations")
print("=" * 85)
print(f"\n{'combo':18s} {'FS_echo':>8s} {'DT_echo':>8s} {'DT_deg':>8s} {'NE_deg':>8s}")
for t in LOGS:
    fs = parse_mean(texts[t], 'FAREND SINGLETALK', '--- echo_mos ---')
    dt = parse_mean(texts[t], 'DOUBLETALK', '--- echo_mos ---')
    dt_deg = parse_mean(texts[t], 'DOUBLETALK', '--- deg_mos ---')
    ne_deg = parse_mean(texts[t], 'NEAREND SINGLETALK', '--- deg_mos ---')
    print(f"{t:18s} {fs:8.3f} {dt:8.3f} {dt_deg:8.3f} {ne_deg:8.3f}")

print(f"\n=== Δ vs (0,0) baseline ===")
base_fs = parse_mean(texts['(0,0) baseline'], 'FAREND SINGLETALK', '--- echo_mos ---')
base_dt = parse_mean(texts['(0,0) baseline'], 'DOUBLETALK', '--- echo_mos ---')
base_dtd = parse_mean(texts['(0,0) baseline'], 'DOUBLETALK', '--- deg_mos ---')
base_ne = parse_mean(texts['(0,0) baseline'], 'NEAREND SINGLETALK', '--- deg_mos ---')
print(f"{'combo':18s} {'ΔFS':>8s} {'ΔDT_echo':>10s} {'ΔDT_deg':>9s} {'ΔNE_deg':>9s}")
for t in list(LOGS.keys())[1:]:
    fs = parse_mean(texts[t], 'FAREND SINGLETALK', '--- echo_mos ---') - base_fs
    dt = parse_mean(texts[t], 'DOUBLETALK', '--- echo_mos ---') - base_dt
    dtd = parse_mean(texts[t], 'DOUBLETALK', '--- deg_mos ---') - base_dtd
    ne = parse_mean(texts[t], 'NEAREND SINGLETALK', '--- deg_mos ---') - base_ne
    print(f"{t:18s} {fs:+8.3f} {dt:+10.3f} {dtd:+9.3f} {ne:+9.3f}")

# Movement / static
print(f"\n=== Movement vs static subset (FS + DT) ===")
for scn_d, scn_n in [('farend_singletalk','FAREND SINGLETALK'),('doubletalk','DOUBLETALK')]:
    cases, is_mv = list_cases(scn_d)
    scores = {t: parse_cases(texts[t], scn_n) for t in texts}
    n = min(len(cases), min(len(v) for v in scores.values()))
    print(f"\n  {scn_n} (n={n}, mv={sum(is_mv[:n])}, static={n-sum(is_mv[:n])})")
    print(f"  {'subset':10s} " + " ".join(f"{t:>14s}" for t in scores))
    mv = np.array(is_mv[:n])
    for sub, mask in [('ALL', np.ones(n,bool)), ('movement', mv), ('static', ~mv)]:
        if mask.sum() == 0: continue
        row = [f"{sub:10s}"]
        base_mean = float(np.mean(np.array(scores['(0,0) baseline'][:n])[mask]))
        for t in scores:
            v = float(np.mean(np.array(scores[t][:n])[mask]))
            if t == '(0,0) baseline':
                row.append(f"{v:14.3f}")
            else:
                row.append(f"{v-base_mean:+14.3f}")
        print(" ".join(row))

# Big wins / losses for each combo on FS
print(f"\n=== FS big wins (Δ>+0.2) / big losses (Δ<-0.2) vs (0,0) ===")
cases_fs, is_mv_fs = list_cases('farend_singletalk')
base_fs_cases = np.array(parse_cases(texts['(0,0) baseline'], 'FAREND SINGLETALK'))
for t in list(LOGS.keys())[1:]:
    vals = np.array(parse_cases(texts[t], 'FAREND SINGLETALK'))
    n = min(len(vals), len(base_fs_cases))
    deltas = vals[:n] - base_fs_cases[:n]
    wins = sum(1 for d in deltas if d > 0.2)
    losses = sum(1 for d in deltas if d < -0.2)
    mv_wins = sum(1 for i,d in enumerate(deltas) if d > 0.2 and is_mv_fs[i])
    mv_losses = sum(1 for i,d in enumerate(deltas) if d < -0.2 and is_mv_fs[i])
    print(f"  {t:18s} wins={wins:3d} (mv={mv_wins:2d}) losses={losses:3d} (mv={mv_losses:2d})  net={wins-losses:+d}")

# Cat-A class wav-level
CAT_A = ['JteZUZ4JYkeD4k2rcVbqHg','VGlWeOPC6UiXSq4SYPiKpw','JLNgGcvTNEqbTDbc28wLkg',
         'VJfVUwJs4k25ziMNvJb43A','r7U6JmcRl0ibIh0mN3CP9g','9xjhiFbGo06hdQIsHTS6qA',
         'lV0kQN0hR0ySmE0bQhuYbw','sLWe8bfYbkGwX1W3PzI1PQ','wr54weKzNkOcZ07hB04kzA',
         'IxgmaPghzUGnR6sxrbGU3Q','s0oJqM6Y1UCHSVmHmgsx4Q','HIMqDWjSoECJFtIP0TM9bg',
         'PZ7V0SfxUkem4IalTp1YgA']
print(f"\n=== Cat-A 13-case wav-level (full-file RMS leak dB) ===")
print(f"  {'combo':18s} {'mean':>8s} {'PZ7V full':>10s} {'PZ7V onset':>11s}")
for t, d in DIRS.items():
    vals = []
    for c in CAT_A:
        p = f'{d}/{c}_farend_singletalk_ours.wav'
        if os.path.exists(p):
            x, _ = sf.read(p, dtype='float32')
            vals.append(rms_db(x))
    pz = f'{d}/PZ7V0SfxUkem4IalTp1YgA_farend_singletalk_ours.wav'
    x, sr = sf.read(pz, dtype='float32')
    pf, po = rms_db(x), rms_db(x[int(9.5*sr):int(10.5*sr)])
    print(f"  {t:18s} {np.mean(vals):+8.3f} {pf:+10.3f} {po:+11.3f}")
