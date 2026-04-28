"""Phase η Stage 1C — Optuna joint tune on 10 HIGH-sensitivity Linear AEC params.

Objective (minimize):
    obj = mean(cat_a_leak, fs_static_leak, fs_mv_leak) - 0.5 * dt_rms_db

Lower = better (more negative leaks AND higher dt preservation).

Baseline obj ≈ -20.18 dB (cat_a=-25.86, fs_stat=-41.49, fs_mv=-37.33, dt=-29.43).

50-case subset, wav-level RMS proxy. AECMOS validation at Stage 1C-2 (800-case).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import optuna
import soundfile as sf

REPO = Path('/Users/mingyu/Desktop/novatek/SE/AEC')
sys.path.insert(0, str(REPO / 'python'))

from aec import AEC, AecConfig, AecMode
from eval_aec_challenge import estimate_delay

MANIFEST = REPO / 'docs/benchmarks/phase_eta_1b/case_subset_50.json'
OUT_DIR = REPO / 'docs/benchmarks/phase_eta_1c'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Search ranges — derived from Stage 1B sweep ±50% bounds.
SEARCH_SPACE = {
    'epc_large_shadow_adv':         ('float', 1.0, 3.0),
    'shadow_err_alpha':             ('float', 0.40, 0.95),
    'epc_large_total_rise':         ('float', 1.5, 4.5),
    'epc_small_shadow_adv':         ('float', 0.65, 1.95),
    'epc_total_rise':               ('float', 0.75, 2.25),
    'shadow_dtd_advantage_scale':   ('float', 1.5, 4.5),
    'shadow_mu_ratio':              ('float', 0.5, 1.5),
    'shadow_dtd_offset':            ('float', 0.75, 2.25),
    'kalman_q_high':                ('log',   5e-4, 1.5e-3),
    'shadow_to_main_copy_threshold':('float', 0.325, 0.975),
}


def load_case(case):
    cid, scn, is_mv = case['id'], case['scn'], case['is_movement']
    suf = '_with_movement' if is_mv else ''
    d = REPO / f'wav/aec_challenge_blind/{scn}'
    mic, sr = sf.read(str(d / f'{cid}_{scn}{suf}_mic.wav'), dtype='float32')
    lpb, _ = sf.read(str(d / f'{cid}_{scn}{suf}_lpb.wav'), dtype='float32')
    return mic, lpb, sr


def run_aec(mic, lpb, sr, overrides):
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]
    delay = estimate_delay(mic, lpb, sr)
    ref = np.zeros(n, dtype=np.float32)
    if 0 < delay < n:
        ref[delay:] = lpb[:n - delay]
    else:
        ref = lpb[:n]
    kw = dict(sample_rate=sr, mode=AecMode.PBFDKF, filter_length=512,
              enable_dtd=False, enable_shadow=True, enable_res=True,
              use_kalman=True, enable_delay_est=False)
    kw.update(overrides)
    config = AecConfig.from_preset('balanced', **kw)
    aec = AEC(config)
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos + hop] = aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        pos += hop
    return out[:n]


def rms_db(x):
    p = float(np.mean(x.astype(np.float64) ** 2) + 1e-20)
    return 10.0 * np.log10(p)


def run_trial(cases_loaded, overrides):
    buckets = {'cat_a': [], 'fs_movement': [], 'fs_static': [], 'dt': []}
    for case, (mic, lpb, sr) in cases_loaded:
        out = run_aec(mic, lpb, sr, overrides)
        buckets[case['class']].append(rms_db(out))
    m = {
        'cat_a': float(np.mean(buckets['cat_a'])),
        'fs_static': float(np.mean(buckets['fs_static'])),
        'fs_mv': float(np.mean(buckets['fs_movement'])),
        'dt': float(np.mean(buckets['dt'])),
    }
    obj = (m['cat_a'] + m['fs_static'] + m['fs_mv']) / 3.0 - 0.5 * m['dt']
    return obj, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--storage', default=str(OUT_DIR / 'study.db'))
    ap.add_argument('--study-name', default='phase_eta_1c')
    args = ap.parse_args()

    cases = json.loads(MANIFEST.read_text())
    print(f'Loading {len(cases)} cases...')
    t0 = time.time()
    cases_loaded = [(c, load_case(c)) for c in cases]
    print(f'  loaded in {time.time()-t0:.1f}s')

    # Baseline
    print('\nBaseline (defaults)...')
    obj_base, m_base = run_trial(cases_loaded, {})
    print(f'  obj={obj_base:.4f}  {m_base}')

    storage = f'sqlite:///{args.storage}'
    study = optuna.create_study(
        direction='minimize',
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial):
        overrides = {}
        for name, spec in SEARCH_SPACE.items():
            kind, lo, hi = spec
            if kind == 'float':
                overrides[name] = trial.suggest_float(name, lo, hi)
            elif kind == 'log':
                overrides[name] = trial.suggest_float(name, lo, hi, log=True)
        t0 = time.time()
        obj, m = run_trial(cases_loaded, overrides)
        dt = time.time() - t0
        trial.set_user_attr('cat_a', m['cat_a'])
        trial.set_user_attr('fs_static', m['fs_static'])
        trial.set_user_attr('fs_mv', m['fs_mv'])
        trial.set_user_attr('dt', m['dt'])
        trial.set_user_attr('elapsed_s', dt)
        print(f'  trial {trial.number:3d}: obj={obj:.4f}  cat_a={m["cat_a"]:+.3f} '
              f'fs_stat={m["fs_static"]:+.3f} fs_mv={m["fs_mv"]:+.3f} dt={m["dt"]:+.3f}  [{dt:.1f}s]')
        return obj

    study.optimize(objective, n_trials=args.trials)

    print(f'\nBest obj: {study.best_value:.4f}  (baseline: {obj_base:.4f})')
    print('Best params:')
    for k, v in study.best_params.items():
        print(f'  {k} = {v}')

    # Save best + baseline
    (OUT_DIR / 'best_params.json').write_text(json.dumps(
        {'baseline_obj': obj_base, 'baseline_metrics': m_base,
         'best_obj': study.best_value, 'best_params': study.best_params,
         'best_trial_metrics': study.best_trial.user_attrs},
        indent=2))
    print(f'\nwrote {OUT_DIR / "best_params.json"}')


if __name__ == '__main__':
    main()
