"""Stage -1.3: Signal Independence Analyzer.

Records per-frame trace of 8 existing post-filter estimates and 5 candidate
filter-independent signals (render-side). Computes per-case correlation
matrix and aggregates max correlation across cases.

A candidate is "independent" if max(|corr|) with all existing < 0.7.
"""
import os, sys, glob, json, random
from collections import defaultdict
import numpy as np
import soundfile as sf

sys.path.insert(0, '.')
from aec import AEC, AecConfig

DATASET = '../wav/aec_challenge_blind'
OUT_DIR = '/tmp/signal_correlation'

EXISTING = ['coh2_mean', 'erle_corrected_mean', 'erl_estimate',
            'saturation_level', 'echo_psd_mean', 'residual_mean',
            'effective_dt', 'linear_failed']
CANDIDATES = ['crest_far', 'render_rms_slow', 'mic_far_ratio',
              'far_spec_flatness', 'far_stationarity']

TOP_LOSERS = [
    ('farend_singletalk', 'iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement'),
    ('farend_singletalk', 'JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk'),
    ('farend_singletalk', 'JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement'),
    ('farend_singletalk', 'VJfVUwJs4k25ziMNvJb43A_farend_singletalk'),
    ('farend_singletalk', '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk'),
    ('doubletalk', 'nyT6FUUdu0W8UpvjP1rRgQ_doubletalk_with_movement'),
    ('doubletalk', 'wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement'),
    ('doubletalk', 'yc5bFUGsR0GSfiGwTTpRWg_doubletalk'),
    ('doubletalk', 'XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement'),
    ('doubletalk', 'QK70KpLuZ0O43BBSWEZvHg_doubletalk'),
]


def trace_case(mic_path, lpb_path):
    """Run AEC and return dict of per-frame arrays for all 13 signals."""
    mic, sr = sf.read(mic_path); far, _ = sf.read(lpb_path)
    n = min(len(mic), len(far))
    cfg = AecConfig.from_preset('balanced'); cfg.filter_length = 448
    aec = AEC(config=cfg)
    hop = aec.hop_size
    n_freqs = aec.res.n_freqs

    trace = {k: [] for k in EXISTING + CANDIDATES}
    far_psd_history = []
    rms_ema = 0.0; alpha_rms = 0.95

    for pos in range(0, n - hop, hop):
        mic_blk = mic[pos:pos+hop].astype(np.float32)
        far_blk = far[pos:pos+hop].astype(np.float32)
        aec.process(mic_blk, far_blk)

        # === Existing 8 ===
        d = aec._diag
        trace['coh2_mean'].append(float(d.get('coh2_mean', 0.0)))
        # erle_corrected_mean is not in _diag; approximate via filter erle
        trace['erle_corrected_mean'].append(float(aec.res._filter_erle_est.erle.mean()))
        trace['erl_estimate'].append(float(aec._erl_estimate))
        trace['saturation_level'].append(float(getattr(aec, '_saturation_level', 0.0)))
        trace['echo_psd_mean'].append(float(d.get('echo_psd_mean', 0.0)))
        trace['residual_mean'].append(float(d.get('residual_mean', 0.0)))
        trace['effective_dt'].append(float(d.get('effective_dt', 0.0)))
        trace['linear_failed'].append(float(d.get('linear_failed', False)))

        # === Candidate 5 ===
        # crest_factor_far (from current SaturationDetector.crest_sat_score on far)
        # We compute it here directly to avoid sharing detector state with mic detector
        rms = float(np.sqrt(np.mean(far_blk.astype(np.float64) ** 2) + 1e-10))
        peak = float(np.abs(far_blk).max())
        crest = peak / rms if rms > 1e-4 else 10.0
        crest_score = float(1.0 / (1.0 + np.exp(5.0 * (crest - 1.8))))
        trace['crest_far'].append(crest_score)

        # render_rms_slow (EMA)
        rms_ema = alpha_rms * rms_ema + (1 - alpha_rms) * rms
        trace['render_rms_slow'].append(rms_ema)

        # mic_far_ratio (raw)
        mic_pwr = float(np.mean(mic_blk ** 2) + 1e-10)
        far_pwr = float(np.mean(far_blk ** 2) + 1e-10)
        trace['mic_far_ratio'].append(mic_pwr / far_pwr)

        # far_spectral_flatness (geometric_mean / arithmetic_mean of far PSD)
        far_spec = np.fft.rfft(far_blk * np.hanning(len(far_blk)).astype(np.float32))
        far_psd = np.abs(far_spec) ** 2 + 1e-12
        flatness = float(np.exp(np.mean(np.log(far_psd))) / np.mean(far_psd))
        trace['far_spec_flatness'].append(flatness)

        # far_stationarity (variance of far_psd across last 50 frames)
        far_psd_history.append(far_psd)
        if len(far_psd_history) > 50:
            far_psd_history.pop(0)
        if len(far_psd_history) >= 5:
            stack = np.stack(far_psd_history)
            mean_psd = stack.mean(axis=0) + 1e-10
            cv = (stack.std(axis=0) / mean_psd).mean()
            stationarity = float(1.0 / (1.0 + cv))  # high = stationary
        else:
            stationarity = 0.0
        trace['far_stationarity'].append(stationarity)

    return {k: np.asarray(v, dtype=np.float32) for k, v in trace.items()}


def correlations(trace):
    """Per-case correlation matrix: candidates × existing."""
    M = np.zeros((len(CANDIDATES), len(EXISTING)), dtype=np.float32)
    for i, c in enumerate(CANDIDATES):
        for j, e in enumerate(EXISTING):
            x, y = trace[c], trace[e]
            if x.std() < 1e-8 or y.std() < 1e-8:
                M[i, j] = 0.0
            else:
                M[i, j] = float(abs(np.corrcoef(x, y)[0, 1]))
    return M


def main(n_random=30):
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = []
    cases.extend(TOP_LOSERS)
    # Random sample
    random.seed(0)
    all_files = []
    for sc in ['farend_singletalk', 'doubletalk', 'nearend_singletalk']:
        all_files.extend([(sc, os.path.basename(p).replace('_mic.wav', ''))
                          for p in glob.glob(os.path.join(DATASET, sc, '*_mic.wav'))])
    random.shuffle(all_files)
    cases.extend(all_files[:n_random])

    all_corrs = []
    for sc, cid in cases:
        mic_p = os.path.join(DATASET, sc, cid + '_mic.wav')
        lpb_p = os.path.join(DATASET, sc, cid + '_lpb.wav')
        if not (os.path.exists(mic_p) and os.path.exists(lpb_p)):
            print(f'SKIP {cid}')
            continue
        try:
            trace = trace_case(mic_p, lpb_p)
            M = correlations(trace)
            all_corrs.append({'sc': sc, 'cid': cid, 'corr': M.tolist()})
            print(f'{sc[:5]} {cid[:30]:30s} ' +
                  ' '.join(f'{M[i].max():.2f}' for i in range(len(CANDIDATES))))
        except Exception as e:
            print(f'ERR {cid}: {e}')

    with open(os.path.join(OUT_DIR, 'per_case_corr.json'), 'w') as f:
        json.dump(all_corrs, f)

    # Aggregate: max across cases, per (candidate, existing)
    if not all_corrs:
        print('NO DATA')
        return
    stack = np.stack([np.asarray(r['corr']) for r in all_corrs])
    max_corr = stack.max(axis=0)        # max across cases
    median_corr = np.median(stack, axis=0)

    print()
    print('=== Aggregate max(|corr|) across cases (candidate × existing) ===')
    print(f'{"":22s} ' + ' '.join(f'{e[:8]:>8s}' for e in EXISTING))
    for i, c in enumerate(CANDIDATES):
        print(f'{c:22s} ' + ' '.join(f'{max_corr[i, j]:8.2f}' for j in range(len(EXISTING))))

    print()
    print('=== Independence verdict (Stage A pass criterion: max < 0.7) ===')
    for i, c in enumerate(CANDIDATES):
        worst = max_corr[i].max()
        worst_e = EXISTING[max_corr[i].argmax()]
        verdict = 'INDEPENDENT' if worst < 0.7 else ('PARTIAL' if worst < 0.85 else 'CORRELATED')
        print(f'{c:22s} max_corr={worst:.2f} (vs {worst_e}) -> {verdict}')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n_random=n)
