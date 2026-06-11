#!/usr/bin/env python3
"""Track F probe: trace H_error leakage during DT segment (aec_record 6.2-7s).

Runs the AEC on aec_record_{mic,ref}_10s.wav and prints per-hop trace of:
  - _last_leakage_div_frac (fraction of bins using diverged leakage)
  - H_error_per_bin mean
  - mu_aec3 mean (computed from formula)
  - _disallow_leakage_diverged flag
  - dt_from_energy

Focus window: 6.0-7.5s to cover DT transition.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python'))

import numpy as np
import soundfile as sf
from aec import AEC, AecConfig, AecMode

SR = 16000
HOP = 160
FOCUS_START = 5.5   # seconds
FOCUS_END   = 8.0

mic, sr = sf.read('../AEC/wav/aec_record/aec_record_mic_10s.wav', dtype='float32')
ref, sr2 = sf.read('../AEC/wav/aec_record/aec_record_ref_10s.wav', dtype='float32')
assert sr == SR and sr2 == SR
mic = mic[:, 0] if mic.ndim > 1 else mic
ref = ref[:, 0] if ref.ndim > 1 else ref
n = min(len(mic), len(ref))
mic, ref = mic[:n], ref[:n]

cfg = AecConfig.from_preset('balanced', sample_rate=SR, mode=AecMode.PBFDKF,
                             filter_length=832, enable_shadow=True,
                             enable_res=True, enable_cng=True)
np.random.seed(0)
aec = AEC(cfg)

# Frame-by-frame processing to intercept internal state
n_hops = n // HOP
trace = []

for i in range(n_hops):
    t = i * HOP / SR
    mic_block = mic[i*HOP:(i+1)*HOP]
    ref_block = ref[i*HOP:(i+1)*HOP]
    _ = aec.process(mic_block, ref_block)

    if FOCUS_START <= t <= FOCUS_END:
        filt = aec.filter  # PBFDKF main filter
        leakage_div = float(getattr(filt, '_last_leakage_div_frac', -1.0))
        h_error_mean = float(np.mean(filt.H_error_per_bin))
        h_error_min  = float(np.min(filt.H_error_per_bin))
        disallow = bool(getattr(filt, '_disallow_leakage_diverged', False))
        dt_energy = float(aec._dt_from_energy)

        # Compute instantaneous mu_aec3 mean for diagnostic
        # mu[k] = H_error[k] / (0.5·H_error[k]·X²[k] + n·E²[k])
        X2 = (np.abs(filt.X_buf) ** 2).sum(axis=0).astype(np.float32)
        e2 = (np.abs(filt.error_spec) ** 2).astype(np.float32)
        n_part = float(filt.n_partitions)
        denom = 0.5 * filt.H_error_per_bin * X2 + n_part * e2 + 1e-8
        mu_vec = filt.H_error_per_bin / denom
        mu_mean = float(np.mean(mu_vec))
        mu_p95 = float(np.percentile(mu_vec, 95))

        e2_coarse_per_bin = getattr(filt, '_e2_coarse_per_bin', None)
        if e2_coarse_per_bin is not None:
            # fraction of bins with e2_refined > e2_coarse (diverged branch)
            div_frac_direct = float(np.mean(e2 > e2_coarse_per_bin))
        else:
            div_frac_direct = -1.0

        trace.append({
            't': t, 'leakage_div': leakage_div, 'h_err_mean': h_error_mean,
            'h_err_min': h_error_min, 'disallow': disallow,
            'dt_energy': dt_energy, 'mu_mean': mu_mean, 'mu_p95': mu_p95,
            'div_frac_direct': div_frac_direct,
        })

print(f"{'t(s)':>6}  {'ld_frac':>7}  {'H_err_mu':>9}  {'H_err_mn':>9}  "
      f"{'disallow':>8}  {'dt_e':>5}  {'mu_mu':>7}  {'mu_p95':>7}  {'e2>coa':>6}")
print('-' * 90)
for r in trace:
    print(f"{r['t']:6.3f}  {r['leakage_div']:7.3f}  {r['h_err_mean']:9.2f}  "
          f"{r['h_err_min']:9.4f}  {str(r['disallow']):>8}  {r['dt_energy']:5.3f}  "
          f"{r['mu_mean']:7.4f}  {r['mu_p95']:7.4f}  {r['div_frac_direct']:6.3f}")

# Summary stats for 6.2-7.0 DT region vs 7.0-7.5 post-DT region
dt_rows = [r for r in trace if 6.2 <= r['t'] <= 7.0]
post_rows = [r for r in trace if 7.0 < r['t'] <= 7.5]
if dt_rows:
    print(f"\nDT region (6.2-7.0s) mean leakage_div_frac: "
          f"{np.mean([r['leakage_div'] for r in dt_rows]):.3f}")
    print(f"DT region dt_energy range: "
          f"{np.min([r['dt_energy'] for r in dt_rows]):.3f} - "
          f"{np.max([r['dt_energy'] for r in dt_rows]):.3f}")
if post_rows:
    print(f"Post-DT region (7.0-7.5s) mean leakage_div_frac: "
          f"{np.mean([r['leakage_div'] for r in post_rows]):.3f}")
    print(f"Post-DT mu_p95 max: {np.max([r['mu_p95'] for r in post_rows]):.4f}")
