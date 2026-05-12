#!/usr/bin/env python3
"""F3.1 outlier audit — per-frame state diff on FS_movement losers.

Goal: find what makes Lsa5Wpw / s0oJqM6Y / wr54weK / pG9Bikvr / Khk1qeM
worse than the median FS_movement case under F3.1 flag-ON. Capture
per-frame:

  * frame_idx, t (s)
  * far_power, mic_power, raw_err_power
  * epc_active, filter_converged, filter_once_converged
  * erl_estimate, _long_window_n_updates
  * F3.1 fired? (flag_on AND filter_converged AND lw_ready)
  * res gain mean / HF mean (synth output spectral magnitude / mic
    spectral magnitude proxy)

Dumps a CSV per case + a short ASCII summary so a single command
gives a comparable trace across the 5 cases.

Run:
    python3 tools/research/f3_1_outlier_audit.py --out /tmp/f3_1_audit
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))

OUTLIERS = [
    ('Lsa5WpwTpUeb7C9dc9RXuQ', -0.301),
    ('s0oJqM6Y1UCHSVmHmgsx4Q', -0.213),
    ('wr54weKzNkOcZ07hB04kzA', -0.167),
    ('pG9Bikvr40Ct1kUtch95kw', -0.140),
    ('Khk1qeMXFUuvFhw3YRSm0w', -0.075),
]
DATASET = 'wav/aec_challenge_blind/farend_singletalk'
SCENARIO_SUFFIX = '_farend_singletalk_with_movement'


def _delay(mic, lpb, sr, max_ms=250):
    n = min(len(mic), len(lpb))
    max_d = int(sr * max_ms / 1000)
    nfft = 1 << int(np.ceil(np.log2(n + max_d)))
    xc = np.fft.irfft(np.fft.rfft(mic, n=nfft) *
                       np.conj(np.fft.rfft(lpb, n=nfft)), n=nfft)
    return int(np.argmax(xc[: max_d + 1]))


def _trace_one(stem: str, use_mic_excess: bool, out_csv: str) -> dict:
    """Trace one case, write per-frame CSV, return summary dict."""
    from aec import AEC, AecConfig, AecMode, AecPreset

    base = os.path.join(DATASET, stem + SCENARIO_SUFFIX)
    mic, sr = sf.read(base + '_mic.wav')
    lpb, _  = sf.read(base + '_lpb.wav')
    mic = mic.astype(np.float32); lpb = lpb.astype(np.float32)

    d = _delay(mic, lpb, sr)
    n = min(len(mic), len(lpb))
    lpb_a = np.zeros(n, dtype=np.float32)
    if 0 < d < n: lpb_a[d:] = lpb[: n - d]
    else: lpb_a = lpb[:n]
    mic = mic[:n]

    np.random.seed(0)
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=832, enable_dtd=False, enable_shadow=True,
        enable_res=True, enable_cng=True, use_kalman=True,
        enable_delay_est=False,
        use_mic_excess_evidence=use_mic_excess,
    )
    aec = AEC(cfg)
    hop = aec.hop_size

    rows: list[tuple] = []
    pos = 0
    while pos + hop <= n:
        mic_h = mic[pos: pos + hop]
        lpb_h = lpb_a[pos: pos + hop]
        out = aec.process(mic_h, lpb_h)
        # Pull diagnostic state immediately AFTER process. AecStats path
        # not turned on — read attributes directly.
        epc_active = bool(aec._epc_det.active)
        fc = bool(aec._filter_converged)
        foc = bool(aec._filter_once_converged) if hasattr(aec, '_filter_once_converged') else False
        erl = float(aec._erl_estimate)
        lw_n = int(aec.res._residual_est._long_window_n_updates) if aec.res else 0
        gain_mean = float(np.mean(aec.res.gain_smooth)) if aec.res else 1.0
        hf2k = aec.res._hf_cap_bin_2k if aec.res else 0
        gain_hf = float(np.mean(aec.res.gain_smooth[hf2k:])) if (aec.res and aec.res.gain_smooth.shape[0] > hf2k) else 1.0
        dt_pb_mean = float(np.mean(aec.res._dt_per_bin_last)) if aec.res else 0.0
        dt_pb_hf = float(np.mean(aec.res._dt_per_bin_last[hf2k:])) if (aec.res and aec.res._dt_per_bin_last.shape[0] > hf2k) else 0.0
        far_p = float(np.mean(lpb_h ** 2))
        mic_p = float(np.mean(mic_h ** 2))
        err_p = float(np.mean(out ** 2))
        f31_fire = bool(use_mic_excess and fc and lw_n > 0)
        rows.append((pos / sr, far_p, mic_p, err_p,
                      int(epc_active), int(fc), int(foc),
                      erl, lw_n, int(f31_fire),
                      gain_mean, gain_hf, dt_pb_mean, dt_pb_hf))
        pos += hop

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('t', 'far_p', 'mic_p', 'err_p', 'epc_active', 'filter_converged',
                     'filter_once_converged', 'erl_estimate', 'lw_n_updates',
                     'f31_fired', 'gain_mean', 'gain_hf', 'dt_pb_mean', 'dt_pb_hf'))
        for r in rows:
            w.writerow(r)

    arr = np.array(rows, dtype=np.float64)
    n_rows = arr.shape[0]
    return {
        'stem': stem,
        'flag': 'ON' if use_mic_excess else 'OFF',
        'n_frames': n_rows,
        'epc_active_pct': 100 * arr[:, 4].sum() / n_rows,
        'fc_pct': 100 * arr[:, 5].sum() / n_rows,
        'foc_pct': 100 * arr[:, 6].sum() / n_rows,
        'erl_mean': float(arr[:, 7].mean()),
        'erl_std':  float(arr[:, 7].std()),
        'erl_range': (float(arr[:, 7].min()), float(arr[:, 7].max())),
        'lw_min': float(arr[:, 8].min()),
        'lw_max': float(arr[:, 8].max()),
        'f31_fire_pct': 100 * arr[:, 9].sum() / n_rows,
        'gain_mean_mean': float(arr[:, 10].mean()),
        'gain_hf_mean': float(arr[:, 11].mean()),
        'dt_pb_mean_mean': float(arr[:, 12].mean()),
        'dt_pb_hf_mean': float(arr[:, 13].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/f3_1_audit')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'{"stem":24} {"flag":4} {"frames":>6} {"epc%":>6} {"fc%":>6} {"foc%":>6} '
          f'{"erl μ":>7} {"erl±":>6} {"lw_max":>7} {"fire%":>6} '
          f'{"gain":>6} {"gainHF":>7} {"dt_pb":>6} {"dt_pbHF":>7}')
    summaries = []
    for stem, expected_de in OUTLIERS:
        for use_flag, lbl in ((False, 'OFF'), (True, 'ON ')):
            csv_path = os.path.join(args.out, f'{stem}_{lbl.strip()}.csv')
            s = _trace_one(stem, use_flag, csv_path)
            summaries.append(s)
            print(f"{stem:24} {lbl:4} {s['n_frames']:>6} "
                  f"{s['epc_active_pct']:>6.1f} {s['fc_pct']:>6.1f} {s['foc_pct']:>6.1f} "
                  f"{s['erl_mean']:>7.4f} {s['erl_std']:>6.4f} "
                  f"{s['lw_max']:>7.0f} {s['f31_fire_pct']:>6.1f} "
                  f"{s['gain_mean_mean']:>6.3f} {s['gain_hf_mean']:>7.3f} "
                  f"{s['dt_pb_mean_mean']:>6.3f} {s['dt_pb_hf_mean']:>7.3f}")
    print()
    print('Δ summary (ON vs OFF):')
    print(f'{"stem":24} {"frames":>6} {"Δgain":>7} {"ΔgainHF":>8} {"Δdt_pb":>7} {"Δdt_pbHF":>9}')
    for i in range(0, len(summaries), 2):
        off = summaries[i]; on = summaries[i + 1]
        print(f"{off['stem']:24} {off['n_frames']:>6} "
              f"{on['gain_mean_mean'] - off['gain_mean_mean']:>+7.3f} "
              f"{on['gain_hf_mean'] - off['gain_hf_mean']:>+8.3f} "
              f"{on['dt_pb_mean_mean'] - off['dt_pb_mean_mean']:>+7.3f} "
              f"{on['dt_pb_hf_mean'] - off['dt_pb_hf_mean']:>+9.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
