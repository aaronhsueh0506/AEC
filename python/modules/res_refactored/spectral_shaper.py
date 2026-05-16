"""spectral_shaper — P52 Phase B Module 3.

Verbatim extraction of `ResFilter._stage_gain_postprocess` (aec.py:1973-2101)
into a free function operating on a ResFilter-shaped state container `rf`.
Logic copied byte-for-byte per anti-loophole §5.5.

v3.16 C1 update: `epc_dt_cap` cap action removed (parallel removal in legacy
`ResFilter._stage_gain_postprocess`). Slot 03 / diag stage 2 preserved as
aliases of stage 02 for diagnostic backward compat. Byte-equal vs legacy
preserved because the cap action fired 0/2,032,022 frames in v3.13 + v3.14
audits.
"""

from __future__ import annotations

import numpy as np


def spectral_shaper(rf, *, g_in, quiet_mask, far_power,
                    effective_dt, is_stationary_dt, divergence,
                    erl_estimate=0.01):
    """Stage 3: quiet mask, 3-bin smooth, HF cap, divergence override.

    Returns updated g (post-divergence-override). Mutates
    `rf._diag_round5_stages[2..6]`. Slot 2 is preserved as alias of slot 1
    post v3.16 C1 (epc_dt_cap removed).
    """
    g = g_in
    # v3.16 C1 — epc_dt_cap removed. Slot 03 / stage 2 alias of stage 02.
    if getattr(rf, '_capture_stages', False):
        rf._stage_gains['03_epc_dt_cap'] = g.copy()
    rf._diag_round5_stages[2] = float(np.mean(g[rf._voice_band_idx])) if rf._voice_band_idx.size > 0 else 0.0

    g[quiet_mask] = 1.0
    if getattr(rf, '_capture_stages', False):
        rf._stage_gains['04_quiet_mask'] = g.copy()
    rf._diag_round5_stages[3] = float(np.mean(g[rf._voice_band_idx])) if rf._voice_band_idx.size > 0 else 0.0

    if far_power > 1e-4:
        if rf._plan_a_kernel_tight:
            kernel = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        else:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        g = np.convolve(g, kernel, mode='same').astype(np.float32)
        if getattr(rf, '_capture_stages', False):
            rf._stage_gains['05_3bin_smooth'] = g.copy()
        rf._diag_round5_stages[4] = float(np.mean(g[rf._voice_band_idx])) if rf._voice_band_idx.size > 0 else 0.0
        if rf.n_freqs > 2:
            g[:2] = np.minimum(g[1], g[2])
        if rf._hf_cap_conditional:
            try:
                far_lw = rf._residual_est._long_window_far_psd
                err_hb = rf.error_psd[rf._hf_cap_bin_2k:]
                far_hb = far_lw[rf._hf_cap_bin_2k:]
                erl_e = float(erl_estimate)
                excess = np.maximum(err_hb - 1.0 * far_hb * erl_e, 0.0)
                err_hb_mean = float(np.mean(err_hb)) + 1e-10
                metric = float(np.mean(excess)) / err_hb_mean
            except Exception:
                metric = 0.0
            if metric < rf._hf_cap_metric_threshold:
                hf_cap_bin = rf._hf_cap_bin
                if (rf.n_freqs > hf_cap_bin + 1
                        and effective_dt < 0.5
                        and not is_stationary_dt):
                    hf_cap = g[hf_cap_bin]
                    g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)
        elif rf._plan_a_hf_cap_2k:
            hf_cap_bin = rf._hf_cap_bin_2k
            if (rf.n_freqs > hf_cap_bin + 1
                    and effective_dt < 0.3
                    and not is_stationary_dt):
                high_ne_conf = float(np.mean(rf._dt_per_bin_last[hf_cap_bin:]))
                if high_ne_conf < 0.3:
                    hf_cap = g[hf_cap_bin]
                    g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)
        else:
            hf_cap_bin = rf._hf_cap_bin
            if (rf.n_freqs > hf_cap_bin + 1
                    and effective_dt < 0.5
                    and not is_stationary_dt):
                hf_cap = g[hf_cap_bin]
                g[hf_cap_bin + 1:] = np.minimum(g[hf_cap_bin + 1:], hf_cap)
        if getattr(rf, '_capture_stages', False):
            rf._stage_gains['06_hf_cap'] = g.copy()
        rf._diag_round5_stages[5] = float(np.mean(g[rf._voice_band_idx])) if rf._voice_band_idx.size > 0 else 0.0

    if divergence > 0.3:
        divergence_gain = 0.01 + (1.0 - 0.01) * (1.0 - divergence)
        g = np.minimum(g, divergence_gain)
    if getattr(rf, '_capture_stages', False):
        rf._stage_gains['07_pre_temporal'] = g.copy()
    rf._diag_round5_stages[6] = float(np.mean(g[rf._voice_band_idx])) if rf._voice_band_idx.size > 0 else 0.0

    _hf_2k = rf._hf_cap_bin_2k
    if g.shape[0] > _hf_2k:
        rf._p4b_gain_hf_mean = float(np.mean(g[_hf_2k:]))
    else:
        rf._p4b_gain_hf_mean = 0.0
    _re = getattr(rf, '_diag_residual_echo_psd_last', None)
    if _re is not None and _re.shape[0] > _hf_2k:
        rf._p4b_res_echo_hf_mean_db = (
            10.0 * float(np.log10(float(np.mean(_re[_hf_2k:])) + 1e-12))
        )
    else:
        rf._p4b_res_echo_hf_mean_db = -120.0
    return g
