"""gain_computer — P52 Phase B Module 2.

Verbatim extraction of `ResFilter._stage_gain_compute` (aec.py:1835-1971) into a
free function operating on a ResFilter-shaped state container `rf`. Logic is
copied byte-for-byte (no numerical reordering, no algebra simplification, no
threshold change) per anti-loophole §5.5.

Scope note (deviation from v1.1 §3.3 logical mapping): §3.3 places
`epc_dt_cap` (diag index [2]) in Module 2. In legacy code `epc_dt_cap` lives
inside `_stage_gain_postprocess` as its first operation. To preserve byte-equal
under the subclass-and-delegate pattern (no orchestrator signature change),
`epc_dt_cap` is left physically in Module 3's `_stage_gain_postprocess`. The
§3.3 logical mapping is honored at the time-ordered sequence level (`g`
flows from `gain_compute` → `epc_dt_cap` → ... unchanged); the code-locality
shift to Module 2 is deferred to the post-five-extraction `ResState` migration
pass (§3.4), at which point the orchestrator owns frame ingestion and module
boundaries are fully under refactor control.
"""

from __future__ import annotations

import numpy as np


def gain_computer(rf, *, residual_echo_psd, eer, coh2, effective_dt,
                  is_stationary_dt, far_power, filter_once_converged,
                  spectral_g_min, eps,
                  erl_estimate: float = 0.01,
                  filter_converged: bool = False,
                  epc_active: bool = False,
                  filter_state: str = 'idle'):
    """Stage 2: ENR / Wiener / spectral_sub gain compute + EMR + spectral floor lift.

    Returns g (post-spectral-floor). Mutates `rf.dominant_ne` / Round 4 / Round 5
    diag caches and `rf._stats_last_*` fields.

    `erl_estimate` / `filter_converged` / `epc_active` are consumed only
    by the F3.1 mic-excess-evidence branch (gated on
    `rf._use_mic_excess_evidence`, default-OFF). Legacy / P4B paths
    remain byte-identical to the pre-F3.1 extraction.
    """
    if rf.gain_type == "enr" and residual_echo_psd is not None:
        raw_nearend_est = np.maximum(rf.error_psd - residual_echo_psd, 0.0)
        noise_floor_psd = np.mean(rf.error_psd) * 0.01 + 1e-10

        # F3.1: per-bin mic-energy excess metric. See ResFilter._stage_gain_compute
        # for full rationale. Kept structurally identical to the legacy class so
        # both code paths stay in lockstep.
        _lw_ready = (
            rf._residual_est is not None
            and rf._residual_est._long_window_n_updates > 0
        )
        # F3.1 v3 (2026-05-12): mic-in-echo-envelope gate + legacy blend.
        # See aec.py ResFilter._stage_gain_compute for full rationale.
        # F3.1 v3: drop envelope gate, keep blend only. See aec.py for
        # full rationale on why binary gating on mic/far ratio is unsafe.
        if (getattr(rf, '_use_mic_excess_evidence', False)
                and filter_converged
                and _lw_ready
                and not epc_active):
            far_lw = rf._residual_est._long_window_far_psd
            erl_e = float(erl_estimate)
            excess = np.maximum(rf.error_psd - far_lw * erl_e, 0.0)
            excess_ratio = np.clip(
                excess / (rf.error_psd + 1e-10), 0.0, 1.0,
            ).astype(np.float32)
            legacy = (1.0 - coh2).astype(np.float32)
            _BLEND_F31 = 0.7
            dt_per_bin = _BLEND_F31 * excess_ratio + (1.0 - _BLEND_F31) * legacy
            if effective_dt > 0.5:
                floor_lift = float((effective_dt - 0.5) * 2.0)
                dt_per_bin = np.maximum(dt_per_bin, floor_lift)
        elif rf._plan_b_dt_per_bin_gamma:
            dt_per_bin = (1.0 - coh2).astype(np.float32)
            if effective_dt > 0.5:
                floor_lift = float((effective_dt - 0.5) * 2.0)
                dt_per_bin = np.maximum(dt_per_bin, floor_lift)
        else:
            dt_per_bin = np.maximum(
                np.full(rf.n_freqs, effective_dt, dtype=np.float32),
                1.0 - coh2
            )
        if is_stationary_dt:
            dt_per_bin = np.maximum(dt_per_bin, rf._stat_dt_mask)
        rf._dt_per_bin_last = dt_per_bin

        _hf_2k = rf._hf_cap_bin_2k
        rf._p4b_dt_per_bin_mean = float(np.mean(dt_per_bin))
        rf._p4b_dt_per_bin_hf_mean = (
            float(np.mean(dt_per_bin[_hf_2k:]))
            if dt_per_bin.shape[0] > _hf_2k else 0.0
        )
        rf._p4b_coh2_hf_mean = (
            float(np.mean(coh2[_hf_2k:])) if coh2.shape[0] > _hf_2k else 0.0
        )
        rf._p4b_effective_dt = float(effective_dt)
        rf._p4b_is_stationary_dt = int(is_stationary_dt)

        dt_shaped_per_bin = dt_per_bin ** 1.1
        nearend_est = np.maximum(raw_nearend_est * dt_shaped_per_bin, noise_floor_psd)

        min_ne_from_dt = rf.error_psd * dt_shaped_per_bin
        _startup_dt_cond = (effective_dt > 0.35 and far_power > 1e-4
                            and not filter_once_converged)
        if _startup_dt_cond and rf.startup_dt_min_ne_scale != 1.0:
            min_ne_from_dt = min_ne_from_dt * rf.startup_dt_min_ne_scale
        if rf._residual_est.using_render_based:
            min_ne_from_dt = min_ne_from_dt * getattr(rf, 'render_min_ne_factor', 0.5)
        nearend_est = np.maximum(nearend_est, min_ne_from_dt)
        if rf._stats is not None:
            rf._stats_last_min_ne = float(np.mean(min_ne_from_dt))

        ne_physical_floor = rf.error_psd * 0.05
        nearend_est = np.maximum(nearend_est, ne_physical_floor)

        rf._diag_nearend_est_last = nearend_est
        rf._diag_residual_echo_psd_last = residual_echo_psd

        enr = residual_echo_psd / (nearend_est + 1e-10)

        blend = rf._enr_blend
        scale = rf.enr_scale
        ne_confidence = dt_per_bin
        effective_scale = scale
        enr_t_ne = (1 - blend) * 2.0 + blend * 1.5
        enr_s_ne = (1 - blend) * 3.0 + blend * 2.5
        enr_t_fs = (1 - blend) * (0.3 * effective_scale) + blend * (0.07 * effective_scale)
        enr_s_fs = (1 - blend) * (0.4 * effective_scale) + blend * (0.1 * effective_scale)
        if effective_dt > 0.4:
            dt_enr_relax = 1.0 + (effective_dt - 0.4) / 0.6 * 0.5
            enr_t_ne = enr_t_ne * dt_enr_relax
            enr_s_ne = enr_s_ne * dt_enr_relax
        enr_t = ne_confidence * enr_t_ne + (1 - ne_confidence) * enr_t_fs
        enr_s = ne_confidence * enr_s_ne + (1 - ne_confidence) * enr_s_fs
        min_gate_width = 0.2
        enr_s_safe = np.maximum(enr_s, enr_t + min_gate_width)

        g = np.where(enr > enr_t,
                     np.clip((enr_s_safe - enr) / (enr_s_safe - enr_t + eps), 0.0, 1.0),
                     1.0)

        if np.sum(rf.noise_psd) > 0:
            emr = residual_echo_psd / (rf.noise_psd + 1e-10)
            emr_transparent = 0.3
            g_emr = np.clip(emr_transparent / (emr + 1e-10), 0.0, 1.0)
            g = np.maximum(g, g_emr)

        if rf._stats is not None:
            rf._stats_last_gain_before_floor = float(np.mean(g))
        if getattr(rf, '_capture_stages', False):
            rf._stage_gains = {'01_softgate_emr': g.copy()}
        rf._diag_round5_stages[0] = float(np.mean(g[rf._voice_band_idx])) if rf._voice_band_idx.size > 0 else 0.0
        g = np.maximum(g, spectral_g_min)
        if rf._stats is not None:
            rf._stats_last_gain_after_floor = float(np.mean(g))
        if getattr(rf, '_capture_stages', False):
            rf._stage_gains['02_spectral_floor'] = g.copy()
        rf._diag_round5_stages[1] = float(np.mean(g[rf._voice_band_idx])) if rf._voice_band_idx.size > 0 else 0.0

        if rf._stats is not None:
            rf._stats_last_enr = float(np.mean(enr))
            rf._stats_last_nearend = float(np.mean(nearend_est))
            rf._stats_last_res_psd = float(np.mean(residual_echo_psd))

        _ne_mean = float(np.mean(nearend_est))
        _res_mean = float(np.mean(residual_echo_psd))
        _noise_mean = float(np.mean(rf.noise_psd)) + 1e-10
        rf._diag_dominant_nearend_raw = bool(
            _ne_mean > 3.0 * _res_mean
            and _ne_mean > 5.0 * _noise_mean
        )

    elif rf.gain_type == "wiener" and residual_echo_psd is not None:
        noise_floor_psd = np.mean(rf.error_psd) * 0.01 + eps
        nearend_est = np.maximum(rf.error_psd - residual_echo_psd, noise_floor_psd)
        beta = rf.over_sub
        g = nearend_est / (nearend_est + beta * residual_echo_psd + eps)
        g = np.maximum(g, spectral_g_min)
    elif residual_echo_psd is not None:
        eer_direct = residual_echo_psd / (rf.error_psd + eps)
        g = np.maximum(1.0 - rf.over_sub * eer_direct, spectral_g_min)
    else:
        g = np.maximum(1.0 - rf.over_sub * eer, spectral_g_min)
    return g
