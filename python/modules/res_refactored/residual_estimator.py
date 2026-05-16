"""residual_estimator — P52 Phase B Module 1.

Verbatim extraction of `ResFilter._stage_residual_model` (aec.py:1732-1833) into a
free function operating on a ResFilter-shaped state container `rf`. Logic is
copied byte-for-byte (no numerical reordering, no algebra simplification, no
threshold change) per anti-loophole §5.5.

`rf` exposes the same mutable attributes as the legacy `ResFilter` instance —
`echo_psd`, `error_psd`, `near_psd_buf`, `near_psd_idx`, `near_psd`,
`_residual_est`, `_nonlinear_frames`, `_harm_lf_start/end`, `_harm_hf_start/end`,
`enable_reverb`, `reverb_decay`, `reverb_psd`, `far_activity`, `reverb_gain`,
`echo_method`, `_stats`, `_stats_last_*`, `_filter_erle_est`, `_fb_erle_est`.

In Phase B.3 the function takes `rf` directly so state lives where it always
has; later modules will migrate state onto `ResState` (§3.4) once all five
extractions are complete and the orchestrator owns frame ingestion.
"""

from __future__ import annotations

import numpy as np


def residual_estimator(rf, *, coh2, far_spec, far_power, erle_factor,
                       dt_for_fs, epc_active, saturation_level,
                       filter_converged, erl_estimate, near_spec,
                       is_stationary_dt, aec_state, echo_pwr_linear):
    """Stage 1: residual_echo_psd attribution + 4 caps + reverb tail + echo boost.

    Returns residual_echo_psd (np.ndarray or None). Mutates `rf.near_psd_buf`,
    `rf.near_psd_idx`, `rf.near_psd`, `rf.reverb_psd`, `rf._nonlinear_frames`,
    and `rf._stats_last_*` fields.
    """
    residual_echo_psd = None
    if rf.echo_method == "direct" and near_spec is not None:
        near_pwr = np.abs(near_spec) ** 2
        rf.near_psd_buf[rf.near_psd_idx] = near_pwr
        rf.near_psd_idx = (rf.near_psd_idx + 1) % 4
        rf.near_psd = np.mean(rf.near_psd_buf, axis=0)

        residual_echo_psd = rf._residual_est.attribute(
            echo_psd=rf.echo_psd, error_psd=rf.error_psd,
            coh2=coh2, far_spec=far_spec, far_power=far_power,
            erle_factor=erle_factor, dt_for_fs=dt_for_fs,
            far_activity=rf.far_activity,
            epc_active=epc_active, saturation_level=saturation_level,
            filter_converged=filter_converged, erl_estimate=erl_estimate,
            filter_erle=rf._filter_erle_est, fb_erle=rf._fb_erle_est,
            aec_state=aec_state,
        )

        if saturation_level > 0.3:
            rf._nonlinear_frames += 1
        else:
            rf._nonlinear_frames = max(0, rf._nonlinear_frames - 1)
        is_nonlinear = rf._nonlinear_frames > 5
        if is_nonlinear and far_power > 1e-4:
            nonlinear_boost = 1.0 + 1.0 * saturation_level
            residual_echo_psd = residual_echo_psd * nonlinear_boost

        if saturation_level > 0.05 and far_power > 1e-4:
            lf_start, lf_end = rf._harm_lf_start, rf._harm_lf_end
            hf_start, hf_end = rf._harm_hf_start, rf._harm_hf_end
            if lf_end > lf_start and hf_end > hf_start:
                lf_echo_mean = float(np.mean(residual_echo_psd[lf_start:lf_end]))
                distortion_factor = 0.1 + 0.4 * saturation_level
                harmonic_floor = lf_echo_mean * distortion_factor
                residual_echo_psd[hf_start:hf_end] = np.maximum(
                    residual_echo_psd[hf_start:hf_end], harmonic_floor)

        if rf._stats is not None:
            rf._stats_last_res_after_attribute = float(np.mean(residual_echo_psd))

        if not rf._residual_est.using_render_based:
            residual_echo_psd = np.minimum(residual_echo_psd, rf.echo_psd * 2.0)
        if rf._stats is not None:
            rf._stats_last_res_after_echo_cap = float(np.mean(residual_echo_psd))

        err_cap_mult = 1.5 if rf._residual_est.using_render_based else 1.0
        residual_echo_psd = np.minimum(residual_echo_psd, rf.error_psd * err_cap_mult)
        if rf._stats is not None:
            rf._stats_last_res_after_error_cap = float(np.mean(residual_echo_psd))

        if not rf._residual_est.using_render_based:
            dt_suppress = np.clip(1.0 - dt_for_fs**2, 0.1, 1.0)
            residual_echo_psd = np.minimum(residual_echo_psd, rf.error_psd * dt_suppress)
        if rf._stats is not None:
            rf._stats_last_res_after_dt_cap = float(np.mean(residual_echo_psd))

        if far_spec is not None and far_power > 1e-4 and erl_estimate > 0.0:
            far_psd_k = np.abs(far_spec) ** 2
            render_ceil = far_psd_k * min(erl_estimate * 2.0, 1.0)
            if rf._stats is not None:
                rf._stats_last_render_ceil_mean = float(np.mean(render_ceil))
                rf._stats_last_erl_estimate = float(erl_estimate)
            if not rf._residual_est.using_render_based:
                residual_echo_psd = np.minimum(residual_echo_psd, render_ceil)
        if rf._stats is not None:
            rf._stats_last_res_after_render_ceil = float(np.mean(residual_echo_psd))

        if rf.enable_reverb:
            far_psd = np.abs(far_spec) ** 2 if far_spec is not None else echo_pwr_linear
            rf.reverb_psd = (rf.reverb_decay * rf.reverb_psd
                             + (1 - rf.reverb_decay) * far_psd)
            if not is_stationary_dt:
                ne_reverb_factor = 0.7 + 0.3 * rf.far_activity * (1.0 - dt_for_fs)
                reverb_gate = rf.far_activity * ne_reverb_factor
                if rf.far_activity < 0.1:
                    reverb_gate = 0.0
                residual_echo_psd = (residual_echo_psd
                                     + rf.reverb_gain * rf.reverb_psd * reverb_gate)

    if far_power > 1e-4 and erle_factor > 0.3 and residual_echo_psd is not None:
        echo_boost = 1.0 + 0.5 * coh2 if dt_for_fs < 0.2 else np.ones_like(coh2)
        residual_echo_psd = residual_echo_psd * echo_boost

    return residual_echo_psd
