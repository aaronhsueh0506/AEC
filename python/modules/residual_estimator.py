"""ResidualEchoEstimator — per-bin residual-echo PSD estimator.

Extracted from ``aec.py`` during refactor R.8. Lives in its own
module (separate from ``res_filter.py``) because the C-side mirror
is residual_estimator.{h,c} — file boundaries match Python ↔ C
cross-reference.

Depends on numpy + erle.compute_erle_confidence.
"""
import numpy as np

from .erle import compute_erle_confidence


class ResidualEchoEstimator:
    """Two-path residual echo PSD attribution (stage 1 + stage 2 of ResFilter).

    Replaces ResFilter's inline ~100 lines of residual_echo_psd computation with
    an explicit class that owns the render-based switching state. Provides two
    modes:

      'legacy': bit-exact reproduction of v2.8.1 inline logic — ERLE-blended
                linear estimate with optional render-based blend driven by
                an ENR-adaptive switching threshold + hysteresis + min hold.
                Used as default for parity validation.

      'split' : explicit AEC3-style branch on `aec_state.usable_linear_estimate`.
                Linear path: `S2_linear / ERLE`. Nonlinear path: `X2 * echo_path_gain`.
                Used by Phase B2 ablation (R1 variant). Disabled by default.

    Owns: _using_render_based, _render_based_hold (legacy state machine).
    Caller (ResFilter) keeps echo_psd / error_psd / near_psd / coh2 / far_activity
    on itself; we pass them in by reference per call.
    """
    LEGACY = 'legacy'
    SPLIT = 'split'

    def __init__(self, n_freqs: int, mode: str = 'legacy',
                 long_window_alpha: float = 0.993):
        """v3.10.0: long-window render PSD EMA mirrors WebRTC AEC3's
        kRenderTransferQueueSizeFrames=100 (= 1000 ms at 10ms/frame). When
        delay alignment is unreliable, RES uses this long-time-averaged
        far PSD as the echo PSD basis instead of the instantaneous far_psd
        (which is sensitive to misalignment). Speech long-term PSD is
        roughly stationary over 1 s, so the smoothed estimate gives a
        usable echo template even when the filter taps are wrong.

        TC ≈ 100 frames at alpha=0.993 (alpha^100 ≈ 0.5).
        """
        self.n_freqs = n_freqs
        self.mode = mode
        self._using_render_based = False
        self._render_based_hold = 0
        self._long_window_alpha = long_window_alpha
        self._long_window_far_psd = np.zeros(n_freqs, dtype=np.float32)
        self._long_window_n_updates = 0
        # P3g Phase 0 dry-run scalars
        self._last_linear_residual_psd_mean = 0.0
        self._last_render_residual_psd_mean = 0.0
        self._last_render_blend = 0.0

    def reset(self, preserve_long_window_ema: bool = False) -> None:
        """Clear residual-echo estimator state.

        preserve_long_window_ema (v3.10.2): when True, keep the long-window
        far-PSD EMA (`_long_window_far_psd` + `_long_window_n_updates`).
        Used by filter-derived-state resets (plateau recovery, delay_first):
        the EMA is input-side context — its accumulated long-term render
        spectrum is independent of the bad filter taps and should survive
        the reset, otherwise the freshly reset filter spends 100 frames
        in pre-warmup-fallback mode again.
        """
        self._using_render_based = False
        self._render_based_hold = 0
        if not preserve_long_window_ema:
            self._long_window_far_psd.fill(0.0)
            self._long_window_n_updates = 0

    @property
    def using_render_based(self) -> bool: return self._using_render_based

    def attribute(self, *, aec_state=None, **kw) -> np.ndarray:
        """Mode-dispatch entry. Caller passes the union of legacy+split kwargs;
        method picks the relevant subset."""
        if self.mode == self.SPLIT and aec_state is not None:
            return self.attribute_split(
                echo_psd=kw['echo_psd'], error_psd=kw['error_psd'],
                far_spec=kw['far_spec'], far_power=kw['far_power'],
                erle_factor=kw['erle_factor'], erl_estimate=kw['erl_estimate'],
                filter_erle=kw['filter_erle'], fb_erle=kw['fb_erle'],
                aec_state=aec_state,
            )
        return self.attribute_legacy(aec_state=aec_state, **kw)

    def attribute_legacy(self, *, echo_psd: np.ndarray, error_psd: np.ndarray,
                         coh2: np.ndarray, far_spec, far_power: float,
                         erle_factor: float, dt_for_fs: float, far_activity: float,
                         epc_active: bool, saturation_level: float,
                         filter_converged: bool, erl_estimate: float,
                         filter_erle, fb_erle, aec_state=None) -> np.ndarray:
        """Stage 1 (ERLE-blended linear) + Stage 2 (render-based switch).

        Bit-exact port of ResFilter.process() residual-echo block from v2.8.1
        (lines ~1456-1555). Mutates self._using_render_based / _render_based_hold.
        """
        # Multi-ERLE residual estimation (Phase 2)
        confidence = compute_erle_confidence(filter_erle.erle, fb_erle.fb_erle)
        erle_corrected = (confidence * filter_erle.erle
                          + (1.0 - confidence) * 1.0)
        erle_corrected = np.maximum(erle_corrected, 0.5)

        erle_est = echo_psd / erle_corrected
        direct_est = echo_psd

        if far_power > 1e-4:
            dt_weight = 1.0 - dt_for_fs
            nonlinear_floor = error_psd * coh2 * far_activity * dt_weight
            direct_est = np.maximum(direct_est, nonlinear_floor)
            erle_est = np.maximum(erle_est, nonlinear_floor)

        residual_echo_psd = (1.0 - erle_factor) * direct_est + erle_factor * erle_est
        # P3g Phase 0: stash the linear-only residual (Stage-1 ERLE-blended)
        # for dry-run audit before any Stage-2 render-based override.
        # Read by AEC.process() into _diag — purely diagnostic.
        self._last_linear_residual_psd_mean = float(np.mean(residual_echo_psd))

        # v3.10.0: maintain long-window far-PSD EMA every frame regardless of
        # render mode — so it's ready immediately when delay drops out and
        # we need a delay-agnostic echo template. Skip on silence to avoid
        # poisoning the EMA with the noise floor.
        if far_spec is not None and far_power > 1e-6:
            inst_far_psd = (np.abs(far_spec) ** 2).astype(np.float32)
            if self._long_window_n_updates == 0:
                self._long_window_far_psd[:] = inst_far_psd
            else:
                a = self._long_window_alpha
                self._long_window_far_psd = (a * self._long_window_far_psd
                                              + (1.0 - a) * inst_far_psd)
            self._long_window_n_updates += 1

        if far_power > 1e-4:
            error_power_mean = float(np.mean(error_psd)) + 1e-10
            enr = far_power / error_power_mean
            switching_threshold = 0.5 * np.clip(enr / (enr + 1.0), 0.3, 0.7)
            hysteresis = 0.05
            if self._using_render_based:
                effective_threshold = switching_threshold + hysteresis
            else:
                effective_threshold = switching_threshold
            force_render = (
                epc_active
                or saturation_level > 0.5
                or not filter_converged
            )
            want_render = (erle_factor < effective_threshold) or force_render
            if want_render and not self._using_render_based:
                self._render_based_hold = 5
            if self._using_render_based:
                self._render_based_hold = max(self._render_based_hold - 1, 0)
            can_exit = (not want_render and self._render_based_hold == 0)
            self._using_render_based = want_render or (self._using_render_based and not can_exit)

            if self._using_render_based:
                # v3.10.1: long-window EMA far-PSD blended with instantaneous.
                # The EMA is updated EVERY far-active frame (regardless of
                # mode — see block above) so its time constant is constant,
                # not dependent on how long fallback has been engaged.
                # We READ it only here (fallback-only). Two refinements
                # vs v3.10.0's hard-replace:
                #   (a) Warmup gate: 100 updates (= 1 EMA TC at alpha=0.993)
                #       so the EMA has formed a real long-term estimate
                #       before being trusted. Below 100 updates: instantaneous
                #       only.
                #   (b) Blend, not hard replace: even after warmup, mix 70%
                #       long-window + 30% instantaneous. The instantaneous
                #       component preserves response to fast far-end onsets;
                #       the long-window component carries delay-agnostic
                #       structure. This avoids the FS smearing risk of
                #       hard-replacing inst with stale EMA when far is
                #       changing fast (Codex finding 3).
                if far_spec is not None:
                    inst_far_psd = (np.abs(far_spec) ** 2).astype(np.float32)
                    warmup_weight = float(min(self._long_window_n_updates / 100.0, 1.0))
                    lw_weight = 0.7 * warmup_weight
                    far_psd = ((1.0 - lw_weight) * inst_far_psd
                               + lw_weight * self._long_window_far_psd)
                else:
                    far_psd = np.zeros(self.n_freqs, dtype=np.float32)
                # v3.8 ABL-1 (ablate v3.3 error_based_floor): error_psd contains
                # NE during DT, so using it as residual_echo floor structurally
                # over-suppresses near-end (same lesson as v3.7.1 PR-B). Use only
                # render_based_echo = far × ERL — AEC3-aligned.
                render_based_echo = far_psd * erl_estimate
                blend = 1.0 - erle_factor / effective_threshold
                blend = np.clip(blend, 0.0, 1.0)
                residual_echo_psd = ((1.0 - blend) * residual_echo_psd
                                     + blend * render_based_echo)
                # P3g Phase 0: stash the render-based component and the
                # blend used, for dry-run audit. (linear residual mean is
                # already stashed above before any override.)
                self._last_render_residual_psd_mean = float(np.mean(render_based_echo))
                self._last_render_blend = float(np.mean(blend))

        # If render-based path didn't run this frame, leave previous
        # render-residual stale; consumers should gate on using_render_based.
        return residual_echo_psd

    def attribute_split(self, *, echo_psd: np.ndarray, error_psd: np.ndarray,
                        far_spec, far_power: float,
                        erle_factor: float, erl_estimate: float,
                        filter_erle, fb_erle, aec_state) -> np.ndarray:
        """AEC3-style two-path R2: linear if `aec_state.usable_linear_estimate`,
        else render-based. Used by Phase B2 R1 ablation."""
        # Always update _using_render_based to reflect current decision
        self._using_render_based = not aec_state.usable_linear_estimate
        if aec_state.usable_linear_estimate:
            confidence = compute_erle_confidence(filter_erle.erle, fb_erle.fb_erle)
            erle_corrected = (confidence * filter_erle.erle + (1.0 - confidence) * 1.0)
            erle_corrected = np.maximum(erle_corrected, 0.5)
            return echo_psd / erle_corrected
        # Render-based path: X2 * echo_path_gain
        far_psd = np.abs(far_spec) ** 2 if far_spec is not None else np.zeros(self.n_freqs)
        return far_psd * erl_estimate
