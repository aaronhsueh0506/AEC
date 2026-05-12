"""F3.1 — per-bin mic-energy excess as NE evidence (unit tests).

Verifies the new `AecConfig.use_mic_excess_evidence` branch in
`ResFilter._stage_gain_compute`. The branch replaces the `(1 - coh2)`
component of `dt_per_bin` with the validated excess-ratio metric
`max(error_psd − far_lw·ERL_est, 0) / error_psd` when:

  • flag is True AND
  • filter_converged is True AND
  • `_long_window_far_psd` has at least one update.

In all other cases the legacy / P4B path is taken and behaviour is
byte-identical to the pre-F3.1 implementation.

Run:
    PYTHONPATH=python python3 -m unittest python.test_f3_1_mic_excess
or:
    python3 -m unittest python/test_f3_1_mic_excess.py
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aec import ResFilter


def _make_res(*, use_flag: bool, n_freqs: int = 65, block_size: int = 128) -> ResFilter:
    """Construct a ResFilter with `gain_type='enr'` and `echo_method='direct'`
    so the F3.1 branch is reachable."""
    return ResFilter(
        block_size=block_size,
        n_freqs=n_freqs,
        echo_method='direct',
        gain_type='enr',
        use_mic_excess_evidence=use_flag,
    )


def _set_lw_state(rf: ResFilter, *, far_lw: np.ndarray, n_updates: int = 100) -> None:
    """Populate the long-window far-PSD state on the residual estimator,
    so the F3.1 readiness gate (`_long_window_n_updates > 0`) opens."""
    rf._residual_est._long_window_far_psd[:] = far_lw.astype(np.float32)
    rf._residual_est._long_window_n_updates = int(n_updates)


def _call_gain_compute(rf: ResFilter, *, error_psd: np.ndarray,
                        residual_echo_psd: np.ndarray,
                        coh2: np.ndarray, effective_dt: float = 0.0,
                        filter_converged: bool = True,
                        erl_estimate: float = 0.1) -> np.ndarray:
    """Drive `_stage_gain_compute` with controlled state; return per-bin
    `dt_per_bin` that the F3.1 / legacy branches produced (read from the
    `_dt_per_bin_last` cache populated inside the method)."""
    rf.error_psd[:] = error_psd
    rf.echo_psd[:] = error_psd  # not relevant to dt_per_bin branch
    g = rf._stage_gain_compute(
        residual_echo_psd=residual_echo_psd,
        eer=None,
        coh2=coh2,
        effective_dt=effective_dt,
        is_stationary_dt=False,
        far_power=1e-2,
        filter_once_converged=filter_converged,
        spectral_g_min=np.full(rf.n_freqs, 0.01, dtype=np.float32),
        eps=1e-10,
        erl_estimate=erl_estimate,
        filter_converged=filter_converged,
    )
    # `_dt_per_bin_last` is the canonical post-branch artefact; capture it
    return rf._dt_per_bin_last.copy(), g.copy()


class TestF31FlagOffParity(unittest.TestCase):
    """Flag-OFF (default) must reproduce legacy `(1-coh2)` behaviour exactly."""

    def test_flag_off_matches_legacy(self):
        n = 65
        rng = np.random.default_rng(0)
        coh2 = rng.uniform(0.0, 1.0, n).astype(np.float32)
        error_psd = rng.uniform(1e-4, 1.0, n).astype(np.float32)
        far_lw = rng.uniform(1e-4, 1.0, n).astype(np.float32)
        res_echo = error_psd * 0.5

        rf_off = _make_res(use_flag=False)
        _set_lw_state(rf_off, far_lw=far_lw)
        dtp_off, _ = _call_gain_compute(
            rf_off, error_psd=error_psd, residual_echo_psd=res_echo,
            coh2=coh2, effective_dt=0.1, filter_converged=True,
        )

        # Legacy: dt_per_bin = max(effective_dt, 1 - coh2)
        expected = np.maximum(np.full(n, 0.1, dtype=np.float32),
                              1.0 - coh2)
        np.testing.assert_allclose(dtp_off, expected, rtol=0, atol=0,
                                    err_msg="Flag-OFF dt_per_bin must be byte-identical to legacy")


class TestF31FlagOnFallbacks(unittest.TestCase):
    """Flag-ON must still fall back to legacy when guards fail."""

    def test_flag_on_but_not_converged_falls_back(self):
        n = 65
        rng = np.random.default_rng(1)
        coh2 = rng.uniform(0.0, 1.0, n).astype(np.float32)
        error_psd = rng.uniform(1e-4, 1.0, n).astype(np.float32)
        far_lw = rng.uniform(1e-4, 1.0, n).astype(np.float32)
        res_echo = error_psd * 0.5

        rf_on = _make_res(use_flag=True)
        _set_lw_state(rf_on, far_lw=far_lw)
        dtp_on, _ = _call_gain_compute(
            rf_on, error_psd=error_psd, residual_echo_psd=res_echo,
            coh2=coh2, effective_dt=0.1, filter_converged=False,  # NOT converged
        )

        expected = np.maximum(np.full(n, 0.1, dtype=np.float32),
                              1.0 - coh2)
        np.testing.assert_allclose(dtp_on, expected, rtol=0, atol=0,
                                    err_msg="Flag-ON + filter_converged=False must fall back")

    def test_flag_on_but_lw_not_ready_falls_back(self):
        n = 65
        rng = np.random.default_rng(2)
        coh2 = rng.uniform(0.0, 1.0, n).astype(np.float32)
        error_psd = rng.uniform(1e-4, 1.0, n).astype(np.float32)
        res_echo = error_psd * 0.5

        rf_on = _make_res(use_flag=True)
        # Do NOT call _set_lw_state — long-window stays at zero updates.
        dtp_on, _ = _call_gain_compute(
            rf_on, error_psd=error_psd, residual_echo_psd=res_echo,
            coh2=coh2, effective_dt=0.1, filter_converged=True,
        )

        expected = np.maximum(np.full(n, 0.1, dtype=np.float32),
                              1.0 - coh2)
        np.testing.assert_allclose(dtp_on, expected, rtol=0, atol=0,
                                    err_msg="Flag-ON + LW not ready must fall back")


class TestF31FsConverged(unittest.TestCase):
    """In FS converged scenario, residual decorrelates → coh2 ≈ 0 →
    legacy dt_per_bin saturates to 1 (the bug). F3.1 must instead give
    a small dt_per_bin, because mic energy is fully explained by
    `far_lw·ERL_est`."""

    def test_fs_converged_metric_does_not_saturate(self):
        n = 65
        rng = np.random.default_rng(3)
        # FS converged: residual is small and uncorrelated with far.
        # coh2 is low everywhere (the broken case).
        coh2 = rng.uniform(0.0, 0.05, n).astype(np.float32)
        far_lw = rng.uniform(0.5, 1.0, n).astype(np.float32)
        erl = 0.1
        # error_psd ≈ far_lw × erl (well-cancelled echo); tiny added noise.
        error_psd = (far_lw * erl + rng.uniform(1e-5, 5e-5, n)).astype(np.float32)
        res_echo = error_psd * 0.9  # most of error attributed to echo

        # Legacy path:
        rf_off = _make_res(use_flag=False)
        _set_lw_state(rf_off, far_lw=far_lw)
        dtp_off, _ = _call_gain_compute(
            rf_off, error_psd=error_psd, residual_echo_psd=res_echo,
            coh2=coh2, effective_dt=0.0, filter_converged=True,
            erl_estimate=erl,
        )

        # F3.1 path:
        rf_on = _make_res(use_flag=True)
        _set_lw_state(rf_on, far_lw=far_lw)
        dtp_on, _ = _call_gain_compute(
            rf_on, error_psd=error_psd, residual_echo_psd=res_echo,
            coh2=coh2, effective_dt=0.0, filter_converged=True,
            erl_estimate=erl,
        )

        # Legacy dt_per_bin should be near 1 (the bug we're fixing).
        self.assertGreater(float(np.mean(dtp_off)), 0.9,
                            msg="Legacy: dt_per_bin should be ~1 in FS post-cancel (the bug)")
        # F3.1 dt_per_bin should be small (NE not present).
        self.assertLess(float(np.mean(dtp_on)), 0.2,
                         msg=f"F3.1: dt_per_bin should be small in FS post-cancel, got mean={float(np.mean(dtp_on))}")


class TestF31DoubleTalk(unittest.TestCase):
    """In DT scenario, mic energy exceeds echo model → excess_ratio rises
    → F3.1 dt_per_bin lifts toward 1, correctly indicating NE."""

    def test_dt_scenario_metric_rises(self):
        n = 65
        rng = np.random.default_rng(4)
        # DT: mic energy is dominated by near-end speech; echo small.
        coh2 = rng.uniform(0.0, 0.2, n).astype(np.float32)
        far_lw = rng.uniform(0.1, 0.5, n).astype(np.float32)
        erl = 0.1
        # error_psd >> far_lw × erl (mic dominated by NE).
        ne_energy = rng.uniform(0.5, 2.0, n).astype(np.float32)
        error_psd = (far_lw * erl + ne_energy).astype(np.float32)
        res_echo = far_lw * erl  # residual echo small

        rf_on = _make_res(use_flag=True)
        _set_lw_state(rf_on, far_lw=far_lw)
        dtp_on, _ = _call_gain_compute(
            rf_on, error_psd=error_psd, residual_echo_psd=res_echo,
            coh2=coh2, effective_dt=0.0, filter_converged=True,
            erl_estimate=erl,
        )

        self.assertGreater(float(np.mean(dtp_on)), 0.7,
                            msg=f"F3.1: dt_per_bin should rise in DT, got mean={float(np.mean(dtp_on))}")


class TestF31EffectiveDtFloor(unittest.TestCase):
    """effective_dt > 0.5 must still lift the F3.1 dt_per_bin floor."""

    def test_effective_dt_floor_lift_applies(self):
        n = 65
        # FS converged scenario where F3.1 base would be ~0.
        far_lw = np.full(n, 1.0, dtype=np.float32)
        erl = 0.1
        error_psd = (far_lw * erl).astype(np.float32)
        coh2 = np.full(n, 0.0, dtype=np.float32)
        res_echo = error_psd.copy()

        rf_on = _make_res(use_flag=True)
        _set_lw_state(rf_on, far_lw=far_lw)
        # effective_dt = 0.8 → floor_lift = (0.8 - 0.5) * 2 = 0.6
        dtp_on, _ = _call_gain_compute(
            rf_on, error_psd=error_psd, residual_echo_psd=res_echo,
            coh2=coh2, effective_dt=0.8, filter_converged=True,
            erl_estimate=erl,
        )

        # All bins should be lifted to at least 0.6 by the floor_lift.
        self.assertGreaterEqual(float(np.min(dtp_on)), 0.59,
                                msg=f"floor_lift should set dt_per_bin >= 0.6, got min={float(np.min(dtp_on))}")


if __name__ == '__main__':
    unittest.main()
