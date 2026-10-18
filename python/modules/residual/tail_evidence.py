"""TailEvidenceModel — a filter-independent, per-bin echo-evidence accumulator fed into R^2
as an upper bound only (``r2 = max(r2, evidence)``), never into the adaptive filter, the
delay estimator, or ERLE.

Rationale (plan section 五, AMEND_1): the linear filter's own residual (S2_lin/ERLE) is blind
to echo energy the filter has not estimated. Per-band energy-decay analysis of a known-good
far-end-only estimator (f13/a20/out/s3b/fir.npz) showed the missing low-frequency energy on
one tested recording sits at a SHORT lag relative to the current render, not in a long
reverberant tail -- so this model tracks render power at a short, configurable lag, not
AEC3's own reverb-tail lag (which this file does not touch; ``ReverbModel``/
``ReverbFrequencyResponse`` are unchanged).

Mechanism, one bin at a time:
  U[k]   = (U[k] + X2_tail[k]) * decay[k]        -- physical render-tail tracker, every hop
  g[k]   a robust CENTRAL estimate of obs[k] = max(E2[k] - N2[k], 0) / U[k], tracked as a
         symmetric EMA in the log domain on hops a caller judges trustworthy (far active,
         linear filter usable, not suspected near-end). This is deliberately NOT a
         min-statistics lower bound: obs has several dB of hop-to-hop spread even on genuinely
         far-only audio (filter/windowing/estimation noise, not level changes), and a running
         minimum of a noisy ratio settles well below its true central value and, once a long
         hold has passed, can only descend further -- useless as an evidence LEVEL, however
         well it would behave as a cautious lower bound. Safety against near-end contamination
         is the caller's gate (never update on a hop judged untrustworthy), not pessimism in
         the tracker; see the caller for the gate and why it must never read the live
         dt-protect flag.
  T[k]   = strength * g[k] * U[k], zero until a bin has accumulated enough trustworthy
           observations (warmup) and outside the configured band (default LF-only).

Caller owns: selecting X2_tail (which lag, which coordinate), E2/N2 (which coordinate -- the
plan requires the SG nearend coordinate, i.e. windowed E2, not raw render Y2), the learning
gate, and the injection (``max(r2, T)`` / optionally ``max(r2_unb, T)``). This class holds no
opinion on any of those; it only does the per-bin accumulate/learn/report arithmetic so the
gating logic in the caller stays readable.
"""
from __future__ import annotations

import numpy as np


class TailEvidenceModel:
    def __init__(
        self,
        n_bins: int,
        decay_per_hop: np.ndarray,
        max_bin: int,
        g_min: float = 1e-6,
        g_max: float = 10.0,
        ema_alpha: float = 0.99,
        warmup_hops: int = 1,
        x2_min: float = 1e-6,
        u_min: float = 1e-6,
    ) -> None:
        self._n_bins = int(n_bins)
        self._decay = np.asarray(decay_per_hop, dtype=np.float64)
        if self._decay.shape == ():
            self._decay = np.full(self._n_bins, float(self._decay))
        assert self._decay.shape == (self._n_bins,)
        self._max_bin = int(max_bin)
        self._g_min = float(g_min)
        self._g_max = float(g_max)
        self._log_g_min = np.log10(self._g_min)
        self._log_g_max = np.log10(self._g_max)
        self._ema_alpha = float(ema_alpha)
        self._warmup_hops = int(warmup_hops)
        self._x2_min = float(x2_min)
        self._u_min = float(u_min)
        self.U = np.zeros(self._n_bins, dtype=np.float64)
        self._log_g = np.full(self._n_bins, self._log_g_min, dtype=np.float64)
        self._n_obs = np.zeros(self._n_bins, dtype=np.int64)

    def reset(self) -> None:
        self.U.fill(0.0)
        self._log_g.fill(self._log_g_min)
        self._n_obs.fill(0)

    @property
    def g(self) -> np.ndarray:
        return np.power(10.0, self._log_g)

    def update(
        self,
        x2_tail: np.ndarray,
        error_psd: np.ndarray,
        noise_psd: np.ndarray,
        learn_mask: np.ndarray,
    ) -> None:
        """One hop. ``learn_mask`` is a per-bin bool array (caller already ANDed in every
        frame-level condition); U accumulates every hop regardless (it is tracking the
        physical render tail, not a learned quantity), but only bins where ``learn_mask`` is
        True AND the bin's own evidence floor (``x2_min``/``u_min``) is cleared update ``g``.
        """
        x2_tail = np.asarray(x2_tail, dtype=np.float64)
        self.U = (self.U + x2_tail) * self._decay
        eligible = (
            np.asarray(learn_mask, dtype=bool)
            & (x2_tail > self._x2_min)
            & (self.U > self._u_min)
        )
        if not np.any(eligible):
            return
        obs = np.maximum(
            np.asarray(error_psd, dtype=np.float64) - np.asarray(noise_psd, dtype=np.float64), 0.0
        ) / np.maximum(self.U, 1e-30)
        log_obs = np.log10(np.clip(obs, 10.0 ** self._log_g_min, 10.0 ** self._log_g_max))
        first = eligible & (self._n_obs == 0)
        follow = eligible & (~first)
        self._log_g = np.where(first, log_obs, self._log_g)
        self._log_g = np.where(
            follow,
            self._ema_alpha * self._log_g + (1.0 - self._ema_alpha) * log_obs,
            self._log_g,
        )
        self._n_obs = np.where(eligible, self._n_obs + 1, self._n_obs)

    def evidence(self, strength: float = 1.0) -> np.ndarray:
        t = (self.g * self.U * float(strength)).astype(np.float32)
        t[self._n_obs < self._warmup_hops] = 0.0
        if self._max_bin < self._n_bins:
            t[self._max_bin:] = 0.0
        return t
