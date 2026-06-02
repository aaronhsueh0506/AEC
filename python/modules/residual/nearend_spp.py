"""Per-bin near-end speech-presence probability (IMCRA-style), reference-based.

Motivation
----------
Every DT lever we have (min-gain floor depth, the cohxd reference-coherence
floor release, the SER floor, the soft near-end blend) is a Pareto *trade* on
the two double-talk axes: more echo cancellation costs near-end preservation,
because they all push a single ``Y²``-derived / ``Γ²`` quantity that couples
echo-suppression to near-suppression. Raising *both* DT axes needs a mechanism
that *moves* the frontier — a per-bin estimate of "is genuine near-end present
in this bin right now" so suppression can be released for echo only where the
near-end is absent.

Single-lag coherence (Γ²(X,Y) used by cohxd, Γ²(Ŷ,Y) used by the ERLE gate)
cannot do this: a reverb-tail far-end decorrelates the residual from the
filter estimate exactly like genuine near-end does, so it false-positives on
echo-only reverb bins — the wall that closed every prior coherence
discriminator here (Jung-2011, Enzner-Vary, ...). It also leans on Ŷ, which is
unreliable on the under-converged hard cases where we most want to suppress.

This estimator instead tracks, per bin, the *minimum over a window* of the
residual-to-reference power ratio ``|E|² / |X|²`` — the residual echo-path
transfer. During far-end-only activity (incl. a decaying reverb tail) that
ratio is slowly varying and well explained by the known, reliable reference
``X``; the minimum tracks the echo/reverb floor. Genuine near-end adds energy
to ``E`` that is *not* explained by ``X``, spiking the ratio above its tracked
minimum. The spike (in dB above the floor) maps through a sigmoid to a per-bin
near-end-presence probability ``p_ne ∈ [0, 1]``.

Key properties:
  * uses ``X`` (true reference), NOT ``Ŷ`` → robust to filter under-convergence;
  * multi-frame minima tracking → separates a persistent reverb/echo floor from
    a transient near-end onset, which single-lag coherence cannot;
  * conservative warm-up: until a floor is learned ``p_ne ≈ 0`` (assume
    echo-only), so as a suppression gate it can only *withhold* extra
    cancellation, never add it.

Literature: Cohen 2002 (IMCRA minima-controlled recursive averaging);
decision-directed SPP; DTD-aided RES spectrogram masking (MDPI Acoustics 2022).

Default-OFF research substrate. ``update`` is pure per-frame state; ``reset``
clears it. Inputs are power spectra already on the orchestrator's ``_PSD_SCALE``
(the E/X ratio is scale-invariant, so the absolute scale is irrelevant).
"""
from __future__ import annotations

import numpy as np


class NearendSpp:
    """Per-bin near-end speech-presence probability via reference-normalized
    residual minima tracking.

    Parameters
    ----------
    n_bins : int
        Number of frequency bins (rfft length).
    alpha : float
        EMA coefficient for smoothing |E|² and |X|² (per hop). 0.2 ≈ ~50 ms at
        a 10 ms hop.
    minima_subwindow : int
        MCRA sub-window length D in frames. The tracked minimum spans
        [D, 2D) frames; at a 10 ms hop, D=60 → a ~0.6–1.2 s floor window
        (long enough to hold across a far-end pause, short enough to follow
        slow filter convergence).
    spike_thr_db, spike_soft_db : float
        Sigmoid centre / softness, in dB of ratio above the tracked floor.
        A near-end onset typically lifts the residual several dB above the
        echo floor; spike_thr_db≈4, spike_soft_db≈3.
    eps : float
        Small power floor to keep ratios well-defined on near-silent bins.
    """

    def __init__(
        self,
        n_bins: int,
        *,
        alpha: float = 0.2,
        minima_subwindow: int = 60,
        spike_thr_db: float = 5.0,
        spike_soft_db: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        self._n_bins = int(n_bins)
        self._alpha = float(alpha)
        self._D = int(minima_subwindow)
        self._thr_db = float(spike_thr_db)
        self._soft_db = float(spike_soft_db)
        self._eps = float(eps)
        self.reset()

    def reset(self) -> None:
        n = self._n_bins
        self._s_e = np.zeros(n, dtype=np.float64)
        self._s_x = np.zeros(n, dtype=np.float64)
        # Minima trackers start +inf so the first window adapts the floor down
        # to the observed transfer (warm-up → ratio ≈ 1 → p_ne ≈ 0).
        self._t_min = np.full(n, np.inf, dtype=np.float64)
        self._t_tmp = np.full(n, np.inf, dtype=np.float64)
        self._j = 0
        self._initialized = False
        self._p_ne = np.zeros(n, dtype=np.float32)

    def update(
        self,
        e_psd: np.ndarray,
        echo_ref_psd: np.ndarray,
        far_active: bool,
    ) -> np.ndarray:
        """Advance one frame and return the per-bin near-end probability.

        Parameters
        ----------
        e_psd : np.ndarray
            |E|² — linear-residual (error) power spectrum (the residual the
            suppressor sees).
        echo_ref_psd : np.ndarray
            The expected residual-echo power per bin. Prefer the RES's
            reverb/echo-path-aware estimate ``R²`` (so ``|E|²/R² ≈ 1`` during
            far-only echo and spikes only when near-end adds energy R² does not
            model). A raw delay-aligned ``|X|²`` also works in principle but on
            real speech the instantaneous ``|E|²/|X|²`` transfer is too variable
            (the residual is X through an imperfect reverberant path), so its
            windowed minimum under-reads the floor and p_ne saturates high —
            use R².
        far_active : bool
            Whether the far-end is active this frame. The ratio and its minima
            are only meaningful while the far-end drives the residual, so state
            is frozen and ``p_ne`` returns all-zeros when the far-end is silent
            (the cohxd consumer only reads ``p_ne`` under its own far-active
            latch; a far-silent near-end branch is a separate concern for the
            general-floor use).
        """
        if not far_active:
            # Hold state; report "no decision" (echo-confident-irrelevant).
            self._p_ne = np.zeros(self._n_bins, dtype=np.float32)
            return self._p_ne

        e = np.asarray(e_psd, dtype=np.float64)
        x = np.asarray(echo_ref_psd, dtype=np.float64)

        a = self._alpha
        if not self._initialized:
            # Seed the EMAs with the first far-active frame to avoid a cold
            # transient that would briefly read as a near-end spike.
            self._s_e[:] = e
            self._s_x[:] = x
            self._initialized = True
        else:
            self._s_e = (1.0 - a) * self._s_e + a * e
            self._s_x = (1.0 - a) * self._s_x + a * x

        # Per-bin residual echo-path transfer (scale-invariant in _PSD_SCALE).
        t = self._s_e / (self._s_x + self._eps)

        # MCRA-style minimum tracking over a sliding sub-window.
        np.minimum(self._t_min, t, out=self._t_min)
        np.minimum(self._t_tmp, t, out=self._t_tmp)
        self._j += 1
        if self._j >= self._D:
            np.minimum(self._t_tmp, t, out=self._t_min)
            self._t_tmp = t.copy()
            self._j = 0

        # Spike of the current transfer above its tracked floor, in dB.
        ratio = t / (self._t_min + self._eps)
        ratio = np.maximum(ratio, 1e-3)
        sr_db = 10.0 * np.log10(ratio)

        # Sigmoid → near-end-presence probability.
        z = np.clip((sr_db - self._thr_db) / max(self._soft_db, 1e-6), -50.0, 50.0)
        self._p_ne = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        return self._p_ne

    @property
    def p_ne(self) -> np.ndarray:
        return self._p_ne


def _selftest() -> None:
    """Synthetic discrimination check: the mask must light up on a genuine
    near-end onset and stay low on far-only (incl. a decaying reverb tail).

    Builds a 3-segment per-bin power timeline at one representative bin:
      seg 1 (far-only):      E ∝ X with a stable residual transfer + noise
      seg 2 (double-talk):   near-end adds energy to E that is NOT in X
      seg 3 (reverb-tail):   X and E both decay together (transfer stable)
    """
    rng = np.random.RandomState(0)
    n_bins = 8
    spp = NearendSpp(n_bins, minima_subwindow=30)

    def frame(x_level, e_extra=0.0, jitter=0.05):
        x = np.full(n_bins, x_level, dtype=np.float64) * (1.0 + jitter * rng.randn(n_bins))
        x = np.abs(x)
        transfer = 0.1  # stable residual echo-path gain
        e = transfer * x + e_extra * (1.0 + jitter * np.abs(rng.randn(n_bins)))
        return e, x

    p_far, p_dt, p_rev = [], [], []
    # seg 1: far-only, learn the floor (120 frames)
    for _ in range(120):
        e, x = frame(1.0)
        p = spp.update(e, x, far_active=True)
        p_far.append(float(p.mean()))
    # seg 2: double-talk — near-end adds energy uncorrelated with X (40 frames)
    for _ in range(40):
        e, x = frame(1.0, e_extra=0.5)   # +5x residual not explained by X
        p = spp.update(e, x, far_active=True)
        p_dt.append(float(p.mean()))
    # seg 3: reverb tail — X decays, E decays with it (same transfer) (40 frames)
    for k in range(40):
        lvl = 1.0 * (0.92 ** k)
        e, x = frame(lvl)
        p = spp.update(e, x, far_active=True)
        p_rev.append(float(p.mean()))

    far_mean = np.mean(p_far[-40:])   # after the floor is learned
    dt_mean = np.mean(p_dt[-20:])     # steady double-talk
    rev_mean = np.mean(p_rev)
    print(f"[selftest] far-only p_ne={far_mean:.3f}  "
          f"double-talk p_ne={dt_mean:.3f}  reverb-tail p_ne={rev_mean:.3f}")
    assert far_mean < 0.2, f"far-only should read echo (low p_ne), got {far_mean:.3f}"
    assert dt_mean > 0.7, f"double-talk should read near (high p_ne), got {dt_mean:.3f}"
    assert rev_mean < 0.3, f"reverb-tail should read echo (low p_ne), got {rev_mean:.3f}"
    print("[selftest] PASS — mask discriminates near onset from far-only/reverb-tail")


if __name__ == "__main__":
    _selftest()
