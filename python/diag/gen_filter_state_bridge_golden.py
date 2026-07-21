"""Generate a binary golden for the C filter_state_bridge port (WS5 Phase 5.2).

filter_state_bridge.build_filter_state_bridge is a thin Kalman->state adapter.
Rather than spin up the real PBFDKF/Shadow/RegimeHandler, we drive the REAL
``build_filter_state_bridge`` with synthetic duck-typed stand-ins that expose
exactly the attributes the function reads (W, fft_size, P, get_error_energy,
main_paused). This isolates the adapter LOGIC — which is the parity surface.

Per frame we evolve W (complex64), P (float32), and the two error energies so
the snapshot's irfft / np.mean / e_ratio paths all vary, then snapshot every
bridge field for the C side to replay.

Layout (LE):
  int32 fft_size, int32 n_freqs, int32 p_len, int32 n_frames
  n_frames × [
     has_W       int32  (1: feed W; 0: W absent -> taps {0}, len 1)
     has_shadow  int32
     W0          n_freqs × Complex (interleaved f32 r,i)   (always written;
                                                            ignored if !has_W)
     P           p_len   × f32
     main_e      f64
     shadow_e    f64
     filter_converged int32
     main_paused      int32
     mu_final         f64
     external_delay   int32
     any_coarse       int32
     all_diverged     int32
     -- expected outputs --
     taps_len     int32
     taps         taps_len × f32      (fft_size, or 1 when W absent)
     div          f64
     regime       int32
     filter_conv  int32 (echo)
     main_paused  int32 (echo)
     mu_final     f64   (echo)
     ext_delay    int32 (echo)
     any_coarse   int32 (echo)
     all_div      int32 (echo)
  ]

Run: python3 python/diag/gen_filter_state_bridge_golden.py /tmp/fsb_golden.bin

Round-4 review (Task C): ``div``/``mu_final`` used to be written straight from
``build_filter_state_bridge``'s return value, which computes both in Python
float64 (the algorithm SPEC, per this repo's float32-campaign policy --
python/modules/filter/filter_state_bridge.py is intentionally left as-is).
But ``FilterStateBridge.divergence_indicator``/``.mu_final`` are declared
``float`` (float32) in the C struct, and ``filter_state_bridge_build``'s
``main_e``/``shadow_e``/``mu_final`` PARAMETERS are likewise ``float``, so
the C computes the whole e_ratio/div chain in native float32 arithmetic --
not "float64 then truncate once at the end". Comparing the C's float32
result bit-exact against the float64 golden was therefore comparing two
genuinely different computations (not just a wider-vs-narrower cast of the
SAME computation), which parity_filter_state_bridge.c's `!=` checks caught
as a permanent ~3-4.5e-8 divergence on every frame -- expected float32-vs-
float64 drift, not a bug in either side.
``_c_contract_div_mu()`` below reproduces the C's exact op sequence
(fsb_f32_mean's pairwise sum via numpy's own float32-preserving np.sum,
then every subsequent op on np.float32 scalars, which numpy keeps in
float32 -- unlike raw Python floats, always float64) so the golden's
expected div/mu values are what the CURRENT f32 C contract actually
produces, restoring true bit-exact-vs-itself regression coverage.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.filter.filter_state_bridge import build_filter_state_bridge  # noqa: E402


def _c_contract_div_mu(P, main_e, shadow_e, has_shadow, mu_final):
    """Mirror filter_state_bridge.c's filter_state_bridge_build() EXACTLY,
    op-for-op, in native float32 (np.float32 scalar arithmetic stays
    float32 -- unlike Python floats, which are always float64). Returns
    (div, mu) as Python floats holding EXACT float32 values, ready to
    write into the f64 golden slots the C-side (float, widened to double
    for storage) will bit-exact-match.
    """
    mu32 = np.float32(mu_final)  # out->mu_final = mu_final; (float param)

    if P.size == 0:
        return float(np.float32(0.0)), float(mu32)

    # p_trace = fsb_f32_mean(P, p_len): float32 pairwise sum / float32(n).
    # P is already float32; np.sum over a float32 array stays float32
    # (numpy preserves dtype for floating input) and numpy 1.26's pairwise
    # algorithm is the one fsb_f32_pairwise_sum is built to match bit-exact
    # (same claim this codebase already relies on for erl_estimator.c /
    # filter_analyzer.c / fullband_erle.c's own f32 pairwise-sum kernels).
    p_trace32 = np.float32(np.sum(P, dtype=np.float32)) / np.float32(P.size)

    main_e32 = np.float32(main_e)      # C's `float main_e` parameter
    shadow_e32 = np.float32(shadow_e)  # C's `float shadow_e` parameter
    e_ratio32 = np.float32(1.0)
    if has_shadow and main_e32 > np.float32(1e-12):
        e_ratio32 = np.float32(shadow_e32 / main_e32)  # float32 / float32 -> float32

    t32 = np.float32(e_ratio32 - np.float32(1.0))
    if t32 < np.float32(0.0):
        t32 = np.float32(0.0)
    div32 = np.float32(p_trace32 * t32)

    return float(div32), float(mu32)

# Production geometry: hop=160 -> block=320 -> fft_size=512, n_freqs=257.
FFT_SIZE = 512
N_FREQS = FFT_SIZE // 2 + 1
N_PARTITIONS = 6
P_LEN = N_PARTITIONS * N_FREQS
N_FRAMES = 16


class FakePBFDKF:
    """Duck-typed stand-in: exposes only W / fft_size / P / get_error_energy."""

    def __init__(self, W, fft_size, P, error_spec):
        self.W = W
        self.fft_size = fft_size
        self.P = P
        self._error_spec = error_spec

    def get_error_energy(self):
        # Mirror PBFDKF.get_error_energy exactly:
        #   float(np.sum(np.abs(error_spec) ** 2))   (error_spec complex64)
        return float(np.sum(np.abs(self._error_spec) ** 2))


class FakeShadow:
    def __init__(self, error_spec):
        self._error_spec = error_spec

    def get_error_energy(self):
        return float(np.sum(np.abs(self._error_spec) ** 2))


class FakeRegime:
    def __init__(self, main_paused):
        self.main_paused = main_paused


def write_complex(f, c):
    inter = np.empty(c.shape[0] * 2, dtype=np.float32)
    inter[0::2] = c.real.astype(np.float32)
    inter[1::2] = c.imag.astype(np.float32)
    inter.tofile(f)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/fsb_golden.bin'
    rng = np.random.RandomState(2026)

    # Evolving state across frames (a multi-frame sequence with state evolution
    # in W, P, and the error energies).
    W = (rng.randn(N_PARTITIONS, N_FREQS)
         + 1j * rng.randn(N_PARTITIONS, N_FREQS)).astype(np.complex64)
    P = (np.ones((N_PARTITIONS, N_FREQS), dtype=np.float32) * np.float32(0.01))

    with open(out, 'wb') as f:
        np.array([FFT_SIZE, N_FREQS, P_LEN, N_FRAMES], dtype=np.int32).tofile(f)

        for i in range(N_FRAMES):
            # --- evolve state ---
            W = (W + (rng.randn(N_PARTITIONS, N_FREQS)
                      + 1j * rng.randn(N_PARTITIONS, N_FREQS)).astype(np.complex64)
                 * np.complex64(0.1)).astype(np.complex64)
            P = (P * np.float32(0.97)
                 + (rng.rand(N_PARTITIONS, N_FREQS).astype(np.float32)
                    * np.float32(0.02))).astype(np.float32)

            # error spectra (complex64), varied so e_ratio sweeps <1 and >1.
            main_es = (rng.randn(N_FREQS) * 50.0
                       + 1j * rng.randn(N_FREQS) * 50.0).astype(np.complex64)
            shadow_scale = 0.5 + 1.5 * (i / max(1, N_FRAMES - 1))  # 0.5 .. 2.0
            shadow_es = (main_es * np.complex64(shadow_scale)).astype(np.complex64)

            has_W = 1 if i != 7 else 0          # frame 7 exercises W-absent path
            has_shadow = 1 if i != 3 else 0     # frame 3 exercises no-shadow path
            # frame 11: main_e tiny -> e_ratio stays 1.0 (1e-12 guard)
            if i == 11:
                main_es = (main_es * np.complex64(1e-9)).astype(np.complex64)
                shadow_es = (shadow_es * np.complex64(1e-9)).astype(np.complex64)

            main_paused = bool(i % 5 == 0)
            filter_converged = bool(i % 2 == 0)
            mu_final = float(0.001 + 0.003 * (i / N_FRAMES))
            external_delay = int(-1 if i == 0 else 120 + 7 * i)
            any_coarse = bool(i % 4 == 1)
            all_div = bool(i % 6 == 2)

            pbfdkf = FakePBFDKF(W if has_W else None,
                                FFT_SIZE if has_W else None,
                                P, main_es)
            shadow = FakeShadow(shadow_es) if has_shadow else None
            regime = FakeRegime(main_paused)

            # The divergence path reads P (independent of W) and the error
            # energies (independent of W); only filter_taps depends on W. So
            # main_e/shadow_e mirror exactly what the function computes
            # internally regardless of has_W.
            main_e = pbfdkf.get_error_energy()
            shadow_e = shadow.get_error_energy() if shadow is not None else 0.0

            br = build_filter_state_bridge(
                filter_converged=filter_converged,
                pbfdkf=pbfdkf,
                regime_handler=regime,
                mu_final=mu_final,
                external_delay_samples=external_delay,
                shadow_filter=shadow,
                any_coarse_filter_converged=any_coarse,
                all_filters_diverged=all_div,
            )

            # --- write inputs ---
            np.array([has_W, has_shadow], dtype=np.int32).tofile(f)
            write_complex(f, W[0])      # always write a full W0 (C ignores if !has_W)
            P.astype(np.float32).ravel().tofile(f)
            np.array([main_e], dtype=np.float64).tofile(f)
            np.array([shadow_e], dtype=np.float64).tofile(f)
            np.array([int(filter_converged), int(main_paused)], dtype=np.int32).tofile(f)
            np.array([mu_final], dtype=np.float64).tofile(f)
            np.array([external_delay], dtype=np.int32).tofile(f)
            np.array([int(any_coarse), int(all_div)], dtype=np.int32).tofile(f)

            # --- write expected outputs ---
            # div/mu: recomputed via the C f32 contract (see
            # _c_contract_div_mu's docstring) -- NOT br.divergence_indicator/
            # br.mu_final, which are build_filter_state_bridge's float64 SPEC
            # values and no longer what the float32 C struct fields hold.
            exp_div, exp_mu = _c_contract_div_mu(P, main_e, shadow_e, has_shadow, mu_final)
            taps = np.asarray(br.filter_taps, dtype=np.float32)
            np.array([taps.shape[0]], dtype=np.int32).tofile(f)
            taps.tofile(f)
            np.array([exp_div], dtype=np.float64).tofile(f)
            np.array([int(br.regime)], dtype=np.int32).tofile(f)
            np.array([int(br.filter_converged), int(br.main_paused)], dtype=np.int32).tofile(f)
            np.array([exp_mu], dtype=np.float64).tofile(f)
            np.array([int(br.external_delay_samples)], dtype=np.int32).tofile(f)
            np.array([int(br.any_coarse_filter_converged),
                      int(br.all_filters_diverged)], dtype=np.int32).tofile(f)

    print(f"wrote {out}  ({N_FRAMES} frames, fft_size={FFT_SIZE}, "
          f"n_freqs={N_FREQS}, p_len={P_LEN})")


if __name__ == '__main__':
    main()
