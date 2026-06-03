"""Generate a binary golden for the C reverb_decay_estimator port (WS5 5.1).

Drives the Python ReverbDecayEstimator over deterministic float32 input
sequences for BOTH operating paths and records per-update state so the C parity
test can replay and assert bit-exact:

  * Legacy partition path (use_aec3_block_energy=False) — float32
    partition_energies, varying delay / quality / usable / stationary flags.
  * AEC3-strict TD path (use_aec3_block_energy=True) — float32 time-domain
    impulse response (multiple of 64 samples), driven long enough to cross the
    _analyze_filter → _estimate_decay handoff and actually MOVE the decay value
    (exercises BlockEnergyAverage/Peak, Early + Late regressors, the slope→decay
    EMA, and all the gating branches).

Per update we record the post-update observable state (decay, smoothing,
block_to_analyze, late_start, late_end, candidate_size, region_identified,
tail_gain) plus the late-regressor (nz/nn/n/N) — enough to catch any divergence
in the accumulate / regression math, not just the final scalar.

Layout (LE):
  int32 magic = 0x52444543
  --- legacy section ---
  int32 n_partitions, int32 hop_size, int32 n_legacy
  float64 default_decay, float64 mild_decay
  n_legacy × [ float32 pe[n_partitions]
               float64 filter_quality | int32 fq_is_none
               int32 filter_delay_blocks | int32 usable | int32 stationary
               --- expected post-update state --- (STATE_DOUBLES f64 + STATE_INTS i32)
             ]
  --- aec3-strict section ---
  int32 td_size, int32 n_aec3
  float64 default_decay2, float64 mild_decay2, int32 n_partitions2, int32 hop2
  float32 td_filter[td_size]   (static across updates; recorded once)
  n_aec3 × [ float64 filter_quality | int32 fq_is_none
             int32 filter_delay_blocks | int32 usable | int32 stationary
             --- expected post-update state ---
           ]

State block per update (order fixed):
  f64: decay, smoothing_constant, late.nz, late.nn, tail_gain
  i32: block_to_analyze, late_start, late_end, candidate_size,
       region_identified, late.n, late.N

Run: python3 python/diag/gen_reverb_decay_estimator_golden.py /tmp/rde_golden.bin
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.residual.reverb_decay_estimator import ReverbDecayEstimator  # noqa: E402

MAGIC = 0x52444543


def _pack_state(f, est):
    # f64 block
    np.array([est._decay, est._smoothing_constant,
              est._late_reverb_decay_estimator._nz,
              est._late_reverb_decay_estimator._nn,
              est._tail_gain], dtype='<f8').tofile(f)
    # i32 block
    np.array([est._block_to_analyze, est._late_reverb_start,
              est._late_reverb_end, est._estimation_region_candidate_size,
              int(est._estimation_region_identified),
              est._late_reverb_decay_estimator._n,
              est._late_reverb_decay_estimator._N], dtype='<i4').tofile(f)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rde_golden.bin'
    with open(out, 'wb') as f:
        np.array([MAGIC], dtype='<i4').tofile(f)

        # ───────────────────────────── legacy ─────────────────────────────
        n_part = 13
        hop = 160
        default_decay = 0.85
        mild_decay = 0.5
        rng = np.random.RandomState(11)
        n_legacy = 16
        np.array([n_part, hop, n_legacy], dtype='<i4').tofile(f)
        np.array([default_decay, mild_decay], dtype='<f8').tofile(f)

        est = ReverbDecayEstimator(n_partitions=n_part, hop_size=hop,
                                   default_decay=default_decay,
                                   mild_decay=mild_decay,
                                   use_adaptive=True,
                                   use_aec3_block_energy=False)
        for i in range(n_legacy):
            base = np.exp(-np.arange(n_part) * 0.3).astype(np.float32) * 100.0
            pe = (base * (1.0 + 0.1 * rng.rand(n_part)).astype(np.float32)
                  ).astype(np.float32)
            # exercise: None quality at i==2, not-usable at i==5,
            # stationary at i==9, delay variation
            fq_is_none = (i == 2)
            fq = None if fq_is_none else float(0.5 + 0.4 * rng.rand())
            delay = (i % 4)                      # 0,1,2,3 — includes 0
            usable = (i != 5)
            stationary = (i == 9)

            pe.tofile(f)
            np.array([fq if fq is not None else 0.0], dtype='<f8').tofile(f)
            np.array([1 if fq_is_none else 0], dtype='<i4').tofile(f)
            np.array([delay, 1 if usable else 0, 1 if stationary else 0],
                     dtype='<i4').tofile(f)

            est.update(partition_energies=pe,
                       time_domain_filter=None,
                       filter_quality=fq,
                       filter_delay_blocks=delay,
                       usable_linear_filter=usable,
                       stationary_signal=stationary)
            _pack_state(f, est)

        # ─────────────────────────── aec3-strict ──────────────────────────
        n_blocks = 13
        td_size = n_blocks * 64           # 832
        hop2 = 160
        default_decay2 = 0.85
        mild_decay2 = 0.5
        n_aec3 = 60

        t = np.arange(td_size)
        h = np.exp(-(t - 130.0) ** 2 / (2 * 40.0 ** 2)) * 0.5
        h = h + np.exp(-(t) / 300.0) * 0.01 * np.sin(t * 0.3)
        h = h.astype(np.float32)

        np.array([td_size, n_aec3], dtype='<i4').tofile(f)
        np.array([default_decay2, mild_decay2], dtype='<f8').tofile(f)
        np.array([n_blocks, hop2], dtype='<i4').tofile(f)
        h.tofile(f)

        est = ReverbDecayEstimator(n_partitions=n_blocks, hop_size=hop2,
                                   default_decay=default_decay2,
                                   mild_decay=mild_decay2,
                                   use_adaptive=True,
                                   use_aec3_block_energy=True)
        rng2 = np.random.RandomState(23)
        for i in range(n_aec3):
            fq_is_none = (i == 4)
            fq = None if fq_is_none else float(0.7 + 0.2 * rng2.rand())
            delay = 2
            usable = (i != 50)            # exercise a not-usable → reset mid-run
            stationary = (i == 45)        # exercise stationary skip

            np.array([fq if fq is not None else 0.0], dtype='<f8').tofile(f)
            np.array([1 if fq_is_none else 0], dtype='<i4').tofile(f)
            np.array([delay, 1 if usable else 0, 1 if stationary else 0],
                     dtype='<i4').tofile(f)

            est.update(partition_energies=None,
                       time_domain_filter=h,
                       filter_quality=fq,
                       filter_delay_blocks=delay,
                       usable_linear_filter=usable,
                       stationary_signal=stationary)
            _pack_state(f, est)

    print(f"wrote {out}  (legacy {n_legacy} + aec3 {n_aec3} updates, "
          f"final aec3 decay={est._decay!r})")


if __name__ == '__main__':
    main()
