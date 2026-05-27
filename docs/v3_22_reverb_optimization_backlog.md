# v3.22 Reverb Optimisation Backlog

After the v3.21 AEC3-strict reverb chain alignment (commit chain ending in
the reverb render-history + decay-default port), the cohort DT case
showed:

- Painted-black HF frames: 24% → 15%
- rev/dir ratio median: 6.2 → 0.13
- Healthy frames: 60% → 75%

The remaining 15% painted-black + 12% non-healthy frames represent
behaviours that are AEC3-correct at our scale but still produce audible
HF damage on certain DT/movement cases. The directions below are
**beyond-AEC3 optimisations** that should be evaluated independently
in v3.22 — they MUST NOT be folded into v3.21.x alignment.

---

## Backlog items

### A. Cap reverb relative to direct R² (low risk, fast iteration)

```python
reverb[k] = min(reverb[k], reverb_max_ratio * direct[k])
```

Bound the reverb contribution so it can never exceed the linear-path
direct estimate by more than `reverb_max_ratio` × (e.g. 4×). Mirrors the
philosophical role of `MAX_ERLE_HF = 1.5` on the direct path — a sanity
cap that AEC3 does not have but our finer fft resolution arguably needs.

Risk: low (purely a hard cap, can be flag-gated).  
Expected win: cuts the rev/dir p95 outliers (currently 170) that drive
the worst painted-black frames.

### B. Per-band `_average_decay` scalar (LF / MF / HF separately)

AEC3 / current Python: single scalar across all bins,
`average_decay = sum(tail[1:]) / sum(direct[1:])`. Dominated by
LF energy. At our 257-bin spectrum, HF has 192 bins that all share an
LF-driven scalar.

Replace with three scalars:
```python
avg_decay_lf = sum(tail[lf_slice]) / sum(direct[lf_slice])
avg_decay_mf = ...
avg_decay_hf = ...
tail_response[lf_slice] = direct[lf_slice] * avg_decay_lf
...
```

Physical motivation: room reverb is frequency-dependent — air absorption
+ surface absorption attenuate HF faster than LF. A single scalar
mis-attributes LF tail/direct ratio to HF.

Risk: medium (changes per-bin tail_response shape).  
Expected win: HF reverb naturally smaller, less inflation in spectral
valleys.

### C. Frequency-dependent decay multiplier (HF decays faster)

```python
decay_per_bin = decay_base * exp(-k * hf_attenuation_rate)
```

Apply per-bin decay so HF bins forget faster than LF. Room-acoustic
common knowledge (RT60 typically 0.5× at HF vs LF for ordinary rooms).

Risk: requires care — too aggressive HF decay re-introduces echo leak.  
Status: NOT AEC3 strict; pure beyond-AEC3 tuning. Defer to v3.22.

### D. ReverbDecayEstimator full per-bin port (canonical AEC3 strict)

AEC3 `ReverbDecayEstimator` (reverb_decay_estimator.cc) estimates decay
from the filter impulse response per filter region. Currently:
- Our `ReverbDecayEstimator` exists and is wired (Phase C.4)
- v3.21 strict alignment switches `use_adaptive_decay = False` because
  AEC3 default `ep_strength.default_len = 0.83 > 0` disables the
  estimator in production AEC3.

To enable adaptive estimation aligned to AEC3 field-trial pathway,
need to:
1. Set `ep_strength.default_len < 0` (AEC3's "use estimator" sentinel)
2. Audit `ReverbDecayEstimator.decay()` returns per-bin (currently
   appears to return scalar — verify)
3. Wire the per-bin returns into `_update_reverb_linear` instead of the
   scalar `_reverb_decay()` path

Risk: high (changes core decay dynamics).  
Status: AEC3-aligned but non-default. Defer.

---

## Cross-cutting work

### E. ReverbFrequencyResponse fft-resolution-aware smoothing (reverted from v3.21)

Commit `8aafe61` (now reverted) expanded the neighbour-max window from
±1 bin to ±4 bins (= 125 Hz physical width, matching AEC3 fft=128
neighbour-width). The intent was: smooth out voice-harmonic-driven
spurious peaks at our finer 31.25 Hz/bin resolution.

The change INFLATED HF reverb instead of smoothing it (sparse HF energy
gets spread across the wider window), causing rev/dir = 1e6 spikes →
painted-black HF.

Revert kept the AEC3-strict ±1 bin literal. The "fft-resolution-aware
window" concept may still be worth pursuing in v3.22 via a different
mechanism (e.g. mean-window centered on the bin instead of max-window
+ self-exclusion). Document for future investigation.

---

## SuppressionGain tuning that diverges from AEC3 strict (revisit in v3.22)

### F. DominantNearendDetector.hold_duration_ms = 500 (AEC3 default: 200)

Our default is 500 ms; AEC3 `hold_duration` = 50 blocks × 4 ms = 200 ms.
Comment in `suppression_gain.py:DominantNearendConfig.hold_duration_ms`
notes the 500 ms was "empirically co-tuned on our 800-case cohort with
the v3.21 SuppressionGain mask shapes". 2.5× longer NE-state dwell
means the gain rule stays in nearend mode longer after the last NE
trigger — keeps mask thresholds gentle (preserves NE speech) at the
cost of slower fall-back to normal mode when echo returns.

Status: intentional v3.21 tuning, NOT a port bug. Re-evaluate at v3.22:
- Try AEC3-strict 200 ms + nearend_tuning thresholds tightened to
  compensate (matched-magnitude trade) on 800-case
- Or document the 500 ms as a permanent tuning divergence with the
  measured FS/DT bench delta vs 200 ms

---

## Evaluation methodology

Use `python/run_one_case.py --trace-hf-chain` for per-frame attribution
(commit 1209fa0). Compare per-bucket:
- `gain_hf_median` distribution
- `rev/dir` ratio (`r2_reverb_hf_median / r2_direct_hf_median`)
- Painted-black fraction (`gain_hf_median < 0.1`)

Cohort baseline (after v3.21 strict alignment) on
`wav/aec_challenge_blind/doubletalk` first case:
- 15.1% painted-black HF
- rev/dir median 0.13, p95 170, max 1.2e4

Any v3.22 candidate must improve on this baseline without regressing
healthy frame fraction (currently 75%) or FS_static cohort (separate
check).
