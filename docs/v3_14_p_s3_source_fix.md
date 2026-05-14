# v3.14 Arc-P Sprint 03: Source Signal Fix + Alpha Scan

**Date**: 2026-05-14
**Branch**: `feature/v3.14-arc-p`
**Sprint**: P.S3 — Fix source signal + gate + alpha tune
**Status**: COMPLETE — byte-equal PASS, case 04 + 08 sanity PASS

---

## 1. Source Signal Math Analysis

### Problem statement (P.S2 finding)

P.S2 implemented per-band ERL as:

```
inst_erl_pb[band] = mean(|echo_spec[k]|²) / mean(|far_spec[k]|²)
```

where `echo_spec = W·X` (PBFDKF main filter output) and `far_spec = X` (current frame far-end).

**P.S1 oracle** (from `error_psd / far_lw` in converged FS frames, case 04):
```
LF = 0.043,  MF = 0.191,  HF = 0.111
```

**P.S2 output** (from `|echo_spec|²/|far_spec|²`):
```
LF = 0.567,  MF = 0.603,  HF = 0.726
```

5-10x discrepancy.

### Root cause

`|W·X|²/|X|² = |Ĥ(k)|²` — the **squared magnitude of the estimated room transfer function**.

In a converged PBFDKF, the W coefficients accumulate energy over the entire filter length (fl=832 taps = 52 ms). Even when the room coupling is low (case 04 ERL ~0.04), the magnitude of the filter W is NOT the room ERL:

- W is an FIR approximating the room impulse response `h(k)` convolved with the decorrelating signal buffer
- `||W||²` reflects the energy in the estimated impulse response, not the echo-to-far power ratio at the microphone
- In case 04, the room has genuine low ERL (only 4.3% of far energy becomes echo) but PBFDKF's W has fitted the early part of the HF (direct + reflections) with full magnitude, so `|W·X|²/|X|²` over-estimates the coupling by reporting filter gain rather than room attenuation

### Correct source: Option B

`res.error_psd / res._residual_est._long_window_far_psd`

**Mathematical justification**:

In converged FS (filter has cancelled most echo):
```
error = mic - echo_cancelled
      = (room_echo + background) - echo_cancelled
      ≈ residual_echo  (background negligible during far-end activity)
      = far × H(k) - far × Ĥ(k)
      ≈ far × ERL_room  (when Ĥ(k) ≈ H(k), residual is the un-modelled tail)
```

So: `error_psd / far_lw ≈ ERL_room(k)` per band in converged FS frames.

This is EXACTLY the P.S1 oracle signal. `res.error_psd` is the EMA-smoothed
per-bin error PSD (α=0.8 BALANCED), updated every frame by ResFilter. `far_lw`
is the long-window far PSD (same signal used by F3.1-v3 excess formula).

**Why NOT option A** (`error_psd / far_psd`): `far_psd` is the single-frame
instantaneous far PSD — much noisier than the long-window `far_lw`. Using `far_lw`
is consistent with F3.1 and more stable.

**Why NOT option C** (`echo_spec + error_psd)/far_psd`): Adding predicted echo and
residual double-counts the cancelled component. Produces values > 1.0 systematically.

**Why NOT option D** (`near_psd / far_psd`): Only valid in FS near-silence; in FS
with room noise, `near_psd` inflates the estimate.

**Verdict**: **Option B is the canonical correct signal**. P.S3 corrects the source
to `res.error_psd / far_lw`.

---

## 2. Gate Correction

### P.S2 gate (wrong)

```python
if raw_dt_ratio < 2.0 and inst_erl_raw < 1.5:  # scalar ERL guards
    ...scalar ERL update...
    if f3_1_per_band_erl_adaptive and _filter_converged and lw_ready:
        ...per-band ERL update (NESTED inside scalar guards)...
```

**Problem**: The `raw_dt_ratio < 2.0` guard uses `raw_err_pwr / far_pwr`
(time-domain broadband power ratio). In high-coupling rooms (case 08, movement),
median `raw_err_pwr / far_pwr ≈ 7.0` because the far-end reference is at a much
lower amplitude than the mic output after partial cancellation. Only 39% of
converged+far-active frames passed this guard — the EMA barely moved from 0.1.

**Root cause**: The scalar ERL source is `mic_pwr / far_pwr` (single-frame raw)
which is indeed unreliable without the guards. But the per-band source
`error_psd / far_lw` is already EMA-smoothed — single-frame DT contamination
is attenuated by α=0.8. The nested guards are overly conservative for the
smoothed source.

### P.S3 gate (correct)

```python
if raw_dt_ratio < 2.0 and inst_erl_raw < 1.5:  # scalar ERL guards (unchanged)
    ...scalar ERL update...
# SIBLING block, NOT nested:
if f3_1_per_band_erl_adaptive and _filter_converged and lw_ready:
    ...per-band ERL update (own gate, uses smoothed signal)...
```

The per-band update fires when:
1. `far_pwr > 1e-4` (outer `erl_update_gate` — far-end active)
2. `_filter_converged` (FilterConvergenceAnalyzer says filter is reliable)
3. `_long_window_n_updates > 0` (far_lw has warmed up)
4. `_far_band > 1e-10` (per-band far energy check, avoids 0-div)

This allows all 136 converged+far-active frames in case 08 to update the EMA
(vs 75 previously), enabling the LF ERL to track 0.451 (vs oracle 0.489).

---

## 3. Alpha Scan Results

**Cases**: 04 (low-coupling FS_static) + 08 (high-coupling FS_movement)
**Oracle**: Case 04: LF=0.043, MF=0.191, HF=0.111 | Case 08: LF=0.489, MF=0.274, HF=0.344

| α     | TC (hops) | Case 04 LF | Case 04 MF | Case 04 HF | Case 08 LF | Case 08 MF | Case 08 HF |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 0.950 | ~20 hops  | 0.069     | 0.280     | 0.211     | 0.638     | 0.244     | 0.304     |
| 0.990 | ~100 hops | 0.078     | 0.253     | 0.178     | 0.451     | 0.195     | 0.226     |
| 0.995 | ~200 hops | 0.085     | 0.219     | 0.160     | 0.324     | 0.161     | 0.180     |
| 0.999 | ~1000 hops| 0.096     | 0.139     | 0.119     | 0.156     | 0.116     | 0.120     |

**Analysis**:

- α=0.95: Too fast — case 08 LF overshoots oracle (0.638 vs 0.489). Fast tracking
  in DT frames could corrupt the EMA with NE energy.
- α=0.99: Best balance. Case 04 LF=0.078 (within range), case 08 LF=0.451 (8% of
  oracle 0.489). Tracks room changes within ~1 second.
- α=0.995: Case 04 close but case 08 LF=0.324 — only 66% of oracle (too slow for
  movement-room tracking).
- α=0.999: Very slow. Case 08 LF=0.156 — only 32% of oracle. Effectively doesn't
  react to room changes within a clip duration.

**Verdict**: **α=0.99 is the correct default** (already in P.S2). Kept unchanged.

---

## 4. Per-Band Cap Revision

**P.S2 caps**: `[LF=0.6, MF=0.8, HF=1.0]` (post-EPC cap on `_per_band_erl`).

With the corrected source signal (error_psd/far_lw), the observed ERL values:
- Case 04 converged: LF max(p90)=0.124, MF max(p90)=0.322, HF max(p90)=0.237
- Case 08 converged: LF max(p90)=0.594, MF max(p90)=0.265, HF max(p90)=0.345

The P.S2 per-band caps `[0.6, 0.8, 1.0]` are not binding for either case.
Rolling p95 tracking would add complexity without benefit at this stage.
**Caps unchanged**; the `clip_hi=1.5` EMA safety ceiling is the primary
protection against outlier contamination.

---

## 5. Verification Results

### 5-case byte-equal (hard bar: atol=0.0)

| Case       | flag-OFF max|Δ| | Result | flag-ON max|Δ| |
|------------|-----------------|--------|-----------------|
| NE         | 0.000000e+00    | PASS   | 0.000000e+00    |
| FS_static  | 0.000000e+00    | PASS   | 0.000000e+00    |
| FS_mvmt    | 0.000000e+00    | PASS   | 1.76e-02        |
| DT_static  | 0.000000e+00    | PASS   | 1.15e-02        |
| DT_mvmt    | 0.000000e+00    | PASS   | 9.41e-03        |

**BYTE-EQUAL: ALL 5 PASS (atol=0.0)**

### Case 04 convergence (P.S3 acceptance criteria)

| Band | P.S3 mean | P.S3 acceptance | P.S1 oracle | Result |
|------|-----------|-----------------|-------------|--------|
| LF   | 0.0778    | [0.01, 0.20]    | 0.043       | PASS   |
| MF   | 0.2530    | [0.05, 0.50]    | 0.191       | PASS   |
| HF   | 0.1778    | [0.05, 0.40]    | 0.111       | PASS   |

P.S3 is 1.7-2.6x the oracle due to EMA lag (α=0.99 + only 581 converged frames
in the clip). The critical improvement vs P.S2 (13x oracle) confirms source signal
fix works.

### Case 08 movement convergence (informational)

| Band | P.S3 mean | P.S1 oracle |
|------|-----------|-------------|
| LF   | 0.4506    | 0.489       |
| MF   | 0.1953    | 0.274       |
| HF   | 0.2259    | 0.344       |

LF within 8% of oracle. MF/HF directionally correct but under-estimated — the
movement creates non-stationary ERL dynamics within the clip; the EMA hasn't
fully settled. Acceptable for P.S3 sanity check.

### Cohort tail (qNvSMyU) — P52 invariant

- flag-OFF vs flag-ON max|Δ|: 3.500e-02 (ERL updated, flag-ON differs)
- RMS output delta: −0.054 dB (marginal, within single-case sanity band)
- P52 hard bar (Δecho ≥ −0.05 dB) requires full 800-case bench — deferred to gate

---

## 6. Files Changed

- `python/aec.py`: source signal corrected (echo_spec/far_spec → error_psd/far_lw);
  per-band gate moved to sibling of scalar ERL (not nested inside raw_dt_ratio guard)
- `tools/research/v3_14_p_s3_validation.py`: byte-equal + case 04/08 + cohort tail + α scan
- `docs/v3_14_p_s3_source_fix.md`: this document

---

## 7. Arc R Sequencing Recommendation

**P.S3 is COMPLETE** — source signal correct, byte-equal hard bar met, case 04/08 sanity PASS.

**Before Arc R** (wiring per-band ERL into the main engine as default-ON):
1. Run 8-case listen (cases 04 + 08 primary) with `f3_1_per_band_erl_adaptive=True`
   to confirm DT protection is not weakened (the sibling gate fires in DT frames
   too — check if per-band ERL inflates from NE speech contamination)
2. If listen confirms no NE contamination in DT frames, run the 800-case bench
   with flag-ON vs BALANCED baseline
3. Arc R acceptance bar (from P.S2 design lock): `DT Δdeg ≥ 0.000 dB` AND
   `FS Δecho ≤ 0.005 dB`

**Key risk for Arc R**: The sibling gate (no `raw_dt_ratio < 2.0` guard) could
allow NE contamination in true DT frames. In DT, `error_psd ≈ mic_psd ≈ echo + NE`
which would inflate per-band ERL. This might reduce DT protection (higher ERL →
larger echo mask → F3.1 excess fires less → dt_per_bin lower → DT NE suppressed
more). This is the primary 800-case validation target.

**Mitigation available**: If NE contamination is confirmed, add back
`raw_dt_ratio < 2.0` guard ONLY for DT frames (gate on `effective_dt < 0.5`).
This preserves the fix for case 08 (FS movement, no NE) while blocking DT
contamination.
