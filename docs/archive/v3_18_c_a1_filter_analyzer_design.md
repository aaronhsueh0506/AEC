# v3.18 Phase C.A.1 — FilterAnalyzer sub-design (2026-05-16)

**Status**: sub-design under [docs/v3_18_c1_aec_state_design.md](v3_18_c1_aec_state_design.md) §2 C.A.
Audit-only port; behavioural no-op when flag off.

## 1. Public API

New file `python/aec_filter_analyzer.py`:

```python
class FilterAnalyzer:
    """AEC3-aligned filter-shape consistency + peak/gain analyzer.

    Operates on the main PBFDKF filter's time-domain impulse response
    (computed once per frame via IFFT of W summed across partitions).
    Outputs:
      - consistent_estimate (bool): peak position has been stable
      - max_echo_path_gain (float): largest |h_hp[k]| across the response
      - peak_index (int): sample index of the main peak (post-HP)
      - peak_index_history (deque): last N peak indices for diagnostics
    """

    def __init__(self, sample_rate: int, hp_cutoff_hz: float = 600.0,
                 peak_window: int = 5, consistent_threshold: int = 30):
        self.sample_rate = sample_rate
        # 1st-order HP IIR coefficient: alpha = exp(-2π·fc/fs)
        self._hp_alpha = float(np.exp(-2.0 * np.pi * hp_cutoff_hz / sample_rate))
        self._peak_window = peak_window      # ± tolerance in samples
        self._consistent_threshold = consistent_threshold   # frames
        self._peak_history = deque(maxlen=consistent_threshold)
        self._consistent_count = 0
        # Outputs (audit-only, no consumer writes these):
        self.consistent_estimate = False
        self.max_echo_path_gain = 0.0
        self.peak_index = -1

    def update(self, w_time: np.ndarray) -> None:
        """Update analyzer from time-domain filter impulse response.

        `w_time`: 1-D array, full impulse response (sum across partitions).
        """
        # HP filter (1st-order IIR)
        h_hp = self._apply_hp(w_time)
        # Peak detection
        abs_h = np.abs(h_hp)
        self.peak_index = int(np.argmax(abs_h))
        self.max_echo_path_gain = float(abs_h[self.peak_index])
        # Consistency check
        self._peak_history.append(self.peak_index)
        if len(self._peak_history) >= self._consistent_threshold:
            recent = list(self._peak_history)
            ref = recent[0]
            stable = all(abs(p - ref) <= self._peak_window for p in recent)
            self.consistent_estimate = stable

    def _apply_hp(self, x: np.ndarray) -> np.ndarray:
        # y[n] = α·(y[n-1] + x[n] - x[n-1])
        y = np.zeros_like(x)
        alpha = self._hp_alpha
        prev_x = 0.0; prev_y = 0.0
        for n in range(len(x)):
            y[n] = alpha * (prev_y + x[n] - prev_x)
            prev_x = x[n]; prev_y = y[n]
        return y

    def reset(self) -> None:
        self._peak_history.clear()
        self._consistent_count = 0
        self.consistent_estimate = False
        self.max_echo_path_gain = 0.0
        self.peak_index = -1
```

## 2. Wiring point in `AEC`

Per-frame update after main filter processes. Audit-only flag:

```python
# AecConfig
filter_analyzer_enabled: bool = False    # C.A audit-only port

# AEC.__init__ (lazy-init when flag is True)
if self.config.filter_analyzer_enabled:
    self._filter_analyzer = FilterAnalyzer(sample_rate=self.config.sample_rate)
else:
    self._filter_analyzer = None

# AEC.process(), right after self._update_misadjustment_estimator() block
# (after main filter has processed; before RES)
if self._filter_analyzer is not None:
    # Sum W across partitions, IFFT to time domain
    W_sum = self.filter.W.sum(axis=0)
    w_time = np.fft.irfft(W_sum, self.filter.fft_size).astype(np.float32)
    self._filter_analyzer.update(w_time)
    self._diag['filter_analyzer_consistent'] = bool(
        self._filter_analyzer.consistent_estimate)
    self._diag['filter_analyzer_peak_index'] = int(
        self._filter_analyzer.peak_index)
    self._diag['filter_analyzer_max_gain'] = float(
        self._filter_analyzer.max_echo_path_gain)
```

## 3. v3.14 Arc P relationship — DEFER subsume to C.A.4+

Per C.1 §3.1 (read-only mirror first), C.A does NOT yet touch v3.14
Arc P (`_per_band_erl`, `f3_1_per_band_erl_adaptive`). Arc P remains
the production per-band ERL EMA in BALANCED. C.A.4+ revisits whether
to fold Arc P into `FilterAnalyzer` once C.A pre-bench gate passes.

This isolates the C.A pre-bench gate decision: we test whether the
new `consistent_estimate` signal is informative *independent of* Arc P
absorption.

## 4. Pre-bench gate (C.A.3)

Per C.1 §6:
> `consistent_estimate` distribution DIFFERS from `_filter_converged`
> on 5-case trace by ≥ 10% in either direction (else port adds no signal)

5-case trace stems:
- `0I0XMl3M_farend_singletalk_with_movement` (movement)
- `qNvSMyU_farend_singletalk` (cohort tail, never converges)
- `0KjzXA3g20qsd8zmSekADw_farend_singletalk` (clean FS)
- `0I0XMl3M_doubletalk` (DT)
- `sUQrHEPA_doubletalk` (DT, 0% refined_usable)

Per case, compute:
- `consistent_estimate` coverage % across frames
- `_filter_converged` coverage % across frames
- Difference per case; cumulative across 5 cases

PASS: max |Δcoverage| ≥ 10% on at least 2 of 5 cases (informative).
FAIL: all 5 cases within ±10% (signals overlap; analyzer redundant).

## 5. Risk + mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | HP filter implementation in pure Python is slow | per-frame cost ~O(fft_size) = O(2048). 0.1 ms / frame; negligible vs PBFDKF cost. If a profile shows >1% overhead, vectorise `_apply_hp` with `scipy.signal.lfilter` |
| R2 | Peak-position drift on warmup might mislabel as inconsistent | `_consistent_threshold=30` (300 ms) gives the filter time to settle before we declare anything consistent |
| R3 | Multi-partition W summing loses phase info | `irfft(W_sum)` is the right operation — corresponds to total filter impulse response. AEC3 does the same conceptually (impulse response across all blocks) |
| R4 | `consistent_estimate` may simply track `_filter_converged` 1-to-1 (port adds no signal) | C.A.3 pre-bench gate exists exactly to catch this. If R4 triggers, close C.A; substrate retained for v3.19+ retry |

## 6. C.A sprint sequence

| Sprint | Action | Output |
|---|---|---|
| C.A.1 | This design doc | doc only |
| C.A.2 | `aec_filter_analyzer.py` + AecConfig flag + wiring + lazy-init + diag fields + 5-case byte-equal flag-OFF | byte-equal flag-OFF md5 PASS |
| C.A.3 | Run 5-case trace flag-ON, compute coverage delta vs `_filter_converged` | pre-bench gate decision (PASS/FAIL) |
| C.A.4 | (only if C.A.3 PASS) 60-case trace; document distributional finding; decide whether to proceed to C.B | trace doc + decision |
| C.A.5 | (only if C.A.4 PASS) Close C.A, hand off to C.B | C.A verdict doc |

## 7. Cross-references

- [docs/v3_18_c1_aec_state_design.md §2.C.A](v3_18_c1_aec_state_design.md) — parent design + ordering
- [docs/v3_18_c1_aec_state_design.md §6](v3_18_c1_aec_state_design.md) — pre-bench gate definition
- [docs/aec3_extracts/src/aec3/filter_analyzer.h](aec3_extracts/src/aec3/filter_analyzer.h) — AEC3 reference API
