# AEC Changelog

Versioning: major.minor.patch. Major changes architecture; minor adds features
or tuning sweeps; patch is bugfix.

---

## v3.4.0 (2026-04-29) — DT_static echo gap closure via render_ceil skip

**Goal**: Continue closing DT_static echo gap vs AEC2 (was −0.285 in v3.2, −0.241
in v3.3). Trace finding: render_ceil cap (line 1554, `residual ≤ far_now × ERL × 2`)
was neutralizing v3.3.0's error-based fallback at hotspots — capping residual
back down to far_now × ERL even though fallback returned much larger values.

**Three coordinated axes**:
1. **Skip render_ceil in render-mode** — the cap exists to bound a wrong linear
   filter, but render-mode IS the fallback for when filter is wrong; cap was
   self-defeating. Tested gating on `not_once_converged` (rc11): didn't help
   DT_movement deg, lost FS echo.
2. **NE-protect reverb hard-cut** — `reverb_gate=0` when `far_activity < 0.1`.
   far_activity EMA decays slowly; without hard-cut, reverb tail lingers
   ~200ms after far ends, hurting NE-only frames.
3. **DT-aware fallback factor** — `fallback_factor = 0.7 - 0.2 × clip(dt_for_fs)`.
   Strong DT (NE present) → less of error_psd is echo → smaller fallback floor
   (down to 0.5×). Protects DT_movement deg.

**800-case AECMOS vs AEC2** (CNG=True, balanced preset):

| bucket | v3.3.0 → v3.4.0 |
|---|---|
| FS_static echo | +0.031 → +0.065 (won AEC2) |
| FS_movement echo | +0.328 → +0.353 |
| DT_static echo | −0.241 → −0.233 |
| DT echo | −0.163 → −0.155 |
| DT_movement deg | −0.189 → −0.213 |
| NE deg | −0.089 → −0.091 (4.008, > 4.0) |

Win rate: Echo 47% (DT 100/300), Deg 42%.

**Iteration history attempted but reverted (v3.5 candidates)**:
- rc12 (Axis 4: fast reverb decay when far silent): hurt leak +30% on echo
  hotspots. Reverb-tail-fallback contradicts fast-collapse-for-NE; cannot do both
  with one reverb signal.
- rc13/rc14 (Axis 5: tail-augmented `far_for_conf` in fallback): smoke −2~−5%
  leak energy, AECMOS Δ <0.01. Diminishing returns within current framework.
- rc12/rc13 (Axis 6: per-bin NE-dominant gain floor): falsely triggered on
  echo leak hotspots where attribute under-estimates true echo.
  `error_psd >> expected_echo` is true for both NE and under-attributed echo
  bins; cannot discriminate without reliable echo estimate.

**Conclusion**: v3.4.0 is at the limit of what the current attribute+reverb
framework can deliver. Further gains require a per-bin echo-vs-NE discriminator
(long-history coherence, NN postfilter, or sub-band coherence — see Route B
in `aec_v3_evolution.md`).

---

## v3.3.0 (2026-04-29) — Reverb tuning + error-based render fallback

Trace-driven (diag_leak_hotspot.py on 5 worst cases): echo at frame t in losing
cases originates 60–320ms ago — render-based estimate (far_now × ERL) captures
only current X (often quiet between speech bursts) and underestimates true echo
by 2 orders of magnitude (e.g., 4.2 vs error_psd 239).

**Two existing-but-misconfigured mechanisms fixed together**:
1. Existing IIR reverb (line ~1568, WebRTC-AEC3-style):
   - Was: `decay=0.65 (TC ~50ms), gain=1.4`, DT-gate kills reverb 70% during DT
   - Now: `decay=0.85 (TC ~130ms, matches typical RT60), gain=1.6`,
     DT-gate base `0.3 → 0.7` (DT is exactly when reverb matters most)
2. attribute_legacy render-based path:
   - Was: `render_based_echo = far × ERL` (single-frame, undersized when filter
     never converges)
   - Now: `max(far × ERL, error_psd × far_conf × 0.7)` where far_conf is
     per-bin "active echo path" indicator

**800-case vs v3.2.0**: FS +0.045 echo / 0 deg, DT +0.030 echo / −0.032 deg,
NE 0 echo / −0.006 deg.

---

## v3.2.0 (2026-04-29) — Axis 1+2+3: ERL outlier protection + render-mode gain ceiling

Trace-driven multi-axis fix targeting DT/FS echo gap vs AEC2.

**Discriminator found**: losing cases have ERL clamped at high values (mean
0.5–0.95) due to NE-corruption, leading to bloated render_ceil → cap at
error_psd → ENR collapse → gain p90=1.0 (10% of frames no suppression).
Winners have ERL < 0.2 (filter learned echo cleanly).

**Axes**:
- **Axis 1 — NE-corruption ERL protection**: Skip ERL update when
  `inst_erl ≥ 1.5` (physically implausible: mic > far means NE dominates).
  Tighten clip from 10.0 → 1.0.
- **Axis 2 — Render-mode gain ceiling**: Hard ceiling 0.6 on smoothed gain
  when `using_render AND far_active > 0.3`, preventing transient leaks where
  gain hits 1.0.
- **Axis 3 — Relax error_psd cap in render-mode**: Allow `residual ≤ error_psd
  × 1.5` instead of `× 1.0` so render-based estimate isn't bounded by
  NE-inflated error signal. Currently no-op since Axis 1 already keeps render
  estimate small, but defensive against future ERL drift.

**800-case vs AEC2 (CNG=True)**: FS echo +0.115 (was +0.085 in v3.1.0),
DT_static echo −0.285 (was −0.302), DT deg +0.058, NE deg −0.082.

---

## v3.1.0 (2026-04-28) — Trace-driven render-mode RES fixes

Targets DT echo gap. Single deep trace on worst static-DT case revealed two
bottlenecks:
1. error_psd cap at line 1545 killed 71% of render-based residual estimate.
2. min_ne_from_dt floor inflated nearend_est, depressing ENR.

**Fixes** (render-mode aware):
- `echo_psd × 2.0` cap skipped when `using_render_based`.
- `dt_suppress` cap skipped when `using_render_based`.
- `min_ne_from_dt × 0.5` factor (configurable via `render_min_ne_factor`).

**800-case vs AEC2 (CNG=True)**: FS echo +0.085, DT echo −0.211 (closed 13%
of gap from v3.0.2 baseline).

---

## v3.0.0 → v3.0.2 (2026-04-28) — Phase A+B+C decoupling refactor

Architectural decoupling into 5 detector classes + AecState aggregator,
preparing the ground for v3.1+ targeted fixes. **800/800 bit-exact vs v2.8.1**.

- **Phase A** (v3.0.0): Extract from `AEC.process()` into independent classes:
  ShadowCopyController, RenderActivityDetector, FilterConvergenceAnalyzer,
  DoubleTalkAnalyzer, EchoPathChangeDetector, AecState aggregator.
  Properties on AEC delegate to detector state for backward compat.
- **Phase B** (v3.0.1): ResidualEchoEstimator extracted (legacy mode).
- **Phase C** (v3.0.2): Ablation hooks (gate_mode, no_reset_sources,
  maybe_mark_diverged), default = bit-exact v2.8.1.

11 ablation variants (B2 R1 split, C1 S1/S2/S3 coherence/streak, C2 E3
no-reset) all failed to improve movement-DT echo. AEC3-borrowed detector
patterns don't translate to PBFDKF main + PBFDKF shadow architecture.

---

## v2.8.1 (2026-04-28) — Movement DT ablation complete + cleanup

Final v2.x release; baseline for v3 refactor. Movement-DT remains in Pareto
disadvantage vs AEC2 (echo −0.040, deg −0.073) — boundary of detector tuning.

---

## v2.5.0 (2026-04) — Final v2.5 line

Multi-phase shadow filter improvements, ERL tracking refinements,
DT-detector coherence integration. See `archive/CHANGELOG_v2.5.0.md` and
`archive/CHANGELOG_v2.5.md` for detailed phase breakdown.

---

## v2.4.0 (2026-03) — Phase fixes + GetStats API

Fix1+Fix2 (pre-conv render-based force + DT-indicator ERL blend), AecFilterState,
AecStats, GetStats(), AecDebugLogger added.

---

## v2.3.0 (2026-03) — Initial PBFDKF release with shadow filter

PBFDKF main + PBFDKF shadow. ResFilter v2 (direct + ENR + reverb).
DTD coherence integration. Initial AECMOS competitive results vs AEC3 baseline.

---

## Earlier (v1.x and pre-v2.3) — see archive/

`archive/` contains: phase reports (κ, η, b15, b16, phase2_stage_1b/1d/2),
specs (raw_dt_delay_alignment, dt_jump_veto, shadow_nlms),
DEVLOG.md (chronological work log), early CHANGELOGs.
