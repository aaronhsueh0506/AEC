# AEC Research Arcs Summary — P25 to P58 (2024–2026)

**Status as of 2026-05-11:** All exploratory arcs closed. Production preset
remains `balanced` (default since pre-P50). No new code or flag from this
research stream shipped. This document consolidates key insights, mechanisms
ruled out, bugs found, and remaining future-work options.

---

## 1. Production state (unchanged through P25–P58)

- **Front-end filter:** PBFDKF main + Q-scaled shadow (`shadow_q_ratio=3.0`,
  `enable_shadow=True`), bidirectional `ShadowCopyController` copy paths,
  `filter_length=832` samples (52 ms @ 16 kHz).
- **Back-end RES:** 9-stage P34b residual chain (ERLE-direct + 4 caps +
  nonlinear boost + harmonic floor + reverb-additive + temporal smoothing +
  spectral floor + DT-suppress + render-ceil).
- **Default preset:** `balanced` (`res_g_min_db=-55`, `res_reverb_decay=0.85`,
  `enable_cng=True`).
- **`balanced` superseded `nearend_protect_dt` candidate per P50 Phase 2.**

---

## 2. Arc-by-arc verdicts

### P25–P29 — early audits (consumer-ladder, render trust, NE attribution)

- P25/P26: Stage gain regression audits. CLOSED — local mechanism, no fix
  produced production change.
- P27: 4-band partitioned subband coarse tracker. Trace-only; never wired to
  consumer. Default-OFF.
- P28: `force_render` trust modifier with k-of-N temporal latch (k=3, N=5).
  Phase 1a dry-run; closed without consumer wiring.
- P29: Case-pick + window analysis utilities. Diagnostic only.

### P30–P34b — residual chain attribution

- P30: `DominantNearendDetector` per-band advisory (near > −35 dB AND
  error−echo ≥ 6 dB). L1 ladder; never consumed.
- P31: Idle RES bypass attempt (Pattern A collapse). CLOSED — fix was
  upstream in stage gain capture, not the bypass itself.
- P32: Render carry-over (`t_carry_s=2.0`). Trace-only.
- P33: Pattern B findings. Confirmed back-end is responsive to upstream
  state; reclassification alone cannot fix degenerate inputs.
- P34 / P34b: Stage-0/1 residual chain attribution. Established that the
  9-stage chain's caps (echo×2 / error×1.5 / DT-suppress / render-ceil)
  carry FS-suppression load; removing them costs ≥ 0.7 dB AECMOS-echo on
  cohort (later re-confirmed by P58).

### P36–P40 — WebRTC parity / minimum-statistics noise

- P36: Phase 1a counterfactual; failure diagnostic recorded.
- P37: Echo gain-change diagnostic. No mechanism delta.
- P38: WebRTC suppressor parity; dry-run verdict — parity not achievable
  without front-end change.
- P39: Voice-band noise-PSD readout (B1: 1–2 kHz, B2: 2–4 kHz). Trace-only.
- P40 I1: Per-bin minimum-statistics tracker (D=16 frames, U=10 subwindows,
  B_min=+1.76 dB Martin K~10). Trace-only.
- P40 I2/I3/I3': Render-projected echo audibility + structural mismatch
  features. Trace-only; verdict — features insufficient for typed-path
  reclassification.

### P41–P44 — Stage-5 smear / phenomenology

- P41: Spec-synth + noise-lift + limiter-gain trace.
- P42: Stage-5 nearend-only spillover audit. Mechanism confirmed; no fix.
- P43: Stage-5 3-bin smear guard. CLOSED FAIL — guard insufficient.
- P44a: Smear phenomenology #2; major re-attribution.
- P44b: Oracle bypass + PSD phenomenology; major re-attribution.
- P44c: Stage-5 clean closure; pure attribution audit, no code change.

### P46 — per-bin attribution + routing

- P46.1: Per-bin attribution driver.
- P46.1b: Subtype decomposition; render-ceil track CLOSED.
- P46.1c: Oracle upstream upper bound; upstream PSD CLOSED as single-axis
  fixable.
- P46.1d: Per-bin attribution on representative case (P10GsQvh).
- P46.2: H3 control listener verdict.
- **Verdict:** Routing is a 2-axis problem (upstream PSD + downstream caps);
  neither axis alone yields a clean fix.

### P47 — reverb-additive + echo-tail guard

- P47 §9.C: Upstream PSD/ENR scoping — input-side bound established.
- P47.1: Reverb-additive audit (Q1/Q4 partition). Reverb-additive is real
  on this cohort but the 9-stage chain already accounts for it via
  `reverb_psd` injection.
- P47.2: Echo-tail guard pre-design audit. Verdict B → C; not implemented.

### P48 — non-NN decision layer / oracle

- P48a: Oracle dataset + baseline. 9-window oracle (`/tmp/p48a_oracle/`)
  established as the standard dry-run cohort. **Preserve this dataset
  pattern for future audits.**
- P48b: Follow-up audits with listener answers; verdict on model-inferred
  labels.
- P48c: Phase 1 trace; input-insufficient at the C2 boundary.

### P49 — WebRTC-equivalent non-NN input pipeline

- Phase 0: Design lock; revision 3 with M3 typed-path refit.
- Phase 1a–1e: M1 (RenderSignalAnalyzer) PASS; M2 cohort-anchor
  pre-convergence finding; M3 typed-path; M4+M5 LOCKSTEP joint-witness PASS;
  M6 spine retest **FAIL**.
- **Verdict:** P49 closes input-insufficient at the spine. The WebRTC-style
  input pipeline does not synthesise enough discrimination on this cohort
  to drive a typed-state classifier.

### P50 — DT-NE preset policy (the only arc that reached AECMOS)

- Phase 1: Dry-run + classifier upgrade verdicts.
- Phase 1.5: Classifier upgrade verdict.
- Phase 1.6: `balanced` vs `nearend_protect_dt` trade-off; pre-Phase 2 gate
  set at 800-case AECMOS.
- Phase 1.7: Broader-eval verdict.
- **Phase 2 (2026-05-11): GREEN_BALANCED_ONLY.**
  - `balanced` ships as Phase 2 default candidate (no code change; it was
    already the default).
  - `nearend_protect_dt` **DROPPED**: G3 FS Δecho −1.328 mean / −2.741 worst
    (~8× past audibility bound). DT NE gain was real (+0.641 AECMOS deg)
    but FS regression too severe. Symmetric pattern repeats in P58 below.

### P51 / P51b — heterogeneous shadow observer

- P51 design: FDAF heterogeneous shadow (different family from PBFDKF main).
- P51 Phase 1 (mode D) + shadow-observer verdicts.
- P51b: WebRTC alignment; Phase 1 verdict CLOSED — destabilisation surface
  (P51b §0.2.2): in-flight coefficient copy between heterogeneous filters
  triggers main divergence.
- **Hard invariant carried forward:** main + shadow MUST be same family
  (PBFDKF). No in-flight bidirectional coefficient copy.

### P52 — linear filter architecture revisit

- Phase 0: PBFDKF audit. **PBFDKF + Kalman state confirmed load-bearing**
  on this cohort. No swap candidate viable.

### P53 — pipeline coordination

- Phase 1 / Phase 1 v2: pipeline coordination redesign. CLOSED
  input-insufficient — degenerate inputs cannot synthesise discrimination
  via reclassification alone.

### P54 — WebRTC AFIR trace

- Phase 0 audit verdict: H_K confirmed (Kalman state is load-bearing).
  Full main swap to AFIR not pursued.

### P55 — PBFDKF-native dual filter with Kalman state to RES

- Phase 0: Design lock with `KalmanStateInterface` exposing 11 per-frame
  fields (innovation_var, kalman_p_trace, residual_ratio, system_distance_proxy,
  under_modeling_indicator, Enzner-Vary Wiener gain, etc.). Bidirectional
  shadow copy retired; one-shot stream-start init only.
- Phase 1: HB1/HB2/HB3/HB5 PASS; **HB4 FAIL** — Jung 2011
  `residual_ratio_voice_db` discriminator separates cohort roles by only
  1.913 dB max-pairwise vs ≥ 3 dB bar.
- Phase 1.2: Enzner-Vary canonical Wiener gain on 800-case: DT−FS dB
  separation +7.01 vs pre-locked 20 dB bar → FAIL.
- **Verdict CLOSED:** Both canonical Kalman-state discriminators (Jung 2011
  ratio + Enzner-Vary Wiener) fail on this cohort.

### P56 — temporal gate / Wiener post-filter

- Design pivoted twice (Wiener canonical → OM-LSA → back). Out of
  Phase 0 scope. Closed without implementation.
- **Lesson:** OM-LSA is a noise-reduction primitive; conflating it with AEC
  residual suppression is a category error. Downstream NR handles noise.

### P57 — Sohn-Kim-Sung 1999 dual-VAD regime classifier

- Phase 1 smoke: 24.9% regime accuracy vs 70% bar → **FAIL**.
- Mechanism: Sohn 1999 statistical VAD on `error_spec` confuses residual
  echo for NE speech; the four-state outer-product (FS/DT/NE/silence)
  collapses because input feature is itself echo-contaminated.

### P58 — AEC3-pattern RES restructure

- Modules implemented: `ErleEstimator` + mode-switched
  `ResidualEchoEstimator` (linear: `echo/ERLE`; nonlinear: `0.5×far`) +
  Wiener `SuppressorGain` with asymmetric EMA smoothing.
- Phase 1 smoke (5 FS): byte-equal OFF; bounded ON; determinism PASS.
- Phase 2 800-case AECMOS: **FAIL**
  - DT Δdeg +0.332 PASS (real NE-preservation gain)
  - **FS Δecho −0.674 FAIL** (bar −0.30)
  - Overall geomean 0.96 FAIL (bar 1.005)
- Mechanism: AEC3 ERLE-divide predicts tiny residual when filter is
  FS-converged → Wiener gain ≈ 1 → no post-suppression. The legacy 9-stage
  chain's caps were doing load-bearing FS work; AEC3 omits them because
  its upstream stack is tighter than ours.
- **Same shape as P50 `nearend_protect_dt`:** DT gain real, FS regression
  too severe; symmetric back-off impossible.

### P58.1 — Fan 2019/2020 under-modeling bias correction

- Hypothesis: add `γ × echo_psd` to `R²_linear` to capture un-modeled IR
  tail beyond `filter_length=52 ms`. γ=0.03 derived from RT60≈400 ms physics.
- 1-case smoke: bias ∈ {0, 0.03, 0.30} all produced identical output energy
  (−60.83 dBFS); legacy: −67.82 dBFS.
- **Verdict CLOSED at smoke** — bias has zero effect because the AEC3 cap
  `min(r², error_psd × 1.5)` saturates the numerator before bias matters.
  The 7 dB legacy/P58 gap originates downstream of the residual-PSD
  predictor (CNG modulation / post-gain stages), not in the predictor itself.

---

## 3. Bugs found

### DRX reverb_psd leak (D-X-reverb)

- Found during Phase-1a audit of P34b residual chain.
- Root cause: `reverb_psd` was not reset on long-silent far frames,
  leading to stale reverb additive injection in the next far-active burst.
- Fix landed: `enable_drx_reverb_psd_reset` flag with R3-only long-silence
  reset (clears `reverb_psd` + `far_activity` to 0 after N silent frames).
- Default-OFF; production retained pre-fix behaviour pending verdict.

### Shadow copy destabilisation (P51b §0.2.2)

- Bidirectional in-flight coefficient copy between heterogeneous (or even
  same-family with different Q) filters can trigger main divergence
  cascades during regime transitions.
- Workaround in current production: `epc_hangover=20` main-pause window.
- Proper fix (locked as P55 invariant): retire in-flight copy entirely,
  permit only one-shot stream-start init. NOT shipped (P55 closed).

### `error_psd × 1.5` cap saturation hides predictor differences

- AEC3-style `min(r², error_psd × 1.5)` cap masks both the Fan 2019 bias
  term and any other residual-PSD additive. Future predictor changes must
  test against output energy directly, not against `r²` pre-cap.

---

## 4. Knowledge accumulated (negative results worth keeping)

On this cohort (`wav/aec_challenge_blind/`, 800 cases, PBFDKF
filter_length=832), the following classes of solution were exhaustively
ruled out:

1. **Heterogeneous filters** (P51) — destabilises main.
2. **Linear-filter swap** (P52, P54) — PBFDKF + Kalman state is local
   optimum; no alternative family yields gain.
3. **Reclassification of degenerate inputs** (P49, P53) — energy/coherence
   features lack discriminating power; no classifier upgrade fixes
   "input-insufficient" inputs.
4. **Kalman-state to RES exposure** (P55) — both Jung 2011 and Enzner-Vary
   canonical discriminators fail HB4 / 800-case bars.
5. **DT-NE preservation policies** (P50 `nearend_protect_dt`, P58
   mode-switched Wiener) — DT gain real but FS regression severe;
   asymmetric trade.
6. **AEC3-pattern back-end** (P58) — porting back-end alone without
   tightening front-end (delay-est, render-buffer, reverb-model) inherits
   tighter-front-end assumption.
7. **Under-modeling bias correction** (P58.1) — bias term is canonical but
   masked by AEC3 cap; needs different consumer formulation to matter.
8. **Statistical VAD as regime classifier** (P57) — Sohn 1999 on echo-
   contaminated error spectrum is fundamentally circular.

---

## 5. Future work — non-NN options remaining (not yet attempted)

### Front-end restructure (highest leverage)
- **HyKF** (IEEE/ACM TASLP 2023, "Hybrid-Frequency-Resolution Adaptive
  Kalman Filter"): dual main + shadow at different resolutions
  (low-res fast-tracker for path-change, high-res long-IR slow-tracker).
  Different differentiation axis than current `shadow_q_ratio=3.0`.
- Tighter delay estimation + render-buffer (AEC3-style) to make the
  front-end's residual prediction trustworthy enough for an AEC3-style
  back-end.
- Filter-length sweep: cohort RT60 estimation suggests 100–200 ms IR span
  needed; current 52 ms under-models. PBFDKF partition / Kalman convergence
  bounds limit how much longer is practical.

### Cohort change (sidesteps the cohort-specific local optimum)
- Re-evaluate against a different dataset (in-house, different room
  acoustics, different mic profiles). The current cohort's FS/DT trade
  may be cohort-specific.

### Hybrid back-end (workaround, not canonical)
- Keep P58 mode-switch for DT NE-preservation gain (+0.332 AECMOS deg
  real) but layer the legacy 9-stage caps as a floor for FS frames:
  `R²_final = max(P58_wiener_prediction, legacy_cap_floor)`. Defensive,
  not canonical, but preserves both halves of the trade.

### NN arc (separate plan)
- Tracked in `~/.claude/plans/jazzy-brewing-castle.md` (Joint NR+RES+
  Dereverb). Independent track; not in this summary's scope.

---

## 6. Reference assets preserved

- **Bench standard:** `feedback_bench_j4.md` — `preset=balanced /
  filter_length=52ms / cng=True / j4 parallel`. All AECMOS A/B in this
  arc family used this.
- **Oracle dataset:** P48a 9-window oracle pattern. If recreated, the
  ground-truth label methodology is in the (now-removed) P48a docs;
  approximate reconstruction possible from MEMORY entries.
- **AECMOS baseline:** P50 Phase 2 `balanced` scores at
  `/tmp/p50_phase2/aecmos_balanced/scores.json` (preserved on this
  workstation; not in repo). Reproducible by running
  `python/bench_aecmos.py` on the dataset with `--preset balanced`.

---

## 7. Reading order for future engineers

If you want to revisit one of the closed arcs:

1. Read this summary.
2. Read `MEMORY.md` entries for the relevant arc (terse rules + project
   notes).
3. Check `git log --all --grep=PXX` for the original commits if you want
   to reconstruct the implementation. (The feature branch
   `feature/p27-subband-coarse-tracker` containing detail commits was
   deleted; consult reflog within 30 days, or trust this summary
   thereafter.)
