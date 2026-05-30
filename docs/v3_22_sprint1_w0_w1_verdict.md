# v3.22 Sprint 1 — W0 + W1' verdict (NE-first)

**Status (2026-05-29):** W0 DONE (keystone refuted for NE-only). W1' DONE
(implemented + byte-equal, but **INERT** per witness — not shipping). Next lever
= **W2 (DT adaptation freeze)**, pending user go. Production `main` unchanged.

Branch: `v3_22` (forked from `v3_21_release` @ `0d6d5c2`, local-only).
Comparison anchor: shipped v3.21 scorecard (`results/v3_21_close/scores.json`),
model `Run_1663915512_Stage_0.onnx`.

---

## W0 — empirical confirmation of the "ERLE-timing desync" keystone

The plan's keystone (Part B.1): ERLE frozen at `min_erle=1.0` during the
200-hop session startup while `usable_linear` opens at ~40 hops → in that window
`R² = s2_linear / 1.0` undivided → `min_gain` collapses → NE wiped.

**Verdict: REFUTED for the NE-only cohort; only partial for DT.**

Read-only probes (in-code `_last_*` observers; throwaway scripts in `/tmp`, NOT
committed): `w0_desync_probe.py`, `w0_ne_breakdown.py`, `w0_ne_miclevel.py`,
`w0_dt_probe.py`.

- **NE-only worst cases run 0 % linear path.** `usable_linear` never opens with
  no echo present → the ERLE-divide path that the keystone blames *never fires*
  on NE-only. The over-suppression there is the **nonlinear R²** path
  (`R² = X² · default_gain(1.0)` + reverb), with **reverb 76–84 % of the
  suppressed-hop R²**, applied against a **quiet** mic.
- The 2 worst NE-only mics measure **−42.7 / −51 dBFS** → these are the
  **low-mic spec cases** (out of spec; per user, do not hard-solve — amplify +
  report only).
- **DT is heterogeneous:** linear-absorption (W2 territory) + suppressor
  ERLE-at-min + per-bin non-convergence — not a single startup-gate cause.

---

## W1' — `erle_startup_follows_convergence` (the startup-gate fix)

**Mechanism:** when ON, `ErleEstimator.update()` drops the fixed 200-hop session
startup gate and updates as soon as `converged_filter` is True (the sub-estimators
already self-gate on convergence), so ERLE-readiness tracks the same convergence
signal `usable_linear` uses instead of an independent clock.

**Implementation (7 edits, default-OFF, byte-equal 12/12 PASS):**
- `config.py` — `erle_startup_follows_convergence: bool = False` (+ rationale block).
- `state/aec_state.py` — `AecStateConfig` field + thread into `ErleEstimator` build.
- `state/erle_estimator.py` — `__init__` param + the gate-skip in `update()`.
- `orchestrator.py` — both `_Aec3StateConfig` constructions (init + reset).

**Verdict: INERT — not shipping.** Witness (`/tmp/w1_witness.py`, OFF vs ON,
voice-band median over startup window blocks<200):

| case | ERLE leaves min @blocks (OFF→ON) | startup gain median (OFF→ON) |
|---|---|---|
| SgKY30 (DT 1.194) | **756 → 26** | 0.667 → **0.667** (unchanged) |
| XV5L2dn (DT-mvmt 1.189) | 241 → 175 | 0.005 → **0.005** (unchanged) |
| 9xjhi (FS 1.743) | 229 → 228 | 1.000 → **1.000** (unchanged) |

The flag works mechanically (ERLE leaves min far earlier on SgKY30, 756→26) **but
the suppression gain is identical** → it will not move AECMOS. The witness saved
a ~15-min 800-case run that would have read ≈0.000 (like the v3.21 C1–C6 NEUTRAL
bundle).

**Why it's inert (the 3rd refinement of the root cause):** in the over-suppressed
hops `converged_filter` is **False** (the filter is not converging per-bin during
DT / startup), so ERLE stays pinned at `min_erle` *regardless* of the session
gate — aligning the session gate can't help when the gate it now depends on
(convergence) is itself False there. Also the startup window is largely the
nonlinear path, and per-bin ERLE convergence in DT is sparse → "leaves min"
reflects one bin crossing 1.05 while the **median** voice-band ERLE stays at min.

**Disposition:** keep the flag as **default-OFF substrate** (clean, byte-equal,
plausibly useful *combined* with a convergence fix later — W9), but inert alone.

---

## Synthesis across W0 + W1' — the suppression-side ERLE lever is a dead end

Two suppression-side hypotheses are now refuted (keystone for NE-only; W1'
startup-gate). The over-suppression is **not** an ERLE-divide timing problem. It
is driven by **R² magnitude vs a quiet / DT near**:

1. **NE-only:** nonlinear R² (`X²·default_gain=1.0` + reverb tail 76–84 %)
   dwarfs a quiet mic. The 2 worst are **low-mic = spec** (don't hard-solve).
2. **DT:** the **linear filter itself absorbs NE** (QkRkwwFK ≈ **−4.7 dB**
   voice-band cut *before* the suppressor) + per-bin non-convergence keeps ERLE
   at min so the suppressor can't recover it.

**The real in-spec, NE-primary lever is W2 — freeze the linear filter's
adaptation when near-end dominates** (stop it absorbing NE in the first place;
this also protects the irreversible `_ours_nores` output). HARD-nores; gate on
`dt_from_energy` (already computed, currently stats-only).

---

## W2 + W4 (2026-05-29) — DT track: four levers refuted, root cause is R² magnitude

**W2 (DT adaptation freeze) — NO-GO.** De-risk trace on QkRkwwFK_doubletalk
(`/tmp/w2_derisk.py`): the main filter IS adapting during DT (main_mu 0.564,
dW_rel 2.41%/hop ≈ echo-only 2.45%), so there is real movement to freeze — but a
counterfactual freeze (mu→0 on DT hops, static echo path) recovered only **+0.17
dB** nores VB energy, *below* the +0.21 dB drift on supposedly-unchanged hops. The
+4.23 dB DT cut is dominantly **legit echo cancellation**, not NE absorption. W2
dropped.

**W4 (SNR-aware DNE ENR-relax) — implemented, near-inert, default-OFF substrate.**
Root-cause probe (`/tmp/w4_dne_probe.py`) on XV5L2dn / SgKY30: in NE-present
collapse hops the DNE NE-state fires only **44–45%** and the ENR test fails 76%
because the NE-inflated error keeps ERLE low (2.10 / 1.00) → R² ≈ ne (enr
0.64–1.39 vs thr 0.25), even though SNR = 240×–100000×. Implemented a flag
(`dne_loud_nearend_enr_relax_enabled`, config.py + DominantNearendConfig +
suppression_gain.update + orchestrator:481; byte-equal 12/12) that relaxes the
ENR threshold (0.25→0.75) when the near-end overwhelmingly dominates the noise
floor. Witness (`/tmp/w4_witness.py`): NE-state rises (44→52%, 45→54%) **but the
gain barely moves** (0.016→0.018, 0.000→0.000). Plus NE-state already false-fires
**93% in 9xjhi FS** → it is NOT a clean NE-vs-FS gate.

**SYNTHESIS — the master variable is the per-bin R² magnitude.** Four indirect
levers are now refuted (keystone ERLE-desync, W1' startup-gate, W2 linear-freeze,
W4 detector-tuning). None move the suppressor gain because the gain faithfully
tracks the per-bin VB residual-echo estimate R², which is large during DT because
**real echo overlaps the near-end in the voice band**. The suppressor is behaving
correctly; it simply cannot separate NE from echo when they co-occur, and AEC3's
design errs toward echo cancellation. The only levers that actually move the gain
are (a) a **gain floor** that overrides the R²-driven collapse, or (b) **reducing
R²/cancellation** during DT — **both leak DT echo** to protect NE. This is the
PRIMARY=NE / SECONDARY=echo Pareto trade, not a bug fix. Gate caveat: it must be
`dt_from_energy`-based (input-determined: high in DT, low in FS), NOT NE-state
(false-fires in FS).

**Dispositions:** W1' + W4 flags kept default-OFF substrate (byte-equal, harmless,
possible combo ingredients with a floor). W2 dropped (no flag added).

## AEC3 R²/ERLE faithfulness audit (2026-05-29) — no win-win porting gap

Three parallel read-only audits (ERLE chain / convergence gate / R² estimator)
vs `docs/aec3_extracts`, asking: is our DT R² inflated **relative to AEC3** (a
porting gap → win-win fix) or faithful (inherent wall)?

**Verdict: FAITHFUL. No win-win porting gap.**
- **R² estimator** (`residual_echo_estimator.py`): faithful — `s2_linear` is the
  correct windowed H·X echo-estimate power (not Y² or X²·gain), ERLE-divide
  matches, reverb equal-or-*deflated*, stationarity downweight exact, no caps
  missing/added.
- **Convergence gate** (`orchestrator.py:2940-2988`): faithful — same e²/y²
  (capture energy), same per-frame gating with **no ever-converged latch in AEC3
  either** (AEC3 latch is `usable_linear`-only), same thresholds/scaling. AEC3
  freezes ERLE during DT for the identical reason (NE inflates e²).
- **ERLE math** (`subband/fullband/erle_estimator`): faithful min/max/reset/onset.

**The one real divergence — and it is intentional + already NO-SHIP.** The
subband-ERLE EMA smoothing runs AEC3's per-*block* alpha per-*hop*
(`subband_wallclock_smoothing` OFF) → ERLE climbs ~2.5× slower in wall-clock →
sits lower on the recovery ramp. An audit agent flagged this as a "win-win"
(faster ERLE, no trade), **but the empirical record refutes that**: the
2026-05-29 800-case bench (`docs/v3_21_800case_bench_report.md`, M_full bundle
with `use_aec3_wallclock_subband_erle_smoothing=ON`) gave **DT_static +0.026 /
DT_mvmt +0.048 deg BUT FS_static −0.049 / FS_mvmt −0.054 echo, 40 catastrophics
(worst −1.514) → NO-SHIP.** Faster ERLE → smaller R² → *less suppression
everywhere* → the SAME NE-vs-echo Pareto trade; the DT lift is the illusory-deg
side. CHANGELOG root cause: "ratio estimators (ERLE) want preserve-count, not
α-conversion." So even the one available *optimization* of the existing mechanism
is on the NO-SHIP Pareto line.

**Locked divergence (not actionable):** AEC3's coarse filter self-heals
(copy-from-refined) giving a live ERLE keep-alive during DT; our shadow doesn't
(coarse_conv≈0%). But shadow is PBFDAF-locked (user 2026-05-29) and the strict
coarse-rescue was already Pareto-FAIL ([[project_v3_21_poor_coarse_rescue_noship]]).

**Plan Part-A correctness concerns — all resolved (re-audited):** EMA-α verdict
CORRECT (NO-SHIP, above); noise_gate (27509.42) + X²_threshold (44015068) =
false-positive scale bugs (pre-scaled to int16² / corrected value used);
float32/float64 leakage mix = C-parity smell only, byte-equal holds (not a Python
bug). **No confirmed magnitude/scale/dtype bug in v3.21.**

**Incidental:** our DT actually BEATS AEC3 (12-case alignment: DT_mvmt +0.666 /
DT_static +0.094 vs AEC3 ref). The only place we trail AEC3 is FS-echo (9xjhi
−1.713 = known exception, locked-shadow domain). So PRIMARY=NE is already at/above
AEC3 — **the DT near-end wall is shared with AEC3, not a defect of our port.**

**CONCLUSION (triply-confirmed):** DT near-end over-suppression is inherent —
(1) W4/dt-gate: no clean DT-vs-FS detector exists; (2) AEC3-audit: the chain is
faithful and the one knob (ERLE rate) is already NO-SHIP on the Pareto. No
free/win-win fix in traditional DSP. → Option 2: pivot to the FS in-spec track +
`aec_record.wav` eval.

## aec_record.wav eval (2026-05-29) — AEC behaves correctly; "low-mic" = echo-only FS

User's own 85 s recording (`aec_record.wav`, 2-ch: ch0=mic, ch1=ref, 3 ms echo
delay, 8 segments split by silence). Full AEC run + per-segment coherence:

| seg | mic/ref RMS | VB coherence | behaviour |
|---|---|---|---|
| seg1/2 (DT) | −27/−22 | 0.05–0.11 (near-dominant) | preserved Δ−0.6/−0.8 dB ✓ |
| seg5 (main) | −23/−23 | 0.93 (echo-dominant) | cancelled Δ−19.5 dB ✓ |
| seg6 (**"low-mic"**) | **−40.6**/−21.1 | **0.97** (echo-ONLY) | silenced (out −72 dB) ✓ |
| seg7 (**"low-mic"**) | **−36.3**/−23.7 | **0.96** (echo-ONLY) | suppressed Δ−9.4 dB ✓ |

**Finding (CORRECTED — user ground-truth: a very quiet near IS present, visible
in the spectrogram).** Near-dominant segments (seg1/2/3) are preserved correctly.
For the "mic 太小聲" segments (seg6/seg7), the energy-weighted coherence (0.96–0.97)
was **misleading** — it is dominated by the loud echo and masked a faint near-end
talker buried ~15 dB underneath. Measured **SER ≈ −14.9 dB (seg6) / −13.8 dB
(seg7)** (near 15 dB below echo). The near is being lost: seg6 mic −40.9 → out
−65.5 (Δ−24.6, WIPED — short far-pauses, suppressor hangover from the loud far
bleeds in); seg7 preserves the near during *long* far-pauses (mic −42.0 → out
−43.8, Δ−1.9) but loses it during simultaneous echo. The mic is quiet because the
**near talker is quiet (low SER)**, NOT because the mic gain is low overall.

**Amplify test:** boosting the mic +20 dB is **scale-invariant** (seg6 Δ−26.4 raw
vs −27.0 amplified) — near and echo scale together, SER unchanged → near still
wiped. So **absolute level is NOT the spec parameter; SER (near-to-echo ratio)
is.** This is the same DT wall at an extreme ratio (−14/−15 dB SER), not an
absolute-level problem.

**SER as spec (user's question — YES).** Near-end preservation in DT is governed
by SER, not absolute level. Recommend specifying a **minimum SER** for guaranteed
near preservation; below it (these segments at −14/−15 dB) is out-of-spec
(don't-hard-solve). Reference: AEC-Challenge DT near-end is typically within
SER ∈ [−10, +10] dB. Precise threshold → run a controlled SER sweep (near+echo
mixes −20…+10 dB, measure near-preservation knee).

Outputs for listening: `out_aec_record/aec_record_ours.wav` (full) +
`aec_record_seg{6,7}_{raw,amp}.wav`.

## LF "gap" vs AEC3 RESOLVED — it was a cold-start artifact; we beat AEC3 (2026-05-29)

Investigation of the seg5 LF (the "藏在後面的小聲 nearend" the user saw in the
spectrogram) found an apparent +9 dB LF near-preservation deficit vs AEC3. After
two wrong turns (energy-weighted coherence masking the quiet near; reading
`r2_direct`=1.98e8 when the suppressor actually uses the full nonlinear
r2=1.42e10), the root looked like the nonlinear-R² path (`usable_linear=False` →
R²=X²·default_gain(1.0)+reverb wipes LF). **But that was a per-segment COLD-START
artifact.** Fair WARM-vs-WARM comparison (both on the full continuous recording,
`bin/aec3_cli` on `full_mic/full_ref`):

| seg5 region (warm) | LF 50–500 | VB 500–3000 |
|---|---|---|
| OURS | **−50.2** | −48.8 |
| AEC3 | −61.6 | −50.3 |
| ours − AEC3 | **+11.4 dB** | +1.6 dB |

In realistic continuous operation `usable_linear` is 97 % True → linear path →
the near LF is preserved (−50.2), **beating AEC3 by +11.4 dB**. The cold
per-segment run (seg5 in isolation) never opened `usable_linear` → nonlinear wipe
(−62.1); AEC3 cold happened to be −52.6, making us look 9 dB worse. Lesson logged:
[[feedback-per-segment-cold-vs-warm]] — always validate on the warm continuous run.

**NET v3.22 status: 0 shippable algorithm improvements found, but the AEC is
HEALTHY** — it beats AEC3 on DT AECMOS and on this recording's near-LF (warm).
The apparent "problems" all explained: DT simultaneous overlap = inherent wall;
quiet-near = out-of-spec (SER); seg5 near "wipe" = cold-start analysis artifact
(realistic warm run preserves it, > AEC3). **One genuine narrow residual lever:**
cold-start robustness — if a stream *starts* on a hard segment, `usable_linear`
may never open → nonlinear wipe at start (beyond-AEC3; distinct from the artifact).

## Next steps (entry point after compact)

- **W2 [ARCH, HARD-nores] — DT adaptation freeze.** Drive `main_mu→0` (per-bin or
  gated) when NE dominates under echo, using `dt_from_energy` as the freeze
  condition the way `_block_stationary` already gates the W-update. Files:
  `orchestrator.py` mu_scale path + `_simple_mu_ratio`, `filters.py` W-update gate.
  - **De-risk first (avoid another implement-then-find-inert):** one targeted
    trace to confirm the QkRkwwFK −4.7 dB is genuinely **NE-absorption** (bad,
    freezing helps) vs **legit echo-cancellation** (freezing leaks echo) before
    coding. The filter should only cancel ref-correlated energy; NE leaks into
    the gradient during DT, so a DTD-style freeze (mu→0 on NE-dominant) is the
    classic remedy — but verify the magnitude is NE, not echo.
- Open alternative the user raised: bank low-mic / pre-processed / discontinuous
  FS cases as **spec** and pivot to the FS gain-change / echo track instead.
- `aec_record.wav`: amplify low-mic segments + report processed output
  (diagnostic only; low-vol → minimum-input-level spec).

Full plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (v3.22 plan).
