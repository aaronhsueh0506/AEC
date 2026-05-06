# Research log — P3 arc roll-up (7GT post-alignment investigation)

Date: 2026-05-06
Code line: v3.10.4 (shipped); all P3e/f/g toggles default off; only
zero-cost diagnostic fields retained.

## Origin

`7GTxyTksSUqCnP5y0ILG4A_doubletalk` — the canonical 788 ms-skew DT
case in the AEC Challenge blind set. v3.10.4 alignment fix moved the
delay search range to 1024 ms (matching WebRTC Old AEC), and
v3.10.4's path-A/B delay gate locks the correct alignment at
t = 4.57 s. Bench scores after that fix: **echo 3.366 / deg 3.895**.

Question that started P3: alignment was clearly working — *why was
the score still mediocre, and what's actually limiting it?*

## Sub-investigations and outcomes

### P3a — alignment math sanity (closed, no change)

GCC-PHAT diagnostics confirmed the mic-ref skew at 12 606 samples
(788 ms) is recoverable; alignment math is correct. No code change.

### P3b — production wiring (closed, no change)

Verified the v3.10.4 production delay estimator does in fact lock to
the correct delay at t = 4.57 s and stays there. No code change.

### P3c — delay acquisition latency (deferred)

The 4.57 s blind window before delay locks is a separate problem
(filter learns garbage during it). Not addressed in this arc.

### P3d — post-alignment trace (root cause, log written)

Per-frame trace post-4.57 s on 7GT showed:

- Filter ERLE peaks ~ +5 dB at 8–12 s, then collapses to 0 / negative
  in 24–36 s (NE bursts).
- DTD signals exist (`dt_shadow` median 0.51, `dt_energy` 0.24) but
  the composite `dt_active` gate that drives `mu_scale` never fires
  (because `enable_dtd = False` zeroes the coherence path).
- RES sits in render-based mode 99 % of the time post-alignment.

Hypothesis: filter taps are being learned against NE-contaminated
error in the back half. A DT advisory gate (consult `dt_shadow` /
`dt_energy` directly to drop `mu_scale`) would protect the taps.

Doc: `docs/research_log_p3d_7gt_post_alignment.md`.

### P3e — DT advisory gate, single-threshold sweep (closed, log written)

Three 800-case bench variants of a `mu *= 0.3` advisory gate:

| Variant | Gate | FS_static Δ | DT_static Δ deg | 7GT |
|---|---|---:|---:|---|
| V1 | `dt_shadow > 0.5` | **−0.095** | +0.074 | unchanged |
| V2 | `dt_shadow > 0.5 OR dt_energy > 0.4` | **−0.128** | +0.104 | unchanged |
| V3 | V2 + `once_converged AND not post_reset_warmup` | −0.011 | +0.007 | unchanged |

V1 / V2 fail FS gate badly. V3 keeps FS but loses the DT action; 7GT
is bit-identical in all three. Showed the V1 / V2 "DT gain" was
predominantly FS-side false fires (shadow-beats-main during early
convergence), not real DT detection.

Doc: `docs/research_log_p3e_dt_advisory_negative.md`.

### P3f — Mini AecState, state-classifier-gated mu reduction (closed, log written)

User-proposed shift: **trace-only state model first, then wire**.
Computed five state layers per frame:
`idle / startup / coarse_learning / refined_usable / suspicious_dt
/ diverged`, plus `usable_linear`. AND-conjunction
suspicious_dt requires refined_latched + NE evidence + main_err_jump
+ shadow_lead.

Phase 2 audit on 5 trace cases:

| Invariant | Result |
|---|---|
| 7GT 4.6–8 s = `coarse_learning` | 87 % PASS |
| 7GT 8–12 s = `refined_usable` ∪ `suspicious_dt` | 59 % PASS |
| 7GT 24–36 s NE-evidence = `suspicious_dt` ∪ `diverged` | 94 % PASS |
| FS no-suspicious_dt | 0 / 1191 + 0 / 2284 PASS |
| DT_static / movement NE flagged | 0 % (`shadow_advantage` ≈ 1.0 — honest negative) |

Phase 3 wired `mu *= 0.3` gated on `filter_state == 'suspicious_dt'`.
800-case bench: FS_static **−0.022** (just outside ±0.02), DT bucket
+0.013 / +0.014, **7GT bit-identical (3.366 / 3.895) despite the gate
firing 827 times**.

Two findings:

1. The per-frame mu-reduction *intervention* does not move the case,
   regardless of whether the gate fires on the right frames.
2. The 5-case audit was sufficient to falsify a classifier (V1 / V2
   wouldn't have survived) but not sufficient to verify FS-safety
   (the −0.022 came from cases not in the 5-case set).

Doc: `docs/research_log_p3f_state_gate_negative.md`.

### P3g Phase 0 — RES linear-vs-render switch dry-run (closed, log written)

User-proposed pivot: use the same state classifier to gate the RES
*residual source* instead of adaptation `mu`. Phase 0 = dry-run
audit only — instrument both linear and render-based residual
estimates per frame, *no behaviour change*, then look at the
comparison per state.

Inversion finding:

- **7GT 24–36 s NE frames**: linear residual median **+4.8 dB**,
  render residual **−11.7 dB**. Render dominance **−16.5 dB**.
  The corrupt taps inflate `echo_psd`; the render override is the
  *protective* path. Switching to linear here would **increase
  suppression** based on corrupt taps and damage NE further.
- **FS_static `usable_linear` frames**: render dominance **+11.7 dB**
  (render over-estimates). Switching to linear would suppress less
  in FS, where there is no NE to preserve — pure echo regression.

The P3d observation "`using_render = 99 %` post-alignment" reads now
as a feature, not a defect — it's the system correctly de-trusting
the bad taps.

Doc: `docs/research_log_p3g_res_switch_negative.md`.

## Net result

```
                                        affects 7GT?        ships?
P3a alignment math sanity               n/a                 no change
P3b production wiring                   n/a                 no change
P3c delay acquisition latency           --- deferred ---
P3d post-alignment trace                diagnosis           no change
P3e DT advisory (single threshold)      no                  no
P3f state-gated mu reduction            no (827 fires)      no
P3g Phase 0 RES source dry-run          would worsen        no
```

## What we learned beyond "all variants negative"

1. **The 7GT score asymptote is real.** The filter taps degrade against
   NE in the back half (linear residual reading +4.8 dB attests),
   the render-based RES correctly de-trusts those taps and uses a
   delay-agnostic estimate (−11.7 dB), and the resulting
   `3.366 / 3.895` is what that architecture produces on this case.
   Both adaptation-side (P3e/f) and RES-side (P3g) interventions
   that act *after* the taps go bad have no leverage.

2. **`dt_shadow` / `dt_energy` are not DT signals on this dataset's
   bench DT bucket.** P3f Phase 2 trace showed `shadow_advantage`
   ≈ 1.0 across the DT_static / DT_movement bucket means; the +0.07
   deg gain V1 reported was entirely FS-side false fires. Future
   plans that rest on these signals need a fresh DT signal first.

3. **A 5-case fixed audit falsifies but does not verify.** Fast for
   killing wrong ideas, slow on FS-safety. Combine with a single
   bucket bench (≤ 200 cases) before claiming FS-safety on any
   classifier.

4. **WebRTC AEC3 architectural reading was right, the actions were
   wrong.** AEC3-style state model (P3f Mini AecState) classifies
   correctly; the mistake was assuming "`suspicious_dt → reduce
   mu` and `usable_linear → linear residual`" were the right
   actions. AEC3 itself does much more (subtractor analyzer,
   stationarity estimator, dominant-nearend-aware suppression),
   and we copied only the labels.

## What's left to try (if 7GT is worth more attention)

The only state-driven action not yet tried is **filter reset on
sustained `diverged`** (P3f originally spec'd it as the diverged
action; we deliberately left it unwired). Risk: any false
`diverged` classification on FS would discard a healthy filter and
re-pay the convergence cost. Per user direction, this is the
followup proposal:

- 7GT-only timeline / score-proxy dry-run with reset injection at
  sustained-diverged points;
- only if 7GT visibly moves do we proceed to 800-case risk bench.

Until that experiment, **7GT 3.366 / 3.895 is the v3.10.4 asymptote**.

## What's retained in v3.10.4

Zero-cost diagnostic fields, default off behaviour-wise:

| Field | Source | Purpose |
|---|---|---|
| `main_err_ratio`, `shadow_err_ratio`, `p3f_shadow_advantage` | P3f Phase 1 | Subtractor relative-error trace |
| `erle_slope_db_per_s`, `post_reset_age_ms` | P3f Phase 1 | Convergence trajectory |
| `filter_state`, `usable_linear` | P3f Phase 2 | Mini AecState classifier output |
| `residual_psd_linear`, `residual_psd_render`, `residual_render_blend` | P3g Phase 0 | RES source comparison |
| `dt_advisory_active`, `dt_advisory_hit`, `mu_scale` | P3e | Advisory gate hits + applied mu |

All seven groups appear in `--trace-aec-state` CSV. None affect runtime
or shipped behaviour.

| Toggle | Default | Status |
|---|---|---|
| `dt_advisory_enabled` | `False` | Retained, no effect (gate body skipped) |
| `dt_advisory_use_p3f_state` | `False` | Retained, no effect |
| `dt_advisory_shadow_th / energy_th / hold_ms / mu_factor` | constants | Retained |

## Cross-references

- `docs/research_log_p3d_7gt_post_alignment.md`
- `docs/research_log_p3e_dt_advisory_negative.md`
- `docs/research_log_p3f_state_gate_negative.md`
- `docs/research_log_p3g_res_switch_negative.md`
