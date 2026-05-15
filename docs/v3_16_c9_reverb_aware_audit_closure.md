# v3.16 C9 — Reverb-aware RES override audit CLOSED (2026-05-15)

**Status**: CLOSED — TRIGGER UNDESIGNED per §0.4.
**Branch**: `feature/v3.16` (commit pending).
**Sprint**: v3.16 Phase 4 audit-first (per `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` §10).
**Substrate**: `tools/research/v3_16_c9_reverb_aware_audit.py` retained.

---

## 1. Headline

**Verdict**: Plain Pearson cross-correlation between mic and
delay-aligned lpb does NOT separate pcb1N (the C9 target case) from
healthy controls. HK-2 reframing's reported `r = 0.071` was likely a
different cross-correlation methodology (envelope / magnitude of peak,
not sample-level Pearson). 1-sprint audit is insufficient to design a
robust per-frame trigger; C9's mechanism arc requires multi-sprint
detector R&D that **exceeds v3.16 scope**.

**No mechanism arc opens.** C9 promoted to v3.17 / research-track
backlog with explicit caveat that pcb1N audible debt remains.

---

## 2. Audit data (Tier A 10 cases, 60-sec wall time)

### 2.1 Per-case mic-lpb Pearson r (delay-aligned, 320-sample windows)

| Stem | Bucket | r mean | r p10 | low-r % | delay > 0.8·fl % |
|---|---|---:|---:|---:|---:|
| qNvSMyU FS (cohort tail) | FS_static | +0.009 | −0.231 | 79.0 % | 84.8 % |
| XqvGR01t DT_mvmt | DT_movement | +0.077 | −0.189 | 65.3 % | 0.0 % |
| **pcb1N FS** (HK-2 target) | FS_static | **−0.182** | −0.593 | 91.8 % | 82.3 % |
| XRTnTUjU DT (xrtntuju) | DT_static | −0.027 | −0.161 | 96.1 % | 79.7 % |
| 0I0XMl3M FS_mvmt | FS_movement | −0.002 | −0.087 | 96.0 % | 72.4 % |
| Hp5g1asac DT_mvmt | DT_movement | −0.060 | −0.422 | 83.6 % | 0.0 % |
| WH0jN3PY DT_mvmt | DT_movement | +0.122 | −0.190 | 64.0 % | 86.2 % |
| jtYTdZm3 DT (F2.4) | DT_static | +0.078 | −0.213 | 64.0 % | 0.0 % |
| OX2l6zV7 FS_mvmt | FS_movement | +0.034 | −0.378 | 73.0 % | 81.1 % |
| **Y91uE2t FS CTRL** | FS_static | **−0.414** | −0.733 | **100.0 %** | 0.0 % |

### 2.2 Critical observation: CTRL (Y91uE2t) has WORSE r than pcb1N

The CTRL case (Y91uE2t — best ERLE in C6 audit at +4.7 dB, well-converged
filter) shows r mean = **−0.414** with **100 %** of frames below the
candidate trigger threshold (r < 0.15). pcb1N has r mean = −0.182 with
91.8 % low-r — both are IN THE SAME RANGE.

**Why**: plain sample-level Pearson r is dominated by:
1. Echo path filter coloring (`y[t] = h * x[t-d]`; multi-tap h reduces
   single-alignment r).
2. Phase shift between mic and lpb (negative correlation possible if
   the room transfer function has phase reversal at dominant frequencies).
3. Short window noise variance (320 samples = 20 ms is short for
   stable correlation estimation).

Pearson r is **NOT a good per-frame coupling indicator** for AEC.

### 2.3 What HK-2 reframing's r=0.071 likely measured

HK-2 listen analysis (2026-05-15) reported pcb1N "mic-lpb cross-correlation
r = 0.071". The exact measurement methodology was not recorded. Plausible
candidates:
- Envelope cross-correlation (magnitude on Hilbert envelope).
- Magnitude of the peak in cross-correlation function (similar to GCC
  but without phase normalisation).
- Long-window (1-second) Pearson on full clip.
- Per-bin coherence² mean.

**This audit cannot reproduce that r=0.071 figure with sample-level
Pearson.** The HK-2 reframing's diagnostic value remains (pcb1N IS a
characteristic difficult case), but the proposed C9 trigger
(`r < 0.15` window) does NOT map to any reproducible per-frame metric
in our trace surface.

### 2.4 Other potential triggers also fail to discriminate

| Candidate trigger | pcb1N | Y91uE2t (CTRL) | Discriminates? |
|---|:---:|:---:|:---:|
| `r < 0.15` (Pearson on 320-sample window) | 91.8 % | 100.0 % | **NO** |
| `delay > 0.8 × fl_samples` | 82.3 % | 0.0 % | YES — but FPs on qNvSMyU / 0I0XMl3M / WH0jN3PY / OX2l6zV7 |
| `top1_par < 5` (DelayEst PAR) | 0 % | 0 % | NO (both solid) |
| `top1/top2 par_ratio > 0.7` (multi-peak ambig) | 0 % (mean 0.36) | 0 % (mean 0.32) | NO |
| `dt_from_coherence > 0.5` | n/a (always near 0 on FS) | n/a | NO (signal not informative for FS-only) |
| `res_gain_db < -10` | 0 % (gp5 = −6.27) | 0 % (gp5 = −5.25) | NO |
| `erle_inst_db < 0` (filter making it worse) | high % | low % | partial |

The closest single-feature discriminator is **`delay > 0.8 × fl_samples`**,
but this fires false-positive on cohort tail / movement / DT cases.

---

## 3. Mechanism arc cost re-estimate

Per HK-2 reframing (v3.16 plan §10 finding 5):

> "no single-knob fix addresses all three causes. Deeper mechanism
> (reverb-aware RES override OR delay-aware filter mode OR NL-detector-
> driven gain boost) requires audit + design + 800-case bench,
> exceeding '≤ 1 sprint' budget."

This audit confirms HK-2's caveat. A robust C9 detector requires:

1. **Multi-feature classifier** — combine `delay > 0.8·fl`,
   coherence-band statistics, `top1/top2_par` ratio, and possibly
   envelope cross-correlation. Calibrate on N positive (pcb1N + similar
   reverb-heavy cases) + M negative (cohort tail / movement / DT) cases.
2. **Cohort expansion** — pcb1N is N=1 today. Need 5+ similar cases
   to learn discriminative features without overfit.
3. **RES override design** — when the trigger fires, what mode change?
   "FS-aggressive" needs concrete tuning (which knobs, by how much,
   per-bin or scalar).
4. **800-case FS / DT FP regression guard** — false trigger on
   converged FS / DT damages cohort-wide AECMOS.

Estimated LOE: **5-8 sprints** (vs original 2-3 estimate). Beyond
single-sprint audit scope.

---

## 4. v3.16 plan implications

### 4.1 C9 disposition

**CLOSED for v3.16** per §0.4. Substrate retained
(`tools/research/v3_16_c9_reverb_aware_audit.py` + this verdict doc)
for v3.17 / research-track re-open. pcb1N audible debt remains
unaddressed.

### 4.2 v3.17 / research-track candidate

`docs/v3_17_backlog.md` (when authored) should include:

> **C9.v2 — Reverb-aware RES override (re-scoped)**
> - 5-8 sprints LOE
> - Multi-feature trigger (envelope cross-corr + coherence + delay
>   coverage + DelayEst multi-peak)
> - Cohort expansion (find 5+ pcb1N-class cases in 800-corpus)
> - RES override design + tuning
> - 800-case FP regression guard
> - Acceptance: pcb1N audible improvement + cohort tail / FS / DT no
>   regression > 0.005

### 4.3 v3.16 candidate tally (post C9 closure)

| ID | Phase | Status |
|---|---|---|
| HK-1 | 0 | ✓ DONE (`ea8d320`) |
| HK-2 | 0 | ✓ REFRAMED → C6 + C9 |
| C1 | 0 | ✓ DONE (`d90efdc`) |
| C1b | 0 | ✓ RECLASSIFIED RETAIN |
| C1c | 0 | ✓ DONE (`37a46b2`) |
| C6 | 1 | ✓ CLOSED H2 (`ac7320e`) |
| v3.16-A | 1.5 | ✓ CLOSED CANNOT SHIP (`d181e17`) |
| C5 | 1.6 | DEFERRED (re-evaluate post-C2) |
| C2 | 2 | candidate (likely subsumed by §1.2 / Arc D wall) |
| C3 | 2 | ✓ CLOSED (`d04ba60`) |
| C4 | 2 | candidate (LOW priority; audit-first) |
| v3.16-B | 3 | gated on C2 |
| C7 | 4 | candidate (Arc M.v3 retry; previously CLOSED v3.15 §1.5b) |
| C8 | 4 | candidate (LOW priority) |
| C9 | 4 | ✓ CLOSED (this doc) — TRIGGER UNDESIGNED |

**9 / 15 candidates disposed**. 6 remain, of which 5 are likely
closures or low-ROI:
- C2: likely subsumed by FS-vs-DT wall (§1.2 + Arc D family).
- C4: noise_floor / CNG. Unknown; LOW priority.
- v3.16-B: gated on C2.
- C7: was already CLOSED in v3.15 §1.5b; retry would need new mechanism.
- C8: LOW priority partial-decay alternative for Arc G.
- C5: architectural foundation (no AECMOS Δ); deferred.

---

## 5. Substrate (committed)

- Audit script:
  [`tools/research/v3_16_c9_reverb_aware_audit.py`](../tools/research/v3_16_c9_reverb_aware_audit.py)
- Per-case JSONs (gitignored): `/tmp/v3_16_c9_audit/per_case/*.json`
- Aggregate: `/tmp/v3_16_c9_audit/analysis.json`

Zero changes to `python/aec.py` — audit consumed existing
`trace_delay_est` + AecStats surface.

---

## 6. Verdict signed-off

**CLOSED for v3.16** — TRIGGER UNDESIGNED. C9.v2 promoted to v3.17
/ research-track candidate (5-8 sprint LOE) with explicit acknowledgement
that pcb1N audible debt remains. **v3.16 has reached natural closeout**:
of 6 remaining candidates, 5 have likely-closure indicators (FS-vs-DT
wall, dead path, low priority). Recommend v3.16 closeout + transition
to v3.17 planning.
