# P52 Task A.0 Post-mortem — Forensic Trace

**Question**: A.0 verdict noted "ShadowCopyController IS load-bearing on at
least one cohort case" without explaining the mechanism. This post-mortem
runs read-only forensic traces on the single regressing case
`qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` to determine which of three
hypotheses about controller function is supported, before Phase B B.3
proceeds on its parallel branch.

**Scope (read-only)**
- No `aec.py` logic changes.
- No A.0 spec modification.
- No Phase A reopen.
- Phase B branch unaffected.

**Deliverable**: this doc. Hypothesis judgment at §4.

**Artefacts**
- `tools/research/p52_a0_postmortem_trace.py` — Steps 1 & 3 controller + filter trace
- `tools/research/p52_a0_acoustic_props.py` — Step 2 cohort acoustic properties
- `/tmp/p52_a0_pm/{pre,post}.csv` — per-frame trace pre / post A.0 retirement
- `/tmp/p52_a0_pm/diff.json` — pre-vs-post diff summary
- `/tmp/p52_a0_pm/props.csv` — 800-case acoustic properties

---

## Step 1 — Controller fire timeline (pre-A.0)

Captured with controller-active (HEAD `3236f6c` revert applied = production behaviour).

| Fire kind | Count | Frames (hop = 160 samples, 100 fps → time ≈ frame/100 s) |
|---|---:|---|
| `boost_q` (main Q-boost reset trigger) | 22 | early cluster `[274, 284, 294, 324, 334, 344, 354, 398]` (≈ 2.7–4.0 s); late cluster `[2106…2360]` (≈ 21.1–23.6 s) |
| `reverse_copy` (shadow ← main re-sync) | 11 | scattered `[100, 143, 214, 597, 1012, 1239, 1332, 1651, 1700, 1930, 2078]` |
| `main_paused` (main weight update gated off) | 159 frames | first→last span `[274, 2361]`; 5.92 % of total 2 686 frames |

**Two-cluster fire pattern.** The early cluster (2.7–4.0 s) and late cluster
(21.1–23.6 s) both coincide with abrupt mic-coupling changes (see §2). Between
clusters, scattered `reverse_copy` fires — the controller is continuously
detecting that main has overshot relative to shadow and re-syncing the shadow
back to main.

---

## Step 2 — Cohort acoustic outlier dimensions

Acoustic properties computed for all 800 cases (`tools/research/p52_a0_acoustic_props.py`).

### Single-number per-case stats (target vs cohort)

| Metric | Target value | All-800 median (p5 / p95) | Rank | FS-subset median | Reading |
|---|---:|---|---|---|---|
| `erl_db` (mean over far-active) | −2.15 | 1.44 (−14.3 / 33.0) | 271 / 765 | −3.15 | normal |
| `sat_frac` (> 0.95) | 0.000 | 0.000 | 601 / 800 | 0.000 | normal |
| `mic_peak_db` | 0.00 | −3.06 | 800 / 800 (= 231 cases at 0) | −2.38 | clipping, but common in dataset |
| `mic_rms_db` | −29.13 | −24.46 | 211 / 800 | −23.97 | quieter mic than median |
| `lpb_rms_db` | −23.00 | −25.93 | 557 / 800 | −22.74 | normal far level |
| `coh2_voice` | 0.012 | 0.029 (0.003 / 0.101) | 151 / 800 | 0.032 | low coherence (more decorrelation, more reverb / nonlinearity) |
| `far_active_frac` | 0.863 | 0.615 | 540 / 800 | 0.968 | FS subset usually higher; target below FS median |
| `max_internal_far_silence_gap_s` | 0.86 | FS-subset p99 = 3.59 s | 102 / 300 in FS | — | NOT an outlier on simple far-silence |

→ On simple per-case scalars the target is unremarkable. It is **not** the
clipping outlier, **not** the long-far-silence outlier, **not** the worst-ERL
outlier.

### Decile ERL trajectory (the load-bearing finding)

Per-decile ERL (mean over each tenth of the recording, when far is active):

| Decile | Time (s) | mic_rms_dB | lpb_rms_dB | ERL = mic−lpb (dB) |
|---|---|---:|---:|---:|
| 0 | 0.0 – 2.7  | −22.1 | −22.6 | **+0.5** |
| 1 | 2.7 – 5.4  | −81.3 | −27.2 | **−54.1** |
| 2 | 5.4 – 8.1  | −81.7 | −20.9 | **−60.8** |
| 3 | 8.1 –10.7  | −81.7 | −23.6 | **−58.0** |
| 4 |10.7 –13.4  | −23.3 | −22.8 | **−0.5** |
| 5 |13.4 –16.1  | −43.8 | −22.8 | −21.0 |
| 6 |16.1 –18.8  | −61.7 | −26.7 | −35.0 |
| 7 |18.8 –21.5  | −28.6 | −21.0 | **−7.6** |
| 8 |21.5 –24.2  | −66.8 | −23.5 | **−43.3** |
| 9 |24.2 –26.9  | −79.4 | −22.8 | −56.6 |

The mic-to-far ratio **swings by 61 dB across deciles** (decile 0 = +0.5 dB
vs decile 2 = −60.8 dB) while the far level stays steady at −20 to −27 dB.
That is **extreme echo-path nonstationarity** — not a passive recording but
an environment where the coupling collapses and recovers repeatedly.

### Cohort comparison on decile-ERL variability

| Metric | Target | All-800 median (p95 / p99 / max) | Rank |
|---|---:|---|---|
| ERL_decile_std | 23.4 dB | 2.87 (12.99 / 21.04 / 32.03) | **655 / 660 (p99.2)** |
| ERL_decile_peak-to-peak | 61.2 dB | 8.43 (38.30 / 60.09 / 73.85) | **655 / 660 (p99.2)** |

→ **Target is top 0.8 % of the cohort by echo-path nonstationarity.** Two
doubletalk cases and 0I0XMl3M0… (FS_with_movement) outrank it; everything
else is below. Among **FS-only** target ranks **298 / 300 (p99.3)**.

This is the outlier dimension. The case is a near-worst-of-cohort instance
of intra-recording echo-path-change. Standard production gate (cohort-wide
mean) doesn't notice; this single case does.

---

## Step 3 — Main filter behaviour difference (pre vs post A.0)

Same case re-captured after `git checkout eac5325 -- python/aec.py` (the
retired controller from A.0 commit).

### Aggregate ERLE_main delta (post − pre)

| Region | Mean Δ ERLE_main (dB) |
|---|---:|
| All 2 686 frames | **−0.559** (matches the verdict's case-level metric) |
| Fire frames only (33 frames) | **+5.67** (post is *better* exactly when controller fired) |
| Fire ± 50 frame window (~970 frames) | **+1.02** |
| Outside fire-window | **−0.87** |
| Decile 0–7 (first 80 %) | ≈ 0 (controller fires here didn't cost ERLE) |
| Decile 8 (80–90 %) | **−7.13** |
| Decile 9 (90–100 %) | **−10.07** |

### Filter weight L2 norm at fire frames

| Frame | Time (s) | pre W L2 | post W L2 | pre ERLE | post ERLE |
|---|---|---:|---:|---:|---:|
| 274 (first fire) | 2.74 | 1.066 | 1.066 | −33.96 dB | −33.96 dB |
| 2361 (last fire) | 23.61 | **0.000** | **0.903** | 0.00 | −24.55 |

### Worst regressing frames in post-A.0 (sample of top 7)

| Frame | Time (s) | pre ERLE | post ERLE | Δ (dB) | pre W norm | post W norm | within ±50 of fire |
|---|---|---:|---:|---:|---:|---:|:---:|
| 2646 | 26.46 | (large pos) | −27.04 | −27.0 | small | 0.5+ | n |
| 2382 | 23.82 | −5.59 | −26.35 | −20.8 | 0.037 | 0.689 | y |
| 2395 | 23.95 | +1.17 | −19.25 | −20.4 | 0.039 | 0.647 | y |
| 2581 | 25.81 | −7.40 | −27.51 | −20.1 | 0.084 | 0.556 | n |
| 2365 | 23.65 | −0.05 | −19.91 | −19.9 | 0.033 | 0.782 | y |
| 2668 | 26.68 | −7.59 | −27.45 | −19.9 | 0.073 | 0.502 | n |
| 2368 | 23.68 | +0.56 | −19.20 | −19.8 | 0.033 | 0.782 | y |

**Mechanism**

- In **pre-A.0** at frames 2300+: controller pauses main + boosts Q
  repeatedly, **driving W L2 to ~0** (0.03–0.08). Output ≈ mic (filter
  produces ≈ zero echo estimate) → ERLE_main ≈ 0 dB. Marginal, but harmless.
- In **post-A.0** at the same frames: with no controller intervention, main
  filter keeps adapting through the wild deciles 1–9. By frame 2365 W L2 has
  grown to 0.78. But the W is for a *prior* echo path that no longer
  exists; applying it generates a wrong echo estimate larger than the
  current mic itself → error_signal `‖mic − echo_est‖² > ‖mic‖²` → **ERLE_main
  goes negative, hitting −27 dB**. The filter has diverged into harmful
  adaptation.

This is the textbook Yang-2017 / Jung-2011 echo-path-change failure mode
under a slow-adapting recursion. The controller's `boost_q` + `pause_main` +
in-flight `reverse_copy` are the production defence against it.

---

## 4. Hypothesis evaluation

Three hypotheses for what the controller actually does:

### (a) Controller prevents divergence — **SUPPORTED**

**Evidence:**
- Step 3 shows the post-A.0 catastrophe is **divergence**, not slow
  tracking: W L2 grows from 0.03 → 0.78 (filter became *more confident*)
  while ERLE drops from 0 → −27 dB (filter became *more wrong*). Classic
  divergence signature.
- Step 1 timing: 22 fires concentrate exactly at the moments where the
  echo path collapses or recovers (early cluster after Decile 0→1 ERL drop
  of −55 dB; late cluster around Decile 7→8 transition).
- Step 2 confirms the case sits in the top 0.8 % of the cohort on
  decile-ERL variability — exactly the regime where divergence is
  expected and an explicit defence is justified.
- Yang 2017 / Jung 2011 literature names this scenario and prescribes
  exactly this defence (Q-boost or main-pause + shadow re-sync).

### (b) Controller maintains R-freshness — **WEAKLY SUPPORTED, not primary**

**Evidence against being the primary mechanism:**
- Production main R-EMA `_alpha_r = 0.95` ([aec.py:1048](../python/aec.py#L1048))
  is already aggressive (TC ≈ 20 frames ≈ 200 ms). R is *already* fresh
  on the time-scale of the deciles (each decile = 2.7 s = ~270 ms tracking).
- The post-A.0 failure is `W` going wrong (filter coefficients), not `R`
  going stale. Even if R were instantly perfect, with a stale `W` the
  innovation is computed against a wrong echo estimate.
- R freshness would matter for *Kalman gain* on the next innovation, but
  the divergence has already happened in the previous frames' coefficient
  update.

R-freshness is not the load-bearing mechanism on this case. R-reset is the
Phase A.4 main-R-reset hook design proposal — a different intervention
that targets a different failure mode (slow recovery after path change
*with* otherwise-correct W).

### (c) Controller squeezes marginal cohort-wide ERLE — **REFUTED**

**Evidence against:**
- Per-subset mean Δ ERLE_main after retirement: DT +0.002, FS +0.003, NE
  0.000. If the controller squeezed cohort margin, retiring it would lose
  a measurable mean; it does not.
- 799 of 800 cases meet the case-level bar (mean Δ < 0.5 dB). The cohort
  is essentially indifferent to the controller's presence.
- Even on the target case, decile 0–7 mean Δ ≈ 0 — the controller's
  contribution is binary (defend against catastrophe) not gradient
  (small squeeze across many frames).

The controller is a *catastrophe defence on the cohort tail*, not a
cohort-wide gain squeezer.

---

## 5. Judgment

**(a) Divergence defence is supported as the primary mechanism.**

The production ShadowCopyController is a **catastrophe defence for the
top-≤1% of cohort echo-path-nonstationarity cases**. On 799 / 800 cases
its mean impact is < 0.5 dB. On the single outlier (target case, p99.2 on
decile-ERL std) it prevents the main filter from diverging by 20–27 dB
during periods where the echo path is wildly nonstationary, by repeatedly
forcing W back to near-zero so the AEC degrades gracefully to pass-through
rather than catastrophically to anti-correlated output.

P52 v1.1 §2.6 framed A.0 fail as "the v1.0 design premise that shadow is
just an information source is invalid on this cohort." This post-mortem
sharpens that into a specific, mechanistic statement:

> The controller's `boost_q` + `pause_main` are doing the same job
> P52's planned `PathChangeDetector` + `reset_observation_noise()`
> would do (Task A.4 / §2.5.2) — but the controller acts on the **coefficients**
> (forcing W → 0), whereas the planned mechanism acts only on the **R-EMA**
> (forcing high Kalman gain). The latter does not converge to the same
> defence: a high Kalman gain on a wildly nonstationary path drives **fast
> mis-adaptation**, not robustness. Yang 2017's fast-recovery prescription
> works because in Yang's setup the *new* path is stable after the change
> — high gain helps converge. Our outlier case has *no* stable post-change
> path; it cycles between coupling regimes. High gain there is harmful.

This is a non-trivial finding. It is consistent with everything in the
prior P-chain about why every "make the linear filter adapt faster" arc
failed: the cohort tail is not a slow-tracker problem; it's a never-
stationary-target problem, and the only defence is *degrade to pass-through*.

---

## 6. Implications (informational only — not decisions)

The post-mortem provides evidence that supports, but does not by itself
authorise, the following:

1. P52 §2.5.2 design (R-reset only, no W intervention) would **not** have
   addressed the target case even if Phase A had passed A.0. The R-reset
   targets a different failure mode (slow recovery after a single clean
   path change). The cohort tail's failure mode is unbounded mis-adaptation
   on a never-converging path.
2. The cohort's "tail dimension" is decile-ERL variability, not
   far-silence-gap or clipping. Any future arc that wants to address this
   tail should design the bench-filter against `ERL_decile_std` outliers,
   not just cohort mean.
3. The 5.92 % `main_paused` footprint is concentrated in a tiny fraction
   of cases. Most production cases never see the controller fire. Any
   architectural change here is a cohort-tail intervention, not a
   cohort-wide improvement.
4. Per the user's constraint, **this doc does not reopen Phase A or propose
   a new design**. The judgment is filed; next steps wait for user direction.

---

## 7. Cross-references

- Verdict: [research_log_p52_task_a0_verdict.md](research_log_p52_task_a0_verdict.md)
- Design lock: [p52_design_lock_v1.1.md §2.5, §2.6, §I3](p52_design_lock_v1.1.md)
- Anomaly notes: [phase_a_anomaly_notes.md](phase_a_anomaly_notes.md)
- Trace tools:
  - `tools/research/p52_a0_postmortem_trace.py` (capture + diff)
  - `tools/research/p52_a0_acoustic_props.py` (cohort metrics)
- Snapshot artefacts: `/tmp/p52_a0_pm/*.csv`, `/tmp/p52_a0_pm/diff.json`
- Cohort source: `wav/aec_challenge_blind/` (800-case AEC challenge)
- Bench standard: `feedback_bench_j4` (preset=balanced, fl=832, cng=True)
