# v3.15 §1.7 — RES audit and refactor plan

**Status**: SKELETON — sections in place; findings + candidates marked
TODO until [`tools/research/v3_15_res_audit.py`](../tools/research/v3_15_res_audit.py)
has been run on the post-§1.2 / §1.3-deferred substrate.

**Date**: 2026-05-15 (skeleton authored)
**Branch**: `feature/v3.15-arc-t` (worktree `v3-15-arc-t`)
**Author**: v3.15 Phase F closeout
**Doc role**: Gates a separate v3.16 RES refactor arc. Hard bar §6.

---

## 1. Header / scope

- §1.7 purpose: comprehensive RES health audit on the post-§1.2 / §1.3
  substrate.
- §1.2 status: closed CANNOT SHIP (placeholder — link verdict).
- §1.3 status: deferred (placeholder — link decision).
- Substrate baseline for diff: v3.13 Phase 3 verdict
  ([docs/v3_13_phase3_res_audit_verdict.md](v3_13_phase3_res_audit_verdict.md)).
  Substrate has changed since v3.13: Arc P, Arc R, and Arc S-orth.A
  landed on `main`. This audit measures whether the 5 RES floor paths
  shifted in fire-rate as a result.
- Code changes inside §1.7: ZERO by default (audit-only). Delete-only
  candidates (e.g. `epc_dt_cap` if still 0/800) ship as standalone
  byte-equal-verified commits inside §1.7.

---

## 2. Audit method

- Script: [`tools/research/v3_15_res_audit.py`](../tools/research/v3_15_res_audit.py).
- Reuses the v3.13 Phase 3 fire-rate-counting pattern
  ([docs/v3_13_phase3_res_audit_verdict.md §Method](v3_13_phase3_res_audit_verdict.md))
  but **does not** require a custom `_diag_floor_fires` aec.py diff —
  the existing `AecConfig.capture_stages=True` substrate
  ([python/aec.py:286](../python/aec.py#L286), `ResFilter.get_stage_gains()`
  at [python/aec.py:2521](../python/aec.py#L2521)) already exposes
  per-bin gain vectors after each of the 5 floor sites. Per-frame
  fire-rate = `np.any(post − pre > 1e-7)`.
- `ne_g_floor` is folded into `spectral_g_min` *before* stage 02 so it
  does not surface as a separate stage; it is inferred from the cached
  scalars `ResFilter._stats_last_ne_g_floor` and
  `_stats_last_spectral_g_min` (frame-level binary, matches v3.13 verdict
  semantic).
- Bench config: BALANCED / fl=832 / cng=True / seed=0 / j=4 — standard
  800-case AEC Challenge corpus per [CLAUDE.md](../CLAUDE.md) "Standard
  800-case config".
- Per-bucket aggregator: FS_static / FS_movement / DT_static / DT_movement
  / NE / cohort_tail / GLOBAL. Bucket assignment via stem suffix +
  `qNvSMyU…` cohort-tail anchor (per v3.13 + p52 docs).
- Outputs: `/tmp/v3_15_res_audit/audit.json` (full per-bucket JSON),
  `summary.csv` (fire-rate + Δ-vs-v3.13), per-case JSONs.
- Diag-hook requirement: NONE — `capture_stages` substrate is sufficient.
  No `aec.py` modification needed for the 5 paths covered. **OPEN**: if
  a future audit wants to instrument the 4-cap chain (Stage 1 v3.13
  `s8_cap*_binding` counters), it can reuse the existing
  `enable_audit_counters()` substrate
  ([python/aec.py:2378](../python/aec.py#L2378)).

### 2.1 v3.13 baseline numbers (for diff column)

Reference values copied from v3.13 verdict §"Per-path fire rate by
bucket" — used as the diff baseline in `summary.csv`:

| Path | FS_static | FS_movement | DT_static | DT_movement | NE | cohort_tail (qNvSMyU) |
|---|---:|---:|---:|---:|---:|---:|
| spectral_floor | 0.894 | 0.881 | 0.524 | 0.529 | 0.097 | 0.974 |
| ne_g_floor    | 0.880 | 0.867 | 0.934 | 0.933 | 0.999 | 0.750 |
| epc_dt_cap    | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| quiet_mask    | 0.509 | 0.495 | 0.298 | 0.307 | 0.060 | 0.679 |
| hf_cap        | (n/a) | (n/a) | (n/a) | (n/a) | (n/a) | (n/a) |

`hf_cap` was not measured discretely in v3.13 (rolled into general
post-process); §1.7 audit establishes its first-pass baseline.

---

## 3. Findings

> **TODO** — fill in after running `v3_15_res_audit.py` on full 800
> cases. Report (a) post-§1.2/§1.3 fire-rate per bucket per path,
> (b) diff vs v3.13 baseline (shifted? unchanged?), (c) load-bearing
> vs dead determination per path.

### 3.1 `spectral_floor`

- **TODO** Post-§1.2/§1.3 fire-rate (per bucket): …
- **TODO** Diff vs v3.13 baseline: …
- **TODO** Cohort tail (qNvSMyU): … (v3.13 baseline 97.4 %).
- **TODO** Determination: load-bearing (KEEP) / dead (REMOVE) / candidate
  for unification.

### 3.2 `ne_g_floor`

- **TODO** Post-§1.2/§1.3 fire-rate (per bucket): …
- **TODO** Diff vs v3.13 baseline: …
- **TODO** Skew (max-min mean rate): … (v3.13 baseline 0.13 → universal floor).
- **TODO** Determination: …

### 3.3 `epc_dt_cap`

- **TODO** Post-§1.2/§1.3 fire-rate (per bucket): …
- **TODO** v3.13 baseline 0/800. If still 0/800 → ship dead-code removal
  inside §1.7 as standalone byte-equal-verified commit.
- **TODO** If non-zero on this substrate (e.g. §1.2 changed `epc_dt`
  trigger or stage-02 floor characteristic) → flag as "newly live" and
  document mechanism.

### 3.4 `quiet_mask`

- **TODO** Post-§1.2/§1.3 fire-rate (per bucket): …
- **TODO** Diff vs v3.13 baseline: …
- **TODO** Determination: physical noise gate (KEEP) / candidate.

### 3.5 `hf_cap`

- **TODO** Post-§1.2/§1.3 fire-rate per bucket: … (no v3.13 baseline).
- **TODO** Sub-mode breakdown: which of the three branches
  (conditional / plan_a_2k / v3.8.3-strict) is hot in BALANCED?
- **TODO** Determination: …

### 3.6 Cross-path observations

- **TODO** Substrate shift (Arc P + R + S-orth.A landed) — did any path's
  bucket skew flip relative to v3.13?
- **TODO** Cohort-tail (qNvSMyU) integrity — does spectral_floor still
  approach 97 %? If so, reaffirm load-bearing for cohort-tail catastrophe
  defence.
- **TODO** Frame totals per bucket — sanity that 800-case run completed
  without case-skew bias.

---

## 4. Refactor candidates

> **TODO** — populate ≥ 5 candidates ranked by
> `predicted AECMOS Δ × implementation cost × cohort-tail risk`
> after findings (§3) are filled in.

Each candidate must have a measurable success criterion (per hard bar
§6). Template:

```
### Candidate <ID>: <name>
- Mechanism: <what code path changes, where>
- Predicted AECMOS Δ: <+0.0xx (mean) / +0.0xx (worst)>
- Implementation LOE: <S / M / L>  (Small ≤ 1 sprint; Medium ≤ 3; Large > 3)
- Cohort-tail risk: <none / low / medium / high>
- Measurable success criterion: <e.g. "DT_movement Δdeg ≥ -0.005 AND
  cohort_tail Δecho ≥ 0">
- Predicted byte-equal cost: <byte-equal | minor drift | new bench>
- v3.13 / v3.12 prior art reference: <links>
```

### Candidate C1: `epc_dt_cap` dead-code removal

- **Mechanism**: delete `if epc_dt: g = min(g, EPC_DT_GAIN_CAP)` block
  + `_stage_gains['03_epc_dt_cap']` capture wrapper at
  [python/aec.py:3199-3204](../python/aec.py#L3199).
- **Predicted AECMOS Δ**: 0.000 (byte-equal — fire-rate 0 in v3.13).
- **Implementation LOE**: S (one commit, ≤ 20 LOC delete + capture-key
  rename in `get_stage_gains()` docstring).
- **Cohort-tail risk**: none (0/800 in v3.13).
- **Measurable success**: 800-case sample-level byte-equal pass
  (`atol=1e-6, rtol=1e-5`).
- **Notes**: ships INSIDE §1.7 as a delete-only commit if §3.3 confirms
  fire-rate still 0 on the v3.15 substrate.

### Candidate C2: TODO (e.g. `ne_g_floor` unification into canonical floor)

- **TODO**

### Candidate C3: TODO (e.g. `hf_cap` sub-mode dead-branch removal)

- **TODO**

### Candidate C4: TODO (e.g. `spectral_floor` cohort-tail-only specialisation)

- **TODO**

### Candidate C5: TODO

- **TODO**

---

## 5. Recommendation — v3.16 arc authorisation

> **TODO** — depends on §4 candidate ranking.

Decision matrix template:

| Candidates with predicted Δ ≥ +0.005 | Recommendation |
|---|---|
| ≥ 3 | Authorise v3.16 RES refactor arc; promote top-3 as Phase 1 sprints. |
| 1 – 2 | Ship the 1 – 2 as standalone v3.15.x commits; **do not** open v3.16. |
| 0 (only delete-only / cosmetic) | **Declare RES architecture stable**; close §1.7; no v3.16. |

- Top-1 candidate (highest predicted Δ): **TODO**
- Top-2: **TODO**
- Top-3: **TODO**
- Cohort-tail risk register (high-risk candidates → require explicit
  qNvSMyU regression check before authorisation): **TODO**

---

## 6. Hard-bar status

Per request:

> Hard bar (doc acceptance): plan doc must rank ≥ 5 refactor candidates
> by (predicted AECMOS Δ × implementation cost × cohort-tail risk); each
> candidate must have a measurable success criterion. If audit finds
> < 3 candidates with predicted AECMOS Δ ≥ +0.005 → declare RES
> architecture stable, no v3.16 refactor.

| Bar | Status |
|---|---|
| ≥ 5 refactor candidates listed | **TODO** (skeleton has 1; need 4 more after audit) |
| Each candidate has measurable success criterion | **TODO** (template in §4) |
| ≥ 3 candidates with predicted Δ ≥ +0.005 → authorise v3.16 | **TODO** |
| Else → declare RES architecture stable, close §1.7, no v3.16 | **TODO** |
| Delete-only candidates (e.g. C1 `epc_dt_cap`) ship inside §1.7 | **TODO** (gated on §3.3 confirming fire-rate still 0) |

---

## 7. Artifacts

- This plan: `docs/v3_15_res_audit_and_refactor_plan.md`
- Audit script: `tools/research/v3_15_res_audit.py`
- v3.13 baseline: [`docs/v3_13_phase3_res_audit_verdict.md`](v3_13_phase3_res_audit_verdict.md)
- v3.12 predecessor (static categorization): `docs/v3_12_res_audit.md`
- Audit raw outputs (gitignored): `/tmp/v3_15_res_audit/audit.json`,
  `summary.csv`, `per_case/<stem>.json`.
- §1.2 verdict (CANNOT SHIP): **TODO** link.
- §1.3 deferred decision: **TODO** link.
