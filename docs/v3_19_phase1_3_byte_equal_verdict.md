# v3.19 Phase 1.3 — implementation + 5-case byte-equal flag-OFF verdict

**Status**: PASS 2026-05-16. 5/5 md5 match between post-patch (flags
all OFF) and pre-patch (no flags) baseline.

## 1. Implementation summary

Patches per [docs/v3_19_phase1_2_per_branch_flag_design.md](v3_19_phase1_2_per_branch_flag_design.md):

### AecConfig — 3 new fields
[python/aec.py:283-298](../python/aec.py#L283):
```python
c_e_branch_force_render_use_fq_usable: bool = False
c_e_branch_dt_per_bin_use_fq_usable: bool = False
c_e_branch_coh2_ema_use_fq_usable: bool = False
```

### Branch R1 — `ResidualEchoEstimator.attribute_legacy` force_render
[python/aec.py:5162-5174](../python/aec.py#L5162):
- Added `aec_state=None` param to `attribute_legacy` signature
- Added pass-through in `attribute()` dispatcher (line 5066)
- Added `_c_e_branch_force_render_use_fq_usable` instance attr to
  `ResidualEchoEstimator.__init__` (line 5060)
- Wired from AecConfig at AEC.__init__ (line 6515)
- Patch shape: 3-guard (flag, aec_state, back_ref) → fq_usable() override

### Branch G1 — `ResFilter._stage_gain_compute` F3.1 v3 dt_per_bin
[python/aec.py:3318-3333](../python/aec.py#L3318):
- Added `aec_state=None` param to `_stage_gain_compute` signature
  (line 3225)
- Updated caller in `process()` to pass `aec_state=aec_state`
  (line 4274)
- Added `_c_e_branch_dt_per_bin_use_fq_usable` instance attr to
  ResFilter `__init__` (line 2433)
- Plumbed via ResFilter init param `c_e_branch_dt_per_bin_use_fq_usable`
- Patch shape: 3-guard → fq_usable() override on F3.1 v3 gate

### Branch P1 — `ResFilter.process()` coh2 EMA asymmetric
[python/aec.py:4067-4080](../python/aec.py#L4067):
- aec_state already in scope (process() param)
- Added `_c_e_branch_coh2_ema_use_fq_usable` instance attr (line 2434)
- Plumbed via ResFilter init param `c_e_branch_coh2_ema_use_fq_usable`
- Patch shape: 3-guard → fq_usable() override on `if filter_converged`

### eval_aec_challenge.py env-var bridges
[python/eval_aec_challenge.py:344-347](../python/eval_aec_challenge.py#L344):
```python
('AEC_C_E_BRANCH_FORCE_RENDER_FQ_USABLE', 'c_e_branch_force_render_use_fq_usable', True),
('AEC_C_E_BRANCH_DT_PER_BIN_FQ_USABLE', 'c_e_branch_dt_per_bin_use_fq_usable', True),
('AEC_C_E_BRANCH_COH2_EMA_FQ_USABLE', 'c_e_branch_coh2_ema_use_fq_usable', True),
```

## 2. Byte-equal verification

Method: render 5 stems (first 5 from `tools/research/v3_15_subset_cases.txt`)
with both:
- **Baseline** (pre-patch, `git stash` of post-patch changes — equivalent
  to v3.18 production HEAD `8c3b468`)
- **Post-patch + all flags OFF** (default config, no env-var overrides)

Compute md5 of each output WAV; expect identical hashes (5/5).

### Run

```bash
mkdir -p /tmp/v3_19_post_patch /tmp/v3_19_baseline
# (1) Render with patches in working tree, all flags False default
python3 /tmp/v3_19_byte_equal_smoke.py /tmp/v3_19_post_patch
# (2) git stash → render baseline
git stash push -m "phase 1.3 byte-equal" python/aec.py python/eval_aec_challenge.py
python3 /tmp/v3_19_byte_equal_smoke.py /tmp/v3_19_baseline
git stash pop
```

### Result (5/5 md5 match)

| Stem | md5 (post-patch) | md5 (baseline) | match |
|---|---|---|---|
| qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk | bc730e413da2069eb9ce80ddcd6b3a26 | bc730e413da2069eb9ce80ddcd6b3a26 | ✓ |
| 0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement | 34993c08b21c9cd70945826bdf7ac74e | 34993c08b21c9cd70945826bdf7ac74e | ✓ |
| 3UAwzzOa40aCXQAmEdpwww_farend_singletalk_with_movement | 3b24a409fa5ee1429974c5db26c0a044 | 3b24a409fa5ee1429974c5db26c0a044 | ✓ |
| XqvGR01tJkan17zltLs38Q_doubletalk_with_movement | 8afa53736717e6d3c806bae189824c6c | 8afa53736717e6d3c806bae189824c6c | ✓ |
| Hp5g1asacUCt5rJVLO1FuQ_doubletalk_with_movement | 7b5ee9b407d060cac08c2d8623ad8adf | 7b5ee9b407d060cac08c2d8623ad8adf | ✓ |

**Verdict**: byte-equal flag-OFF PASS 5/5.

## 3. Wiring sanity check

```python
$ python3 -c "..."
R1 wired on res_est: True
G1 wired on ResFilter: True
P1 wired on ResFilter: True
```

All 3 flags propagate to instance attributes when set ON in
AecConfig.

## 4. Disposition

- Hard bar PASS — proceed to Phase 1.4 single-branch sweep
- 1.4 sweep order per Phase 1.1b §4.3 hypothesis: G1 first, P1
  second, R1 last
- 60-case AECMOS rendering uses
  `tools/research/v3_15_subset_bench.sh` infrastructure (per HK.1
  KEEP list)

## 5. Cross-references

- [docs/v3_19_phase1_2_per_branch_flag_design.md](v3_19_phase1_2_per_branch_flag_design.md) — design lock
- [docs/v3_19_phase1_1a_resfilter_branch_inventory.md](v3_19_phase1_1a_resfilter_branch_inventory.md) — branch enumeration
- [docs/v3_19_phase1_1b_aec3_suppressiongain_annotation.md](v3_19_phase1_1b_aec3_suppressiongain_annotation.md) — AEC3 mapping
- [tools/research/v3_15_subset_cases.txt](../tools/research/v3_15_subset_cases.txt) — 60-case subset (first 5 used here)
- `/tmp/v3_19_byte_equal_smoke.py` — verification script (one-off)
