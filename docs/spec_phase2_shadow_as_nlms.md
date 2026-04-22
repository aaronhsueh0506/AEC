# Spec: Phase 2 — Shadow as NLMS (PBFDAF) in PBFDKF Mode

Status: DRAFT (Stage 1A) — awaiting user audit before Stage 1B implementation.

Feature flag: `AEC_EXP_SHADOW_NLMS` (default `'0'` — OFF).
Branch: `feature/phase2-shadow-as-nlms`.
Related: [B-16 spec](spec_b16_raw_dt_jump_veto.md), [AEC3 architecture analysis](aec3_full_architecture_analysis.md), [B-16 Stage 1D results](b16_stage_1d_results.md).

---

## §1 Background

### §1.1 Phase 1 (B-16) 成果與局限

Phase 1 B-16 (raw_dt jump veto) 已 merged (commit `8d9bd91`, flag-gated default OFF). B-16 在 wav-level 對 PZ7V FS#84 證實有 leak 下降 (−6.59 dB on onset window, Stage 1D), 但 800-case AECMOS NEUTRAL (見 [b16_stage_1d_results.md](b16_stage_1d_results.md))。

B-16 是 **defensive symptom-level veto** — 攔截 raw_dt > 0.6 的 2-frame sustained jump, 用 ~500 ms EMA 取代。它**沒有修**底層 architectural issue:
- 為什麼 shadow-filter 在 echo path gain jump 當下沒觸發 EPC large / copy gate?
- 為什麼 shadow_advantage 永遠 ≈ 1.0?

### §1.2 Root cause — Kalman self-normalization

[aec3_full_architecture_analysis.md §6.1](aec3_full_architecture_analysis.md) 指出 gap #2: AEC3 mainline 是 **refined (Kalman) + coarse (NLMS)** dual-filter。本 repo 在 PBFDKF mode 下 shadow 也用 PBFDKF (Kalman) — 兩個 Kalman filter 被 Kalman gain `K = P|X|² / (P|X|² + R)` 的 saturation 行為鎖在一起。當 `P|X|² ≫ R` (echo event), `K → 1.0`, 兩 filter 吃相同 gradient, shadow_err ≈ main_err, `shadow_advantage ≈ 1.0` forever。

### §1.3 Stage 0a + 0b validation

**Stage 0a (code survey)** 結論 GO (conditional):
- PBFDAF / PBFDKF API 完全兼容 (`__init__` 簽章相同, `W / X_buf / process / reset / copy_weights_from` 共用形態)
- `copy_weights_from` 已以 `hasattr` 跨類防禦
- 9 個 shadow touch points 中 8 個已 polymorphism-safe；僅 L2549 需 FilterClass swap
- 注意: L2904–2905 `_external_saturation_level` **已 isinstance-gated** (post-compact review 確認), **無需**新增 gate

**Stage 0b (live probe D1)** 結論 GO (strong), 量到的 PZ7V FS#84 onset window shadow_advantage:

| 指標 | Baseline (PBFDKF shadow) | Mock (PBFDAF shadow) |
|---|---|---|
| onset-window max | **1.053** | **3.012** |
| onset-window mean | 1.006 | 1.871 |
| first ≥ 1.3 | NEVER | onset +4 frames |
| first ≥ 2.0 | NEVER | onset +16 frames |
| peak Δ (mock − base) | — | **+1.96** |

Baseline main_err 與 shadow_err 於 onset 後 bit-identical (1.288e+3 vs 1.295e+3, ratio 0.995) — K saturation 把兩 filter 鎖成同一個 tracker。Mock PBFDAF shadow (`leak=1e-6, alpha_r_scale=1.03`) 打破鎖, 於 +4 frames 起 shadow_err 從 main_err 分離 15–55%。

Evidence files:
- `/tmp/d1_baseline_pz7v.log`, `/tmp/d1_mock_pz7v.log`, `/tmp/analyze_d1.py`
- probe branch `probe/phase2-d1-shadow-nlms` commit `3682f98`

### §1.4 為什麼現在做

- B-16 把 immediate PZ7V hazard 壓住, 但底層 architecture 不對等於繼續累積 latent bugs
- Stage 0b 的 Δ +1.96 數字級證實 hypothesis, 風險可量化
- 2 hunks (~10 行), flag-gated default OFF, 回滾成本近零

---

## §2 Goal

### §2.1 In scope

PBFDKF mode 下 (main = Kalman), shadow filter 改用 PBFDAF (NLMS):
1. 打破 Kalman self-normalization 造成的 shadow_adv ≈ 1.0 saturation
2. Revive EPC large trigger (依賴 `shadow_adv > epc_large_shadow_adv = 2.0` 或 small `> 1.3`)
3. Revive copy gate 雙向 (shadow_err < main_err × 0.65 有機會觸發)
4. 對齊 AEC3 mainline: refined=Kalman / coarse=NLMS

### §2.2 Not in scope

- **單向 copy (AEC3 refined→coarse only)**: 保留雙向 copy, Stage 2 再驗
- **shadow_q_ratio tune**: PBFDAF shadow 下自然 no-op via L2552 `isinstance` gate；preset 數值作 fallback 保留
- **B-16 移除**: B-16 是正交 defensive layer (symptom-level, 即使 shadow-NLMS 生效仍有價值), 保留
- **PBFDAF mode 的 shadow 行為**: 本 spec 不動 (PBFDAF main 已用 PBFDAF shadow)
- **EPC / copy-gate threshold tuning**: PBFDAF shadow 下 advantage 幅度遠大, 可能需要 tune, 但本 spec **不改 threshold**, 先觀察原 threshold 行為
- **Cross-algo config flag (AecConfig field)**: 採 Stage 0a Task C "策略 C" (env flag only, no dataclass change)

---

## §3 Design

### §3.1 Feature flag

位置: aec.py 頂部常數, 緊鄰 `AEC_FIX_B16`:

```python
# Phase 2: force shadow filter to PBFDAF (NLMS) when main is PBFDKF.
# Breaks Kalman K saturation that pins shadow_advantage ≈ 1.0 during
# echo path gain jumps. See docs/spec_phase2_shadow_as_nlms.md.
AEC_EXP_SHADOW_NLMS = int(os.environ.get('AEC_EXP_SHADOW_NLMS', '0'))
```

Flag 命名前綴 `AEC_EXP_`: "experimental architectural swap", 區別於 `AEC_FIX_` (bug-fix veto class)。

### §3.2 Shadow class swap (L2549 ±)

現行 [aec.py:2541-2549](../python/aec.py#L2541-L2549):

```python
if FilterClass is PBFDAF:
    # PBFDAF has no Q (Kalman shadow uses shadow_q_ratio for
    # convergence-speed differentiation). For PBFDAF we differentiate
    # by making R lag and weights leak slower ...
    shadow_kwargs['alpha_r_scale'] = 1.03
    shadow_kwargs['leak'] = 1e-6
self.shadow_filter = FilterClass(**shadow_kwargs)
```

改為 (**選項 A — pure add, 不動既有 branch**):

```python
if FilterClass is PBFDAF:
    # (既有 block, 完全不動 — 保 bit-exact invariant §5.1)
    shadow_kwargs['alpha_r_scale'] = 1.03
    shadow_kwargs['leak'] = 1e-6

# PHASE 2 START: shadow-as-NLMS for PBFDKF mode
ShadowClass = FilterClass
if AEC_EXP_SHADOW_NLMS and isinstance(self.filter, PBFDKF):
    ShadowClass = PBFDAF
    shadow_kwargs['alpha_r_scale'] = 1.03
    shadow_kwargs['leak'] = 1e-6
# PHASE 2 END

self.shadow_filter = ShadowClass(**shadow_kwargs)
```

差異:
- 既有 `if FilterClass is PBFDAF:` block **一字不改** (invariant §5.1 friendly)
- 新增 `# PHASE 2 START ... END` 區塊, 內部 `alpha_r_scale/leak` 設定與既有 block 數值相同 (harmless redundancy)
- `self.shadow_filter = ShadowClass(**shadow_kwargs)` — 最後一行 `FilterClass` 改成 `ShadowClass` (唯一必改既有行)

選項評估 (見 audit fix 1):
- 選項 A (採用): 既有 branch zero-diff, flag OFF 時 `ShadowClass == FilterClass`, 進既有 L2549 等價 assign 語意
- 選項 B (refactor 合併): 乾淨但觸更多行, invariant §5.1 grep 會顯示既有 block 被動
- 選項 C (post-construct mutate): 不可行, __init__ 已 consume kwargs

**Invariant §5.1 grep check 更新** (見 §5.1):
```bash
git diff main..feature -- python/aec.py | \
  grep -B1 -A1 "^[-+]" | grep "if FilterClass is PBFDAF"
# expected: zero matches (既有 PBFDAF branch 一字不動)
```

### §3.3 L2552-2555 Q 設定 (無需改)

現行:
```python
if isinstance(self.shadow_filter, PBFDKF):
    self.shadow_filter.Q_high = self.filter.Q_high * self.config.shadow_q_ratio
    ...
```

`isinstance(..., PBFDKF)` gate 已在; flag ON 時 `shadow_filter` 是 PBFDAF → 自動跳過。shadow_q_ratio config 保留, no-op 於 PBFDAF shadow path。

### §3.4 L2904-2905 `_external_saturation_level` (無需改)

現行 [aec.py:2904-2905](../python/aec.py#L2904-L2905):
```python
if self.shadow_filter is not None and isinstance(self.shadow_filter, PBFDKF):
    self.shadow_filter._external_saturation_level = self._saturation_level
```

已 isinstance-gated (Phase 1 維護狀態)。flag ON 時自動跳過, 因為 PBFDAF shadow 無 B-7 freeze 路徑。**Stage 0a 初版報告誤判**為需加 gate, Stage 0b probe 實際驗證時確認已 gated, 此 spec 修正。

### §3.5 Touch point summary

| 行 | 動作 | 是否本 spec 改動 |
|---|---|---|
| §3.1 頂部常數 | 新增 `AEC_EXP_SHADOW_NLMS` flag | ✅ +1 hunk, ~4 lines |
| L2541-2549 | ShadowClass swap + condition rewrite | ✅ +1 hunk, ~6 lines net |
| L2552-2555 Q 設定 | 已 isinstance-gated | ❌ 無需改 |
| L2904-2905 saturation | 已 isinstance-gated | ❌ 無需改 |
| L2679 / L2932 reset | 多型 | ❌ |
| L3094 process / L3097 get_error_energy | 繼承 / 共簽章 | ❌ |
| L3148 / L3154 copy_weights_from | hasattr 跨類防禦 | ❌ |

**Total: 2 hunks, ~10 lines net change.**

---

## §4 Expected behavior

### §4.1 PBFDKF shadow (flag=0, baseline)

完全不變。Bit-exact Phase 1 merged main (`8d9bd91`)。§5 invariant 1 強制。

### §4.2 PBFDAF shadow (flag=1, experiment) — Stage 0b 實測

PZ7V FS#84 onset (t≈9.95 s):

| phase | frames from onset | shadow_adv (mock) | baseline 對比 |
|---|---|---|---|
| pre-onset stable | < 0 | 0.88–1.12 | 0.95–1.06 (近似) |
| onset (frame 0) | 0 | 0.99 | 0.99 (同步 reset) |
| early adapt | +2 to +10 | 1.12–1.52 | 0.99–1.00 (baseline 平) |
| mid adapt | +10 to +16 | 1.53–2.04 | 1.01–1.02 |
| diverged | +16 to +33 | 2.04–3.01 | 1.02–1.03 |
| post-adapt decay | +33+ | → 2.20 asymptote | → 1.022 asymptote |

### §4.3 Downstream effect chain

Shadow responsive unlocks 兩條 chain:

**Chain A — EPC large trigger** ([aec.py:3172+](../python/aec.py#L3172) 附近):
```
shadow_adv > 2.0 (epc_large_shadow_adv) AND
dt_signal < 0.3 AND sat_safe AND non-stationary
→ EPC large: Q boost, P stretch, R reset, RES forced
→ filter 快速 re-learn
```
Baseline: shadow_adv < 1.3 forever → EPC large 不觸發。
Phase 2: shadow_adv 於 +16 frames 破 2.0 → EPC large **可能**觸發 (視其他 guard)。

**Chain B — Copy gate shadow→main** ([aec.py:3131+](../python/aec.py#L3131)):
```
shadow_err < main_err × 0.65 AND hysteresis ≥ 3 AND streak ≥ 10
→ filter.copy_weights_from(shadow_filter)
```
Baseline: shadow_err ≈ main_err → 不觸發。
Phase 2: onset +4 frames 起 shadow_err ≤ 0.85 × main_err, peak 達 0.40 × main_err → streak 可累積 → copy 可能觸發。

### §4.4 Non-obvious bonus

Phase 2 的價值 **不只是 shadow filter 本身**, 是 revive **整個 shadow-based detection subsystem** (EPC large chain + copy gate chain + 所有依賴 `shadow_advantage / dt_from_shadow` 的 downstream guards)。Stage 1D matrix 會量化這個 downstream amplification。

---

## §5 Invariants (mandatory)

每條附 grep-verifiable 判準。

### §5.1 Flag OFF = bit-exact baseline

**Rule**: `AEC_EXP_SHADOW_NLMS=0` 時, aec.py 的所有 runtime call path 與 Phase 1 merged main (commit `8d9bd91`) 一致。

**Grep verify**:
```bash
git diff main..feature/phase2-shadow-as-nlms -- python/aec.py | \
  grep -E "^[+-]" | grep -v "^[+-]{3}" | wc -l
# expected: ≤ 15

# Option A structural: 既有 "if FilterClass is PBFDAF:" branch 一字不動
git diff main..feature/phase2-shadow-as-nlms -- python/aec.py | \
  grep -E "^[-+].*if FilterClass is PBFDAF"
# expected: zero lines

# FilterClass → ShadowClass 只在最後 assign 行 flipped
git diff main..feature/phase2-shadow-as-nlms -- python/aec.py | \
  grep -E "^-.*self\.shadow_filter = FilterClass"
# expected: exactly 1 line (the single flipped assign)
```

所有 runtime 改動必須包在 `if AEC_EXP_SHADOW_NLMS and ...:` 或同等 condition 內。Flag=0 時 `ShadowClass == FilterClass`, §3.2 最後行 `ShadowClass(**shadow_kwargs)` 與原 `FilterClass(**shadow_kwargs)` 等價。

### §5.2 PBFDAF mode unaffected

**Rule**: main = PBFDAF 時, shadow 仍走 PBFDAF 原 path。不因 flag=1 而改變。

**Grep verify**:
```bash
grep -n "AEC_EXP_SHADOW_NLMS and FilterClass is PBFDKF" python/aec.py
# expected: exactly 1 match (§3.2 swap gate)
```

`FilterClass is PBFDKF` 是必要 gate。main = PBFDAF 時條件 False, ShadowClass 保持 PBFDAF (symmetric baseline)。

### §5.3 Cross-class copy via hasattr (no new function)

**Rule**: 不新增 cross-class copy 邏輯。完全沿用 existing [aec.py:782-789](../python/aec.py#L782-L789) 的 PBFDAF `copy_weights_from` hasattr 防禦與 [aec.py:980-986](../python/aec.py#L980-L986) PBFDKF 的 W-only copy。

**Grep verify**:
```bash
git diff main..feature/phase2-shadow-as-nlms -- python/aec.py | \
  grep -E "^\+.*def copy_weights_from"
# expected: zero matches
```

### §5.4 Config API unchanged

**Rule**: AecConfig dataclass 不新增/刪除/改 default 任何欄位。`shadow_q_ratio` 保留, PBFDAF shadow 下透過 L2552 `isinstance` gate 自然 no-op。Preset defaults 不改。

**Grep verify**:
```bash
git diff main..feature/phase2-shadow-as-nlms -- python/aec.py | \
  grep -E "^\+.*shadow_(mu_ratio|copy_threshold|err_alpha|mu_min|copy_hysteresis|q_ratio|dtd_advantage_scale|dtd_offset)"
# expected: zero matches
```

### §5.5 Orthogonal to B-16

**Rule**: `AEC_FIX_B16` 與 `AEC_EXP_SHADOW_NLMS` 可獨立 on/off。Code 中兩 flag 不在同一 `if` condition, 不 cross-reference。

**Grep verify**:
```bash
grep -n "AEC_FIX_B16" python/aec.py | grep "AEC_EXP_SHADOW_NLMS"
grep -n "AEC_EXP_SHADOW_NLMS" python/aec.py | grep "AEC_FIX_B16"
# expected: zero matches on both
```

兩 flag 各自獨立 `if`, 測試 matrix §7.3 才檢視 interaction。

---

## §6 Smoke test (Stage 1C)

8 cases × 4 flag combinations (2×2 matrix)。

### §6.1 Cases

| Case | 類型 | 來源 | 目的 |
|---|---|---|---|
| `PZ7V0SfxUkem4IalTp1YgA` | FS onset | Phase 1 target | Phase 2 primary |
| `nyT6...` (B-16 Task 5 #1) | DT no-movement | B-16 Stage 1C | DT false-trigger guard |
| `W0zK3d...` (B-16 Task 5 #2) | DT no-movement | B-16 Stage 1C | 同上 |
| `Tgtk...` (B-16 Task 5 #3) | DT no-movement | B-16 Stage 1C | 同上 |
| `XTqo...` (B-16 Task 5 #6 with_movement) | DT movement | B-16 Stage 1C 邊界 | Movement × DT 穩定性 |
| `HIMq...` (worst_fs_gap top) | FS non-onset | worst_fs_gap.csv | 非 onset FS 不退步 |
| `r7U6...` (worst_fs_gap top) | FS non-onset | 同上 | 同上 |
| `XXz0qk...` (Stage 1D movement top-win) | Movement | b16_stage_1d per_case_delta.csv | B-16 受益 case 不失 |

> 具體 case ID 由 Stage 1B 執行者從 `b16_stage_1c_cases.txt` 與 Stage 1D top-win/top-regression csv 補齊。

### §6.2 Matrix (per case)

| Run | B16 | SHADOW_NLMS | 目的 |
|---|---|---|---|
| (0,0) | 0 | 0 | Baseline (Phase 1 merged main) |
| (0,1) | 0 | 1 | Pure Phase 2 effect |
| (1,0) | 1 | 0 | Pure B-16 (等同 Stage 1D (1,0)) |
| (1,1) | 1 | 1 | Stacked |

### §6.3 Metrics per run

- Full-file RMS leak (dB) — FS scenarios
- Onset window RMS leak (dB, t=±1 s around onset) — PZ7V only
- DT window RES gain mean — DT scenarios
- Wav-level max |Δ| vs baseline

### §6.4 Pass criteria

| 檢查 | 判準 |
|---|---|
| (0,0) vs post-merge main | bit-exact (max |Δ| < run-to-run noise ~2e−3) |
| (0,1) PZ7V onset window | leak ≤ B-16 alone (−20.42 dB from Stage 1D); ideal ≤ −22 dB |
| (1,1) PZ7V onset | ≥ (0,1) (stacking 至少不退) |
| (0,1) DT cases | RES gain regression < 0.5 dB vs (0,0) |
| (0,1) FS non-onset | full-file leak Δ < 0.1 dB |
| (0,1) XXz0qk | Δ ≥ 0 vs (0,0) (Phase 2 不能吃掉 B-16 救的 case) |
| (1,1) vs (0,1)/(1,0) | ≥ max of them |

任一 fail → STOP, 進 §9 rollback。

### §6.5 EPC activation telemetry (new, audit fix 2)

§6.4 是純 performance metrics。為驗證 §4.3 downstream chain hypothesis, 對 **PZ7V flag=(0,1)** combination monkey-patch `AEC.process` dump per-frame telemetry in onset window `t=9.80–10.50 s` (~70 frames):

```python
# 探測變數 (全部從 aec instance 取值)
shadow_adv    = main_err_smooth / (shadow_err_smooth + 1e-10)
dt_signal     = _dt_from_shadow
sat_safe      = (_saturation_level < 0.3)
non_stat      = not _is_stationary_far   # 視實際屬性名調整
epc_level     = 'large' if epc_active_large else 'small' if epc_active else 'none'
```

產出表:

| frame | t | shadow_adv | dt_signal | sat_safe | non_stat | epc_level |
|---|---|---|---|---|---|---|
| … | … | … | … | … | … | … |

### §6.5.1 Hypothesis checks (§4.3 chain)

- **Shadow responsive**: shadow_adv 於 onset +16 frames 內 ≥ 1.3 (Stage 0b 已測; sanity 再驗)
- **EPC large fire**: `epc_level == 'large'` 至少出現 1 frame
- **Guard 排查**: 若 shadow_adv ≥ 1.3 但 EPC 未 fire, 查是哪個 guard blocker:
  - dt_signal > 0.3 (DT 誤判)
  - sat_safe == False (saturation 擋)
  - non_stat == False (被判 stationary)

### §6.5.2 GO criteria

- shadow_adv cross 1.3 within onset +16 frames → PASS (Phase 2 shadow responsive)
- EPC large fire ≥ 1 frame → PASS (revive confirmed)
- 若 shadow_adv PASS 但 EPC large 不 fire, 且 §6.4 performance metric 都 PASS: **不 block merge**, 但 Stage 1D 結果解讀註明: "performance gain 來自 copy gate revive / RES 層, 非 EPC"

此 telemetry 的價值: 若 Stage 1C/1D 整體 MARGINAL (§9), §6.5 data 可以定位 bottleneck 在 shadow filter 本身 vs downstream guard。

### §6.6 Execution order (new, audit fix 3)

2×2 matrix × 8 cases 須依序跑, 每步為下一步的 sanity baseline:

| 順序 | Flag combo | 目的 | Fail 則 STOP |
|---|---|---|---|
| 1 | (0,0) 8 cases | Verify bit-exact vs main HEAD `8d9bd91` | code 污染 baseline → STOP |
| 2 | (1,0) 8 cases | Verify match `docs/benchmarks/b16_stage_1d/` per-case | Phase 2 影響 B-16 path → STOP |
| 3 | (0,1) 8 cases | Pure Phase 2 effect | 進 §6.5 telemetry + §6.4 pass |
| 4 | (1,1) 8 cases | Stacked layers | 進 §6.4 (1,1) 判準 |

步驟 1-2 fail 表示 Phase 2 code 不當侵入 flag=0 path, 必須修 code 再重跑。步驟 3-4 是 new observation path。

---

## §7 800-case benchmark (Stage 1D)

### §7.1 Movement subset separation (Stage 1D lesson)

**Movement label source** (Stage 1A audit fix 4 pre-probe 確認, 2026-04-22):
- 無 metadata 檔 (`AEC/wav/aec_challenge_blind/` 下無 .csv / .json / .tsv)
- Label = **filename substring `_with_movement`** before `_mic.wav` / `_lpb.wav`
- Verified counts:
  - `farend_singletalk/`: 131 with_movement cases / 300 total (262/600 wav files)
  - `doubletalk/`: 114 with_movement cases / 300 total (228/600 wav files)
  - `nearend_singletalk/`: (not probed, movement label applies same way if present)
- 認定規則 (Stage 1B probe script 沿用):
  ```python
  is_movement = '_with_movement_' in filename
  ```

800-case 依此 label 切兩組:
- `movement_subset` (~245 cases across FS+DT)
- `static_subset` (~555 cases)

各組獨立 report: mean ΔFS_echo, mean ΔDT_echo, std, big-wins (Δ > +0.2), big-losses (Δ < −0.2)。

**GO criteria movement subset**:
- std 不超過 baseline +20% (variance 可略增)
- big-losses 不多於 Phase 1 B-16 alone 的 movement big-losses 數量

### §7.2 wav-level secondary metric (v3 dual-window, revised post Stage 1B)

AECMOS 不 reward 1-sec improvement in 21-sec file。對 Cat-A 13 cases ([bisect_analysis_and_plan.md](bisect_analysis_and_plan.md) top-20 FS gap 的 12 non-movement + PZ7V) 獨立算 **dual-window** leak:
- **Full-file** RMS leak (dB, whole 21 s)
- **Onset window** RMS leak (dB, t=±1 s around onset via energy detect)

Stage 1B 觀察: Phase 2 mechanism 在 post-onset copy gate, 不在 onset 本身 (onset 是 B-16 mechanism)。單窗 criterion bias 向 B-16-style fix。v3 改 dual-window:

**GO criteria wav-level (v3)**:
- PZ7V onset leak ≤ −22 dB (B-16 level maintained) **OR** PZ7V full-file leak ≤ baseline −5 dB (Phase 2 mechanism)
- Cat-A 13 cases full-file mean improvement ≥ 2 dB
- Cat-A 13 cases onset window mean improvement ≥ 3 dB
- 任一窗 PASS 即 wav-level PASS; 兩窗皆 PASS 為 **PASS++**

### §7.3 2×2 matrix full run

4 個 flag combinations 全跑 800-case:

| Run | B16 | SHADOW_NLMS | Baseline |
|---|---|---|---|
| (0,0) | 0 | 0 | 可直接用 Phase 1 benchmark `docs/benchmarks/b16_stage_1d/baseline_aecmos.log.gz` |
| (1,0) | 1 | 0 | 可直接用 `b16on_aecmos.log.gz` |
| (0,1) | 0 | 1 | 需新跑 |
| (1,1) | 1 | 1 | 需新跑 |

即 Stage 1D 本輪實際新跑 2 runs (節省一半時間)。

### §7.4 Matrix interaction 分析

定義 `ΔX = X(flags) − X(0,0)` for AECMOS FS_echo。

| pattern | 說明 | 解讀 |
|---|---|---|
| Δ(1,1) ≈ Δ(0,1) + Δ(1,0) | Orthogonal | 兩 layer 無 interaction, 可獨立評估 |
| Δ(1,1) > Δ(0,1) + Δ(1,0) | Synergy | 兩 layer 互補加乘 |
| Δ(1,1) ≈ max(Δ(0,1), Δ(1,0)) | Substitution | 一個 fix 吸收另一個 (預期情境) |
| Δ(1,1) < max(...) | Cancel | **Worst case**, STOP 調查 |

Pareto plot: FS_echo vs DT_echo, 4 點 + baseline, 識別 Pareto frontier。

---

## §8 Stage gates

- **Stage 1A**: spec review + user audit (本文件)
- **Stage 1B**: implement (遵 §5 invariants 1–5)
- **Stage 1C**: 8-case smoke test, 2×2 matrix, §6.4 pass criteria
- **Stage 1D**: 800-case benchmark, 2×2 matrix, §7 GO criteria
- **Stage 2**: merge decision — flag default 仍 OFF (Phase 1 風格), spec & benchmark artifact preserve

任一 stage fail → STOP, 不得私自 relax criteria。

---

## §9 Rollback plan (audit fix 5: 3-tier)

### §9.1 Stage 1C tiered verdict (v3 dual-window, revised post Stage 1B)

**Rationale for dual-window**: B-16 mechanism acts in the onset window (raw_dt veto blocks cascade at frame 995+)。Phase 2 mechanism is architecturally different — shadow-as-NLMS enables post-onset copy gate revive, manifesting in the post-onset adaptation period (t≈11-15 s typically)。Single-window (onset-only) criterion biases toward B-16-style fixes, underestimates architectural fixes operating post-onset。v2 §9.1 used onset-only; v3 changes to dual-window.

**Two-window assessment (both examined)**:
- **Primary**: Onset window (t=10–11 s, 1 s) leak vs baseline
- **Secondary**: Full-file (21 s) leak vs baseline

| Tier | 判準 | 行動 |
|---|---|---|
| **PASS** | (onset ≤ baseline −5 dB) OR (full-file ≤ baseline −5 dB) AND no metric > baseline +0.5 dB | Hypothesis validated, 進 Stage 1D |
| **MARGINAL** | best-window improvement 2–5 dB, no regression | 進 Stage 1D, 結果解讀附 §6.5 bottleneck 說明 |
| **FAIL** | no window shows ≥ 2 dB improvement, OR any window shows > 0.5 dB regression | STOP, 不進 Stage 1D, branch 保留 debug reference |

#### §9.1.1 MARGINAL bottleneck 查核 (用 §6.5 telemetry)

Hypothesis 部分對 (shadow responsive confirmed §6.5.1), 但 downstream 卡住。依 telemetry 分路:

| §6.5 observation | 可能 bottleneck | Stage 2 action (不含本 spec) |
|---|---|---|
| shadow_adv ≥ 1.3 但 EPC large 未 fire | guard 擋 (dt_signal / sat_safe / non_stat) | Stage 2 tune EPC thresholds |
| EPC large fire 但 leak 未 drop 足 | RES 響應慢 | Phase 3 議題 |
| Copy gate 未 trigger | hysteresis/streak 太保守 | Stage 2 tune shadow_copy_hysteresis |

MARGINAL **不 block** Stage 1D, 但 Stage 1D 結果 interpretation 要帶這個 downstream limitation context。

#### §9.1.2 FAIL 深度 debug

Stage 0b PZ7V shadow_adv +1.96 Δ 的 advantage 沒 translate 成 leak fix, 可能原因:
- Shadow filter 本身 OK 但 shadow-based subsystem 整體有其他 bottleneck (非 §9.1.1 列出的單一 guard)
- Phase 2 code bug (invariant §5 漏 check)
- Kalman main 在 Phase 2 下行為改變導致 shadow/main 耦合意外斷裂

**FAIL 行動**: 回 [aec3_full_architecture_analysis.md §6](aec3_full_architecture_analysis.md) 重評估優先級。可能 Gap #5 (filter misadjustment) 更 impactful, 或 Gap #2 需要先做 §2.2 "not in scope" 的 prerequisite (單向 copy 或 threshold tune)。

### §9.2 Stage 1D tiered verdict

| Tier | 判準 | 行動 |
|---|---|---|
| **PASS** | 整體 AECMOS + wav-level GO (§7.3-4), 無系統性 regression | Merge flag-gated to main (follow B-16 pattern), flag default OFF |
| **MARGINAL** | Pareto edge (某 metric 略退, 某 metric 明顯進) | User review 決策; 可能先 tune params 再決 |
| **FAIL** | Overall regress 或 §9.3 任一條件命中 | 不 merge, branch 保留 reference |

MARGINAL 處理細節:
- **Movement subset std > +20%**: 調 PBFDAF shadow `alpha_r_scale` (1.03 → 1.01 更保守) 或 `leak` (1e−6 → 5e−6)。B-14 原值針對 PBFDAF main, PBFDKF-main 環境下當 shadow 可能需重調。可以 Stage 1D 重跑 1 次 param-tuned variant 後再決。
- **(1,1) cancel pattern (§7.4 Δ(1,1) < max)**: 兩 layer interaction 探討, 可能要調整 B-16 flag default 或退 Phase 2。

### §9.3 Not acceptable conditions (hard fail, 任一命中 → FAIL tier)

- (0,1) vs (0,0) 在 non-PZ7V metric 有 > 0.05 dB **系統性** regression (個別 case 例外可接受)
- (1,1) < (1,0) (stacking 讓事情變糟)
- §7.4 matrix interaction == "cancel" pattern (Δ(1,1) < max(Δ(0,1), Δ(1,0)))
- Invariant §5.1 bit-exact baseline broken

---

## §10 Not in scope (reiteration)

- 單向 shadow→main copy (AEC3 refined→coarse only)
- PBFDAF main + PBFDKF shadow 對稱模式
- shadow_q_ratio tune (自然 no-op)
- EPC guard threshold tune (epc_large_shadow_adv=2.0, epc_small_shadow_adv=1.3)
- Copy gate threshold tune (shadow_copy_threshold=0.65)
- B-16 移除或 deprecate
- AecConfig dataclass 擴張 (維持 env flag only)

---

## §11 Stage 1B completion checklist

- [ ] `AEC_EXP_SHADOW_NLMS` env flag 加入 aec.py 頂部, default `'0'`
- [ ] L2541-2549 ShadowClass swap hunk applied
- [ ] Invariant §5.1 grep check (diff ≤ 15 lines)
- [ ] Invariant §5.2 grep check (exactly 1 match of swap gate)
- [ ] Invariant §5.3 grep check (zero new copy_weights_from)
- [ ] Invariant §5.4 grep check (zero shadow_* config field diffs)
- [ ] Invariant §5.5 grep check (flags 獨立)
- [ ] Flag=0 bit-exact sanity: PZ7V 單 case run vs post-merge main, max|Δ| < 2e−3
- [ ] Flag=1 smoke: PZ7V 單 case 跑通不 crash, shadow_filter 型別 `isinstance(PBFDAF) == True`
- [ ] 8-case × 4-flag smoke test matrix runnable (Stage 1C handoff)

---

## §12 References

- [spec_b16_raw_dt_jump_veto.md](spec_b16_raw_dt_jump_veto.md) — Phase 1 veto spec (template source)
- [b16_stage_1d_results.md](b16_stage_1d_results.md) — Phase 1 NEUTRAL verdict (Stage 1D lesson source)
- [aec3_full_architecture_analysis.md](aec3_full_architecture_analysis.md) §3.5, §6.1 — architectural rationale
- [bisect_analysis_and_plan.md](bisect_analysis_and_plan.md) — Cat-A 13-case list source for §7.2
- Stage 0b probe evidence: `/tmp/d1_baseline_pz7v.log`, `/tmp/d1_mock_pz7v.log`, branch `probe/phase2-d1-shadow-nlms@3682f98`
- Line refs verified against main commit `8d9bd91` (post B-16 merge)
