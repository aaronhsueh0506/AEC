# C AEC Rewrite Plan — v2.5 → v3.8.1 Sync

**Status**: legacy `c_impl_v25_legacy/` preserved; new `c_impl/` rebuilt from Python v3.8.1.
**Author / date**: 2026-05-01.
**Scope**: full architectural sync of C AEC implementation to match Python `aec.py` v3.8.1.

---

## Why a rewrite, not incremental port

C `c_impl/` is at v2.5.0 (`5407e71`). Python is at v3.8.1 (`3844169`). The gap spans:

| Version | Architectural change | Port shape |
|---|---|---|
| v2.6 ~ v2.8 | preset re-tuning, movement-DT improvements | parameter sync + tuning sweep |
| v3.0 ~ v3.5 | PR-A/B/C/D/E experiments + multi-ERLE refactor | partial / abandoned branches |
| v3.6.0 | filter_length 32→52ms | C default tweak (1 line) |
| v3.6.1 | DT-from-frame-zero stats detector | new C class `DtAnalyzer` |
| v3.7.0 | G1 KX blended P-update | rewrite `pbfdkf.c` Q/P update |
| v3.7.1 | drop render-based linear_failed (PR-B) | branch removal in `res_filter.c` |
| v3.8.0 | drop error_based_floor (ABL-1) + drop Y2 fallback (ABL-2) | branch removal in `res_filter.c` |
| v3.8.1 | drop dead `erl > 1.2` (ABL-4) + diag clear | branch removal + reset() addition |

Incremental port = re-doing ~10 commit cycles with bench validation each. Rewrite = single milestone targeting v3.8.1 directly.

---

## Public API stability (preserve)

Audio_ALG `pipelines/aec_nr_pipeline.c` consumes these symbols. **MUST remain stable** during rewrite:

```c
/* aec.h */
Aec*  aec_create(const AecConfig*);
void  aec_destroy(Aec*);
void  aec_reset(Aec*);
int   aec_process(Aec*, const float* mic, const float* ref, float* out);
int   aec_process_ex(Aec*, const float*, const float*, float*, AecResContext*);
AecResContext* aec_context_create(const Aec*);
void  aec_context_destroy(AecResContext*);
int   aec_get_hop_size(const Aec*);
int   aec_get_delay(const Aec*);
struct ResFilter* aec_get_res(const Aec*);

/* res_filter.h */
ResFilter* res_create(const ResConfig*);
void res_destroy(ResFilter*);
void res_reset(ResFilter*);
void res_process(ResFilter*, const float* error, const Complex* echo,
                 const Complex* far, const Complex* near, ...);
```

**Existing pipeline integration pattern**:
```
HPF → aec_process_ex(...) → ctx populated
   → mmse_lsa_process(nr, aec_out, nr_out)        // NR
   → corrected_echo[k] = ctx->echo_spec[k] × nr_gain[k]
   → res_process(res, nr_out, corrected_echo, ...) // RES with NR-aware echo
```

NR insertion contract (DO NOT BREAK):
1. `AecResContext` exposes `echo_spec_re/im`, `far_spec_re/im`, `near_spec_re/im`, `far_power`, `filter_converged`, `erle_factor`, `dt_indicator`, `is_stationary_dt`, `saturation_level`, `erl_estimate`.
2. Caller can mutate `echo_spec` (multiply by NR gain per bin) before passing to `res_process()`.
3. 1-frame OLA delay is caller's responsibility (use `prev_ctx`).

---

## File structure (new c_impl/)

```
c_impl/
├── include/
│   ├── aec.h               -- Public main API (preserved)
│   ├── aec_types.h         -- AecConfig + AecResContext (extended for v3.8)
│   ├── pbfdkf.h            -- PBFDKF main filter (G1 KX blended)
│   ├── res_filter.h        -- RES (v3.8.1 cleaned)
│   ├── shadow_filter.h     -- NEW: separated from aec.c (was inline)
│   ├── dt_analyzer.h       -- NEW: from Python DtAnalyzer + StationaryDtDetector
│   ├── multi_erle.h        -- NEW: FilterErleEstimator + FullbandErleEstimator
│   ├── delay_estimator.h   -- preserved
│   ├── render_activity.h   -- NEW: from Python RenderActivityDetector
│   ├── epc_detector.h      -- NEW: from Python EchoPathChangeDetector
│   ├── shadow_copy.h       -- NEW: from Python ShadowCopyController
│   ├── fft_wrapper.h       -- preserved (kiss_fft backend)
│   ├── fast_math.h         -- preserved (sqrtf/log/exp approximations)
│   └── hpf.h               -- preserved
├── src/
│   ├── aec.c               -- Main orchestration (was 1440 → est. 1800)
│   ├── pbfdkf.c            -- Frequency-domain Kalman + G1 KX blended
│   ├── res_filter.c        -- v3.8.1 (no ABL'd branches)
│   ├── shadow_filter.c     -- NEW
│   ├── dt_analyzer.c       -- NEW
│   ├── multi_erle.c        -- NEW
│   ├── render_activity.c   -- NEW
│   ├── epc_detector.c      -- NEW
│   ├── shadow_copy.c       -- NEW
│   ├── delay_estimator.c   -- port from Python (GCC-PHAT)
│   ├── fft_wrapper.c       -- preserved
│   └── hpf.c               -- preserved
├── lib/                    -- kiss_fft (preserved)
├── example/                -- aec_wav demo (regenerate)
├── Makefile                -- regenerate (lib + bin targets)
└── README.md               -- new (rewrite from scratch)
```

Estimated total: ~5500 LOC (vs current 4488).

---

## Phase split

### Phase 0 — Scaffolding (this session, low risk)

- [x] Backup `c_impl_v25_legacy/`
- [ ] Update `c_impl/include/aec_types.h`:
  - filter_length default 32→52ms (v3.6.0)
  - Extend `AecResContext` with `e2_main`, `e2_shadow`, `y2`, `filter_once_converged`
  - Add v3.8.x preset retuning constants (Q_low, etc. from Python)
  - Bump version comment to "Matches Python v3.8.1"
- [ ] Doc `c_integration_guide.md` (NR insertion + restrictions)
- [ ] Mark v2.5 algorithm internals as TODO with explicit reference to Python source line numbers

### Phase 1 — PBFDKF G1 KX blended (1-2 days)

- [ ] Rewrite `pbfdkf.c` `_update_weights()` to mirror Python `_update_weights_kalman` (line 763 area)
- [ ] G1 critical change: `KX = mu_mean × KX_optimal + (1-mu_mean) × KX_scaled` for P update
- [ ] Verify with parity test: feed 1000 frames synthetic → diff vs Python <1e-5

### Phase 2 — Multi-ERLE + Render activity + DT analyzer (2-3 days)

- [ ] Port `FilterErleEstimator` (Python line 924)
- [ ] Port `FullbandErleEstimator` (Python line 963)
- [ ] Port `RenderActivityDetector` (search Python for class)
- [ ] Port `DtAnalyzer` + stationary DT (Python `dt_analyzer`, `StationaryDtDetector`)
- [ ] Wire into AEC.process

### Phase 3 — ResFilter v3.8.1 algorithmic core (2-3 days)

- [ ] Rewrite `res_filter.c` from Python v3.8.1
- [ ] Critical: ensure no ABL'd structures regenerated (e2-floor / mic_psd × 0.5 / error × 0.9 / error_based_floor)
- [ ] `ResidualEchoEstimator` separation (Python class)
- [ ] Spectral floor + CNG + ENR mask
- [ ] Reverb tail model

### Phase 4 — Shadow filter + EPC + delay (1-2 days)

- [ ] Shadow as parallel PBFDKF (separate Q schedule)
- [ ] `EchoPathChangeDetector`
- [ ] `ShadowCopyController` (5-state machine)
- [ ] Delay estimator port (existing `delay_estimator.c` is OK)

### Phase 5 — Parity verification (1-2 days)

- [ ] Build a `parity_smoke.c` that mirrors Python `parity_smoke.py`
- [ ] 800-case bench through C: `./bin/aec_bench wav/aec_challenge_blind --preset balanced`
- [ ] Compare per-bucket scores ±0.005 vs Python v3.8.1 baseline JSON
- [ ] If mismatch >0.005 in any bucket: trace + fix

### Phase 6 — Integration tests (1 day)

- [ ] `aec_nr_pipeline.c` rebuild and verify
- [ ] Static memory branch port (separate engagement)

---

## Parity validation gates (HARD REQUIREMENT — float32-level)

**Acceptance bar**: C output must match Python output within numerical
precision attributable solely to float32 (single precision) vs float64
(double precision) drift. NOT preset re-tuning, NOT "close enough", NOT
"800-case bucket scores agree". The floor is **per-sample bit-level**
comparison modulo unavoidable float ordering differences.

Each phase must pass before next:

1. **Build**: `make clean && make` (no warnings)
2. **Smoke**: `./bin/aec_wav example/mic.wav example/ref.wav out.wav` runs without segfault
3. **Per-component parity** (every phase): synthetic input → C output array
   vs Python output array, numpy `np.allclose(rtol=1e-5, atol=1e-7)`. For
   intermediate state (P, W, error_psd, etc.): same tolerance.
4. **End-to-end audio parity** (after Phase 5): all 800 case `out.wav` files
   produced by C → diff against Python `out.wav`, mean abs sample diff
   < 1e-5 across full file. AECMOS scores per-case match Python within
   ±0.001 (= AECMOS noise floor).
5. **Bucket-level parity** (after Phase 5): preset BALANCED 5-bucket scores
   identical to Python within ±0.001.

**Implications for implementation**:
- All AEC internal compute is float32 (matches Python's `.astype(np.float32)`
  casts; user explicitly cast `np.float32(self.delta)` etc. to prevent
  float64 promotion).
- FFT path: kiss_fft float32 (matches Python `np.fft.rfft` on float32 array).
  Python returns complex64 (float32 pairs).
- Order-of-operations matters: matrix-vector products (`np.matmul`,
  `np.einsum`) reduce in a specific order; C must replicate.
- Integer indexing: `partition_idx`, `(partition_idx - p) % n_partitions`
  semantics (negative mod in Python returns positive; C `%` returns
  signed). Must explicitly handle.

If a phase exceeds the rtol=1e-5 budget, halt and root-cause before next
phase. Common drift sources to inspect:
- `np.float32 vs float`: scalar promotion in mixed arithmetic
- `np.real()` casting: complex → float32, NOT float64
- Reduction order: `sum()` accumulator type
- FFT scaling: numpy normalizes IFFT by 1/N; kiss_fft does not — caller
  must scale.

---

## NR integration — explicit contract for Audio_ALG

```c
/* In aec_nr_pipeline.c (DO NOT CHANGE except for new ctx fields): */
aec_process_ex(aec, mic, ref, aec_out, ctx);    // Stage 1: linear AEC
mmse_lsa_process(nr, aec_out, nr_out);          // Stage 2: NR
const float* nr_gain = mmse_lsa_get_gain(nr, NULL);
for (int k = 0; k < n_freqs; k++) {
    corrected_echo[k].r = prev_ctx->echo_spec_re[k] * nr_gain[k];
    corrected_echo[k].i = prev_ctx->echo_spec_im[k] * nr_gain[k];
}
res_process(res, nr_out, corrected_echo,
            prev_far_spec, prev_near_spec,
            prev_ctx->far_power, prev_ctx->filter_converged,
            prev_ctx->erle_factor, prev_ctx->dt_indicator,
            prev_ctx->over_sub, prev_ctx->divergence,
            prev_ctx->is_stationary_dt, prev_ctx->shadow_dt,
            prev_ctx->erl_estimate, prev_ctx->epc_active,
            prev_ctx->saturation_level, res_out);
```

**Key invariants for NR insertion**:

1. **Echo correction is mandatory**: `corrected_echo = echo_spec × nr_gain` per bin, because NR has already attenuated those frequencies. Skipping correction → RES double-suppresses → audible NE damage.
2. **1-frame OLA delay alignment**: NR's MMSE-LSA introduces 1-frame OLA. RES inputs (echo, far_spec, near_spec, ctx) MUST come from `prev_ctx` (delayed copy), not current frame.
3. **Don't insert NR before linear AEC**: linear filter learns echo path from raw mic; NR-corrupted mic breaks adaptation.
4. **Don't run RES standalone (without linear)**: RES uses filter's `echo_spec` as primary echo estimate; standalone RES has no echo basis.

---

## Restrictions / things you CANNOT do (post-rewrite)

1. **Don't reintroduce e2-floor variants** — see `signal_flow_constraints.md` section 7d "REMOVED" warning. Four separate ablation studies confirmed structural NE damage:
   - `residual = max(residual, error_psd × 0.9)` ← v3.7.1 PR-B
   - `residual = max(render_based, error_psd × far_conf × 0.7)` ← v3.8.0 ABL-1
   - `residual = max(residual, mic_psd × 0.5)` ← v3.8.0 ABL-2
   - `linear_failed if erl > 1.2` ← v3.8.1 ABL-4 (was dead code)

2. **Don't use scalar DT detector to override per-bin gain calculations** — PR-F session showed this is architecturally hopeless (DT bucket coh2 overlap = 0.84). Either use per-bin evidence directly, or use scalar gate as outer envelope only (not inner override).

3. **Don't tune presets to dataset** — current presets passed 800-case validation. Re-tuning for new dataset must include cross-preset robustness check (4 presets × all metrics).

4. **Don't bypass `aec_reset()` between independent calls** — diagnostic counters and detector states leak otherwise. Static memory port must respect the same lifecycle.

---

## Static memory branch (separate engagement)

After main C is parity-verified:
1. `feature/static-memory` branch: replace all `malloc()` with statically-sized struct fields
2. Configurable via compile-time `AEC_MAX_FRAME_SIZE` macro
3. Sample rate fixed at compile time (8/16/48 kHz target)
4. RAM budget per-instance: estimate ~80 KB @ 16kHz / 52ms / 257 freqs

Defer until main C is functional.

---

## Open questions

- Does Novatek deployment target use 16kHz exclusively? (affects partition count + buffer sizes)
- Is float32 sufficient or do we need int32-fixed-point port?
- Is kiss_fft acceptable on target HW or do we need vendor FFT?

---

## Estimated total effort

7-10 working days for main C rewrite + parity bench. Static-memory branch +2-3 days. Integration with Audio_ALG and Novatek SDK +2-3 days. **Total ~12-16 working days**.
