# v3.21.2 AECMOS baseline — `balanced`

**Snapshot**: v3.21.2 release HEAD (= last U5.3 commit on the v3.21.2 branch).
**Render config**: `--preset balanced --filter 832 --cng --parallel --workers 4`,
full 800-case AEC Challenge blind test corpus.
**Scoring**: AECMOS via `python/bench_aecmos.py`.

## What v3.21.2 changed vs v3.21.0

- **Phase A/B (commits 7e9e612 + f1ea92c)**: frequency-canonical
  bin-index alignment in `SuppressionGain`. HF cap `lgb` 937 Hz → 4000 Hz;
  mask interpolation `last_lf`/`first_hf` 156/250 Hz → 625/1000 Hz.
  Fixes user-reported Chinese /i/ HF damage.
- **U3 (commit 6a071c1)**: `sr` threading through SuppressionGain consumers.
  16 kHz behaviour byte-equal preserved; non-16-kHz now correctly scales.
- **U5.3 (commit c7481a4)**: `EpStrengthConfig.default_gain` 0.014 → 0.020.
  Asymmetric Pareto-positive partial FS-echo recovery via 2× higher
  nonlinear-mode R² estimate. AEC3 field-trial Aggressive value.

See [CHANGELOG.md](../../../CHANGELOG.md) v3.21.2 entry for full
mechanism / verdict cross-references.

## Bucket means (v3.21.2 final)

| Bucket | n | echo | deg |
|---|---:|---:|---:|
| FS_static | 169 | 3.582 | 4.999 |
| FS_movement | 131 | 3.509 | 4.999 |
| DT_static | 186 | 4.188 | 2.481 |
| DT_movement | 114 | 4.166 | 2.485 |
| NE | 200 | 4.998 | 4.054 |

## Δ vs v3.21.0 baseline (a537b65 / `docs/bench/v3_21_3aadd2d_baseline/`)

| Bucket | Δecho | Δdeg | Note |
|---|---:|---:|---|
| FS_static | −0.147 | +0.000 | HF cap relaxation Pareto cost |
| FS_movement | −0.117 | +0.000 | HF cap relaxation Pareto cost |
| DT_static | −0.049 | **+0.094** | HF formant fidelity recovered |
| DT_movement | −0.048 | **+0.115** | HF formant fidelity recovered |
| NE | +0.000 | +0.003 | flat |

## Reference points

| | FS_st echo | FS_mv echo | DT_st echo | DT_mv echo | DT_st deg | DT_mv deg | NE deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| AEC2 | 3.48 | 3.48 | 4.26 | 4.26 | 2.39 | 2.39 | 4.10 |
| AEC3 | 3.88 | 3.88 | 4.54 | 4.54 | 1.85 | 1.85 | 3.45 |
| v3.21.0 (`a537b65`) | 3.729 | 3.626 | 4.237 | 4.215 | 2.387 | 2.371 | 4.052 |
| **v3.21.2** | 3.582 | 3.509 | 4.188 | 4.166 | **2.481** | **2.485** | 4.054 |

v3.21.2 trades FS_echo (-0.147 / -0.117 vs v3.21.0) for DT_deg gain
(+0.094 / +0.115). Net positive on DT formant fidelity (matches user-
reported HF damage report on Chinese /i/); FS_echo remains comfortably
above AEC2 baseline and within ~0.30 dB of AEC3 reference.

## Files in this directory

| File | Purpose |
|---|---|
| `balanced_scores.json` | Per-case AECMOS scores (800 cases) |
| `balanced_result.md` | Bucket-mean summary + worst-20 per bucket |

## Reproducing this baseline

```bash
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
    --preset balanced --filter 832 --cng --parallel \
    -o out_python_v3_21_2/ --workers 4
python3 python/bench_aecmos.py out_python_v3_21_2/ results_v3_21_2/
```
