# 3aadd2d AECMOS baseline — `balanced` vs `balanced_aec3`

**Snapshot**: HEAD `3aadd2d` on `feature/v3.21-aec3-clean` (2026-05-19).
**Render config**: `--filter 832 --cng --parallel --workers 4`, full 800-case AEC Challenge blind test corpus.
**Scoring**: AECMOS via `python/bench_aecmos.py`.

This baseline is the byte-equal anchor for the v3.21 release cleanup cycle. After each cleanup
round, `python/check_byte_equal.py` must report 25/25 PASS against
[byte_equal_reference.json](byte_equal_reference.json) before the round can land.

## Bucket means

| Bucket | n | **`balanced`** echo | deg | **`balanced_aec3`** echo | deg | Δecho | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS_static    | 169 | 3.253 | 5.000 | **3.729** | 4.999 | **+0.476** | -0.001 |
| FS_movement  | 131 | 3.315 | 5.000 | **3.626** | 4.999 | **+0.311** | -0.001 |
| DT_static    | 186 | 3.956 | **2.861** | **4.237** | 2.387 | **+0.281** | **-0.474** |
| DT_movement  | 114 | 3.929 | **2.727** | **4.215** | 2.371 | **+0.286** | **-0.356** |
| NE           | 200 | 4.998 | 3.985 | 4.998 | **4.052** | 0.000 | +0.067 |

(Δ = `balanced_aec3` − `balanced`.)

## Reference points

| | FS_st echo | FS_mv echo | DT_st echo | DT_mv echo | DT_st deg | DT_mv deg | NE deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| AEC2 | 3.48 | 3.48 | 4.26 | 4.26 | 2.39 | 2.39 | 4.10 |
| AEC3 | 3.88 | 3.88 | 4.54 | 4.54 | 1.85 | 1.85 | 3.45 |
| v3.20 Phase 0 (`balanced` @ 792c0b7) | 3.769 | 3.730 | 4.244 | 4.082 | 2.297 | 2.318 | 4.005 |
| **3aadd2d `balanced`** | 3.253 | 3.315 | 3.956 | 3.929 | **2.861** | **2.727** | 3.985 |
| **3aadd2d `balanced_aec3`** | **3.729** | **3.626** | **4.237** | **4.215** | 2.387 | 2.371 | **4.052** |

`balanced_aec3` wins on echo across all four FS+DT buckets and on NE_deg. `balanced` wins on
DT_deg by a wide margin (>0.35) but pays for it with significant echo regressions vs both AEC2 and
the v3.20 Phase 0 production baseline.

## Ship decision

`balanced_aec3` is the v3.21 production default. Other presets (`mild`, `soft`, `balanced`,
`aggressive`, `maximum`) are slated for removal in subsequent cleanup rounds; the AEC3 chain
becomes the sole rendering path.

## Files in this directory

| File | Purpose |
|---|---|
| `balanced_scores.json` | Per-case AECMOS scores for legacy `balanced` preset (800 cases) |
| `balanced_result.md` | Bucket-mean summary + worst-20 per bucket for `balanced` |
| `balanced_aec3_scores.json` | Per-case AECMOS scores for `balanced_aec3` (800 cases) |
| `balanced_aec3_result.md` | Bucket-mean summary + worst-20 per bucket for `balanced_aec3` |
| `byte_equal_reference.json` | md5 of `_ours.wav` + `_ours_nores.wav` for 25 representative cases (5 per bucket, sorted by echo at 0 / 25% / 50% / 75% / 100% percentile within each bucket) — the anchor for `python/check_byte_equal.py`. **Updated 2026-05-26**: hashes regenerated with `use_partition_summed_x2_for_h_error_gain=False` (the v3.21 closure default; prior JSON had accidentally used True from v3.21.20 ALL-ON experiment). |

## Reproducing this baseline

```bash
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
    --preset balanced_aec3 --filter 832 --cng --parallel \
    -o out_3aadd2d_balanced_aec3/ --workers 4
python3 python/bench_aecmos.py out_3aadd2d_balanced_aec3/ results_3aadd2d_balanced_aec3/
```

## Re-running the byte-equal check after cleanup

```bash
python3 python/check_byte_equal.py            # uses preset=balanced_aec3
python3 python/check_byte_equal.py --preset balanced   # after rename
```

Expected output: `=== 25/25 PASS, 0 FAIL ===`.
