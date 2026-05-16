# v3.19 HPF reference-path flip AECMOS verdict (2026-05-16)

**Status**: CANNOT SHIP at default OFF — revert to default ON, flag
retained for ablation. Revisit after Item 2 RES re-audit.

## Context

This session's refactor cycle (R.0–R.12) bundled a single algorithm
change (commit `ab44842`): flipped `enable_highpass_ref` default from
**True** (legacy: 80 Hz HPF on reference signal, always on when
`enable_highpass=True`) to **False** (AEC3-aligned: gated behind
WebRTC field trial `Aec3HighPassFilterEchoReference`, default off).

Mic-path HPF (`enable_highpass`) was unchanged, matching WebRTC APM
`Config::high_pass_filter.enabled = true` default.

## Bench setup

Both 800-case BALANCED preset, fl=832 (52 ms), CNG on, `--parallel
--workers 4`. Single-flag delta:
- **HPF-OFF render**: `/tmp/r12_800/` (post-flip, `enable_highpass_ref=False`)
- **HPF-ON baseline render**: `/tmp/r12_800_hpfon/` (revert override, `enable_highpass_ref=True`)
- AECMOS via `python/bench_aecmos.py` (FastAECMOS single-session)

HPF-OFF render: 779 / 800 _ours.wav files; AECMOS scored 769.
HPF-ON render: 800 / 800 _ours.wav files; AECMOS scored 800.
The 31 missing in HPF-OFF render were silently dropped during the
earlier `_BLEND_F31_MIC_EXCESS` NameError crash (the bug fix landed
mid-bench in commit `025af8a`). Bucket counts compared on the
overlap (165–169 FS_static, 131 FS_movement, 168–186 DT_static,
101–114 DT_movement, 200 NE).

## Bucket means (HPF-OFF − HPF-ON, sign re-framed for verdict clarity)

| Bucket       | n     | Δecho   | Δdeg    | Hard-bar verdict                  |
|--------------|------:|--------:|--------:|-----------------------------------|
| FS_static    | 169   | +0.022  | +0.000  | PASS                              |
| FS_movement  | 131   | +0.017  | +0.000  | PASS                              |
| DT_static    | 168   | +0.011  | +0.017  | PASS (both dimensions positive)   |
| DT_movement  | 101   | +0.030  | **−0.034** | **FAIL** (~7× worst-case bar) |
| NE           | 200   | +0.000  | −0.002  | borderline neutral                |

Hard bars (per CLAUDE.md preset chain + `feedback_aec_code_review_accuracy.md` discipline):
- Δdeg ≥ −0.005 per bucket (worst-case)
- Δecho ≥ −0.000 (FS bars; can absorb minor regression elsewhere)

## Mechanism — Pareto trade-off as predicted

The user predicted this exact shape on 2026-05-16 (per memory:
"aec3的設定會砍比較多nearend"). Mechanistically:

- **HPF-OFF gives the filter more low-freq reference energy** (below
  80 Hz). Since real-room echo has substantial sub-100 Hz content,
  the filter has more information to cancel echo → **FS echo
  improvement** (+0.017 to +0.030 across all 4 echo-bearing buckets).
- **HPF-ON limits low-freq ref reaching the filter and the residual
  estimator**. Less ref energy ⇒ less aggressive low-freq suppression
  in the residual stage ⇒ **DT NE preservation** in low-freq bins
  where speech fundamental energy lives (especially male voice
  pitch ~100–200 Hz partials).
- Movement cohort amplifies the asymmetry: movement creates rapid
  ref-vs-mic re-alignment events. Without HPF on ref, the filter's
  brief mismatched state during re-alignment over-suppresses NE.

This is structurally identical to the v3.18 D-γ Pareto wall and
v3.13 E5 saturation arc trade-off line (~0.5–1 dB DT loss per
+1 dB FS gain). The DT_movement bucket Δdeg −0.034 vs Δecho +0.030
sits exactly on that slope.

## Verdict

**CANNOT SHIP at default OFF**. Revert `enable_highpass_ref` default
to True. Flag retained for ablation testing.

### Why not "ship as MIXED with per-preset gating"

Considered but rejected:
- BALANCED preset is the only production-meaningful target right now
  (per `feedback_bench_j4.md`); per-preset gating doesn't help if
  BALANCED itself fails the bar.
- DT_movement deg cost (−0.034) is large and consistent (101 cases),
  not a single-case outlier — it's a real bucket-level regression.

### Why retain the flag (not delete)

- Substrate has structural value: when **Item 2 RES re-audit** lands
  consumer-side improvements that protect DT NE in low-freq bins
  (e.g. dominant_ne_detect → over_sub soft-feature wiring per Item 2c,
  or per-band ENR control per existing `res_per_band_enr` substrate),
  the HPF-OFF FS gain may become attainable without the DT cost.
- Re-bench at that point. If the trade-off slope flattens (DT loss
  ≤ −0.005 per +0.020 FS gain), HPF-OFF becomes ship-able.
- Flag falls into Group A (RES-blocked) of the substrate inventory.

## Revert action

`python/modules/config.py:225-229` — `enable_highpass_ref` default
back to `True`, comment updated to reference this verdict doc.

## Ship-side artefact

This is the final commit on `feature/v3.18-aec3-fetch` before the
branch is renamed to `refactor/aec-py-modules` (frozen archive).
The new branch `feature/v3.19-closeout` opens at this commit and
proceeds with Items 2 / 3 / 4.
