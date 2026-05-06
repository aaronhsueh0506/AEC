# Research log — P3d 7GT post-alignment trace

Date: 2026-05-05
Code line: v3.10.4 release.
Trace input: `wav/aec_challenge_blind/doubletalk/7GTxyTksSUqCnP5y0ILG4A_doubletalk_*.wav`
Diag CSV: `/tmp/7gt_p3d_diag.csv` (3667 frames, hop 160, 16 kHz).

## Question

P3a confirmed the 7GT skew (788 ms) is recoverable. P3b confirmed the
production delay estimator does in fact lock to 12606 samples at
t = 4.57 s and stays there. The bench echo score (3.366) is still
disappointing. Is the back half being correctly handled, or is the
filter / DTD / RES misbehaving after alignment?

## Method

Ran `run_one_case.py` on 7GT doubletalk no-pad, BALANCED, cng=on,
with `--diag-csv` capturing per-frame state. Split the trace at
t = 4.59 s (first frame after Path A delay-first commit) and
analysed the post-alignment region.

## Findings

### 1. Alignment itself is fine

- delay_samp jumps 0 → 12606 between t=4.58 s and t=4.59 s (Path A
  fires).
- delay_ms holds 787.9 ms across all 3209 post-alignment frames; no
  jitter, no Path B re-trigger.

### 2. Filter once-converged but degrades over time

| t window | far-active n | erle_win mean | once_conv | conv |
|----------|-------------:|--------------:|----------:|-----:|
| 4–8 s    | 373          | +1.04 dB      | 0 / 373   | 0    |
| 8–12 s   | 386          | **+5.02 dB**  | 374 / 386 | high |
| 24–28 s  | 65           | **−1.00 dB**  | 65 / 65   | 0    |
| 28–32 s  | 400          | **−1.82 dB**  | 400 / 400 | 0    |
| 32–36 s  | 377          | **−0.39 dB**  | 377 / 377 | 0    |

ERLE peaks at +5 dB by t=8–12 s (about 4–8 s after Path A reset),
then **goes negative** in the back half — i.e. filter output is worse
than mic input on those segments. Once_converged stays 1 (the latch
doesn't reset), but the running ERLE collapses.

### 3. DTD has signal but composite gate never fires

Per-frame DTD distribution over 1611 far-active post-alignment frames:

| signal      | median | p75   | frames > 0.3   |
|-------------|-------:|------:|---------------:|
| dt_conf     | 0.000  | 0.000 | 0 / 1611       |
| dt_coh      | 0.000  | 0.000 | 0 / 1611       |
| dt_energy   | 0.238  | 0.438 | 666 / 1611 (41%) |
| dt_shadow   | 0.514  | 0.897 | 1045 / 1611 (65%) |
| **dt_active > 0.5 (composite)** | — | — | **0 / 1611 (0%)** |

`dt_conf` / `dt_coh` zero is expected — BALANCED has
`enable_dtd=False`, so the coherence-based DTD path is intentionally
inactive. The shadow-filter and energy DTD signals do fire, often
strongly (`dt_shadow` is over 0.5 most of the time). But the
composite `dt_active` flag never crosses 0.5 — its gate logic is
either AND-too-strict, or it consults `dt_conf` directly.

The practical consequence: `mu_scale` is not being reduced during
DT, even though shadow / energy DTD plainly indicates DT is present.
The filter learns against NE-contaminated error in those frames,
pulling the taps off the echo path. That is the ERLE collapse seen
in the trajectory above.

### 4. RES is stuck in render-based mode

- using_render = 99% across post-alignment.
- Driven by `force_render = ... or not filter_converged` in
  `attribute_legacy`. With current ERLE oscillating around 0–5 dB
  and `_filter_converged` flipping off in the back half, render-based
  is the active path almost always.
- Effect: echo suppression is carried by far-PSD × ERL estimate, not
  by the filter's `echo_spec`. Acts as a backstop, but also presses
  on NE high band → contributes to deg score not being higher.

## Bench-level evidence

7GT_doubletalk score in v3.10.4 BALANCED: echo **3.366**, deg
**3.895**. The deg score is very high (NE preserved well — render-
based is gentle on NE), the echo score is mediocre (filter not
helping; render-based suppresses but only partially).

Compare to WebRTC AEC3 same case (no-pad): echo 3.625, deg 2.268.
AEC3 sacrifices NE to suppress echo more aggressively. WebRTC old
AEC: echo 3.387, deg 3.794 — the closest profile to ours, suggesting
that even when alignment works (old AEC has 1 s search range), the
filter's adaptation against this case is the limiting factor, not the
post-filter.

## Conclusion

The 7GT 788-ms-skew alignment is solved at the v3.10.4 layer. The
remaining gap on this case is **adaptation protection**:

- DT signals (`dt_shadow`, `dt_energy`) are present and correct.
- But the composite `dt_active` gate that drives `mu_scale` and
  shadow-copy decisions is consulting only `dt_conf` (which is zero
  with `enable_dtd=False`), so the available DT evidence is unused.
- Without DT-driven mu reduction, the filter adapts during DT and
  the taps degrade. ERLE goes negative.
- RES then sees `not filter_converged` and parks in render-based
  mode permanently, which limits both echo suppression and NE
  preservation.

Two paths suggested by the trace:

1. **Adaptation-only DT advisory gate (P3e)**: route `dt_shadow` /
   `dt_energy` directly into `mu_scale` reduction and shadow-copy
   inhibition, without touching RES. Lowest-risk fix because it
   addresses the upstream root cause (filter divergence during DT)
   without re-entering the suppressor Pareto wall the v3.10.4
   fallback hit.
2. (Considered, deferred) Fix the composite `dt_active` gate to
   honour `dt_shadow` even when `enable_dtd=False`. Similar effect
   but touches more code paths and risks regressing other presets.

P3e is the cleaner first step. Variants to try:

- `dt_shadow > 0.5` only
- `dt_shadow > 0.5 OR dt_energy > 0.4`
- both + 300–500 ms hysteresis (hit-then-hold) to avoid per-frame
  flicker

Bench acceptance per current plan:
- 7GT doubletalk echo / deg both must improve
- FS_static / FS_movement no worse than −0.02
- NE flat
- 800-case DT_movement deg gain not driven by a single case

## Files

- `/tmp/7gt_p3d_diag.csv` (raw per-frame trace)
- `/tmp/7gt_p3d_gains.npz` (per-stage gain dump if needed for stage-6 inspection)

Reproduce:

```bash
python python/run_one_case.py \
  wav/aec_challenge_blind/doubletalk/7GTxyTksSUqCnP5y0ILG4A_doubletalk_mic.wav \
  wav/aec_challenge_blind/doubletalk/7GTxyTksSUqCnP5y0ILG4A_doubletalk_lpb.wav \
  /tmp/7gt_p3d.wav --preset balanced --cng \
  --diag-csv /tmp/7gt_p3d_diag.csv \
  --res-gain-dump /tmp/7gt_p3d_gains.npz
```
