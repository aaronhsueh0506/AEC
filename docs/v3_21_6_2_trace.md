# v3.21.6.2 — 3-case local A/B trace

**Date**: 2026-05-28
**Baseline**: v3.21.6.1 (commit a98fc30)
**Candidate**: v3.21.6.2 (4 audit ships — see [CHANGELOG.md](../CHANGELOG.md))

## Method

Render the same `mic.wav` / `lpb.wav` pair twice (baseline vs candidate),
both `--mode pbfdkf --preset balanced --enable-res --cng`. Compute total
+ per-band (LF 50-500 Hz / MF 500-2000 Hz / HF 2000-7000 Hz) energy
delta. For DT cases, additionally compute the NE-active-segment delta
(mic > −25 dBFS over 100-ms windows) to isolate near-end preservation
from echo cancellation.

Δ sign convention: negative = candidate is **quieter** (more
suppression / less residual energy). On FS_static (no NE speech),
negative = better echo cancellation unambiguously. On DT, sign needs
NE-active comparison to interpret.

## Results

### Case 1 — `9xjhiFbGo06hdQIsHTS6qA` (FS_static stress)

```
mic   power: −9.77 dBFS
base  power: −17.07 dBFS   (cancel = +7.30 dB)
new   power: −19.27 dBFS   (cancel = +9.50 dB)
delta (new − base): −2.196 dB
  LF (50-500):    base=−28.60  new=−29.71  delta=−1.103 dB
  MF (500-2000):  base=−19.61  new=−22.12  delta=−2.512 dB
  HF (2000-7000): base=−23.06  new=−25.11  delta=−2.050 dB
```

**Verdict**: candidate cancels **2.20 dB more echo** total; MF/HF gain
roughly +2 dB each. 9xjhi is a known stress case where the shadow filter
underperforms the refined; the corrected poor_coarse hangover (10 hops
instead of 16, no shadow freeze) lets the rescue copy fire on the proper
AEC3 timing and the refined `disallow_leakage_diverged` window match the
strict 100-ms wall-clock.

### Case 2 — `0I0XMl3M0ECO0U1N0cJvpg` (DT_static)

```
mic   power: −19.50 dBFS
base  power: −20.51 dBFS   (cancel = +1.01 dB)
new   power: −20.55 dBFS   (cancel = +1.05 dB)
delta (new − base): −0.037 dB
  LF: delta=−0.021 dB    MF: delta=−0.146 dB    HF: delta=+0.014 dB
  NE-active (851 chunks, 85.1 s): base=−17.86  new=−17.89  delta=−0.036 dB
```

**Verdict**: NE-active delta −0.036 dB is well inside the per-frame
noise floor — NE preservation unchanged. No Pareto cost for the
FS_static win.

### Case 3 — `49IIo03GZ0CYQOmeA3A0BA` (DT_movement)

```
mic   power: −22.99 dBFS
base  power: −23.72 dBFS   (cancel = +0.73 dB)
new   power: −23.81 dBFS   (cancel = +0.82 dB)
delta (new − base): −0.094 dB
  LF: delta=−0.145 dB    MF: delta=+0.075 dB    HF: delta=−0.018 dB
  NE-active (523 chunks, 52.3 s): base=−19.28  new=−19.37  delta=−0.092 dB
```

**Verdict**: −0.09 dB NE-active delta inside noise floor. DT_movement
robustness preserved.

## Cross-case summary

| Case | Bucket | Total Δ (dB) | NE-active Δ (dB) | Verdict |
|---|---|---|---|---|
| 9xjhi | FS_static stress | −2.20 | — | better cancel |
| 0I0XMl3M | DT_static | −0.04 | −0.04 | flat |
| 49IIo03GZ | DT_movement | −0.09 | −0.09 | flat |

**Local verdict**: pure win on the FS_static stress case, no measurable
NE cost on the two DT cases sampled. Net consistent with the four
shipped items being AEC3-strict corrections that affect the
poor_coarse rescue path (only fires when refined is genuinely cleaner
than coarse, i.e. far-end-dominant frames).

## Caveats

1. **3 cases ≠ 800-case**. Per [[feedback-aecmos-pareto-comparison]],
   only the full bench tells us the Pareto picture. User has not yet
   authorised the 800-case re-bench.
2. **No internal HF-painted-black case in this trace**. The original
   v3.21.6.1 trace symptom (HF wiped above ~1 kHz during NE speech in
   internal cohort spectrogram) cannot be verified directly here. The
   DT_static HF delta of +0.014 dB is statistically zero — not a
   confirmation that the painted-black symptom is fixed. Needs the
   user-supplied internal case file to verify.
3. **Tier B #8 is the only behavioural ship**. Tier A #3 / #5 + Tier C
   #9 byte-equal vs v3.21.6.1 on the same 0I0XMl3M case (separately
   verified by temporarily reverting #8).

## Next steps

1. User listen-validate v3.21.6.2 on internal HF-painted-black case.
2. If clean: user authorises 800-case re-bench.
3. If still shows HF wipe: trace the user case per the Phase 1 replan
   gate in [v3_21_alignment_roadmap.md](v3_21_alignment_roadmap.md).
