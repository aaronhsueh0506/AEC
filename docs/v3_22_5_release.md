# AEC v3.22.5 — release summary
2026-06-07 · branch `feature/render-correlation` · BALANCED algorithm byte-equal to 3.22.4

## What v3.22.5 is
A **release-hygiene + API** version on top of the 3.22.4 algorithm. The BALANCED audio path is **byte-equal
to 3.22.4** (and 27/27 byte-equal across gentle/balanced/aggressive); the version bump is for:
1. **Streaming render/capture C API** (`aec_analyze_render` / `aec_process_capture`, async FIFO over the
   bit-exact engine — lockstep is byte-identical to `aec_process`).
2. **Dead-code removal** (both impls): −1312 lines Python, −423 lines C. Removed 10 default-OFF research
   flags (none enabled by any preset) + the opt-in DTD detector subsystem + `NearendSpp`. Python↔C remain
   **bit-exact** (peak|Δ|=0, ±CNG, all 3 presets).
3. **Python CLI** now exposes all three presets (`--preset gentle|balanced|aggressive`), matching the C CLI.

## 800-case AECMOS — ours vs WebRTC AEC3 vs Speex (MDF)
Standard bench: `balanced / fl=832 / --cng`, local AECMOS ONNX, full 800-case blind corpus. echo↑ deg↑.

| Bucket | n | **ours** echo / deg | AEC3 echo / deg | Speex echo / deg |
|---|---:|---:|---:|---:|
| FS_static | 169 | 3.576 / 4.999 | 3.821 / 4.999 | 2.847 / 5.000 |
| FS_movement | 131 | 3.512 / 4.999 | 3.790 / 4.999 | 2.757 / 5.000 |
| DT_static | 186 | 4.201 / 2.156 | 4.531 / 1.815 | 3.427 / 3.179 |
| DT_movement | 114 | 4.082 / 2.228 | 4.456 / 1.816 | 3.272 / 3.301 |
| NE | 200 | 4.998 / 4.047 | 4.999 / 3.410 | 4.998 / 4.128 |

### Reading (Pareto-aware)
- **Echo cancellation: AEC3 > ours > Speex.** AEC3 cuts the most echo; Speex is a weak canceller
  (FS ~2.8, DT ~3.3); ours sits in between (FS ~3.5, DT ~4.1).
- **Near-end preservation (deg): Speex > ours > AEC3.** AEC3 pays for its echo cancellation with the worst
  DT deg (1.82) and NE deg (3.41); Speex preserves best but barely cancels; ours is the balanced middle.
- **Ours beats AEC3 on every degradation axis** — DT deg 2.16/2.23 vs 1.82, NE deg 4.047 vs 3.410 — while
  **approaching AEC3 on echo** (FS_movement 3.512 vs 3.790, within 0.28). This is the v3.22 ship target:
  *approach AEC3 on echo, beat AEC3 on deg.* All four ship bars met (NE deg ≥4, FS echo >3.5,
  DT echo >4, DT deg >2); **FS_movement 3.512 > 3.5** (the primary hard gate).
- The Pareto picture is identical to the single-case breath finding: Speex "sounds clean" because it is a
  weak canceller, not because it is smart; AEC3 is echo-priority and over-suppresses the near-end; ours
  holds the middle. The three **presets** (gentle −20 / balanced −28 / aggressive −38 dB far-active floor)
  let a product slide along exactly this echo↔deg axis.

## Verification ledger (this release)
- Python cleanup byte-equal: **27/27** wavs (9 cases × 3 presets, incl. movement) identical to pre-cleanup; 25/25 unit tests.
- 800-case scores **identical to 3.22.4** (FS_static 3.576 / DT_static 4.201·2.156 / NE 4.047) → cleanup changed nothing.
- C↔Python: **bit-exact (peak|Δ|=0)** on no-cng and cng, all 3 presets; per-module golden harnesses + e2e all pass.
- Three presets differ in **exactly one** field (`min_gain_floor_far_active_db`), 70-field config.

## Roadmap pointers (doc-only, no NN shipped)
- **NN integration seams**: [`nn_integration_interface.md`](nn_integration_interface.md) — how a model
  replaces the residual / NR / both on the shared 257-bin·hop-160·16 kHz grid via `AecResContext`.
- **Frequency-domain pipeline**: `Audio_ALG/docs/freq_domain_pipeline_design.md` — `AEC(linear) → NR →
  AEC(residual)` in one freq domain (window+FFT once, IFFT+OLA once); the seams already exist, the refactor
  removes 2 of 3 FFT/IFFT round-trips. Item-13 sensitivity: lock the grid at 16 kHz / fft=512 / hop=160
  (SR/hop are most baked into the delay subsystem + PSD calibration).
