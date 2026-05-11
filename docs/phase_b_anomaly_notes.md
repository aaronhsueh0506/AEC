# Phase B Anomaly Notes

Per P52 v1.1 §8.3: anomalies discovered mid-Phase are logged here, not acted upon.

## B.1 — Inventory anomalies

**A2** (2026-05-11). `_stage_gain_compute` reads `self.noise_psd` at
[aec.py:1919-1920](../python/aec.py#L1919); `noise_psd` is owned by Module 5
(`_stage_noise_floor_and_cng`). I.e. Module 2 reads Module 5's previous-frame
state. Refactor must plumb `noise_psd` as an input to Module 2 from `ResState`;
not a bug, but a documented cross-module dependency.

**A3** (2026-05-11). `_stage_gain_postprocess` reads
`self._residual_est._long_window_far_psd` at
[aec.py:2030](../python/aec.py#L2030). Module 3 reads internal state of Module
1's estimator. B.2 must expose this as a public accessor
(`residual_estimator.long_window_far_psd`); §5.5 forbids changing the
underlying logic.

**A4** (2026-05-11). The "9 stages" naming in v1.1 §3.3 covers only the gain
pipeline (Modules 2-4 cover the 9 `_diag_round5_stages` slots plus Module 5).
`_stage_residual_model` (Module 1) has no `_diag_round5_stages` write. This is
a naming artefact, not a refactor blocker.
