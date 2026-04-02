---
name: aec-benchmark
description: "Run AEC benchmark on aec_challenge_blind test set and score with AECMOS. Use when: tuning AEC parameters, running AEC benchmark, evaluating AEC quality, checking aecmos scores, testing echo cancellation performance, or after adjusting AEC algorithm settings like mu, filter length, preset, RES parameters, etc."
argument-hint: "[--preset balanced|mild|aggressive|maximum] [--filter N] [extra args...]"
allowed-tools: Bash, Read, Grep, Glob
effort: high
---

# AEC Benchmark Skill

Run the full AEC benchmark pipeline: process the blind test set, then score with AECMOS.

## Project Paths

- **AEC Python dir**: `${CLAUDE_SKILL_DIR}/../../../python`
- **Blind test set**: `${CLAUDE_SKILL_DIR}/../../../wav/aec_challenge_blind`
- **Output dir**: `${CLAUDE_SKILL_DIR}/../../../wav/aec_challenge_blind/output`
- **Core algorithm**: `${CLAUDE_SKILL_DIR}/../../../python/aec.py` (AecConfig dataclass defines all tunable parameters)

## Workflow

### Step 1: Confirm Parameter Changes

Before running, briefly summarize what parameter changes were made (if any). Check if the user modified `aec.py` (AecConfig) or wants to pass CLI arguments.

Key tunable parameters in AecConfig:
- `filter_length` (default 512): Echo path modeling window
- `mu` (default 0.3): Adaptive filter step size
- `enable_res` / `res_g_min_db` / `res_over_sub`: Residual Echo Suppressor settings
- `enable_shadow` / `shadow_mu_ratio`: Shadow filter settings
- `kalman_q_high` / `kalman_q_low`: Kalman filter Q values
- Presets: mild, balanced, aggressive, maximum

### Step 2: Run AEC Processing

Execute `eval_aec_challenge.py` on the blind test set:

```bash
cd <AEC_ROOT>/python && python3 eval_aec_challenge.py ../wav/aec_challenge_blind/ $ARGUMENTS
```

Default arguments if none provided: `--filter 3200 --parallel`

If the user specified a preset, add `--preset <name>`. If comparing presets, use `--all-presets`.

This processes all scenarios (farend_singletalk, nearend_singletalk, doubletalk) and outputs:
- ERLE scores for farend_singletalk and doubletalk
- SDR scores for nearend_singletalk
- Processed WAV files in the output directory

### Step 3: Run AECMOS Scoring

Execute `eval_aecmos_local.py` to get objective quality scores:

```bash
cd <AEC_ROOT>/python && python3 eval_aecmos_local.py ../wav/aec_challenge_blind/
```

This uses the local ONNX model at `/tmp/AEC-Challenge/AECMOS/AECMOS_local/Run_1663915512_Stage_0.onnx`.

Output metrics:
- **echo_mos** (1-5): Echo presence score (higher = less echo)
- **deg_mos** (1-5): Degradation score (higher = less quality loss)

### Step 4: Present Results

After both steps complete, present a clear summary:

1. **ERLE/SDR metrics** from eval_aec_challenge.py (per scenario)
2. **AECMOS scores** (echo_mos, deg_mos per scenario and method)
3. If previous results exist, highlight improvements/regressions
4. Suggest next parameter adjustments if scores indicate room for improvement

## Important Notes

- Always use `python3` (macOS)
- The blind test set has no ground truth clean speech — use ERLE/SDR from eval_aec_challenge.py and echo_mos/deg_mos from AECMOS
- If AECMOS model is not found, remind user to clone: `git clone https://github.com/microsoft/AEC-Challenge /tmp/AEC-Challenge`
- Processing can take several minutes depending on filter length and number of test cases
