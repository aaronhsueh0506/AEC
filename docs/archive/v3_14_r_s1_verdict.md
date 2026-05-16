# v3.14 Arc-R Sprint S1 — per-band ENR threshold wire verdict

**Status**: WIRE COMPLETE. Default-OFF, byte-equal verified against
parent commit on 5-case sample (atol=0.0, 5/5 PASS). Full 800-case
byte-equal verification running in background via
`tools/research/v3_14_r_s1_byte_equal.py --full` (baseline gen-render
of all 800 cases at parent commit 9162d78 → `--full` sweep compares
modified code flag-OFF against per-case `.npy` baselines).

The wire correctness is also **provable by static analysis**:

- The `else:` branch at the use site (`python/aec.py:2853-2857`) is a
  character-for-character copy of the pre-R.S1 code.
- The new precompute step (`python/aec.py:1916-1936`) only allocates
  new attributes — no existing state is touched.
- When `_per_band_enr=False`, those new attributes are never read.

The full 800-case sweep is empirical confirmation, expected to PASS by
construction. If it FAILS, that would indicate a hidden side-effect
the 5-case sample didn't surface, which would require investigation.

**Date**: 2026-05-14

**Branch**: `feature/v3.14-arc-r` (from `feature/v3.14-arc-p` @ 9162d78)

## TL;DR

R.S1 adds a `res_per_band_enr` flag (default `False`) plus two
per-band tuple fields (`enr_t_ne_per_band`, `enr_s_ne_per_band`) on
`AecConfig`, plumbs them through `ResFilter.__init__`, precomputes
per-bin substitution arrays at init, and inserts a single
`if self._per_band_enr:` branch at the use site in
`ResFilter._stage_gain_compute`. The same branch is mirrored in the
extracted `python/res_refactored/gain_computer.py` so both code paths
remain in lockstep.

**Hard bar (byte-equal flag-OFF vs parent commit)**: 5/5 PASS on
sample, atol=0.0, all per-case `max|Δ|` is exactly zero. Full 800-case
running.

## Motivation

ENR thresholds `enr_t_ne` / `enr_s_ne` in
`ResFilter._stage_gain_compute` (`python/aec.py:2792-2793` pre-R.S1)
use a `_enr_blend`-driven interpolation between two scalar endpoints
(`2.0/1.5` for `t_ne`, `3.0/2.5` for `s_ne`). The blend ramps over
bins 5-10 (~150-300 Hz @ 16 kHz / 256-fft); bins 11+ asymptote to
the high-band scalar. After P-revised P.S3's adaptive per-band ERL
(LF/MF/HF, which can vary 4× per-room — case 04 had LF=0.043,
MF=0.191, HF=0.111 vs scalar capped 0.3), the downstream ENR
thresholds remain band-flat. This **clips P.S3's signal before it
reaches the gain decision** — band-tuned ERL feeds a band-flat gate.

Per-bin audit findings memo (`project_per_bin_audit_findings.md`)
ranks `enr_t_ne / enr_s_ne` as the **#2 per-band candidate** right
after `erl_estimate` (which P-revised P.S3 addresses).

## Files modified

| File | Lines added/changed | Purpose |
| --- | --- | --- |
| `python/aec.py` | +69 -3 | `AecConfig` fields (lines 731-736); `ResFilter.__init__` params (lines 1864-1882) + precompute (lines 1916-1936); use-site branch (lines 2845-2858); AEC instantiation plumbing (lines 5252-5254). |
| `python/res_refactored/gain_computer.py` | +9 -2 | Mirror the same branch in the extracted module (lines 121-135) so `ResFilterRefactored` stays byte-equal. |
| `python/eval_aec_challenge.py` | +5 | New env override `AEC_RES_PER_BAND_ENR` (lines 171-174). |
| `tools/research/v3_14_r_s1_byte_equal.py` | new file (~210 lines) | 5-case sanity + 800-case full byte-equal sweep with gen-baseline / compare-against-baseline flow. |
| `docs/v3_14_r_s1_verdict.md` | new file (this) | R.S1 verdict + R.S2 handoff. |

## Plug-in points

**`AecConfig`** (`python/aec.py:731-736`):
```python
res_per_band_enr: bool = False
enr_t_ne_per_band: tuple = (1.5, 1.5, 1.5)
enr_s_ne_per_band: tuple = (2.5, 2.5, 2.5)
```

**`ResFilter.__init__`** (`python/aec.py:1864-1882, 1916-1936`):
- New params `per_band_enr`, `enr_t_ne_per_band`, `enr_s_ne_per_band`
  (with the same defaults as `AecConfig`).
- Precompute per-bin arrays `_enr_t_ne_pb[n_freqs]` and `_enr_s_ne_pb[n_freqs]`
  from the 3-band tuples using the **same band boundaries (1 kHz / 4 kHz)
  as P.S3 adaptive per-band ERL EMA** so the two arcs operate on
  consistent frequency partitions.
- Bin boundaries cached: `_enr_per_band_b1k` (= 32 at 16 kHz / 512-fft) /
  `_enr_per_band_b4k` (= 128).

**Use site** (`python/aec.py:2845-2858`):
```python
if self._per_band_enr:
    enr_t_ne = self._enr_t_ne_pb
    enr_s_ne = self._enr_s_ne_pb
else:
    enr_t_ne = (1 - blend) * 2.0 + blend * 1.5
    enr_s_ne = (1 - blend) * 3.0 + blend * 2.5
```

**Refactored module mirror**
(`python/res_refactored/gain_computer.py:121-135`): identical branch
keyed on `rf._per_band_enr`.

**AEC instantiation** (`python/aec.py:5252-5254`): forwards three new
config fields into the `ResFilter` constructor.

**Env override** (`python/eval_aec_challenge.py:171-174`):
`AEC_RES_PER_BAND_ENR` → `config_overrides['res_per_band_enr']`.

## Default values rationale

Defaults are **uniform** = the post-blend high-band asymptote of the
legacy formula. After bin 10 (~300 Hz @ 16 kHz / 256-fft), the legacy
blend has fully ramped → `enr_t_ne = 1.5`, `enr_s_ne = 2.5` for all
bins 11+. So:

| Tuple | Default value | Origin |
| --- | --- | --- |
| `enr_t_ne_per_band` | `(1.5, 1.5, 1.5)` | high-band asymptote of legacy `(1-blend)*2.0 + blend*1.5` (= 1.5 for blend=1) |
| `enr_s_ne_per_band` | `(2.5, 2.5, 2.5)` | high-band asymptote of legacy `(1-blend)*3.0 + blend*2.5` (= 2.5 for blend=1) |

Flag-OFF byte-equal is preserved by construction (the `else:` branch
is line-for-line identical to the pre-R.S1 code). Flag-ON with the
default tuples differs from flag-OFF only in bins 0-10 (DC-300 Hz),
where the legacy blend has `enr_t_ne` in `[1.5, 2.0]` (linear ramp
from blend=0 → blend=1) and the uniform default gives `1.5`
throughout. The flag-ON deviation is empirically observed in the
sample run:

| Case | Flag-ON `max|Δ|` vs flag-OFF |
| --- | --- |
| NE | 0.000000e+00 |
| FS_static | 2.878147e-06 |
| FS_mvmt | 9.622348e-03 |
| DT_static | 9.465870e-03 |
| DT_mvmt | 1.736448e-02 |

(Magnitudes are larger than just the LF blend mismatch because the
per-bin `dt_per_bin` × `enr_t_ne` blend with `enr_t_fs` propagates the
small LF change through every downstream stage; this is expected and
R.S1 makes no claim about flag-ON neutrality.)

## Byte-equal flag-OFF result

### 5-case sample (sanity)

Methodology: stashed R.S1 changes → rendered baselines on parent
commit (9162d78, `feature/v3.14-arc-p` HEAD) → popped stash → re-ran
with R.S1 code and `res_per_band_enr=False` → compared.

| Bucket | Case stem (prefix) | Baseline source | flag-OFF `max|Δ|` | Verdict |
| --- | --- | --- | --- | --- |
| NE | `014AzuqPZku2004NbTTmcA_nearend_singletalk` | parent 9162d78 | 0.000000e+00 | PASS |
| FS_static | `0KjzXA3g20qsd8zmSekADw_farend_singletalk` | parent 9162d78 | 0.000000e+00 | PASS |
| FS_mvmt | `0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement` | parent 9162d78 | 0.000000e+00 | PASS |
| DT_static | `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | parent 9162d78 | 0.000000e+00 | PASS |
| DT_mvmt | `49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement` | parent 9162d78 | 0.000000e+00 | PASS |

**Result: 5/5 PASS at atol=0.0**.

### 800-case full sweep

See "Status" header. Running in background via
`tools/research/v3_14_r_s1_byte_equal.py --full`. Results appended
after completion. The byte-equal property is proven by construction
(the flag-OFF `else:` branch is character-for-character identical to
the pre-R.S1 code, and the precompute step only allocates new
attributes that are never read when the flag is OFF), so the full
sweep is empirical confirmation — failure would indicate a hidden
side-effect that 5-case sample did not catch.

#### Additional byte-equal property: `ResFilterRefactored` parity

The wire is mirrored in `python/res_refactored/gain_computer.py`,
which `ResFilterRefactored` delegates to. A separate sanity run
verified that on the same case (FS_static), output from
`ResFilter` and `ResFilterRefactored` is bit-identical for both
flag states (`max|Δ| = 0.0` for both OFF and ON). This means the
P52 Phase B subclass-and-delegate parity continues to hold under
R.S1.

## What R.S2 (tuning) will need to do

R.S1 ships the wire. R.S2 (next sprint) will tune the per-band
tuple values via 800-case A/B / AECMOS scoring. Expected work:

1. **Baseline characterisation**: render 800-case with
   `res_per_band_enr=True` and the default `(1.5, 2.5)` uniform
   tuples. Diff against pre-R.S1 baseline. Confirm the diff is
   dominated by bins 0-10 (LF only).
2. **Coarse sweep**: 3-level grid for each of LF/MF/HF over each of
   `t_ne` and `s_ne`. Suggested anchor points:
   - `enr_t_ne_per_band`: `{(2.0, 1.5, 1.0), (1.5, 1.5, 1.5),
     (1.0, 1.5, 2.0)}` — keep `MF` central, sweep LF and HF.
   - `enr_s_ne_per_band`: mirror the above scaled by 1.67×.
3. **AECMOS-anchored selection**: BALANCED, fl=832, cng=True, j=4.
   FS Δecho bar = -0.005 (no per-bucket regression), DT Δdeg bar =
   -0.025 (per-bucket).
4. **xrtntuju regression check**: re-render 5 DT NE-positive
   windows and confirm no degradation (CLAUDE.md feedback file).
5. **Per-band ERL × per-band ENR co-action audit**: with
   P.S3 flag ON simultaneously (the intended productisation pair),
   verify the two flags interact constructively. P.S3 changes
   `erl_estimate` per-bin → affects F3.1-v3 mic-excess → affects
   `dt_per_bin` → affects the `ne_confidence` blend that gates
   `enr_t_ne` (which R.S2 will tune). Compound effects need a
   dedicated R.S2/P.S3 paired A/B.

## Constraints honoured

- DO NOT tune the per-band values: defaults left as uniform.
- DO NOT touch other arcs (P / D / H / S-orth): no edits outside
  R.S1 surface.
- DO NOT push to remote / merge to main: not done.
- HEREDOC commit message with `Co-Authored-By` footer.
- File-specific git add (no `-A` / `.`).

## Surprises / open notes

- **None functional**. The legacy blend formula at bin 11+ asymptotes
  cleanly to the per-band default — this made the default selection
  trivial. Flag-ON with defaults still differs from flag-OFF in
  bins 0-10 (DC-300 Hz), which is non-zero but expected; R.S1 does
  not claim flag-ON neutrality (that is a tuning property, not a
  wire property).
- **Bench command pattern note**: the task description quoted
  `--fl 832 --cng True --j 4 --limit 5`, but `python/eval_aec_challenge.py`
  exposes `--filter` (not `--fl`), `--cng` (boolean flag, no value),
  `--parallel` (3-way scenario fork, no `--j N` worker count), and no
  `--limit`. Byte-equal verification therefore uses the dedicated
  research script (`tools/research/v3_14_r_s1_byte_equal.py`) which
  does its own ProcessPoolExecutor parallelism and matches against
  `.npy` baselines rendered from the parent commit.
- **Baseline JSON drift**: not applicable — R.S1 does not run AECMOS,
  only byte-equal. R.S2 will run AECMOS and must verify baseline JSON
  is fresh post-9162d78.
