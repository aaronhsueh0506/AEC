# P3 arc research scripts

Audit / analysis helpers used during the P3 arc (v3.10.4 post-release
7GT investigation). Inputs are CSVs produced by `python/run_one_case.py
--trace-aec-state` or `python/trace_delay_acquisition.py`. Findings are
rolled up in `docs/SUMMARY.md` Round 8.

| Script | Purpose |
|---|---|
| `p3c_summarize.py` | 800-case TTFS summary (per-bucket median, fast-path-pass rate) from `trace_delay_acquisition.py` output |
| `p3c_c_parity_check.c` | Standalone C harness: confirms `delay_est_confidence()` fast-path matches Python on 6 cases |
| `p3f_audit.py` | Per-state invariant audit on 5-case Phase 2 trace set (Mini AecState classifier) |
| `p3g_audit.py` | Linear-vs-render residual PSD median per state (RES dry-run) |
| `p3h_compare.py` | Pre/post diverged-reset trace comparison (ERLE_win / state distribution) |
| `p3h_aecmos.py` | Single-case AECMOS scoring helper for 7GT reset dry-run |

These scripts are research tools, not part of the shipped pipeline.
Re-running them requires regenerating the input CSVs from the AEC
Challenge blind set.
