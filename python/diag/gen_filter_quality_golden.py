"""Generate a binary golden for the C filter_quality port (WS5 Phase 5.2).

Runs the Python FilteringQualityAnalyzer over a deterministic multi-frame
boolean/int input sequence and writes, for each per-hop step, the inputs plus
the full observable state after the update:
    overall flags : linear_filter_usable, gate1_pass, gate2_pass
    counters      : startup_blocks, reset_blocks (int)
    internal      : convergence_seen, convergence_hops_counter, overall_usable

This module is PURE INTEGER COUNTERS + bool gates (no float math), so byte-equal
means exact int/bool agreement. The sequence deliberately exercises:
  - counter accumulation past _STARTUP_HOPS (40) and _RESET_HOPS (20),
  - filter_update gating (active_render && not saturated_capture),
  - transparent_mode masking the final usable result,
  - gate3 via external_delay presence vs convergence_seen latch,
  - reset() semantics: blocks_since_start NOT reset, blocks_since_reset cleared,
    convergence_seen latches across reset.

REAL input semantics (captured from aec_state.py:375 call site, balanced
16 kHz / fl=832 pipeline). All inputs are Python bools / Optional[DelayEstimate]
(consumed by presence only) — there are no float arrays in this module:
    active_render          : bool
    transparent_mode       : bool
    saturated_capture      : bool
    external_delay         : Optional[DelayEstimate]  (presence -> int flag)
    any_filter_converged   : bool

Layout (LE):
  int32 n_steps
  n_steps × [
    int32 active_render
    int32 transparent_mode
    int32 saturated_capture
    int32 external_delay_present
    int32 any_filter_converged
    int32 do_reset                 (1 -> call reset() BEFORE this update)
    -- expected observable state after (optional reset +) update --
    int32 linear_filter_usable
    int32 gate1_pass
    int32 gate2_pass
    int32 startup_blocks
    int32 reset_blocks
    int32 convergence_seen
    int32 convergence_hops_counter
    int32 overall_usable
  ]

Run: python3 python/diag/gen_filter_quality_golden.py /tmp/fq_golden.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.state.filter_quality import FilteringQualityAnalyzer  # noqa: E402
from modules.delay.delay_types import DelayEstimate, DelayQuality  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/fq_golden.bin'

    # A real-shaped external delay object (consumed by presence only).
    DELAY = DelayEstimate(quality=DelayQuality.REFINED, delay=320)

    # Build the deterministic step list. Each entry:
    #   (active_render, transparent_mode, saturated_capture,
    #    external_delay, any_filter_converged, do_reset)
    steps = []

    # 1) 45 active hops, no external delay, no convergence yet.
    #    startup crosses 40 at hop 41; reset crosses 20 at hop 21. But gate3
    #    has neither external_delay nor convergence -> usable stays False.
    for _ in range(45):
        steps.append((True, False, False, None, False, False))

    # 2) introduce convergence (latches) -> gate3 satisfied; counters already
    #    past thresholds -> usable should flip True (TM off).
    for _ in range(3):
        steps.append((True, False, False, None, True, False))

    # 3) transparent_mode on -> usable forced False even though gates pass.
    for _ in range(2):
        steps.append((True, True, False, None, False, False))

    # 4) saturated_capture on -> filter_update suppressed (counters frozen),
    #    but convergence latch already set; usable stays True (no TM).
    for _ in range(2):
        steps.append((True, False, True, None, False, False))

    # 5) active_render off -> filter_update suppressed too; counters frozen.
    for _ in range(2):
        steps.append((False, False, False, None, False, False))

    # 6) reset() before update: blocks_since_reset cleared, blocks_since_start
    #    preserved, convergence_seen latches. reset_ok fails until reset counter
    #    climbs back past 20, while startup_ok stays True.
    steps.append((True, False, False, None, False, True))   # reset here
    for _ in range(19):                                     # reset counter 1..20
        steps.append((True, False, False, None, False, False))
    # at this point reset_blocks == 20 -> reset_ok still False (> 20 needed)
    for _ in range(3):                                      # cross 20 -> usable
        steps.append((True, False, False, None, False, False))

    # 7) external_delay present but convergence already latched (still gate3).
    for _ in range(2):
        steps.append((True, False, False, DELAY, False, False))

    # 8) external_delay present, fresh analyzer-like path: drop convergence
    #    relevance by relying on external_delay for gate3 (latch still set, but
    #    this exercises the presence flag path explicitly).
    for _ in range(2):
        steps.append((True, False, False, DELAY, True, False))

    # 9) second reset, then immediately transparent — usable False two ways.
    steps.append((True, True, False, None, False, True))    # reset + TM
    for _ in range(2):
        steps.append((True, True, False, None, False, False))

    # 10) tail: a few more active converged hops to confirm recovery.
    for _ in range(25):
        steps.append((True, False, False, None, True, False))

    fq = FilteringQualityAnalyzer(use_linear_filter=True)

    n_steps = len(steps)
    with open(out, 'wb') as f:
        np.array([n_steps], dtype=np.int32).tofile(f)
        for (active_render, transparent_mode, saturated_capture,
             external_delay, any_filter_converged, do_reset) in steps:
            ext_present = 0 if external_delay is None else 1
            np.array([
                1 if active_render else 0,
                1 if transparent_mode else 0,
                1 if saturated_capture else 0,
                ext_present,
                1 if any_filter_converged else 0,
                1 if do_reset else 0,
            ], dtype=np.int32).tofile(f)

            if do_reset:
                fq.reset()
            fq.update(
                active_render=active_render,
                transparent_mode=transparent_mode,
                saturated_capture=saturated_capture,
                external_delay=external_delay,
                any_filter_converged=any_filter_converged,
                filter_analyzer_consistent=False,  # no-op in source
            )

            np.array([
                1 if fq.linear_filter_usable() else 0,
                1 if fq.gate1_pass() else 0,
                1 if fq.gate2_pass() else 0,
                int(fq.startup_blocks()),
                int(fq.reset_blocks()),
                1 if fq._convergence_seen else 0,
                int(fq._convergence_hops_counter),
                1 if fq._overall_usable else 0,
            ], dtype=np.int32).tofile(f)

    print(f"wrote {out}  ({n_steps} steps, pure-int gates)")


if __name__ == '__main__':
    main()
