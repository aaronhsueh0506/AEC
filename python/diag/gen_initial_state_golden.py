"""Generate a binary golden for the C initial_state port (WS5 Phase 5.2).

Drives the Python InitialState through a deterministic active_render /
saturated_capture sequence across THREE configs and snapshots the full state
(counter + initial_state + transition_triggered) per frame so state evolution
— including the single-frame transition edge, the saturation-gated counter, and
a mid-run reset — is verified, not just a final scalar.

Configs exercised:
  cfg 0: conservative=False, seconds=2.5  -> threshold 250 hops
  cfg 1: conservative=True,  seconds=2.5  -> threshold 500 hops (conservative)
  cfg 2: conservative=False, seconds=0.07 -> threshold 7 hops (fast edge,
          tests int() truncation: int(0.07*100)=6 in f64) + a reset() at frame.

Each config runs N_FRAMES frames. The active_render / saturated_capture pattern
is deterministic (period-based) so the counter only advances on
(active_render and not saturated_capture).

Layout (LE):
  int32 n_cfgs
  per cfg:
    int32 conservative, float64 seconds, int32 threshold_hops,
    int32 conservative_hops, int32 n_frames, int32 reset_at (-1 = none)
    n_frames × [ u8 active_render, u8 saturated_capture,
                 i32 strong_not_saturated_render_blocks,
                 u8 initial_state, u8 transition_triggered ]

Run: python3 python/diag/gen_initial_state_golden.py /tmp/initial_state_golden.bin
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.state.initial_state import InitialState  # noqa: E402

N_FRAMES = 620  # > 500 conservative threshold so all configs transition


def render_pattern(i):
    """Deterministic (active_render, saturated_capture) for frame i.

    active_render high most of the time; saturated_capture pulses every 4th
    active frame — so the counter must SKIP those frames. A short idle gap
    (active_render False) every 50 frames stalls the counter too.
    """
    active = 1 if (i % 50) >= 3 else 0          # idle gap frames 0,1,2 mod 50
    sat = 1 if (active and (i % 4) == 0) else 0  # saturation pulse
    return active, sat


CONFIGS = [
    # (conservative, seconds, reset_at)
    (0, 2.5,  -1),
    (1, 2.5,  -1),
    (0, 0.07, 300),  # tiny threshold (int(0.07*100)=6) + mid-run reset()
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/initial_state_golden.bin'
    with open(out, 'wb') as f:
        f.write(struct.pack('<i', len(CONFIGS)))
        for (cons, secs, reset_at) in CONFIGS:
            s = InitialState(conservative_initial_phase=bool(cons),
                             initial_state_seconds=secs)
            f.write(struct.pack('<idiii i', cons, secs,
                                s._initial_state_hops, s._conservative_hops,
                                N_FRAMES, reset_at))
            for i in range(N_FRAMES):
                if reset_at >= 0 and i == reset_at:
                    s.reset()
                active, sat = render_pattern(i)
                s.update(bool(active), bool(sat))
                f.write(struct.pack('<BBiBB', active, sat,
                                    s._strong_not_saturated_render_blocks,
                                    1 if s.initial_state_active() else 0,
                                    1 if s.transition_triggered() else 0))
    print(f"wrote {out}  ({len(CONFIGS)} configs, {N_FRAMES} frames each)")


if __name__ == '__main__':
    main()
