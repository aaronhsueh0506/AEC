#!/usr/bin/env python3
"""Phase 4 parity baseline: EchoPathChangeDetector + ShadowCopyController."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aec import EchoPathChangeDetector, ShadowCopyController, AecConfig

N_FRAMES = 200
SEED = 42
EPC_HANGOVER = 20
EPC_TOTAL_RISE = 1.5
EPC_DELTA_THR = 0.3
SHADOW_COPY_THRESHOLD = 0.7
SHADOW_COPY_HYSTERESIS = 5

OUT_FILE   = Path(__file__).parent / 'parity_phase4_python.npz'
INPUT_FILE = Path(__file__).parent / 'parity_phase4_input.bin'


def main():
    rng = np.random.default_rng(SEED)
    # EPC inputs
    far_pwr_global = (rng.uniform(0, 0.05, N_FRAMES)).astype(np.float32)
    # Inject a sudden gain change at frame 80
    far_pwr_global[80:120] *= 5.0
    main_err  = (rng.uniform(1e-5, 1e-2, N_FRAMES)).astype(np.float32)
    shadow_err= (rng.uniform(1e-5, 1e-2, N_FRAMES)).astype(np.float32)
    # Inject error rise around frame 130
    main_err[130:140] *= 3.0
    shadow_err[130:140] *= 3.0
    is_stationary_arr = np.zeros(N_FRAMES, dtype=np.int32)
    filter_converged_arr = np.array([1] * N_FRAMES, dtype=np.int32)
    filter_converged_arr[:50] = 0
    main_paused_arr = np.zeros(N_FRAMES, dtype=np.int32)

    # ShadowCopy inputs
    sat_level = (rng.uniform(0, 0.2, N_FRAMES)).astype(np.float32)
    dt_energy = (rng.uniform(0, 0.5, N_FRAMES)).astype(np.float32)
    epc_active_arr = np.zeros(N_FRAMES, dtype=np.int32)

    cfg = AecConfig(sample_rate=16000,
                    epc_hangover=EPC_HANGOVER,
                    epc_total_rise=EPC_TOTAL_RISE,
                    epc_delta_threshold=EPC_DELTA_THR,
                    shadow_copy_threshold=SHADOW_COPY_THRESHOLD,
                    shadow_copy_hysteresis=SHADOW_COPY_HYSTERESIS)

    epc = EchoPathChangeDetector(cfg)
    sc = ShadowCopyController(cfg, gate_mode='energy')

    # EPC outputs
    epc_active = np.zeros(N_FRAMES, dtype=np.int32)
    epc_hangover = np.zeros(N_FRAMES, dtype=np.int32)
    epc_gain_fast = np.zeros(N_FRAMES, dtype=np.float32)
    epc_gain_slow = np.zeros(N_FRAMES, dtype=np.float32)
    # Per-frame fired source: 0 none, 1 delay, 2 epv, 3 shadow_rise
    epc_event_source = np.zeros(N_FRAMES, dtype=np.int32)

    # ShadowCopy outputs
    sc_pause = np.zeros(N_FRAMES, dtype=np.int32)
    sc_boost_q = np.zeros(N_FRAMES, dtype=np.int32)
    sc_reverse = np.zeros(N_FRAMES, dtype=np.int32)
    sc_baseline = np.zeros(N_FRAMES, dtype=np.float32)

    for f in range(N_FRAMES):
        # Run EPC: epv, then shadow_rise (skip force_delay path)
        epv = epc.update_epv(
            far_pwr_global=float(far_pwr_global[f]),
            filter_converged=bool(filter_converged_arr[f]),
            main_paused=bool(main_paused_arr[f]),
        )
        sr = epc.update_shadow_rise(
            main_err_smooth=float(main_err[f]),
            shadow_err_smooth=float(shadow_err[f]),
            is_stationary=bool(is_stationary_arr[f]),
        )
        if epv.fired:
            src = 2
        elif sr.fired:
            src = 3
        else:
            src = 0
            # Tick hangover only when no fire
            epc.tick_hangover()
        epc_event_source[f] = src
        epc_active[f]   = int(epc.active)
        epc_hangover[f] = int(epc.hangover_count)
        epc_gain_fast[f]= epc.epv_gain_fast
        epc_gain_slow[f]= epc.epv_gain_slow

        # Run ShadowCopy
        decision = sc.update(
            shadow_frame_count=f,
            far_pwr=float(far_pwr_global[f]),
            main_err_smooth=float(main_err[f]),
            shadow_err_smooth=float(shadow_err[f]),
            epc_active=bool(epc_active_arr[f]),
            saturation_level=float(sat_level[f]),
            dt_from_energy=float(dt_energy[f]),
        )
        sc_pause[f]    = int(decision.pause_main)
        sc_boost_q[f]  = int(decision.boost_q)
        sc_reverse[f]  = int(decision.reverse_copy)
        sc_baseline[f] = sc.copy_err_baseline

    np.savez(OUT_FILE,
             epc_active=epc_active, epc_hangover=epc_hangover,
             epc_gain_fast=epc_gain_fast, epc_gain_slow=epc_gain_slow,
             epc_event_source=epc_event_source,
             sc_pause=sc_pause, sc_boost_q=sc_boost_q,
             sc_reverse=sc_reverse, sc_baseline=sc_baseline,
             config=np.array([N_FRAMES], dtype=np.int32))

    with open(INPUT_FILE, 'wb') as fp:
        fp.write(far_pwr_global.tobytes())
        fp.write(main_err.tobytes())
        fp.write(shadow_err.tobytes())
        fp.write(is_stationary_arr.tobytes())
        fp.write(filter_converged_arr.tobytes())
        fp.write(main_paused_arr.tobytes())
        fp.write(sat_level.tobytes())
        fp.write(dt_energy.tobytes())
        fp.write(epc_active_arr.tobytes())

    print(f"Saved Python baseline → {OUT_FILE}")
    print(f"Saved C input bin     → {INPUT_FILE}")
    print(f"  epc final: active={epc_active[-1]} hangover={epc_hangover[-1]}")
    print(f"      fast={epc_gain_fast[-1]:.3e} slow={epc_gain_slow[-1]:.3e}")
    print(f"  sc final:  pause={sc_pause[-1]} baseline={sc_baseline[-1]:.3e}")
    print(f"  EPC fires: epv={int(np.sum(epc_event_source==2))} sr={int(np.sum(epc_event_source==3))}")


if __name__ == '__main__':
    main()
