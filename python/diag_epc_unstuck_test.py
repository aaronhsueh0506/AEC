"""A/B test: does ticking EPC hangover unconditionally fix the worst movement-DT cases?

Hypothesis (from worst-10 analysis):
  In v2.8.1, EPC hangover only ticks down inside `(shadow_filter and filter_converged)` gate.
  Cases that never converge (case sx6m) or converge late (cases 1, 2, 6) get stuck with
  epc_active=True for the entire clip, blocking shadow→main copy and capping ERL.

Fix:
  Move tick_hangover() to run every frame, regardless of convergence state.
  Implemented via monkey-patching AEC.process for clean A/B without permanent code change.

Variants:
  V0_baseline: v2.8.1 behavior (tick gated on converged)
  V1_unstuck:  tick every frame

Compares AECMOS on the same worst-10 cases that were identified in
diag_worst_movement_dt.py. If V1 closes the +1.0 echo gap, the fix is real.
"""
import json
import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV  = REPO / 'wav/aec_challenge_blind/doubletalk'
OUT  = Path(__file__).parent / 'output_worst'
MODEL = REPO / 'model/Run_1663915512_Stage_0.onnx'
sys.path.insert(0, str(REPO / 'model'))


def _load_worst10():
    rank = json.load(open(OUT / 'movement_dt_ranking.json'))
    return rank[:10]


def _make_aec():
    cfg = AecConfig.from_preset('balanced', sample_rate=16000, mode=AecMode.PBFDKF,
                                enable_dtd=False, enable_shadow=True, enable_res=True,
                                use_kalman=True, enable_cng=False,
                                enable_delay_est=True, delay_est_period_s=0.25,
                                delay_est_init_s=0.2)
    return AEC(cfg)


def _run(case, unstuck: bool):
    """Run AEC on one case. If unstuck=True, hangover ticks every frame."""
    mic, sr = sf.read(str(WAV / f'{case["stem"]}_mic.wav'), dtype='float32')
    lpb, _  = sf.read(str(WAV / f'{case["stem"]}_lpb.wav'), dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))

    aec = _make_aec()
    hop = aec.hop_size

    out = np.zeros(n, dtype=np.float32)
    pos = 0
    epc_changes = []  # (frame, active)
    prev_active = False
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        # FIX: tick hangover always (not just inside converged gate)
        if unstuck:
            # Only tick if no fire happened this frame.
            # (we approximate by ticking unconditionally; if a fire happened,
            # hangover was just set to N, ticking once → N-1, harmless one-frame off)
            aec._epc_det.tick_hangover()
        cur = aec.epc_active
        if cur != prev_active:
            epc_changes.append((pos // hop, cur))
        prev_active = cur
        pos += hop

    return out[:pos], sr, epc_changes


def main():
    sys.path.insert(0, str(REPO / 'model'))
    from aecmos import AECMOSEstimator
    aecmos = AECMOSEstimator(str(MODEL))

    cases = _load_worst10()
    print(f'{"stem":<55s} {"V0 echo":>8s} {"V1 echo":>8s} {"Δ":>7s}'
          f'  {"V0 deg":>7s} {"V1 deg":>7s} {"Δ":>7s}  v0_tr v1_tr')
    print('-' * 130)
    sums = {'v0_e': 0, 'v1_e': 0, 'v0_d': 0, 'v1_d': 0}
    for case in cases:
        mic, sr = sf.read(str(WAV / f'{case["stem"]}_mic.wav'), dtype='float32')
        lpb, _  = sf.read(str(WAV / f'{case["stem"]}_lpb.wav'), dtype='float32')
        if mic.ndim > 1: mic = mic[:, 0]
        if lpb.ndim > 1: lpb = lpb[:, 0]

        v0, _, v0_trans = _run(case, unstuck=False)
        v1, _, v1_trans = _run(case, unstuck=True)

        # Trim everything to common length for AECMOS
        n = min(len(mic), len(lpb), len(v0), len(v1))
        m = mic[:n]; l = lpb[:n]; v0p = v0[:n]; v1p = v1[:n]

        v0_e, v0_d = aecmos.run('dt', l, m, v0p)
        v1_e, v1_d = aecmos.run('dt', l, m, v1p)

        sums['v0_e'] += v0_e; sums['v1_e'] += v1_e
        sums['v0_d'] += v0_d; sums['v1_d'] += v1_d

        print(f'{case["stem"]:<55s} {v0_e:>8.3f} {v1_e:>8.3f} {v1_e-v0_e:>+7.3f}'
              f'  {v0_d:>7.3f} {v1_d:>7.3f} {v1_d-v0_d:>+7.3f}'
              f'    {len(v0_trans):>4d}     {len(v1_trans):>4d}')

    n = len(cases)
    print('-' * 130)
    print(f'{"MEAN (worst-10)":<55s} {sums["v0_e"]/n:>8.3f} {sums["v1_e"]/n:>8.3f}'
          f' {(sums["v1_e"]-sums["v0_e"])/n:>+7.3f}'
          f'  {sums["v0_d"]/n:>7.3f} {sums["v1_d"]/n:>7.3f} {(sums["v1_d"]-sums["v0_d"])/n:>+7.3f}')


if __name__ == '__main__':
    main()
