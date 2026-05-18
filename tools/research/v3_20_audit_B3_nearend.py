"""B.3 nearend wiring smoke — 5 cohort cases.

For each cohort worst-case, dump:
  - SG amp_gain MIN / MEDIAN distribution (p10/p50/p90)
  - final gain_smooth distribution
  - ENR sum ratio
  - mic_rms vs out_rms (per-case ERLE)

PRE-fix expected: amp_gain_med p50 = 1.000 (SG never fires on median)
POST-fix expected: amp_gain_med p50 < 1.000 on FS-active frames
"""
import os, sys, numpy as np, soundfile as sf

# AEC3 full chain on
os.environ['AEC_USE_AEC3_SUPPRESSION_GAIN'] = '1'
os.environ['AEC_USE_AEC3_RESIDUAL_ESTIMATOR'] = '1'
os.environ['AEC_AEC_STATE_FULL_ENABLED'] = '1'
os.environ['AEC_RES_AEC3_SKIP_LEGACY_POST'] = '1'
os.environ['AEC_AEC3_DELAY_CONTROLLER_ENABLED'] = '1'

sys.path.insert(0, '/Users/mingyu/Desktop/novatek/SE/AEC/python')
from aec import AEC
from modules.config import AecConfig, AecPreset

CORPUS = '/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind'
CASES = [
    ('FS_static',   f'{CORPUS}/farend_singletalk/9xjhiFbGo06hdQIsHTS6qA_farend_singletalk'),
    ('FS_movement', f'{CORPUS}/farend_singletalk/0luXwWjGEEC9G5nf0yTVXw_farend_singletalk_with_movement'),
    ('DT_static',   f'{CORPUS}/doubletalk/p0mhFbhV6kGJgjd0RTTIIw_doubletalk'),
    ('DT_movement', f'{CORPUS}/doubletalk/wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement'),
    ('NE',          f'{CORPUS}/nearend_singletalk/SR68lGQwTUy508j0P8BKZQ_nearend_singletalk'),
]


def run_one(stem_prefix):
    mic, sr = sf.read(stem_prefix + '_mic.wav')
    lpb, _ = sf.read(stem_prefix + '_lpb.wav')

    cfg = AecConfig.from_preset(AecPreset.BALANCED)
    cfg.cng = True
    cfg.res_use_aec3_estimator = True
    cfg.res_use_aec3_suppression_gain = True
    cfg.res_aec3_skip_legacy_post_stages = True
    cfg.aec_state_full_enabled = True

    np.random.seed(42)
    aec = AEC(cfg)
    orch = aec
    hop = orch._hop_size

    # Patch SG to capture per-frame inputs / outputs
    from modules.suppression_gain_aec3 import Aec3SuppressionGain
    _orig = Aec3SuppressionGain.get_gain
    cap = {'amp_mins': [], 'amp_meds': [], 'near_sums': [],
           'echo_sums': [], 'R2_sums': []}

    def patched(self, *, nearend_spectrum, echo_spectrum,
                residual_echo_spectrum, comfort_noise_spectrum,
                render_block, saturated_echo, clock_drift):
        amp, hbg = _orig(
            self, nearend_spectrum=nearend_spectrum,
            echo_spectrum=echo_spectrum,
            residual_echo_spectrum=residual_echo_spectrum,
            comfort_noise_spectrum=comfort_noise_spectrum,
            render_block=render_block,
            saturated_echo=saturated_echo, clock_drift=clock_drift)
        cap['amp_mins'].append(float(amp.min()))
        cap['amp_meds'].append(float(np.median(amp)))
        cap['near_sums'].append(float(nearend_spectrum.sum()))
        cap['echo_sums'].append(float(echo_spectrum.sum()))
        cap['R2_sums'].append(float(residual_echo_spectrum.sum()))
        return amp, hbg

    Aec3SuppressionGain.get_gain = patched

    out_audio = np.zeros_like(mic)
    nfr = len(mic) // hop
    mic_rms = np.zeros(nfr); out_rms = np.zeros(nfr)
    final_min = np.zeros(nfr); final_med = np.zeros(nfr)
    usable_lin_rate = 0
    for i in range(nfr):
        m = mic[i*hop:(i+1)*hop]
        r = lpb[i*hop:(i+1)*hop]
        if len(m) < hop:
            break
        o = aec.process(m, r)
        out_audio[i*hop:(i+1)*hop] = o[:len(m)]
        mic_rms[i] = float(np.sqrt(np.mean(m**2)))
        out_rms[i] = float(np.sqrt(np.mean(o[:len(m)]**2)))
        gs = orch.res.gain_smooth
        final_min[i] = float(gs.min())
        final_med[i] = float(np.median(gs))
        try:
            if orch._aec3_full_state is not None and orch._aec3_full_state.usable_linear_estimate():
                usable_lin_rate += 1
        except Exception:
            pass

    # Restore class patch
    Aec3SuppressionGain.get_gain = _orig

    # Stats — active frames only
    active = mic_rms > 5e-3
    am = np.array(cap['amp_mins'])
    ad = np.array(cap['amp_meds'])
    nm = np.array(cap['near_sums'])
    ec = np.array(cap['echo_sums'])
    rr = np.array(cap['R2_sums'])
    enr_ratio = np.where(nm > 1e-8, rr / nm, 0.0)
    erle = np.where(out_rms > 1e-8, mic_rms**2 / out_rms**2, 0.0)

    def p(arr, mask=None):
        sel = arr[mask] if mask is not None else arr
        if len(sel) == 0:
            return (0.0, 0.0, 0.0)
        return (float(np.percentile(sel, 10)),
                float(np.percentile(sel, 50)),
                float(np.percentile(sel, 90)))

    return {
        'frames': nfr, 'active': int(active.sum()),
        'usable_lin_rate': usable_lin_rate, 'usable_lin_pct': 100.0 * usable_lin_rate / max(nfr, 1),
        'amp_min': p(am, active),
        'amp_med': p(ad, active),
        'final_min': p(final_min, active),
        'final_med': p(final_med, active),
        'enr_ratio': p(enr_ratio, active),
        'erle_lin': p(erle, active),
        'mic_rms': p(mic_rms, active),
        'out_rms': p(out_rms, active),
    }


b3_on = os.environ.get('AEC_B3_USE_ERROR_PSD', '0') == '1'
print(f"=== AEC_B3_USE_ERROR_PSD={'1 (fix ON)' if b3_on else '0 (baseline)'}\n")
print(f"{'case':12s}  {'usable%':>8s}  {'amp_min p50':>11s}  {'amp_med p50':>11s}  "
      f"{'final_min p50':>13s}  {'final_med p50':>13s}  "
      f"{'ENR p50':>9s}  {'ERLE p50 (dB)':>13s}  {'out_rms p50':>11s}")
print("-" * 140)
for name, stem in CASES:
    r = run_one(stem)
    erle_db_p50 = 10 * np.log10(max(r['erle_lin'][1], 1e-10))
    print(f"{name:12s}  {r['usable_lin_pct']:7.1f}%  {r['amp_min'][1]:11.4f}  {r['amp_med'][1]:11.4f}  "
          f"{r['final_min'][1]:13.4f}  {r['final_med'][1]:13.4f}  "
          f"{r['enr_ratio'][1]:9.4f}  {erle_db_p50:13.2f}  {r['out_rms'][1]:11.4f}")
