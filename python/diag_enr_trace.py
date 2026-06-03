"""Read-only per-bin ENR/R2/near trace — locate WHY the ENR ramp kills near in DT.

Replicates eval_aec_challenge.run_ours preprocessing, monkeypatches the
SuppressionGain ENR ramp + get_gain to snapshot per-frame internals, then asks:
  in near-active DT frames where the ramp fires (suppresses), is ENR = R2/(near+1)
  high because R2 OVER-ESTIMATES echo (L1: inflation) or because the thresholds
  are just aggressive (L2)?

Usage: python3 diag_enr_trace.py <mic.wav> <lpb.wav> [is_movement]
"""
import sys, numpy as np, soundfile as sf
from aec import AEC, AecConfig, AecPreset, AecMode
from eval_aec_challenge import estimate_delay


def _mono(x):
    return x[:, 0] if x.ndim > 1 else x


def main():
    mic_p, lpb_p = sys.argv[1], sys.argv[2]
    is_mv = len(sys.argv) > 3 and sys.argv[3] == '1'
    mic, sr = sf.read(mic_p); ref, _ = sf.read(lpb_p)
    mic = _mono(mic).astype(np.float32); ref = _mono(ref).astype(np.float32)
    n = min(len(mic), len(ref)); mic, ref = mic[:n], ref[:n]

    delay = estimate_delay(mic, ref, sr)
    ref_al = np.zeros(n, dtype=np.float32)
    if 0 < delay < n:
        ref_al[delay:] = ref[:n - delay]
    else:
        ref_al = ref[:n]

    delay_kw = dict(enable_delay_est=True, delay_est_period_s=0.25,
                    delay_est_init_s=0.2) if is_mv else dict(enable_delay_est=False)
    import os
    extra = {}
    _sub = os.environ.get('AEC_ERLE_SUBCONV')
    if _sub is not None:
        extra['erle_gate_subtractor_converged'] = float(_sub) > 0
        extra['erle_gate_subtractor_threshold'] = float(_sub) if float(_sub) <= 1 else 0.5
        _fl = os.environ.get('AEC_ERLE_SUBCONV_FLOOR')
        if _fl: extra['erle_gate_subtractor_y2_floor'] = float(_fl)
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=832, enable_dtd=False, enable_shadow=True,
        enable_res=True, use_kalman=True, enable_cng=True, **delay_kw, **extra)
    np.random.seed(0)
    aec = AEC(cfg)
    sg = aec._aec3_sg
    ree = aec._aec3_ree

    est = []
    orig_est = ree.estimate
    def wrap_est(*a, **k):
        st = k.get('aec_state')
        s2 = k.get('s2_linear'); xp = k.get('render_psd')
        try:
            usable = bool(st.usable_linear_estimate())
            erle = np.asarray(st.erle(), np.float64).copy()
        except Exception:
            usable = False; erle = np.ones(257)
        try:
            conv = bool(aec._filter_converged)
        except Exception:
            conv = False
        est.append(dict(usable=usable, conv=conv, erle=erle,
                        s2=np.asarray(s2, np.float64).copy() if s2 is not None else None,
                        xp=np.asarray(xp, np.float64).copy() if xp is not None else None))
        return orig_est(*a, **k)
    ree.estimate = wrap_est

    rec = []
    orig_ramp = sg._gain_to_no_audible_echo
    def wrap_ramp(nearend, echo, masker):
        g = orig_ramp(nearend, echo, masker)
        rec.append(dict(
            near=np.asarray(nearend, np.float64).copy(),
            r2=np.asarray(echo, np.float64).copy(),          # weighted_residual fed to ramp
            r2_dir=np.asarray(getattr(ree, '_last_r2_direct_component', np.zeros_like(echo)), np.float64).copy(),
            r2_rev=np.asarray(getattr(ree, '_last_r2_reverb_component', np.zeros_like(echo)), np.float64).copy(),
            ramp_g=np.asarray(g, np.float64).copy(),
            fire=np.asarray(sg._last_fire_mask, bool).copy(),
            enr=np.asarray(sg._last_enr_raw, np.float64).copy(),
        ))
        return g
    sg._gain_to_no_audible_echo = wrap_ramp

    orig_gain = sg.get_gain
    finals = []
    def wrap_gain(*a, **k):
        G = orig_gain(*a, **k)
        finals.append(np.asarray(G, np.float64).copy())
        return G
    sg.get_gain = wrap_gain

    hop = aec.hop_size
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos+hop], ref_al[pos:pos+hop])
        pos += hop

    if not rec:
        print("no frames captured"); return
    near = np.stack([r['near'] for r in rec])      # (F, K)
    r2   = np.stack([r['r2']   for r in rec])
    r2d  = np.stack([r['r2_dir'] for r in rec])
    r2r  = np.stack([r['r2_rev'] for r in rec])
    fire = np.stack([r['fire'] for r in rec])
    rampg= np.stack([r['ramp_g'] for r in rec])
    enr  = np.stack([r['enr']  for r in rec])
    F, K = near.shape
    nb = min(len(finals), F)
    finalg = np.stack(finals[:nb]) if nb else np.ones_like(near)
    if finalg.shape != near.shape:
        finalg = np.ones_like(near)

    # near-active frames: top 50% by total near energy (DT → near present widely)
    fe = near.sum(axis=1)
    thr = np.median(fe)
    act = fe > thr
    print(f"frames={F} bins={K} hop={hop} delay={delay}  near-active(>median)={act.sum()}")
    print()

    def band(lo, hi):
        b = slice(int(lo*K/(sr//2)), int(hi*K/(sr//2)) if hi else K)
        return b
    bands = {'LF 0-625': band(0,625), 'MF 625-1500': band(625,1500),
             'HF 1500-4000': band(1500,4000), 'VHF 4000+': band(4000,0)}

    A = act  # near-active frames mask
    print("In NEAR-ACTIVE frames:")
    print(f"  ramp fire-rate (bins suppressed):  {fire[A].mean():.1%}")
    print(f"  final gain median: {np.median(finalg[A]):.3f}   ramp-gain median: {np.median(rampg[A]):.3f}")
    print(f"  R2_reverb / R2_total (where R2>0): {(r2r[A].sum()/max(1e-9,(r2d[A].sum()+r2r[A].sum()))):.1%}")
    print()
    # KEY inflation test: in fire bins of near-active frames, how big is R2 vs near?
    fm = fire & A[:,None]
    if fm.any():
        ratio = r2[fm] / (near[fm] + 1.0)   # == ENR in fire bins
        print("  --- INFLATION TEST: R2/near (=ENR) in FIRING bins of near-active frames ---")
        print(f"    median ENR        : {np.median(ratio):.2f}   (>1 ⇒ R2 claims more echo than near has)")
        print(f"    %% fire-bins ENR>1 : {(ratio>1).mean():.1%}")
        print(f"    %% fire-bins ENR>3 : {(ratio>3).mean():.1%}")
        print(f"    median R2_direct fraction in fire bins: {(r2d[fm].sum()/max(1e-9,r2[fm].sum())):.1%}")
    print()
    # get_gain returns AMPLITUDE gain (sqrt of power gain) → square for energy.
    gpow = finalg**2
    print("  per-band: fire-rate | near-ENERGY retained (Σnear·G²/Σnear) | final-gain p10/p25/p50:")
    for name, b in bands.items():
        fr = fire[A][:, b].mean()
        nb_ = near[A][:, b]; gb = gpow[A][:, b]
        retain = nb_.ravel().dot(gb.ravel()) / max(1e-9, nb_.sum())
        gg = finalg[A][:, b]
        p10, p25, p50 = np.percentile(gg, [10, 25, 50])
        print(f"    {name:14}: fire {fr:5.1%}   near-retain {retain:5.1%}   g(amp) {p10:.2f}/{p25:.2f}/{p50:.2f}")
    # Overall energy-weighted near retention (the real near-damage number)
    retain_all = near[A].ravel().dot(gpow[A].ravel()) / max(1e-9, near[A].sum())
    print(f"  >>> OVERALL near-energy retained in near-active frames: {retain_all:.1%}  (100%=no near damage)")
    # Distribution of final gain in FIRING bins (where damage concentrates)
    if fm.any():
        gf = finalg[fm]
        print(f"  final-gain in FIRING bins: p10={np.percentile(gf,10):.2f} p25={np.percentile(gf,25):.2f} "
              f"p50={np.percentile(gf,50):.2f} mean={gf.mean():.2f}")

    # --- L1 INFLATION TEST: is R2 inflated via usable_linear=False or pinned ERLE? ---
    ne = min(len(est), F)
    if ne:
        usable = np.array([est[i]['usable'] for i in range(ne)], bool)
        conv = np.array([est[i]['conv'] for i in range(ne)], bool)
        print(f"    converged_filter (ERLE update gate) =True rate: ALL={conv.mean():.1%}  near-active={conv[act[:ne]].mean():.1%}")
        print(f"      ^ if LOW ⇒ ERLE gate CLOSED ⇒ ERLE can never earn (legacy-latch bug); if HIGH ⇒ gate open but Y²/E² near-poisoned")
        erle = np.stack([est[i]['erle'] for i in range(ne)])     # (ne,K) per-bin credited ERLE
        Ae = act[:ne]
        print()
        print("  --- L1: R2 path & ERLE credit (near-active frames) ---")
        print(f"    usable_linear=True rate: {usable[Ae].mean():.1%}  "
              f"(False ⇒ R2 = X²·gain² crude over-estimate, NOT S²/ERLE)")
        fmE = (fire[:ne] & Ae[:,None])
        if fmE.any():
            er_fire = erle[fmE]
            print(f"    credited ERLE in FIRING bins: p25={np.percentile(er_fire,25):.2f} "
                  f"median={np.median(er_fire):.2f} p75={np.percentile(er_fire,75):.2f}  (erle_min≈1)")
            print(f"    %% firing bins with ERLE<2 (near floor ⇒ under-credit): {(er_fire<2).mean():.1%}")
            print(f"    %% firing bins with ERLE<1.2 (pinned at min):           {(er_fire<1.2).mean():.1%}")
        # collapse-vs-never-earned: ERLE in LOW-near (more far-only) vs HIGH-near frames
        feN = fe[:ne]
        lo = feN < np.percentile(feN, 25)   # quietest-near 25% (closest to far-only)
        hi = feN > np.percentile(feN, 75)
        print(f"    ERLE max over whole case: {erle.max():.1f}  (>>1 ⇒ filter DID earn credit somewhere)")
        print(f"    ERLE median in LOW-near frames:  {np.median(erle[lo]):.2f}  "
              f"p90={np.percentile(erle[lo],90):.1f}")
        print(f"    ERLE median in HIGH-near frames: {np.median(erle[hi]):.2f}  "
              f"p90={np.percentile(erle[hi],90):.1f}")
        print(f"    >>> {'COLLAPSE (earned then lost in DT)' if erle[lo].max()>4 and np.median(erle[hi])<2 else 'NEVER-EARNED (ERLE low even when near is quiet)'}")


if __name__ == '__main__':
    main()
