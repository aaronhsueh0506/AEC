#!/usr/bin/env python3
"""
Module-level state dump for C parity harness.

Runs Python AEC.process on a mic/ref wav file. For each AEC sub-module
(HighPassFilter, SaturationDetector, ..., PBFDKF, ResFilter, AEC),
captures per-frame inputs, outputs, and internal state, and writes them
to module-specific .npz files.

The C parity harness reads these files, drives the C port of the same
module with identical inputs, and asserts each captured field matches
within rtol=1e-4 (or exactly, for ints/bools).

Output layout:
  <out_dir>/
    meta.npz                  hop, sample_rate, n_frames, preset
    hpf_mic.npz               in[F,hop], out[F,hop], z1[F], z2[F]
    hpf_ref.npz               (same)
    saturation_ref.npz        in[F,hop], sat[F], smoothed[F]
    saturation_mic.npz        (same)
    delay_est.npz             in_mic[F,hop], in_ref[F,hop], delay[F], par[F], ...
    pbfdaf.npz                near[F,hop], far[F,hop], error[F,hop],
                              error_spec[F,K], echo_spec[F,K], W[F,P,K]
    pbfdkf.npz                pbfdaf fields + P[F,P,K], R[F,K], _error_psd[F,K]
    res_filter.npz            input[F,hop], output[F,hop], gain_mean[F], ...
    aec.npz                   end-to-end input/output + every diag field

Usage:
  python3 dump_module_state.py <mic.wav> <ref.wav> <out_dir>
                               [--preset balanced] [--cng] [--no-delay-est]
                               [--max-frames N]
"""
import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aec import (AEC, AecConfig, AecPreset,
                 HighPassFilter, SaturationDetector, DelayEstimator,
                 FilterErleEstimator, FullbandErleEstimator,
                 RenderActivityDetector, FilterConvergenceAnalyzer,
                 DoubleTalkAnalyzer)


class ModuleRecorder:
    """Captures per-frame state for one module instance."""
    def __init__(self, module_name: str):
        self.name = module_name
        self.records = []  # list[dict[str,np.ndarray|float|int|bool]]

    def add(self, **fields):
        # Coerce everything into a serialisable form (np.ndarray or scalar)
        rec = {}
        for k, v in fields.items():
            if isinstance(v, np.ndarray):
                rec[k] = v.copy()
            elif isinstance(v, (bool, np.bool_)):
                rec[k] = bool(v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                rec[k] = v
            elif v is None:
                rec[k] = None
            else:
                rec[k] = v
        self.records.append(rec)

    def save(self, path: Path):
        if not self.records:
            return
        # Stack each field across frames
        keys = list(self.records[0].keys())
        out = {}
        for k in keys:
            vals = [r[k] for r in self.records]
            try:
                out[k] = np.stack(vals) if isinstance(vals[0], np.ndarray) \
                         else np.asarray(vals)
            except Exception:
                # Non-uniform shapes — keep as object array
                out[k] = np.asarray(vals, dtype=object)
        np.savez(str(path), **out)


def install_hooks(aec: AEC, recorders: dict):
    """Monkey-patch hooks into AEC sub-modules to capture state per call."""

    # ── HighPassFilter (mic + ref) ────────────────────────────
    for tag, hpf in (('hpf_mic', aec._hp_mic), ('hpf_ref', aec._hp_ref)):
        if hpf is None:
            continue
        rec = recorders[tag]
        orig = hpf.process
        def make(rec_, orig_, hpf_):
            def wrapped(x):
                z1_in, z2_in = hpf_.z1, hpf_.z2
                y = orig_(x)
                rec_.add(input=np.asarray(x, dtype=np.float32),
                         output=np.asarray(y, dtype=np.float32),
                         z1_in=float(z1_in), z2_in=float(z2_in),
                         z1_out=float(hpf_.z1), z2_out=float(hpf_.z2))
                return y
            return wrapped
        hpf.process = make(rec, orig, hpf)

    # ── SaturationDetector (mic + ref) ────────────────────────
    for tag, det in (('sat_ref', aec._sat_detector_ref),
                     ('sat_mic', aec._sat_detector_mic)):
        if det is None:
            continue
        rec = recorders[tag]
        orig = det.detect
        def make(rec_, orig_, det_):
            def wrapped(sig):
                lvl_in = det_.saturation_level
                lvl = orig_(sig)
                rec_.add(input=np.asarray(sig, dtype=np.float32),
                         level_in=float(lvl_in),
                         level_out=float(lvl))
                return lvl
            return wrapped
        det.detect = make(rec, orig, det)

    # ── DelayEstimator ────────────────────────────────────────
    if getattr(aec, 'delay_est', None) is not None:
        de = aec.delay_est
        rec = recorders['delay_est']
        orig_de = de.accumulate
        def make_de(rec_, orig_, de_):
            def wrapped(mic, ref):
                orig_(mic, ref)
                rec_.add(mic=np.asarray(mic, dtype=np.float32),
                         ref=np.asarray(ref, dtype=np.float32),
                         estimated_delay=int(de_.estimated_delay),
                         par=float(getattr(de_, '_last_par', 0.0)),
                         n_updates=int(getattr(de_, '_n_updates', 0)))
            return wrapped
        de.accumulate = make_de(rec, orig_de, de)

    # ── PBFDAF / PBFDKF ───────────────────────────────────────
    for tag, flt in (('pbfdkf_main', aec.filter),
                     ('pbfdkf_shadow', getattr(aec, 'shadow_filter', None))):
        if flt is None:
            continue
        rec = recorders[tag]
        orig = flt.process
        def make(rec_, orig_, flt_):
            def wrapped(near, far, mu_scale=1.0):
                err = orig_(near, far, mu_scale)
                rec_.add(
                    near=np.asarray(near, dtype=np.float32),
                    far=np.asarray(far, dtype=np.float32),
                    mu_scale=np.asarray(mu_scale, dtype=np.float32)
                              if isinstance(mu_scale, np.ndarray)
                              else float(mu_scale),
                    error=np.asarray(err, dtype=np.float32),
                    error_spec=np.asarray(flt_.error_spec, dtype=np.complex64),
                    echo_spec=np.asarray(flt_.echo_spec, dtype=np.complex64),
                    near_spec=np.asarray(flt_.near_spec, dtype=np.complex64),
                    P=np.asarray(getattr(flt_, 'P', np.zeros(1)), dtype=np.float32),
                    R=np.asarray(getattr(flt_, 'R', np.zeros(1)), dtype=np.float32),
                )
                return err
            return wrapped
        flt.process = make(rec, orig, flt)

    # ── ResFilter (full output) ──────────────────────────────
    if getattr(aec, 'res', None) is not None:
        rf = aec.res
        rec = recorders['res_filter']
        orig_rf = rf.process
        def make_rf(rec_, orig_):
            def wrapped(*args, **kwargs):
                out = orig_(*args, **kwargs)
                rec_.add(output=np.asarray(out, dtype=np.float32))
                return out
            return wrapped
        rf.process = make_rf(rec, orig_rf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mic_wav')
    ap.add_argument('ref_wav')
    ap.add_argument('out_dir')
    ap.add_argument('--preset', default='balanced',
                    choices=['mild', 'balanced', 'aggressive', 'maximum'])
    ap.add_argument('--cng', action='store_true', help='Enable CNG (default OFF)')
    ap.add_argument('--no-delay-est', action='store_true')
    ap.add_argument('--max-frames', type=int, default=0,
                    help='Truncate to this many frames (0=all)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mic, sr = sf.read(args.mic_wav)
    ref, sr2 = sf.read(args.ref_wav)
    assert sr == sr2, f"sample-rate mismatch: {sr} vs {sr2}"
    n = min(len(mic), len(ref))
    mic = mic[:n].astype(np.float32)
    ref = ref[:n].astype(np.float32)

    preset_enum = {'mild': AecPreset.MILD,
                   'balanced': AecPreset.BALANCED,
                   'aggressive': AecPreset.AGGRESSIVE,
                   'maximum': AecPreset.MAXIMUM}[args.preset]
    cfg = AecConfig.from_preset(preset_enum, sample_rate=sr,
                                enable_cng=args.cng)
    if args.no_delay_est:
        cfg.enable_delay_est = False

    aec = AEC(cfg)
    hop = aec._hop_size

    n_frames = (n - hop) // hop + 1
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)

    # Set up recorders for every module of interest
    module_tags = ['hpf_mic', 'hpf_ref',
                   'sat_ref', 'sat_mic',
                   'delay_est',
                   'pbfdkf_main', 'pbfdkf_shadow',
                   'res_filter']
    recorders = {t: ModuleRecorder(t) for t in module_tags}
    install_hooks(aec, recorders)

    # End-to-end recorder for the whole AEC
    aec_rec = ModuleRecorder('aec')

    # Run frame loop
    for i_frame in range(n_frames):
        i = i_frame * hop
        m = mic[i:i + hop]
        r = ref[i:i + hop]
        if len(m) < hop or len(r) < hop:
            break
        out = aec.process(m, r)
        d = aec._diag
        aec_rec.add(
            mic_in=m.astype(np.float32),
            ref_in=r.astype(np.float32),
            output=np.asarray(out, dtype=np.float32),
            mu_scale=float(d.get('mu_scale', 1.0)),
            erle_factor=float(d.get('erle_factor', 0.0)),
            dt_indicator=float(d.get('dt_indicator', 0.0)),
            far_activity=float(d.get('far_activity', 0.0)),
            erl_estimate=float(d.get('erl_estimate', 0.1)),
            converged=bool(d.get('converged', False)),
            once_converged=bool(d.get('filter_once_converged', False)),
            using_render=bool(d.get('using_render_based', False)),
            epc_active=bool(d.get('epc_active', False)),
        )

    # Persist all recorders + meta
    np.savez(str(out_dir / 'meta.npz'),
             hop=np.int32(hop),
             sample_rate=np.int32(sr),
             n_frames=np.int32(n_frames),
             preset=args.preset,
             enable_cng=bool(args.cng),
             enable_delay_est=bool(cfg.enable_delay_est))

    for tag, rec in recorders.items():
        rec.save(out_dir / f'{tag}.npz')
    aec_rec.save(out_dir / 'aec.npz')

    print(f'Dumped {n_frames} frames × {len(recorders) + 1} modules → {out_dir}')


if __name__ == '__main__':
    main()
