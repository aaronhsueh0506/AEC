"""Python side of the Python<->C parity harness (c_impl/test/pyc_parity_aec.sh).

Runs the Python AEC (modules.orchestrator.AEC) over raw float32 mic/ref hop by hop and writes the
files c_impl/test/pyc_parity/aec_run.c writes in the SAME layout, so the two sides can be compared
sample for sample:

    <pre>.out.f32 / <pre>.raw.f32     final output / linear output
    <pre>.trace.txt                    per hop: hop obs_call_counter main_call_counter converged
                                       ctx_filter_converged epc_render_forced_remaining
                                       lo_trustworthy lo_corroborated lo_injected dt_from_energy
                                       (first 6 columns == the C trace; the rest are Python-only
                                       evidence for the live-observer injection guard)
    <pre>.ctx.f32 (with --ctx)         per hop: 9 scalars, error_spec, echo_spec, res_gain, r2,
                                       comfort_noise -- same order as the C dump

Mic-path HPF (--hpf)
    fp64  the shipped Python HighPassFilter (float64 coefficients and state). The C library runs the
          audiolib float32 DF2T filter, so a fp64-vs-f32 comparison carries an HPF-precision
          gap that the adaptive loop amplifies. Informational.
    f32   a bit-faithful float32 emulation of audiolib hpf.c (coefficients and DF2T state in
          float32, evaluation order b1*x - a1*y + z2, no FMA) swapped in for
          modules.orchestrator.HighPassFilter before the AEC is constructed. This is what the
          Layer-1 (std-math C vs Python) comparison uses; `--hpf-only` lets the harness prove the
          emulation equals the C filter bit for bit on the actual input.
    off   enable_highpass=False (the C side is run with --no-hpf too).
The patch lives only in this process; nothing under python/modules is modified.

    --perturb SEED  multiplies every mic sample by (1 + EPS * u), u in {-1, 0, +1} drawn from
          RandomState(SEED), EPS = --perturb-eps (default 6e-8, a ~1-ulp float32 nudge) of the INPUT
          (the C side always reads the unperturbed file). A large EPS (1e-4) probes how sensitive a
          scene is to approximation-sized noise, which is what separates a fast-math budget excess
          from a branch flip. The adaptive loop has hard decision points (e.g. a single-bin
          e2_main <= e2_shadow compare) at which a tiny relative nudge selects a different but
          equally valid branch, so the harness treats the unperturbed run and a few perturbed runs
          as the set of branches the Python reference can legitimately take, and asks whether C
          matches one of them tightly.

    usage: pyc_parity_aec_run.py mic.f32 ref.f32 outprefix sr fft preset
                [--hpf fp64|f32|off] [--obs] [--no-res] [--no-cng] [--ctx] [--guard X]
                [--perturb SEED [--perturb-eps EPS]]
           pyc_parity_aec_run.py --hpf-only in.f32 out.f32 sr [cutoff_hz]
"""
import os
import sys

import numpy as np

_PY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PY_ROOT)

f32 = np.float32


class F32HPF:
    """float32 emulation of audiolib hpf.c (same formulas, same evaluation order)."""

    def __init__(self, cutoff_hz, sample_rate):
        wc = f32(2.0) * f32(np.pi) * f32(cutoff_hz) / f32(sample_rate)
        wc_w = f32(np.tan(f32(wc / f32(2.0))))
        k = f32(wc_w * wc_w)
        sqrt2 = f32(1.41421356237)
        norm = f32(f32(1.0) / f32(f32(1.0) + f32(sqrt2 * wc_w) + k))
        self.b0 = norm
        self.b1 = f32(f32(-2.0) * norm)
        self.b2 = norm
        self.a1 = f32(f32(2.0) * f32(k - f32(1.0)) * norm)
        self.a2 = f32(f32(f32(1.0) - f32(sqrt2 * wc_w) + k) * norm)
        self.z1 = f32(0)
        self.z2 = f32(0)

    def process(self, x):
        out = np.empty_like(x)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        z1, z2 = self.z1, self.z2
        for i in range(len(x)):
            xi = f32(x[i])
            yi = f32(f32(b0 * xi) + z1)
            z1 = f32(f32(f32(b1 * xi) - f32(a1 * yi)) + z2)
            z2 = f32(f32(b2 * xi) - f32(a2 * yi))
            out[i] = yi
        self.z1, self.z2 = z1, z2
        return out

    def reset(self):
        self.z1 = f32(0)
        self.z2 = f32(0)


def _hpf_only(argv):
    src, dst, sr = argv[0], argv[1], int(argv[2])
    cutoff = float(argv[3]) if len(argv) > 3 else 80.0
    x = np.fromfile(src, dtype=np.float32)
    h = F32HPF(cutoff, sr)
    hop = 128
    out = np.empty_like(x)
    for i in range(0, len(x), hop):
        out[i:i + hop] = h.process(x[i:i + hop])
    out.tofile(dst)


def _run(argv):
    micp, refp, pre, sr, fft, preset = argv[:6]
    sr, fft = int(sr), int(fft)
    flags = argv[6:]
    hpf = 'fp64'
    if '--hpf' in flags:
        hpf = flags[flags.index('--hpf') + 1]
    if hpf not in ('fp64', 'f32', 'off'):
        raise SystemExit('--hpf must be fp64, f32 or off')

    from modules.config import AecConfig
    import modules.orchestrator as orch

    kw = dict(sample_rate=sr, frame_size=fft)
    if '--obs' in flags:
        kw['live_observer_enabled'] = True
    if hpf == 'off':
        kw['enable_highpass'] = False
    if '--no-res' in flags:
        kw['enable_res'] = False
    if '--no-cng' in flags:
        kw['enable_cng'] = False
    want_ctx = '--ctx' in flags
    if want_ctx:
        kw['return_res_context'] = True
    if '--guard' in flags:
        kw['live_observer_inject_dt_energy_max'] = float(flags[flags.index('--guard') + 1])
    if hpf == 'f32':
        orch.HighPassFilter = F32HPF          # looked up at AEC construction time
    cfg = AecConfig.from_preset(preset, **kw)
    np.random.seed(0)
    aec = orch.AEC(cfg)

    mic = np.fromfile(micp, dtype=np.float32)
    ref = np.fromfile(refp, dtype=np.float32)
    if '--perturb' in flags:
        eps = float(flags[flags.index('--perturb-eps') + 1]) if '--perturb-eps' in flags else 6e-8
        u = np.random.RandomState(int(flags[flags.index('--perturb') + 1])).randint(-1, 2, size=len(mic))
        mic = (mic.astype(np.float64) * (1.0 + eps * u)).astype(np.float32)
    n = min(len(mic), len(ref))
    hop = cfg.hop_size
    outs, raws, rows, ctx_rows = [], [], [], []
    for h, i in enumerate(range(0, n - hop, hop)):
        r = aec.process(mic[i:i + hop].copy(), ref[i:i + hop].copy())
        ctx_conv = -1
        if isinstance(r, tuple):
            o, ctx = r
            ctx_conv = int(bool(ctx.filter_converged))
            K = len(ctx.error_spec) if ctx.error_spec is not None else cfg.frame_size // 2 + 1

            def _c(a):
                if a is None:
                    return np.zeros(2 * K, np.float32)
                a = np.asarray(a, np.complex64)
                return np.stack([a.real, a.imag], axis=1).reshape(-1).astype(np.float32)

            def _f(a):
                return np.zeros(K, np.float32) if a is None else np.asarray(a, np.float32)

            usable = ctx.usable_linear
            sc = np.array([ctx.far_power, ctx.erle_factor, ctx.dt_indicator, ctx.divergence,
                           ctx.saturation_level, ctx.erl_estimate, float(bool(ctx.filter_converged)),
                           float(bool(ctx.res_floor_protect)),
                           0.0 if usable is None else float(bool(usable))], np.float32)
            ctx_rows.append(np.concatenate([sc, _c(ctx.error_spec), _c(ctx.echo_spec),
                                            _f(ctx.res_gain), _f(ctx.r2), _f(ctx.comfort_noise)]))
        else:
            o = r
        outs.append(np.asarray(o, np.float32))
        raws.append(np.asarray(aec._last_raw_output, np.float32))
        d = aec._diag
        of = getattr(aec, '_observer_filter', None)
        rows.append('%d %d %d %d %d %d %d %d %d %.9g' % (
            h,
            of._call_counter if of is not None else -1,
            aec.filter._call_counter,
            int(bool(aec._convergence.converged)), ctx_conv,
            int(getattr(aec, '_epc_render_forced_remaining', 0)),
            int(bool(d.get('lo_trustworthy', False))),
            int(bool(d.get('lo_corroborated', False))),
            int(bool(d.get('lo_injected', False))),
            float(getattr(aec, '_dt_from_energy', 0.0))))
    np.concatenate(outs).astype(np.float32).tofile(pre + '.out.f32')
    np.concatenate(raws).astype(np.float32).tofile(pre + '.raw.f32')
    with open(pre + '.trace.txt', 'w') as f:
        f.write('\n'.join(rows) + '\n')
    if want_ctx:
        np.concatenate(ctx_rows).astype(np.float32).tofile(pre + '.ctx.f32')
    print('pyc_parity_aec_run: %d hops hop=%d sr=%d fft=%d preset=%s hpf=%s kw=%s'
          % (len(rows), hop, sr, fft, preset, hpf, sorted(kw.items())), file=sys.stderr)


def main():
    a = sys.argv[1:]
    if a and a[0] == '--hpf-only':
        _hpf_only(a[1:])
    elif len(a) >= 6:
        _run(a)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
