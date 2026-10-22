#!/usr/bin/env python3
"""Python<->C parity driver for the AEC (called by c_impl/test/pyc_parity_aec.sh).

Two layers, measured with the same method:

  Layer 1  std-math C (SIMD=0, -DUSE_STANDARD_MATH) versus the Python reference, with the mic-path HPF
           made float32 on the Python side (bit-faithful emulation of audiolib hpf.c; proven bit
           exact against the C filter in the "hpf_fidelity" cells), so the documented fp64-vs-f32 HPF
           gap (audit D-01 / S-19) is out of the comparison. Catches sync bugs.
  Layer 2  shipping C build (Makefile default: SIMD=1, fast_math) versus the std-math C build: the
           accepted fast-math / NEON approximation budget.

Tolerances below are FROZEN (2026-10-20, after S-01/S-03/S-06): chosen from a full measurement against
the post-Wave-S real trees (not the pre-fix measurement the first provisional cut used), each with an
explicit margin over the worst currently-GREEN cell for that metric -- see CHANGELOG.md's "tolerance
freeze" entry for the measured numbers, the per-constant margin reasoning and the blind spots this freeze
does not close (the Layer-2 INCONCLUSIVE_CHAOTIC bucket absorbs every Layer-2 cell that would otherwise be
red on the current default build; TOL is still one global value shared by every scenario row regardless of
its own headroom). Every cell still prints the measured numbers next to the tolerance.

Known gaps are registered in EXPECTED_RED (audit S-xx). A scenario listed there may exceed its
tolerance without failing the run, but ONLY if its "shape" evidence is still present (e.g. S-02:
Python's injection guard still blocks at least one hop in the scene). An expected-red scenario that
turns green fails the run ("remove the marker"), and so does one that is red without its shape evidence.

Bistable scenes. The adaptive loop contains hard decisions (single-bin e2_main <= e2_shadow compares,
threshold latches). A relative nudge of 6e-8 (one float32 ulp) on the Python input picks a different but
equally valid branch in some scenes (Python-vs-Python differences of -30 dB for tens of hops are
observed). A C build that lands on the other branch is not wrong, so Layer 1 asks whether C matches ANY of
a few Python branches (unperturbed + up to MAX_PY_VARIANTS-1 one-ulp-perturbed inputs) within tolerance.
Variants are only run when needed. Layer 2 compares two C builds whose approximations (fast_exp is a
3rd-order Taylor, fast_log is good to 1e-4) are far larger than one ulp, so when it is red the harness
re-runs Python with a 1e-4-relative nudge of the mic: if Python itself then moves by more than the Layer-2
tolerance the scene is "chaotic" and Layer 2 is reported INCONCLUSIVE_CHAOTIC (branch selection, not
approximation error) instead of red.

Resolution (measured 2026-10-20 by planting constant changes in scratch copies; this is what Layer 1 can and cannot see).
  A change of a live steady-state constant (AEC_FILTER_NOISE_GATE_POWER_FLOAT, C or the same constant on the Python side):
    5 % and 1 %  caught in 6 of 7 probe scenes (margins 6x..340x over the provisional tolerance);
    1e-4         caught in 1 of 7 (a chaotic scene amplified it); 1e-6 not seen.
  The Python-side +5 % change is caught the same way as the C-side one.
  A call_counter starting one hop behind (the S-01 shape, on the main/shadow/observer PBFDAF): caught in 7 of 7.
  One-ulp constant changes (6e-8 relative) are NOT visible here: they move the C output by ~1e-7, which is the float32
  noise floor between C and an fp64-FFT Python. Bit-exact checks of such constants belong to the per-module C tests.
  Not every cell is informative: INCONCLUSIVE_CHAOTIC (Layer 2) and INFO(...) rows never gate, and the observer cells on the
  rec.16k256 input carry an S-01 effect (8e-3) smaller than that input's own chaos distance (2.9e-2), so their audio numbers
  do not quantify S-01 (the marker rests on the call_counter evidence; preecho/dtb cells quantify it).

Exit: 0 all as expected; 1 a cell failed; 2 setup error.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scenes  # noqa: E402

# ----------------------------------------------------------------------------------------------
# FROZEN tolerances (2026-10-20; supersedes the Wave-S-prep provisional cut). Each value below is
# justified against a full measurement on the post-S-01/S-03/S-06 real tree (44 scenarios, grids
# 8k256/16k256/16k512/48k1024, --repeat-check byte-identical); see CHANGELOG.md for the measured
# worst-GREEN value, the chosen margin and why per constant. Unless noted the number is UNCHANGED
# from the provisional cut because 2x the measured worst-GREEN cell already rounds to it; two CTX
# constants and two L2 constants were tightened because the real margin was much wider than 2x.
# ----------------------------------------------------------------------------------------------
TOL = {
    'status': 'frozen',
    # Layer 1: std-math C vs Python (f32 HPF), best-matching Python branch
    'L1_out_max_abs': 3.0e-4,        # unchanged; worst GREEN 1.453e-4 (dtb.8k256.bal), margin 2.06x
    'L1_raw_max_abs': 3.0e-4,        # unchanged; worst GREEN 1.153e-4 (dtb.8k256.bal), margin 2.60x
    'L1_out_rms_db': -75.0,          # unchanged; worst GREEN -80.07 dB, margin 5.07 dB. 20*log10(rms(C-Py)/rms(Py)); lower is tighter
    # Layer 2: shipping build vs std-math build (fast-math / NEON approximation budget)
    'L2_out_max_abs': 2.0e-3,        # unchanged; worst GREEN 7.893e-4 (recdc.48k1024.bal.nohpf), margin 2.53x
    'L2_raw_max_abs': 2.0e-4,        # tightened 10x (was 2.0e-3); worst GREEN 9.049e-5, margin 2.21x
    'L2_out_rms_db': -61.0,          # tightened 1 dB (was -60.0); worst GREEN -67.75 dB, margin 6.75 dB
    # context seam (aec_get_res_context), Layer 1 only
    'CTX_scalar_rel': 2.0e-5,        # tightened 50x (was 1.0e-3); worst GREEN 2.265e-6 (erle_factor), margin 8.8x
    'CTX_spec_rel_l2': 1.0e-5,       # tightened 100x (was 1.0e-3); worst GREEN 5.28e-7 (echo_spec), margin 18.9x
    'CTX_gain_max_abs': 1.0e-3,      # unchanged; worst GREEN 4.380e-4 (res_gain), margin 2.28x
}
MAX_PY_VARIANTS = 3           # unperturbed + one-ulp-perturbed seeds 1 and 2
L2_PROBE_EPS = 1e-4           # relative mic nudge used to ask 'is this scene chaotic at approximation scale?'
L2_PROBE_SEED = 11

# ----------------------------------------------------------------------------------------------
# Known Python<->C gaps that survive wave 1 (PY_C_SYNC_AUDIT.md). Keyed by the S-id used in scenario
# definitions; the text is what the table prints.
# ----------------------------------------------------------------------------------------------
EXPECTED_RED_INFO = {
    'S-02': 'live observer: C has no live_observer_inject_dt_energy_max injection guard (Python blocks injection when dt_from_energy > 0.5)',
    'S-19': 'parity_aec_e2e 8 kHz row exceeds TOL_E2E (run by make test-parity-e2e, not by this suite)',
    'D-01': 'INFO only (documented, never gates): the shipped Python mic HPF is fp64 while C runs the float32 DF2T filter; the adaptive '
            'loop amplifies that 1e-7 relative difference (audit: -35..-63 dB). Layer 1 therefore uses the bit-exact float32 emulation; '
            'the *.hpf64 rows show the size of the fp64 effect on this harness inputs, judged against the unperturbed Python run only',
    'N-01': 'INFO only (not an audit S-id, never gates): with the mic HPF disabled, aec_record (rec.48k1024, and its DC-free recdc.16k512) '
            'drives C off the Python branch (-36..-40 dB). Both C builds agree with each other and no one-ulp Python nudge reproduces it, '
            'but Python itself flips at 1e-5 relative: float32-FFT noise on a low-frequency-rich unfiltered mic selects another branch. '
            'The same input with the HPF on, and recdc at 8k/16k256/48k, are tight. Root cause not investigated',
    'N-02': 'INFO only (not an audit S-id, never gates): preecho.48k1024.bal.obs_noguard LAYER 1 (std-math C vs Python) has a ~2.9e-3 '
            'max_abs residual at hop ~1790, 9.8x TOL_L1 -- the shipping SIMD=1 build does not hit it and tracks Python to <=1.4e-5 '
            'over this entire scene, so the same residual also shows up as a ship-vs-std Layer 2 reading (hence INCONCLUSIVE_CHAOTIC '
            'there) but is not a Layer-2 fast-math approximation gap. Before the S-01 fix this cell WAS an S-01 case too '
            '(obs_cc_mismatch_hops=1855, L1 max_abs=1.75e-2, 58x TOL_L1); the S-01 fix closed that gap (obs_cc_mismatch_hops=0 now). '
            'This 2.9e-3/9.8x residual is what remains AFTER the S-01 fix, confirmed unchanged by the S-01 fix itself (same value '
            'measured with S-01 reverted and reapplied) -- i.e. it is a separate, still-open gap the S-01 fix did not resolve, not '
            'something S-01 introduced or masked. Root-caused by a separate investigation to a per-bin branch-selection '
            '(leakage-comparison) tie-break specific to the std-math (SIMD=0) diagnostic build; Python itself is comparably '
            'sensitive to a one-ulp input nudge in this scene (self-distance 0.038 >> 2.9e-3), which is the gate this cell is held '
            'to (see the N-02 check below: obs_cc_mismatch_hops==0 and Layer-1 max_abs <= that 1-ulp self-distance)',
}

# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------
REC_MIC = 'wav/aec_record/aec_record_mic_10s.wav'
REC_REF = 'wav/aec_record/aec_record_ref_10s.wav'
SYN_SECONDS = {'echo': 12, 'speechdt': 20, 'dtb': 20, 'preecho': 20}
GRIDS = {            # tag -> (sr, fft)
    '8k256': (8000, 256),
    '16k256': (16000, 256),
    '16k512': (16000, 512),
    '48k1024': (48000, 1024),
}


def input_id(kind, sr):
    return '%s%d' % (kind, sr // 1000)


def build_input(kind, sr, aec_root):
    """-> (mic float32, ref float32)"""
    if kind in ('rec', 'recdc'):
        import soundfile as sf
        mic, sr0 = sf.read(os.path.join(aec_root, REC_MIC), dtype='float32')
        ref, sr1 = sf.read(os.path.join(aec_root, REC_REF), dtype='float32')
        assert sr0 == sr1 == 16000
        if mic.ndim > 1:
            mic = mic[:, 0]
        if ref.ndim > 1:
            ref = ref[:, 0]
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]
        if sr != 16000:
            from scipy.signal import resample_poly
            up, down = {8000: (1, 2), 48000: (3, 1)}[sr]
            mic = resample_poly(mic, up, down)
            ref = resample_poly(ref, up, down)
        if kind == 'recdc':      # what the mic-path HPF would deliver as far as DC goes: mean removed on both channels
            mic = mic - np.mean(mic)
            ref = ref - np.mean(ref)
        return np.ascontiguousarray(mic, dtype=np.float32), np.ascontiguousarray(ref, dtype=np.float32)
    return scenes.make(kind, sr, SYN_SECONDS[kind])


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------------------------
class Scn(object):
    def __init__(self, sid, kind, grid, preset='balanced', hpf='f32', obs=False, guard_off=False, res=True,
                 ctx=False, cng=True, red=(), quick=False, info='', v0_only=False):
        self.id = sid
        self.kind = kind
        self.grid = grid
        self.sr, self.fft = GRIDS[grid]
        self.preset = preset
        self.hpf = hpf               # 'f32' (Python HPF emulated in float32) | 'off' (HPF disabled both sides) |
                                     # 'fp64' (shipped Python HPF; informational rows only)
        self.v0_only = v0_only       # judge against the unperturbed Python run only (documented-effect rows)
        self.obs = obs
        self.guard_off = guard_off   # Python only: live_observer_inject_dt_energy_max = 1e9
        self.res = res
        self.ctx = ctx
        self.cng = cng
        self.red = tuple(red)        # S-ids this scenario is expected to show
        self.quick = quick
        self.info = info             # observation, never a gate (e.g. a chaos-fragile finding that is not an audit S-id)

    @property
    def inp(self):
        return input_id(self.kind, self.sr)

    def c_flags(self):
        f = []
        if self.obs:
            f.append('--obs')
        if self.hpf == 'off':
            f.append('--no-hpf')
        if not self.res:
            f.append('--no-res')
        if not self.cng:
            f.append('--no-cng')
        if self.ctx:
            f.append('--ctx')
        return f

    def py_flags(self):
        f = ['--hpf', self.hpf]
        if self.obs:
            f.append('--obs')
        if self.guard_off:
            f += ['--guard', '1e9']
        if not self.res:
            f.append('--no-res')
        if not self.cng:
            f.append('--no-cng')
        if self.ctx:
            f.append('--ctx')
        return f


def build_scenarios():
    S = []
    add = S.append
    short = {'balanced': 'bal', 'aggressive': 'agg', 'mild': 'mild', 'no_floor': 'nofloor'}
    # A. default configuration (observer off, mic HPF on): real recording and synthetic scenes
    for g in GRIDS:
        for p in ('balanced', 'aggressive'):
            add(Scn('rec.%s.%s' % (g, short[p]), 'rec', g, p, quick=(p == 'balanced' and g in ('16k256', '8k256'))))
    for g in GRIDS:
        add(Scn('dtb.%s.bal' % g, 'dtb', g, quick=(g in ('16k256', '48k1024'))))
        add(Scn('speechdt.%s.bal' % g, 'speechdt', g, quick=(g == '16k256')))
        add(Scn('echo.%s.bal' % g, 'echo', g, quick=(g == '16k512')))
    add(Scn('dtb.16k256.agg', 'dtb', '16k256', 'aggressive'))
    add(Scn('dtb.48k1024.agg', 'dtb', '48k1024', 'aggressive'))
    add(Scn('rec.16k256.mild', 'rec', '16k256', 'mild'))
    add(Scn('dtb.48k1024.mild', 'dtb', '48k1024', 'mild'))
    # B. mic HPF disabled on both sides. recdc = the recording with its DC offset removed (what the HPF would
    # have delivered as far as DC goes); the raw-DC 48k case is the registered N-01.
    for g in ('8k256', '16k256', '48k1024'):
        add(Scn('recdc.%s.bal.nohpf' % g, 'recdc', g, hpf='off', quick=(g == '16k256')))
    add(Scn('dtb.16k256.bal.nohpf', 'dtb', '16k256', hpf='off'))
    add(Scn('rec.48k1024.bal.nohpf', 'rec', '48k1024', hpf='off', info='N-01'))
    # E. informational: the SHIPPED Python mic HPF (fp64) against the float32 C filter, vs unperturbed Python (D-01)
    for sid, kind, g in (('dtb.8k256.bal.hpf64', 'dtb', '8k256'), ('dtb.16k256.bal.hpf64', 'dtb', '16k256'),
                         ('rec.16k256.bal.hpf64', 'rec', '16k256')):
        add(Scn(sid, kind, g, hpf='fp64', info='D-01', v0_only=True))
    add(Scn('recdc.16k512.bal.nohpf', 'recdc', '16k512', hpf='off', info='N-01'))
    # C. live observer on. The observer call_counter ordering gap (S-01, fixed: pbfdaf_process now
    # increments call_counter before the zero-mu early return, matching filters.py's _update_weights)
    # used to move the audio on preecho (Python guard disabled there, which isolated S-01) and on the
    # recording at 8k/16k256 (Python's guard blocks nothing in it); S-02 (injection guard, still open)
    # needs short near-end bursts (dtb) so that Python's guard actually blocks.
    for g in ('16k256', '16k512'):
        add(Scn('preecho.%s.bal.obs_noguard' % g, 'preecho', g, obs=True, guard_off=True, quick=(g == '16k512')))
    # 48k1024 WAS an S-01 case before the fix (obs_cc_mismatch_hops=1855, L1 max_abs=1.75e-2, 58x
    # TOL_L1); the S-01 fix closed that (its observer call_counter now agrees with Python on every
    # hop). What remains AFTER the S-01 fix is a separate, smaller LAYER 1 finding (std-math C vs
    # Python, ~2.9e-3 max_abs at hop ~1790, 9.8x TOL_L1), confirmed unaffected by S-01 itself (same
    # value whether S-01 is applied or reverted) -- a per-bin branch-selection (leakage-comparison)
    # tie-break specific to the std-math (SIMD=0) diagnostic build, root-caused by a separate
    # investigation; the shipping SIMD=1 build does not hit it and tracks Python to <=1.4e-5 over this
    # whole scene. Because ship tracks Python so closely, the same residual also appears as a
    # ship-vs-std Layer 2 reading and gets reported INCONCLUSIVE_CHAOTIC there (Python itself is
    # comparably sensitive to a one-ulp nudge in this scene), which is why this was once mis-described
    # as a Layer-2 fast-math approximation gap. Flagged INFO (gated below: obs_cc_mismatch_hops==0 and
    # Layer-1 max_abs <= Python's own 1-ulp self-distance) so it stays visible instead of disappearing
    # into a plain PASS, without being a free pass either.
    add(Scn('preecho.48k1024.bal.obs_noguard', 'preecho', '48k1024', obs=True, guard_off=True, info='N-02'))
    for g in ('8k256', '16k256', '48k1024'):
        add(Scn('dtb.%s.bal.obs' % g, 'dtb', g, obs=True, red=('S-02',), quick=(g == '16k256')))
    add(Scn('rec.8k256.bal.obs', 'rec', '8k256', obs=True))
    add(Scn('rec.16k256.bal.obs', 'rec', '16k256', obs=True))
    # D. context seam (aec_get_res_context). S-05 (filter_converged one hop
    # late on the rising edge) was fixed 2026-10-20: Python now patches the
    # field after update_convergence runs for the hop, matching C's
    # aec_get_res_context() post-update pull contract. No marker needed.
    add(Scn('dtb.16k256.bal.ctx', 'dtb', '16k256', ctx=True, quick=True))
    add(Scn('dtb.48k1024.bal.ctx', 'dtb', '48k1024', ctx=True))
    add(Scn('speechdt.16k256.bal.ctx_nores', 'speechdt', '16k256', ctx=True, res=False))
    # F. NO_FLOOR (AecPreset value 3 / "no_floor"): construction-time-only on both sides today (the
    # C-side runtime aec_set_preset()/aec_res_set_preset() retarget to/from NO_FLOOR added alongside this
    # row is NOT exercised by this driver -- it constructs fresh per scenario, same as every other preset
    # row here). min_gain_split_floor_enabled=False removes all three suppression-gain floor variants, so
    # this is the deepest point on the strength axis and the one most likely to separate from balanced;
    # one dtb scene (where the floor matters most) plus one on the other FFT family.
    add(Scn('dtb.16k256.nofloor', 'dtb', '16k256', 'no_floor'))
    add(Scn('dtb.48k1024.nofloor', 'dtb', '48k1024', 'no_floor'))
    return S


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def r6(x):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return float('%.6g' % x)


def load_f32(path):
    return np.fromfile(path, dtype=np.float32)


def cmp_metrics(test, ref):
    """test, ref: float32 arrays. Metrics of d = test - ref (float64)."""
    n = min(len(test), len(ref))
    a = test[:n].astype(np.float64)
    b = ref[:n].astype(np.float64)
    d = a - b
    ad = np.abs(d)
    rms_d = math.sqrt(float(np.mean(d * d))) if n else 0.0
    rms_b = math.sqrt(float(np.mean(b * b))) if n else 0.0
    exact = bool(np.array_equal(test[:n], ref[:n])) and len(test) == len(ref)
    if rms_d == 0.0:
        rms_db = None            # bit identical: no finite dB figure
    else:
        rms_db = 20.0 * math.log10(rms_d / (rms_b + 1e-30))
    return {
        'n': int(n),
        'max_abs': r6(float(np.max(ad)) if n else 0.0),
        'p999': r6(float(np.percentile(ad, 99.9)) if n else 0.0),
        'rms_db': r6(rms_db),
        'bitexact': exact,
        'len_equal': len(test) == len(ref),
    }


def worse(x, tol):
    return x is not None and x > tol


# ----------------------------------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------------------------------
class Ctx(object):
    pass


def run(cmd, log=None, env=None, check=True):
    e = dict(os.environ)
    e.update({'OMP_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0'})
    if env:
        e.update(env)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e)
    if log:
        with open(log, 'wb') as f:
            f.write(p.stdout)
    if check and p.returncode != 0:
        sys.stderr.write('FAILED (rc=%d): %s\n%s\n' % (p.returncode, ' '.join(cmd), p.stdout.decode('utf8', 'replace')[-2000:]))
        raise SystemExit(2)
    return p


def read_trace(path):
    rows = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if t:
                rows.append(t)
    return rows


class Runner(object):
    def __init__(self, args):
        self.a = args
        self.work = args.work
        self.pyscript = os.path.join(args.aec_root, 'python', 'diag', 'pyc_parity_aec_run.py')
        self.cache = {}

    def inp_paths(self, kind, sr):
        iid = input_id(kind, sr)
        return (os.path.join(self.work, 'inputs', iid + '_mic.f32'), os.path.join(self.work, 'inputs', iid + '_ref.f32'))

    def c_run(self, build, scn):
        key = ('c', build, scn.id)
        if key in self.cache:
            return self.cache[key]
        binp = self.a.std_bin if build == 'std' else self.a.ship_bin
        mic, ref = self.inp_paths(scn.kind, scn.sr)
        d = os.path.join(self.work, 'run', scn.id)
        os.makedirs(d, exist_ok=True)
        pre = os.path.join(d, 'c_' + build)
        run([binp, mic, ref, pre, str(scn.sr), str(scn.fft), scn.preset] + scn.c_flags(), log=pre + '.log')
        self.cache[key] = pre
        return pre

    def py_run(self, scn, seed, eps=None, tag=None):
        """seed 0 = unperturbed; seed>0 = mic nudged by (1 + eps*u), eps default one float32 ulp."""
        key = ('py', scn.id, seed, eps, tag)
        if key in self.cache:
            return self.cache[key]
        mic, ref = self.inp_paths(scn.kind, scn.sr)
        d = os.path.join(self.work, 'run', scn.id)
        os.makedirs(d, exist_ok=True)
        pre = os.path.join(d, tag or ('py_v%d' % seed))
        flags = scn.py_flags()
        if tag == 'py_noguard_twin':
            flags = flags + ['--guard', '1e9']
        cmd = [sys.executable, self.pyscript, mic, ref, pre, str(scn.sr), str(scn.fft), scn.preset] + flags
        if seed:
            cmd += ['--perturb', str(seed)]
            if eps is not None:
                cmd += ['--perturb-eps', repr(eps)]
        run(cmd, log=pre + '.log')
        self.cache[key] = pre
        return pre


def load_run(pre):
    r = {'out': load_f32(pre + '.out.f32'), 'raw': load_f32(pre + '.raw.f32')}
    r['trace'] = read_trace(pre + '.trace.txt')
    if os.path.exists(pre + '.ctx.f32'):
        r['ctx'] = load_f32(pre + '.ctx.f32')
    return r


def cell(c_run_data, ref_run_data):
    return {'out': cmp_metrics(c_run_data['out'], ref_run_data['out']),
            'raw': cmp_metrics(c_run_data['raw'], ref_run_data['raw'])}


def l1_ok(m):
    return (not worse(m['out']['max_abs'], TOL['L1_out_max_abs'])
            and not worse(m['raw']['max_abs'], TOL['L1_raw_max_abs'])
            and not (m['out']['rms_db'] is not None and m['out']['rms_db'] > TOL['L1_out_rms_db'])
            and m['out']['len_equal'] and m['raw']['len_equal'])


def l2_ok(m):
    return (not worse(m['out']['max_abs'], TOL['L2_out_max_abs'])
            and not worse(m['raw']['max_abs'], TOL['L2_raw_max_abs'])
            and not (m['out']['rms_db'] is not None and m['out']['rms_db'] > TOL['L2_out_rms_db'])
            and m['out']['len_equal'] and m['raw']['len_equal'])


def score(m):
    return m['out']['max_abs'] if m['out']['max_abs'] is not None else 0.0


# ----------------------------------------------------------------------------------------------
# Evidence (shape predicates)
# ----------------------------------------------------------------------------------------------
def obs_counter_evidence(c_tr, p_tr):
    n = min(len(c_tr), len(p_tr))
    mism = sum(1 for i in range(n) if c_tr[i][1] != p_tr[i][1])
    return {'obs_cc_mismatch_hops': mism, 'obs_cc_final_c': int(c_tr[n - 1][1]) if n else None,
            'obs_cc_final_py': int(p_tr[n - 1][1]) if n else None, 'hops': n}


def guard_evidence(p_tr):
    # python trace columns 6,7,8 = lo_trustworthy, lo_corroborated, lo_injected
    blocked = sum(1 for t in p_tr if len(t) > 8 and t[6] == '1' and t[7] == '1' and t[8] == '0')
    injected = sum(1 for t in p_tr if len(t) > 8 and t[8] == '1')
    return {'py_guard_blocked_hops': blocked, 'py_injected_hops': injected}


def trace_mismatch(c_tr, p_tr, col):
    n = min(len(c_tr), len(p_tr))
    return sum(1 for i in range(n) if c_tr[i][col] != p_tr[i][col])


def s05_evidence(c_tr, p_tr):
    """filter_converged (ctx) per hop: C vs Python. The documented gap (S-05) is that Python builds the
    context before update_convergence, so Python's flag is one hop late on every RISING edge while falling
    edges agree. Shape = every mismatching hop is a hop where C has just risen and Python has not yet."""
    n = min(len(c_tr), len(p_tr))
    cc = [int(t[4]) for t in c_tr[:n]]
    pc = [int(t[4]) for t in p_tr[:n]]
    mism = [i for i in range(n) if cc[i] != pc[i]]
    rising = [i for i in range(1, n) if cc[i] == 1 and cc[i - 1] == 0]
    falling = [i for i in range(1, n) if cc[i] == 0 and cc[i - 1] == 1]
    late_rising = [i for i in mism if cc[i] == 1 and pc[i] == 0 and cc[i - 1] == 0]
    py_follows = [i for i in late_rising if i + 1 < n and pc[i + 1] == 1]
    return {'ctx_conv_mismatch_hops': len(mism), 'ctx_conv_rising_edges': len(rising),
            'ctx_conv_falling_edges': len(falling), 'ctx_conv_late_rising_mismatches': len(late_rising),
            'ctx_conv_python_catches_up_next_hop': len(py_follows),
            'ctx_conv_shape_is_late_rising': bool(mism) and len(late_rising) == len(mism)}


def ctx_compare(c, p, K):
    """C and Python ctx dumps (per hop: 9 scalars, error_spec 2K, echo_spec 2K, res_gain K, r2 K, cn K)."""
    rec = 9 + 4 * K + 3 * K
    nh = min(len(c), len(p)) // rec
    C = c[:nh * rec].reshape(nh, rec).astype(np.float64)
    P = p[:nh * rec].reshape(nh, rec).astype(np.float64)
    names = ['far_power', 'erle_factor', 'dt_indicator', 'divergence', 'saturation_level', 'erl_estimate',
             'filter_converged', 'res_floor_protect', 'usable_linear']
    out = {'hops': int(nh), 'scalars': {}}
    for i, nm in enumerate(names):
        d = np.abs(C[:, i] - P[:, i])
        scale = max(float(np.max(np.abs(P[:, i]))), 1e-30)
        out['scalars'][nm] = {'max_abs': r6(float(np.max(d))), 'rel_to_max': r6(float(np.max(d)) / scale),
                              'mismatch_hops': int(np.sum(d > 0))}
    o = 9

    def blk(width):
        nonlocal o
        s = slice(o, o + width)
        o += width
        return C[:, s], P[:, s]

    for nm, w in (('error_spec', 2 * K), ('echo_spec', 2 * K)):
        a, b = blk(w)
        out[nm] = {'rel_l2': r6(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)))}
    for nm in ('res_gain', 'r2', 'comfort_noise'):
        a, b = blk(K)
        out[nm] = {'max_abs': r6(float(np.max(np.abs(a - b)))),
                   'rel_l2': r6(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)))}
    return out


def ctx_ok(cc):
    bad = []
    for nm, v in cc['scalars'].items():
        if nm in ('filter_converged', 'res_floor_protect', 'usable_linear'):
            # S-05 (the one-hop-late rising edge) is fixed; filter_converged
            # is a boolean trace like the other two in this group, so it is
            # held to the same any-mismatch-fails rule, not skipped.
            if v['mismatch_hops'] > 0:
                bad.append(nm)
        elif v['rel_to_max'] is not None and v['rel_to_max'] > TOL['CTX_scalar_rel']:
            bad.append(nm)
    for nm in ('error_spec', 'echo_spec'):
        if cc[nm]['rel_l2'] > TOL['CTX_spec_rel_l2']:
            bad.append(nm)
    if cc['res_gain']['max_abs'] > TOL['CTX_gain_max_abs']:
        bad.append('res_gain')
    for nm in ('r2', 'comfort_noise'):
        if cc[nm]['rel_l2'] > TOL['CTX_spec_rel_l2']:
            bad.append(nm)
    return bad


# ----------------------------------------------------------------------------------------------
# One scenario
# ----------------------------------------------------------------------------------------------
def eval_scenario(R, scn):
    res = {'id': scn.id, 'config': {
        'input': scn.inp, 'sr': scn.sr, 'fft': scn.fft, 'preset': scn.preset, 'observer': scn.obs,
        'py_guard_disabled': scn.guard_off, 'enable_res': scn.res, 'enable_cng': scn.cng, 'context': scn.ctx},
        'hpf': scn.hpf,
        'hpf_type': {'f32': 'f32-emulation (bit-exact to audiolib hpf.c, see hpf_fidelity)', 'off': 'off (both sides)',
                     'fp64': 'fp64 (shipped Python HighPassFilter; informational, audit D-01)'}[scn.hpf],
        'expected_red': list(scn.red)}
    cstd = load_run(R.c_run('std', scn))
    cship = load_run(R.c_run('ship', scn))
    variants = [load_run(R.py_run(scn, 0))]
    seeds = [0]

    def best(c):
        ms = [cell(c, v) for v in variants]
        i = min(range(len(ms)), key=lambda k: score(ms[k]))
        return i, ms

    # ---- Layer 1: std-math C vs the best-matching Python branch ----
    bi, ms = best(cstd)
    while not l1_ok(ms[bi]) and len(variants) < MAX_PY_VARIANTS and not scn.v0_only:
        s_ = len(variants)
        variants.append(load_run(R.py_run(scn, s_)))
        seeds.append(s_)
        bi, ms = best(cstd)
    l1 = ms[bi]
    py_best = variants[bi]
    res['layer1'] = {'build': 'std-math (SIMD=0, -DUSE_STANDARD_MATH)', 'reference': 'python',
                     'py_variant': 'v%d%s' % (seeds[bi], '' if seeds[bi] == 0 else ' (1-ulp perturbed input)'),
                     'variants_evaluated': len(variants), 'out': l1['out'], 'raw': l1['raw'],
                     'margin_out_max_abs': r6((l1['out']['max_abs'] or 0.0) / TOL['L1_out_max_abs']),
                     'within_tolerance': bool(l1_ok(l1))}
    if len(variants) > 1:
        res['layer1']['vs_each_variant_out_max_abs'] = [m['out']['max_abs'] for m in ms]
        res['python_self_distance_1ulp_out_max_abs'] = r6(max(
            cmp_metrics(variants[k]['out'], variants[0]['out'])['max_abs'] for k in range(1, len(variants))))

    # ---- Layer 2: ship vs std ----
    d2 = cell(cship, cstd)
    l2 = {'build': 'ship (Makefile default: SIMD=1, fast_math)', 'reference': 'std-math C', 'out': d2['out'], 'raw': d2['raw'],
          'margin_out_max_abs': r6((d2['out']['max_abs'] or 0.0) / TOL['L2_out_max_abs'])}
    if l2_ok(d2):
        l2['status'] = 'PASS'
    else:
        probe = load_run(R.py_run(scn, L2_PROBE_SEED, eps=L2_PROBE_EPS, tag='py_probe'))
        pd = cmp_metrics(probe['out'], variants[0]['out'])['max_abs']
        l2['python_self_distance_at_%g_out_max_abs' % L2_PROBE_EPS] = pd
        if pd is not None and pd > TOL['L2_out_max_abs']:
            l2['status'] = 'INCONCLUSIVE_CHAOTIC'
            l2['why'] = ('Python itself moves by %.3g when its mic is nudged by %g relative (> the Layer-2 tolerance): '
                         'the ship-vs-std difference in this scene is branch selection, not demonstrably approximation error'
                         % (pd, L2_PROBE_EPS))
        else:
            l2['status'] = 'RED'
    res['layer2'] = l2

    # ---- evidence ----
    ev = {}
    c_tr, p_tr = cstd['trace'], py_best['trace']
    if scn.obs:
        ev.update(obs_counter_evidence(c_tr, p_tr))
        ev.update(guard_evidence(variants[0]['trace']))
        if 'S-02' in scn.red:
            twin = load_run(R.py_run(scn, 0, tag='py_noguard_twin'))
            ev['py_guard_effect_out_max_abs'] = cmp_metrics(variants[0]['out'], twin['out'])['max_abs']
    ev['main_call_counter_mismatch_hops'] = trace_mismatch(c_tr, p_tr, 2)
    ev['converged_mismatch_hops'] = trace_mismatch(c_tr, p_tr, 3)
    if scn.ctx:
        ev.update(s05_evidence(c_tr, p_tr))
        K = scn.fft // 2 + 1
        cc = ctx_compare(cstd['ctx'], py_best['ctx'], K)
        res['context'] = cc
        ev['ctx_fields_over_tolerance'] = ctx_ok(cc)
    res['evidence'] = ev

    # ---- verdict ----
    audio_green = l1_ok(l1)
    ctx_bad = ev.get('ctx_fields_over_tolerance', [])
    shape_failures = []
    for sid in scn.red:
        if sid == 'S-02':
            if not ev.get('py_guard_blocked_hops', 0) > 0:
                shape_failures.append('S-02 shape lost (Python guard blocked no injection in this scene): remove S-02 from the marker')
            elif not (ev.get('py_guard_effect_out_max_abs') or 0.0) > TOL['L1_out_max_abs']:
                shape_failures.append('S-02 shape lost (the guard no longer changes the Python output): remove S-02 from the marker')
        if sid == 'S-05' and not ev.get('ctx_conv_shape_is_late_rising'):
            shape_failures.append('S-05 shape lost (filter_converged mismatch is gone or no longer the one-hop-late rising edge): remove S-05 from the marker')
    audio_expected_red = 'S-02' in scn.red
    if not scn.red:
        status = 'PASS' if (audio_green and not ctx_bad) else 'FAIL (layer 1 red, no marker)'
    elif audio_expected_red:
        if audio_green:
            status = 'FAIL UNEXPECTED_GREEN(%s): remove the marker' % ','.join(scn.red)
        elif shape_failures:
            status = 'FAIL ' + '; '.join(shape_failures)
        else:
            status = 'EXPECTED_RED(%s)' % ','.join(scn.red)
    else:           # S-05: the audio itself must be green; only the flag is allowed to differ
        if not audio_green or ctx_bad:
            status = 'FAIL (red outside the S-05 flag: %s)' % ('output cells' if not audio_green else ','.join(ctx_bad))
        elif shape_failures:
            status = 'FAIL ' + '; '.join(shape_failures)
        else:
            status = 'EXPECTED_RED(S-05)'
    if res['layer2']['status'] == 'RED':
        status = 'FAIL layer 2 red' if status == 'PASS' else 'FAIL layer 2 red; layer 1: ' + status
    if scn.info:
        status = 'INFO(%s): layer 1 %s' % (scn.info, 'within tolerance' if audio_green else 'over tolerance')
        if res['layer2']['status'] == 'RED':
            status = 'FAIL layer 2 red (%s)' % status
        if scn.info == 'N-01':
            ev['n01_python_branches_agree'] = bool(len(variants) > 1 and max(
                cmp_metrics(variants[k]['out'], variants[0]['out'])['max_abs'] for k in range(1, len(variants))) <= TOL['L1_out_max_abs'])
            ev['n01_c_builds_agree'] = bool(l2['status'] == 'PASS')
        if scn.info == 'N-02':
            # N-02 is an INFO marker, not a free pass: require the shape evidence that makes this cell's
            # over-tolerance Layer 1 reading a registered branch-selection tie-break rather than a real
            # regression -- (a) the observer call_counter still agrees with Python on every hop (this
            # scene is not an S-01 case; see scenes.py), and (b) C's distance from Python is no larger
            # than Python's own distance from itself under a one-ulp input nudge (the same self-distance
            # computed above for the Layer-1 branch search). When L1 is already within TOL_BAND
            # (audio_green), the self-distance leg holds trivially. Either condition failing means this
            # specific 48k/1024 cell now carries more than the registered tie-break, so it FAILs instead
            # of silently staying INFO.
            n02_cc_ok = ev.get('obs_cc_mismatch_hops', 0) == 0
            n02_dist = res.get('python_self_distance_1ulp_out_max_abs')
            n02_dist_ok = audio_green or (n02_dist is not None and (l1['out']['max_abs'] or 0.0) <= n02_dist)
            ev['n02_obs_cc_ok'] = n02_cc_ok
            ev['n02_within_1ulp_self_dist'] = n02_dist_ok
            if not (n02_cc_ok and n02_dist_ok) and not status.startswith('FAIL'):
                status = ('FAIL N-02 shape lost (obs_cc_mismatch_hops=%s, L1 out max_abs=%s, py_1ulp_self_dist=%s): '
                          'this exceeds the registered branch-selection tie-break, treat as a real regression'
                          % (ev.get('obs_cc_mismatch_hops'), l1['out']['max_abs'], n02_dist))
    res['status'] = status
    res['py_variants_run'] = len(variants) + (1 if l2['status'] in ('RED', 'INCONCLUSIVE_CHAOTIC') else 0) + \
        (1 if 'py_guard_effect_out_max_abs' in ev else 0)
    res['output_sha256'] = {
        'py_v0_out': sha256_bytes(variants[0]['out'].tobytes()),
        'c_std_out': sha256_bytes(cstd['out'].tobytes()),
        'c_ship_out': sha256_bytes(cship['out'].tobytes()),
    }
    return res


# ----------------------------------------------------------------------------------------------
# Static cells
# ----------------------------------------------------------------------------------------------
def hpf_fidelity(R, a):
    cells = []
    for sr in (8000, 16000, 48000):
        mic, _ = R.inp_paths('rec', sr)
        d = os.path.join(a.work, 'hpf', str(sr))
        os.makedirs(d, exist_ok=True)
        py = os.path.join(d, 'py.f32')
        cs = os.path.join(d, 'c_std.f32')
        csh = os.path.join(d, 'c_ship.f32')
        run([sys.executable, R.pyscript, '--hpf-only', mic, py, str(sr)])
        run([a.std_bin, '--hpf-only', mic, cs, str(sr)])
        run([a.ship_bin, '--hpf-only', mic, csh, str(sr)])
        P, C, S = load_f32(py), load_f32(cs), load_f32(csh)
        m = cmp_metrics(P, C)
        cells.append({'id': 'hpf_fidelity.%dk' % (sr // 1000), 'sr': sr, 'cutoff_hz': 80.0,
                      'py_f32_emulation_vs_c_std': {'max_abs': m['max_abs'], 'bitexact': m['bitexact']},
                      'c_ship_vs_c_std_bitexact': bool(np.array_equal(C, S)),
                      'status': 'PASS' if m['bitexact'] else 'FAIL (the float32 HPF emulation is not bit-exact to the C filter: '
                                                              'Layer-1 "HPF type: f32" labels would be wrong)'})
    return cells


def s03_cell(a):
    """Regenerate aec_balanced_config.h from a scratch copy of the Python tree and diff it against the
    checked-in header -- the two must always be identical (S-03, fixed 2026-10-20: the header is
    regenerated alongside any change to the balanced config or the generator; make test-header-fresh
    covers this same check on an ordinary build without needing the Python<->C parity harness)."""
    root = a.aec_root
    d = os.path.join(a.work, 's03')
    if os.path.exists(d):
        shutil.rmtree(d)
    ign = shutil.ignore_patterns('__pycache__', '.pytest_cache', 'tests', 'output_*', '*.pyc')
    shutil.copytree(os.path.join(root, 'python'), os.path.join(d, 'python'), ignore=ign)
    os.makedirs(os.path.join(d, 'c_impl', 'include'))
    gen = os.path.join(d, 'python', 'diag', 'gen_aec_balanced_config_h.py')
    p = run([sys.executable, gen], log=os.path.join(d, 'gen.log'), check=False)
    new_p = os.path.join(d, 'c_impl', 'include', 'aec_balanced_config.h')
    old_p = os.path.join(root, 'c_impl', 'include', 'aec_balanced_config.h')
    cell_ = {'id': 'static.s03_balanced_config_regen', 'generator': 'python/diag/gen_aec_balanced_config_h.py',
             'header': 'c_impl/include/aec_balanced_config.h'}
    if p.returncode != 0 or not os.path.exists(new_p):
        cell_.update({'status': 'FAIL (generator did not run; see work/s03/gen.log)', 'rc': p.returncode})
        return cell_
    old = open(old_p).read().split('\n')
    new = open(new_p).read().split('\n')

    def key(line):
        t = line.split()
        return t[1] if len(t) > 2 and t[0] == '#define' else None

    od = {key(l): l for l in old if key(l)}
    nd = {key(l): l for l in new if key(l)}
    diff_names = sorted(k for k in set(od) | set(nd) if od.get(k) != nd.get(k))
    other_lines = old != new and not diff_names
    cell_.update({'differing_constants': [{'name': k, 'checked_in': od.get(k, '').split(None, 2)[-1] if k in od else None,
                                           'regenerated': nd.get(k, '').split(None, 2)[-1] if k in nd else None} for k in diff_names],
                  'n_differing': len(diff_names), 'non_define_lines_differ': bool(other_lines)})
    if diff_names or other_lines:
        cell_['status'] = 'FAIL: checked-in header is stale against its generator (re-run gen_aec_balanced_config_h.py)'
    else:
        cell_['status'] = 'PASS'
    return cell_


# ----------------------------------------------------------------------------------------------
# Inputs: generate + pin check
# ----------------------------------------------------------------------------------------------
def prepare_inputs(a, scn_list):
    pins = {}
    pin_path = os.path.join(HERE, 'inputs.sha256')
    if os.path.exists(pin_path):
        for line in open(pin_path):
            line = line.strip()
            if line and not line.startswith('#'):
                h, k = line.split(None, 1)
                pins[k.strip()] = h
    need = sorted(set((s.kind, s.sr) for s in scn_list) | set(('rec', sr) for sr in (8000, 16000, 48000)))
    infos, bad = {}, []
    # source wavs
    for rel in (REC_MIC, REC_REF):
        p = os.path.join(a.aec_root, rel)
        if not os.path.exists(p):
            bad.append('MISSING source wav %s' % rel)
            continue
        h = sha256_file(p)
        infos['wav/' + rel] = h
        if not a.write_pins and pins.get('wav:' + rel) != h:
            bad.append('MISMATCH source wav %s: pinned %s, found %s' % (rel, pins.get('wav:' + rel), h))
    os.makedirs(os.path.join(a.work, 'inputs'), exist_ok=True)
    gen = {}
    for kind, sr in need:
        iid = input_id(kind, sr)
        mic, ref = build_input(kind, sr, a.aec_root)
        mp = os.path.join(a.work, 'inputs', iid + '_mic.f32')
        rp = os.path.join(a.work, 'inputs', iid + '_ref.f32')
        mic.tofile(mp)
        ref.tofile(rp)
        for tag, arr in (('mic', mic), ('ref', ref)):
            h = sha256_bytes(arr.tobytes())
            gen['f32:%s_%s' % (iid, tag)] = h
            if not a.write_pins and pins.get('f32:%s_%s' % (iid, tag)) != h:
                bad.append('MISMATCH generated input %s_%s: pinned %s, generated %s '
                           '(numpy/scipy version drift?)' % (iid, tag, pins.get('f32:%s_%s' % (iid, tag)), h))
        infos[iid] = {'samples': int(len(mic)), 'seconds': r6(len(mic) / float(sr)), 'mic_sha256': gen['f32:%s_mic' % iid],
                      'ref_sha256': gen['f32:%s_ref' % iid]}
    if a.write_pins:
        with open(pin_path, 'w') as f:
            f.write('# Pins for the Python<->C parity harness inputs (test/pyc_parity/pyc_parity.py).\n'
                    '# wav:<path>  sha256 of a tracked source wav (path relative to the aec root).\n'
                    '# f32:<input>_<mic|ref>  sha256 of the float32 array the generator produces\n'
                    '#   (rec8/rec48 = scipy resample_poly of the 16 kHz recording; echo/speechdt/dtb = seeded scenes.py).\n'
                    '# A mismatch means the audio changed (numpy/scipy drift or an edit): the baseline moved, so the run refuses.\n'
                    '# Regenerate (maintainers only): pyc_parity_aec.sh --write-pins\n')
            for rel in (REC_MIC, REC_REF):
                f.write('%s  wav:%s\n' % (infos['wav/' + rel], rel))
            for k in sorted(gen):
                f.write('%s  %s\n' % (gen[k], k))
        print('wrote %s' % pin_path)
    return infos, bad


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------
def fmt(x, w=9):
    if x is None:
        return '-'.rjust(w)
    return ('%.2e' % x).rjust(w)


def fmtdb(x, w=7):
    return ('exact' if x is None else '%.1f' % x).rjust(w)


def print_table(results, statics, hpfc):
    print()
    print('HPF fidelity (Layer-1 prerequisite): float32 Python emulation vs audiolib hpf.c, 80 Hz, input = rec @ rate')
    for c in hpfc:
        print('  %-18s bitexact=%s max_abs=%s ship==std:%s   %s' % (
            c['id'], c['py_f32_emulation_vs_c_std']['bitexact'], c['py_f32_emulation_vs_c_std']['max_abs'],
            c['c_ship_vs_c_std_bitexact'], c['status']))
    print()
    hdr = '%-31s %-3s %-5s %-4s %9s %9s %7s %9s %5s %6s  %s'
    print(hdr % ('scenario', 'L', 'hpf', 'pyV', 'out_max', 'out_p99.9', 'rms_dB', 'raw_max', 'bit', 'margin', 'status'))
    for r in results:
        hp = r['hpf']
        l1, l2 = r['layer1'], r['layer2']
        print(hdr % (r['id'], 'L1', hp, l1['py_variant'].split()[0], fmt(l1['out']['max_abs']), fmt(l1['out']['p999']),
                     fmtdb(l1['out']['rms_db']), fmt(l1['raw']['max_abs']), 'Y' if l1['out']['bitexact'] else 'n',
                     '%.2g' % l1['margin_out_max_abs'], r['status']))
        print(hdr % ('', 'L2', '', '', fmt(l2['out']['max_abs']), fmt(l2['out']['p999']),
                     fmtdb(l2['out']['rms_db']), fmt(l2['raw']['max_abs']), 'Y' if l2['out']['bitexact'] else 'n',
                     '%.2g' % l2['margin_out_max_abs'], l2['status']))
        ev = r['evidence']
        bits = []
        for k in ('obs_cc_mismatch_hops', 'py_guard_blocked_hops', 'py_guard_effect_out_max_abs', 'ctx_conv_mismatch_hops',
                  'ctx_conv_rising_edges', 'ctx_conv_late_rising_mismatches', 'n01_python_branches_agree',
                  'n01_c_builds_agree'):
            if k in ev:
                bits.append('%s=%s' % (k, ev[k]))
        if 'python_self_distance_1ulp_out_max_abs' in r:
            bits.append('py_1ulp_self_dist=%s' % r['python_self_distance_1ulp_out_max_abs'])
        for k in sorted(l2):
            if k.startswith('python_self_distance_at_'):
                bits.append('py_self_dist@%s=%s' % (k.split('_at_')[1].split('_')[0], l2[k]))
        if bits:
            print('%-31s     %s' % ('', '  '.join(bits)))
    for s in statics:
        print()
        print('%s: %s' % (s['id'], s['status']))
        for d in s.get('differing_constants', []):
            print('    %s: checked-in %s  regenerated %s' % (d['name'], d['checked_in'], d['regenerated']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aec-root', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--std-bin', required=True)
    ap.add_argument('--ship-bin', required=True)
    ap.add_argument('--json', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', default=None, help='substring filter on scenario ids (comma separated = any)')
    ap.add_argument('--write-pins', action='store_true')
    ap.add_argument('--skip-static', action='store_true')
    a = ap.parse_args()
    a.aec_root = os.path.abspath(a.aec_root)
    a.work = os.path.abspath(a.work)
    os.makedirs(a.work, exist_ok=True)

    scn_list = build_scenarios()
    if a.quick:
        scn_list = [s for s in scn_list if s.quick]
    if a.only:
        pats = a.only.split(',')
        scn_list = [s for s in scn_list if any(p in s.id for p in pats)]
    if not scn_list:
        print('no scenarios selected')
        return 2
    import scipy
    import soundfile
    env = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
           'soundfile': soundfile.__version__, 'machine': platform.machine(), 'system': platform.system()}
    print('pyc_parity: AEC Python<->C parity  (tolerances %s)' % TOL['status'])
    print('env: python %(python)s numpy %(numpy)s scipy %(scipy)s soundfile %(soundfile)s on %(system)s/%(machine)s' % env)

    infos, bad = prepare_inputs(a, scn_list)
    if bad:
        print('FAIL: input pin check')
        for b in bad:
            print('  ' + b)
        return 1
    if a.write_pins:
        return 0
    R = Runner(a)
    hpfc = hpf_fidelity(R, a)
    results = []
    for s in scn_list:
        results.append(eval_scenario(R, s))
        sys.stdout.flush()
    statics = [] if a.skip_static else [s03_cell(a)]
    print_table(results, statics, hpfc)

    fails = [r['id'] for r in results if r['status'].startswith('FAIL')]
    fails += [c['id'] for c in hpfc if c['status'].startswith('FAIL')]
    fails += [s['id'] for s in statics if s['status'].startswith('FAIL')]
    summ = {
        'scenarios': len(results),
        'pass': sum(1 for r in results if r['status'] == 'PASS'),
        'expected_red': sum(1 for r in results if r['status'].startswith('EXPECTED_RED')),
        'info': sum(1 for r in results if r['status'].startswith('INFO')),
        'fail': len(fails),
        'layer2_inconclusive_chaotic': sum(1 for r in results if r['layer2']['status'] == 'INCONCLUSIVE_CHAOTIC'),
        'layer2_red': sum(1 for r in results if r['layer2']['status'] == 'RED'),
        'python_variant_runs_beyond_v0': sum(r['py_variants_run'] - 1 for r in results),
        'failed_ids': fails,
    }
    doc = {'suite': 'quick' if a.quick else 'full', 'tolerances': TOL, 'env': env, 'expected_red_registry': EXPECTED_RED_INFO,
           'inputs': infos, 'hpf_fidelity': hpfc, 'scenarios': results, 'static': statics, 'summary': summ}
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write('\n')
    print()
    print('summary: %s' % json.dumps(summ, sort_keys=True))
    print('registry of known gaps and observations (S-ids are PY_C_SYNC_AUDIT.md ids; marker => allowed to be red, must stay red):')
    for k in sorted(EXPECTED_RED_INFO):
        print('  %s(%s): %s' % ('INFO' if EXPECTED_RED_INFO[k].startswith('INFO') else 'EXPECTED_RED', k, EXPECTED_RED_INFO[k]))
    print('RESULT: %s' % ('FAIL' if fails else 'PASS'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
