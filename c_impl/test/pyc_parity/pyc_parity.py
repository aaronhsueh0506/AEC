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
The same registry holds INFO-only observations that are not audit S-ids (N-01, N-02, T-01); T-01 marks a cell whose red
comes from a registered float32 tie in a per-bin filter decision and is INFO only while its forced-decision proof holds
(see eval_scenario).

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

Guard seam cells (section G of build_scenarios; --base on both drivers). C and Python dump, per hop, the output guard's decision
(guard_active), the hard post-chain reset counter (post_reset_serial), the rendering base spectrum (output_base_spec /
aec_get_residual().base_spec), the formed hop and the delay state (current delay, bad-relock reject count), next to the analysis seam already
compared (error_spec, echo_spec). Gates: guard_active identical on every hop outside the tie band GUARD_TIE_BAND (derivation at the constant: a mismatch is excused only when both sides' relative energy margin |sum|E|^2 - sum|Y|^2| / sum|E|^2 is
<= 1e-6 on a hop unusable on both sides; tie-band hops and in-band mismatches are counted and printed per cell); post_reset_serial
identical on every hop; current delay identical on every hop and the bad-relock reverts at the same hops (the guard.revert.* cells
also require at least one revert in each language); base_spec (all hops, and the hops where either side's guard fired) within
CTX_spec_rel_l2; the ctx fields within their CTX constants (res_gain three ways: its near-end-power-weighted rel L2,
CTX_gain_wrel_l2; its max_abs over the loud bins, CTX_gain_loud_max_abs; and the rendered product base * res_gain per hop,
CTX_render_hop_rel_l2; its all-bin max_abs is printed only); formed_hop within L1_raw_max_abs; the audio within the Layer-1/2
constants. The hard seam gates
(guard_active outside the band, post_reset_serial, delay events / reverts) have no marker anywhere. Seven cells carry T-01 (a
registered float32 tie in the refined filter's per-bin leakage compare or the filter noise gate): they are INFO only while the
hard seam gates are green as run AND the Python run with the registered decisions forced the way C takes them (driver
--force-leak / --force-gate) is green on every audio, context and guard-seam gate; an unforced green cell FAILs (remove the
marker). The inputs are blind-set clips (untracked datasets, wav/download_blind_test.py): a missing clip is a setup
FAILURE, not a skip; they run in --full / --only only.

Suites: the default is the lean suite (10 scenes, one per mechanism a planted regression is known to move); --full runs
every scenario; --quick is an older, narrower subset; --only SUBSTR selects from the full set.

Dumps are checked before anything is measured (checked_pair): every compared array must be non-empty, of the same length as its
counterpart and finite on both sides (out, raw, trace hops, ctx records, base records, formed hop, delay rows). A violation FAILs
the cell whatever its marker (INFO, T-01, expected-red); there is no deliberate trim, both drivers run the same hop loop. Gates are
written so that a NaN reading fails (worse(): 'not (x <= tol)').

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
    # context seam (aec_get_linear / aec_get_residual views), Layer 1 only
    'CTX_scalar_rel': 2.0e-5,        # tightened 50x (was 1.0e-3); worst GREEN 2.265e-6 (erle_factor), margin 8.8x
    'CTX_spec_rel_l2': 1.0e-5,       # tightened 100x (was 1.0e-3); worst GREEN 5.28e-7 (echo_spec), margin 18.9x
    'CTX_gain_max_abs': 1.0e-3,      # printed only (no longer gates, see CTX_gain_wrel_l2): a per-bin, per-hop maximum is set by the
                                     # quietest bins, where float32 FFT rounding is ~1e-3 of the bin itself 75-85 dB below the hop
                                     # peak and the max-gain ratchet multiplies an earlier hop's difference by sqrt(inc) (x2 at
                                     # 16k/256, x4 at 16k/512); GREEN cells reach 9.96e-4 (guard.hard_reset.16k512.lean)
    'CTX_gain_wrel_l2': 1.0e-4,      # res_gain error weighted by the near-end power |E + echo|^2 of each bin (what the gain
                                     # multiplies): worst C-vs-Python 1.52e-5 (IUp2.16k256 vs its 1-ulp branch; 1.01e-5 unperturbed,
                                     # rOza.16k256), margin 6.6x; a planted 0.1 % enr_suppress change reads 1.29e-3..3.02e-3 (13x over)
    # Two res_gain gates beside the weighted L2, which is a whole-cell aggregate and so dilutes an error confined to a few hops or
    # bins. Measured 2026-10-29 on the 16 guard cells and the 3 other context cells (C std vs the Python run each cell is judged
    # against, and the forced run of every T-01 cell); a planted +0.1 % enr_suppress (Python, Y91u/waxU 16k256 full and
    # hard_reset.16k256.lean) reads 18x..83x over each:
    'CTX_gain_loud_max_abs': 3.0e-3, # max |dgain| over the bins within LOUD_BIN_DB of the hop's peak near-end power |E + echo|^2:
                                     # worst 1.17e-3 (revert.rOza.16k256; the ratchet reaches loud bins too), margin 2.6x; planted
                                     # 0.055..0.25
    'CTX_render_hop_rel_l2': 3.0e-3, # what an application renders, rendering base * res_gain (guard seam cells only: the others do
                                     # not dump the base): max over hops within RENDER_HOP_FLOOR_DB of the cell's loudest product hop
                                     # of ||Pc - Pp|| / ||Pp||; worst 1.06e-3 (revert.waxU.16k256), margin 2.8x (p99 <= 5.6e-5 and
                                     # whole-cell <= 1.0e-5, printed); planted 0.066..0.16
}
LOUD_BIN_DB = 20.0             # loud bin: near-end power >= hop peak - 20 dB (10 dB and 30 dB give the same worst cell)
RENDER_HOP_FLOOR_DB = 60.0     # hops more than 60 dB below the loudest product hop are left out (near silence; 40/80 dB change
                               # the worst green reading by < 1 %)
MAX_PY_VARIANTS = 3           # unperturbed + one-ulp-perturbed seeds 1 and 2
L2_PROBE_EPS = 1e-4           # relative mic nudge used to ask 'is this scene chaotic at approximation scale?'
L2_PROBE_SEED = 11

# ----------------------------------------------------------------------------------------------
# Known Python<->C gaps (registered in docs/c_python_function_map.md, appendix A). Keyed by the S-id used in scenario
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
    'T-01': 'INFO only (not an audit S-id): a registered float32 tie -- a refined-filter per-bin H_error leakage compare '
            '(e2_refined[k] <= e2_coarse[k]) or the filter noise gate (X2[k] >= gate) whose two operands differ by less than their '
            'C-vs-Python rounding difference at that bin -- is decided one way by C and the other by Python, and the adaptive '
            'state carries the one-bin difference to the analysis seam (and, with RES on, the audio). Never a free pass: the cell '
            'FAILs unless the hard seam gates (guard_active outside the tie band, post_reset_serial, delay events / reverts) are '
            'green as run AND Python re-run with exactly the registered decision(s) forced the way C takes them (driver '
            '--force-leak / --force-gate) matches C on every Layer-1, context and guard-seam gate; it also FAILs once the '
            'unforced cell is green (remove the marker). Cells, decisions and margins: CHANGELOG.md, 2026-10-28, C-versus-Python parity contract, and 2026-11-16, pyc-parity: hard_reset.48k1024 lean and full register a T-01 tie',
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
_BLIND16 = 'wav/aec_challenge_blind/doubletalk/'
_BLIND16_FS = 'wav/aec_challenge_blind/farend_singletalk/'
_BLIND48 = 'wav/aec_challenge_blind_icassp2023/farend_singletalk/'


def _clip(folder, stem, sr):
    return (folder + stem + '_mic.wav', folder + stem + '_lpb.wav', sr)


# Blind-set clips that exercise the output guard (picked from the 1900-run scan in test/guard_unify_clips.tsv's note): input key is
# the clip stem; value = (mic wav, far-end wav, sample rate), paths relative to the aec root.
CLIPS = {
    # never usable, guard fires on ~55 % of hops
    'sRCs6SKo6kC0xire475q0A_doubletalk': _clip(_BLIND16, 'sRCs6SKo6kC0xire475q0A_doubletalk', 16000),
    # never usable, guard fires on ~55 % of hops (16 kHz / 512 grid)
    'V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement': _clip(_BLIND16, 'V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement', 16000),
    # hard post-chain resets (delay-change filter restarts) and firing after the first usable verdict
    'X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement': _clip(_BLIND16, 'X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement', 16000),
    # 48 kHz, never usable, guard fires on ~70 % of hops
    '6V90tmuaRkG0luLY5iAwTw_farend-singletalk': _clip(_BLIND48, '6V90tmuaRkG0luLY5iAwTw_farend-singletalk', 48000),
    # 48 kHz, hard post-chain resets and firing after the first usable verdict
    '16970455-e9f7-42ec-b668-3ab3f79ff1de_farend-singletalk': _clip(_BLIND48, '16970455-e9f7-42ec-b668-3ab3f79ff1de_farend-singletalk', 48000),
    # bad-relock soft revert on both 16 kHz grids: far-end single talk, moving / static
    'Y91uE2tRg0SUB2a9XjT30w_farend_singletalk_with_movement': _clip(_BLIND16_FS, 'Y91uE2tRg0SUB2a9XjT30w_farend_singletalk_with_movement', 16000),
    'IUp2c0A4yEyttmLfJ3t0Xw_farend_singletalk': _clip(_BLIND16_FS, 'IUp2c0A4yEyttmLfJ3t0Xw_farend_singletalk', 16000),
    # bad-relock revert, double talk (static): the two clips whose post-revert C-vs-Python difference was the largest
    'rOzaUzFu0kKDiJa1qQVoOg_doubletalk': _clip(_BLIND16, 'rOzaUzFu0kKDiJa1qQVoOg_doubletalk', 16000),
    'waxU019BVEacr7vK6v00mQ_doubletalk': _clip(_BLIND16, 'waxU019BVEacr7vK6v00mQ_doubletalk', 16000),
}
GRIDS = {            # tag -> (sr, fft)
    '8k256': (8000, 256),
    '16k256': (16000, 256),
    '16k512': (16000, 512),
    '48k1024': (48000, 1024),
}


def input_id(kind, sr):
    if kind in CLIPS:
        return kind
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
    if kind in CLIPS:
        import soundfile as sf
        mic_rel, ref_rel, csr = CLIPS[kind]
        chans = []
        for rel in (mic_rel, ref_rel):
            x, xs = sf.read(os.path.join(aec_root, rel), dtype='float32')
            assert xs == csr == sr, '%s: sample rate %s, expected %s (scenario %s)' % (rel, xs, csr, sr)
            chans.append(x[:, 0] if x.ndim > 1 else x)
        n = min(len(chans[0]), len(chans[1]))        # the clips' far-end is shorter than the mic by a few seconds
        return np.ascontiguousarray(chans[0][:n]), np.ascontiguousarray(chans[1][:n])
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
                 ctx=False, cng=True, red=(), quick=False, lean=False, info='', v0_only=False, base=False, reverts=0,
                 ties=()):
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
        self.lean = lean             # member of the default suite (see build_scenarios)
        self.info = info             # observation, never a gate (e.g. a chaos-fragile finding that is not an audit S-id)
        self.base = base             # guard seam cell: both drivers also dump guard_active, post_reset_serial, rendering base, formed hop
        self.reverts = reverts       # guard seam cell: minimum bad-relock reverts BOTH languages must reach (0 = no demand)
        self.ties = tuple(ties)      # T-01: Python driver flags forcing the registered tie decision(s) the way C takes them
        assert not base or ctx, 'a guard seam cell needs the context'

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
        if self.base:
            f.append('--base')
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
        if self.base:
            f.append('--base')
        return f


def build_scenarios():
    # Suites: the default (lean, 10 scenes) keeps one scene per mechanism that a planted regression is known to move;
    # --full runs every scene. Lean membership is an exact id list (the scenes marked lean=True below), so a new
    # variant never joins it by accident. Chosen from planted-regression runs (each of these was caught by the named scene):
    #   H_error decay constant           rec.8k256.bal, rec.16k256.bal, dtb.16k256.bal, speechdt.16k256.bal, echo.16k512.bal
    #   48 kHz resampler phase           dtb.48k1024.bal
    #   observer shift (+1 hop)          preecho.16k256.bal.obs_noguard (the 16k/512 observer scene and the dtb obs scenes miss it)
    #   aggressive-preset floor          dtb.16k256.agg (the rec.* aggressive scenes miss it)
    #   context-seam filter_converged    dtb.16k256.bal.ctx
    #   startup-hops sentinel            rec.16k256.bal, dtb.16k256.bal
    #   mic HPF off path                 recdc.16k256.bal.nohpf
    # --quick is the older, narrower subset; it has no aggressive scene and its observer scene misses the +1 hop shift.
    S = []
    add = S.append
    short = {'balanced': 'bal', 'aggressive': 'agg', 'mild': 'mild', 'no_floor': 'nofloor'}
    # A. default configuration (observer off, mic HPF on): real recording and synthetic scenes
    for g in GRIDS:
        for p in ('balanced', 'aggressive'):
            add(Scn('rec.%s.%s' % (g, short[p]), 'rec', g, p, quick=(p == 'balanced' and g in ('16k256', '8k256')),
                lean=(p == 'balanced' and g in ('16k256', '8k256'))))
    for g in GRIDS:
        add(Scn('dtb.%s.bal' % g, 'dtb', g, quick=(g in ('16k256', '48k1024')), lean=(g in ('16k256', '48k1024'))))
        add(Scn('speechdt.%s.bal' % g, 'speechdt', g, quick=(g == '16k256'), lean=(g == '16k256')))
        add(Scn('echo.%s.bal' % g, 'echo', g, quick=(g == '16k512'), lean=(g == '16k512')))
    add(Scn('dtb.16k256.agg', 'dtb', '16k256', 'aggressive', lean=True))
    add(Scn('dtb.48k1024.agg', 'dtb', '48k1024', 'aggressive'))
    add(Scn('rec.16k256.mild', 'rec', '16k256', 'mild'))
    add(Scn('dtb.48k1024.mild', 'dtb', '48k1024', 'mild'))
    # B. mic HPF disabled on both sides. recdc = the recording with its DC offset removed (what the HPF would
    # have delivered as far as DC goes); the raw-DC 48k case is the registered N-01.
    for g in ('8k256', '16k256', '48k1024'):
        add(Scn('recdc.%s.bal.nohpf' % g, 'recdc', g, hpf='off', quick=(g == '16k256'), lean=(g == '16k256')))
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
        add(Scn('preecho.%s.bal.obs_noguard' % g, 'preecho', g, obs=True, guard_off=True, quick=(g == '16k512'),
                lean=(g == '16k256')))
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
    # D. context seam (aec_get_linear / aec_get_residual views). S-05 (filter_converged one hop
    # late on the rising edge) was fixed 2026-10-20: Python now patches the
    # field after update_convergence runs for the hop, matching C's
    # aec_get_linear() post-update pull contract. No marker needed.
    add(Scn('dtb.16k256.bal.ctx', 'dtb', '16k256', ctx=True, quick=True, lean=True))
    add(Scn('dtb.48k1024.bal.ctx', 'dtb', '48k1024', ctx=True))
    add(Scn('speechdt.16k256.bal.ctx_nores', 'speechdt', '16k256', ctx=True, res=False))
    # F. NO_FLOOR (AecPreset value 3 / "no_floor"): construction-time-only on both sides today (the
    # C-side runtime aec_set_level()/aec_res_set_level() retarget to/from NO_FLOOR added alongside this
    # row is NOT exercised by this driver -- it constructs fresh per scenario, same as every other preset
    # row here). min_gain_split_floor_enabled=False removes all three suppression-gain floor variants, so
    # this is the deepest point on the strength axis and the one most likely to separate from balanced;
    # one dtb scene (where the floor matters most) plus one on the other FFT family.
    add(Scn('dtb.16k256.nofloor', 'dtb', '16k256', 'no_floor'))
    add(Scn('dtb.48k1024.nofloor', 'dtb', '48k1024', 'no_floor'))
    # G. output-guard / reset seam on blind-set clips (see the module docstring). Lean context (enable_res off, return_res_context
    # on) is the API the guard exists for; the two '.full' cells run the same clip with the internal RES on, so the guarded render
    # reaches the final audio that the Layer-1 output gates compare. These need the untracked blind datasets, so they are --full only.
    for sid, clip, g, res in (
            ('guard.never_usable.16k256.lean', 'sRCs6SKo6kC0xire475q0A_doubletalk', '16k256', False),
            ('guard.never_usable.16k256.full', 'sRCs6SKo6kC0xire475q0A_doubletalk', '16k256', True),
            ('guard.never_usable.16k512.lean', 'V0JqgjlrB0Ke9y91r0rxNw_doubletalk_with_movement', '16k512', False),
            ('guard.hard_reset.16k256.lean', 'X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement', '16k256', False),
            ('guard.hard_reset.16k512.lean', 'X7Ua9txMj0aws848JPEbOg_doubletalk_with_movement', '16k512', False),
            ('guard.never_usable.48k1024.lean', '6V90tmuaRkG0luLY5iAwTw_farend-singletalk', '48k1024', False),
            ('guard.hard_reset.48k1024.lean', '16970455-e9f7-42ec-b668-3ab3f79ff1de_farend-singletalk', '48k1024', False),
            ('guard.hard_reset.48k1024.full', '16970455-e9f7-42ec-b668-3ab3f79ff1de_farend-singletalk', '48k1024', True)):
        add(Scn(sid, clip, g, ctx=True, res=res, base=True))
    # bad-relock soft revert (C aec_apply_delay_shift_soft_revert = Python _apply_delay_shift_soft_revert): full mode (RES on, so
    # the post-revert render reaches the audio gates), every clip must revert at least once in BOTH languages at the same hop.
    # Seven guard cells carry a registered float32 tie (T-01; CHANGELOG.md, 2026-10-28, C-versus-Python parity contract; 2026-11-16, pyc-parity: hard_reset.48k1024).
    # rOza.16k512: 74 hops after the revert (hop 543) bin 148's partition-summed render power X2 = 0.01869662292 (Python) / 0.0186966192 (C) straddles the filter noise
    # gate 0.018696621 (margins +-1.0e-7), so the strict 'X2 >= gate' updates that bin in Python and not in C; with that decision
    # forced, the next one is the leakage compare at hop 2099 bin 122 (margins +2.0e-7 / -2.0e-6), and with both forced C and
    # Python agree to 3.9e-7. The revert itself matches (same hop, same delays, same serial).
    for stem, short in (('Y91uE2tRg0SUB2a9XjT30w_farend_singletalk_with_movement', 'Y91u'),
                        ('IUp2c0A4yEyttmLfJ3t0Xw_farend_singletalk', 'IUp2'),
                        ('rOzaUzFu0kKDiJa1qQVoOg_doubletalk', 'rOza'),
                        ('waxU019BVEacr7vK6v00mQ_doubletalk', 'waxU')):
        for g in ('16k256', '16k512'):
            add(Scn('guard.revert.%s.%s.full' % (short, g), stem, g, ctx=True, res=True, base=True, reverts=1))
    # hard_reset.48k1024 (lean and full, std-math build only): the leakage compare at hop 1387 bin 222 and two later ones tie
    # (CHANGELOG.md, 2026-11-16, pyc-parity: hard_reset.48k1024 lean and full register a T-01 tie).
    # T-01 ties, as (driver flags) per cell: each is the FIRST decision C and Python take differently, then the next one on the
    # forced trajectory, until none is left (found with C and Python per-hop decision traces)
    ties = {
        'guard.never_usable.16k512.lean': ('--force-leak', '2327:248:0'),
        'guard.never_usable.48k1024.lean': ('--force-leak', '843:311:1,1091:114:0,1137:382:1,1669:403:1'),
        'guard.revert.IUp2.16k256.full': ('--force-leak', '1110:74:0'),
        'guard.revert.rOza.16k512.full': ('--force-gate', '543:2e-7', '--force-leak', '2099:122:1'),
        'guard.revert.waxU.16k512.full': ('--force-leak', '659:180:1'),
        'guard.hard_reset.48k1024.lean': ('--force-leak', '1387:222:1,1388:302:1,2146:287:1'),
        'guard.hard_reset.48k1024.full': ('--force-leak', '1387:222:1,1388:302:1,2146:287:1'),
    }
    for scn in S:
        if scn.id in ties:
            scn.ties = ties[scn.id]
    return S


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def r6(x):
    """Round for the report. A non-finite value stays non-finite (never None), so a gate written as worse() fails on it."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return x
    return float('%.6g' % x)


def load_f32(path):
    return np.fromfile(path, dtype=np.float32)


class DumpError(Exception):
    """A dump that cannot be compared: empty, non-finite, or of a different length than its counterpart. Fails the cell."""


def checked_pair(name, test, ref):
    """Every C-vs-Python (or build-vs-build) comparison goes through here first: both arrays non-empty, of equal length and
    finite on both sides. Both drivers run the same hop loop (range(0, n - hop, hop)) over the same samples, so there is no
    deliberate trim anywhere and any length difference is a FAIL, never a truncation to the shorter one."""
    test, ref = np.asarray(test), np.asarray(ref)
    if test.size == 0 or ref.size == 0:
        raise DumpError('%s: empty (test %d, ref %d values)' % (name, test.size, ref.size))
    if test.shape != ref.shape:
        raise DumpError('%s: length differs (test %s, ref %s)' % (name, test.shape, ref.shape))
    for side, x in (('test', test), ('ref', ref)):
        if np.issubdtype(x.dtype, np.inexact):
            bad = ~np.isfinite(x)
            if bad.any():
                i = int(np.flatnonzero(bad)[0])
                raise DumpError('%s: %d non-finite value(s) on the %s side (first at flat index %d: %s)'
                                % (name, int(bad.sum()), side, i, np.ravel(x)[i]))
    return test, ref


def cmp_metrics(test, ref, name='audio'):
    """test, ref: float32 arrays (checked by checked_pair). Metrics of d = test - ref (float64)."""
    test, ref = checked_pair(name, test, ref)
    n = len(test)
    a = test.astype(np.float64)
    b = ref.astype(np.float64)
    d = a - b
    ad = np.abs(d)
    rms_d = math.sqrt(float(np.mean(d * d)))
    rms_b = math.sqrt(float(np.mean(b * b)))
    exact = bool(np.array_equal(test, ref))
    if rms_d == 0.0:
        rms_db = None            # bit identical: no finite dB figure
    else:
        rms_db = 20.0 * math.log10(rms_d / (rms_b + 1e-30))
    return {
        'n': int(n),
        'max_abs': r6(float(np.max(ad))),
        'p999': r6(float(np.percentile(ad, 99.9))),
        'rms_db': r6(rms_db),
        'bitexact': exact,
        'len_equal': True,
    }


def worse(x, tol):
    """Gate: True unless x is a number <= tol. None means 'no figure' (e.g. no guard hop in the cell); NaN fails."""
    return x is not None and not (x <= tol)


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


# The C driver takes the level as its enum number; the Python twin keeps the names.
C_LEVEL_DIGIT = {'mild': '0', 'balanced': '1', 'aggressive': '2', 'no_floor': '3'}


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
        run([binp, mic, ref, pre, str(scn.sr), str(scn.fft), C_LEVEL_DIGIT[scn.preset]] + scn.c_flags(), log=pre + '.log')
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
        if tag == 'py_ties':
            flags = flags + list(scn.ties)
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
    if os.path.exists(pre + '.base.bin'):
        r['base_raw'] = np.fromfile(pre + '.base.bin', dtype=np.uint8)
        r['formed'] = load_f32(pre + '.formed.f32')
        r['dly'] = np.loadtxt(pre + '.dly.txt', dtype=np.int64, ndmin=2)
    return r


def cell(c_run_data, ref_run_data):
    nt, nr = len(c_run_data['trace']), len(ref_run_data['trace'])
    if nt == 0 or nt != nr:
        raise DumpError('trace: %d hops (test) vs %d hops (ref)' % (nt, nr))
    return {'out': cmp_metrics(c_run_data['out'], ref_run_data['out'], 'out'),
            'raw': cmp_metrics(c_run_data['raw'], ref_run_data['raw'], 'raw')}


def l1_ok(m):
    return (not worse(m['out']['max_abs'], TOL['L1_out_max_abs'])
            and not worse(m['raw']['max_abs'], TOL['L1_raw_max_abs'])
            and not worse(m['out']['rms_db'], TOL['L1_out_rms_db'])
            and m['out']['len_equal'] and m['raw']['len_equal'])


def l2_ok(m):
    return (not worse(m['out']['max_abs'], TOL['L2_out_max_abs'])
            and not worse(m['raw']['max_abs'], TOL['L2_raw_max_abs'])
            and not worse(m['out']['rms_db'], TOL['L2_out_rms_db'])
            and m['out']['len_equal'] and m['raw']['len_equal'])


def score(m):
    x = m['out']['max_abs']
    return x if x is not None and math.isfinite(x) else math.inf


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


CTX_SCALARS = ('far_power', 'erle_factor', 'dt_indicator', 'divergence', 'saturation_level', 'erl_estimate',
               'filter_converged', 'res_floor_protect', 'usable_linear')


def ctx_records(ctx, K, name):
    """ctx dump -> float64 [hops, 9 + 7K] (per hop: 9 scalars, error_spec 2K, echo_spec 2K, res_gain K, r2 K, cn K)."""
    rec = 9 + 7 * K
    if len(ctx) == 0 or len(ctx) % rec:
        raise DumpError('%s: %d floats is not a whole, non-zero number of %d-float context records' % (name, len(ctx), rec))
    return ctx.reshape(-1, rec).astype(np.float64)


def loud_bin_gain_max_abs(ga, gb, w, x_db):
    """max |ga - gb| over the bins whose near-end power w is within x_db of that hop's peak w (hops with a zero peak skipped)."""
    peak = np.max(w, axis=1, keepdims=True)
    loud = (w >= peak * 10.0 ** (-x_db / 10.0)) & (peak > 0.0)
    return float(np.max(np.abs(ga - gb)[loud])) if loud.any() else 0.0, int(loud.sum())


def render_product_rel(ga, gb, bc, bp, floor_db):
    """The spectrum an application renders, base * gain, C vs Python: per-hop rel L2 ||Pc - Pp|| / ||Pp|| over the hops whose
    product power is within floor_db of the cell's loudest product hop, and the whole-cell rel L2 (all hops)."""
    pc, pp = bc * ga, bp * gb
    e = np.sum(np.abs(pc - pp) ** 2, axis=1)
    q = np.sum(np.abs(pp) ** 2, axis=1)
    keep = q >= np.max(q) * 10.0 ** (-floor_db / 10.0)
    per_hop = np.sqrt(e[keep] / np.maximum(q[keep], 1e-30))
    return {'per_hop_max': float(np.max(per_hop)), 'per_hop_p99': float(np.percentile(per_hop, 99)),
            'per_hop_median': float(np.median(per_hop)), 'hops_gated': int(keep.sum()), 'hops': int(len(q)),
            'whole_rel_l2': float(np.sqrt(np.sum(e) / max(float(np.sum(q)), 1e-30)))}


def ctx_compare(c, p, K, base_c=None, base_p=None):
    """C and Python ctx dumps (per hop: 9 scalars, error_spec 2K, echo_spec 2K, res_gain K, r2 K, cn K); with base_c/base_p (the
    rendering base, complex [hops, K], guard seam cells) also the rendered product base * res_gain."""
    C, P = checked_pair('ctx', ctx_records(c, K, 'ctx (test)'), ctx_records(p, K, 'ctx (ref)'))
    nh = len(C)
    out = {'hops': int(nh), 'scalars': {}}
    for i, nm in enumerate(CTX_SCALARS):
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
    # res_gain weighted by the Python side's near-end power |error_spec + echo_spec|^2 per (hop, bin)
    pe = P[:, 9:9 + 2 * K].reshape(nh, K, 2)
    po = P[:, 9 + 2 * K:9 + 4 * K].reshape(nh, K, 2)
    w = (pe[..., 0] + po[..., 0]) ** 2 + (pe[..., 1] + po[..., 1]) ** 2
    ga, gb = C[:, 9 + 4 * K:9 + 5 * K], P[:, 9 + 4 * K:9 + 5 * K]
    out['res_gain']['wrel_l2'] = r6(float(np.sqrt(np.sum((ga - gb) ** 2 * w) / max(float(np.sum(gb ** 2 * w)), 1e-30))))
    lm, nl = loud_bin_gain_max_abs(ga, gb, w, LOUD_BIN_DB)
    out['res_gain']['loud_max_abs'] = r6(lm)
    out['res_gain']['loud_bins'] = nl
    if base_c is not None:
        bc, bp = checked_pair('base_spec (render product)', base_c, base_p)
        if len(bc) != nh:
            raise DumpError('base_spec: %d hops, context %d hops' % (len(bc), nh))
        rp = render_product_rel(ga, gb, bc.astype(np.complex128), bp.astype(np.complex128), RENDER_HOP_FLOOR_DB)
        out['res_gain']['render'] = {k: (r6(v) if isinstance(v, float) else v) for k, v in rp.items()}
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
        elif worse(v['rel_to_max'], TOL['CTX_scalar_rel']):
            bad.append(nm)
    for nm in ('error_spec', 'echo_spec'):
        if worse(cc[nm]['rel_l2'], TOL['CTX_spec_rel_l2']):
            bad.append(nm)
    g = cc['res_gain']
    if worse(g['wrel_l2'], TOL['CTX_gain_wrel_l2']):
        bad.append('res_gain')
    if worse(g['loud_max_abs'], TOL['CTX_gain_loud_max_abs']):
        bad.append('res_gain(loud bins)')
    if 'render' in g and worse(g['render']['per_hop_max'], TOL['CTX_render_hop_rel_l2']):
        bad.append('res_gain(render product)')
    for nm in ('r2', 'comfort_noise'):
        if worse(cc[nm]['rel_l2'], TOL['CTX_spec_rel_l2']):
            bad.append(nm)
    return bad


def base_records(raw, K):
    """<pre>.base.bin -> (guard_active int32[n], post_reset_serial uint32[n], base_spec complex64[n, K]).
    Per hop: int32 guard_active, uint32 post_reset_serial, K complex float32 (re, im)."""
    dt = np.dtype([('g', '<i4'), ('s', '<u4'), ('b', '<f4', (K, 2))])
    if len(raw) == 0 or len(raw) % dt.itemsize:
        raise DumpError('base dump: %d bytes is not a whole, non-zero number of %d-byte hop records' % (len(raw), dt.itemsize))
    a = raw.view(dt)
    b = (a['b'][..., 0] + 1j * a['b'][..., 1]).astype(np.complex64)
    return a['g'].copy(), a['s'].copy(), b


def ctx_spectra(ctx, K):
    """ctx dump -> (error_spec, echo_spec, usable_linear) per hop, complex128 / int."""
    x = ctx_records(ctx, K, 'ctx')
    nh = len(x)
    e = x[:, 9:9 + 2 * K].reshape(nh, K, 2)
    o = x[:, 9 + 2 * K:9 + 4 * K].reshape(nh, K, 2)
    return e[..., 0] + 1j * e[..., 1], o[..., 0] + 1j * o[..., 1], x[:, 8]


# Guard decision tie band (the guard checks of c_impl/test/test_guard_unify.c). The per-hop rule is a strict '>' between two float32 energy sums,
# sum|E|^2 > sum|Y|^2 (Y = E + echo), and the two languages' upstream spectra legitimately differ by ~1e-7 relative, so a hop whose
# relative margin m = (sum|E|^2 - sum|Y|^2) / sum|E|^2 is that close to 0 can be decided either way. Measured 2026-10-29 on the
# guard cells (16 cells, each side's own dumped float32 spectra, margin in float64), restricted to hops unusable on both sides:
#   * near the threshold (|m| <= 1e-5 on either side) the C-vs-Python margin disagreement |m_c - m_py| has median 2.6e-8..3.4e-8
#     and maximum 3.0e-7 (never_usable.16k256), 1.9e-7 (never_usable.16k512), 3.3e-7 at 48k/1024 (|m| <= 1e-4 there);
#   * every guard mismatch observed (81 + 27 hops on the two 16 kHz never-usable clips) has max(|m_c|, |m_py|) <= 2.48e-7;
#   * on top of that each side rounds its float32 pairwise energy sums (< (log2 K + 1) float32 ulps ~ 6e-7 relative for K = 513,
#     worst case; the dumped-spectra margin does not see it).
# GUARD_TIE_BAND = 1e-6 is 3.3x the largest measured near-threshold disagreement and 4x the largest margin at a mismatch, and
# above the worst-case reduction rounding. A guard_active mismatch is excused ONLY when both sides' |m| are inside the band and the
# hop is unusable on both sides (a usable-verdict difference is never a tie); every other mismatch fails the cell. Hops inside the
# band (unusable on both sides, 0 < max|m| <= band) are counted and printed per cell. The fixed near-threshold fixtures (identical
# inputs, test_guard_unify.c / test_guard_unify.py) keep exact equality; this band applies only to the C-vs-Python cells.
GUARD_TIE_BAND = 1e-6


def guard_margin(E, O):
    """(sum|E|^2 - sum|E+O|^2) / sum|E|^2 per hop in float64: > 0 means the strict guard rule fires (when unusable)."""
    eE = np.sum(np.abs(E) ** 2, axis=1)
    eY = np.sum(np.abs(E + O) ** 2, axis=1)
    return (eE - eY) / np.maximum(eE, 1e-30)


def guard_compare(c, p, K, hop, min_reverts=0):
    """Guard-seam comparison of a C run and a Python run (dicts from load_run, both with --ctx --base)."""
    cg, cs, cb = base_records(c['base_raw'], K)
    pg, ps, pb = base_records(p['base_raw'], K)
    checked_pair('guard_active', cg, pg)
    checked_pair('post_reset_serial', cs, ps)
    checked_pair('base_spec', cb, pb)
    checked_pair('delay events', c['dly'], p['dly'])
    checked_pair('formed_hop', c['formed'], p['formed'])
    for side, r, g in (('test', c, cg), ('ref', p, pg)):
        if len(r['dly']) != len(g) or len(r['formed']) != len(g) * hop or len(r['ctx']) != len(g) * (9 + 7 * K):
            raise DumpError('%s side: base %d hops, dly %d rows, formed %d samples (hop %d), ctx %d floats' % (
                side, len(g), len(r['dly']), len(r['formed']), hop, len(r['ctx'])))
    n = len(cg)
    out = {'hops': int(n), 'hops_c': int(len(cg)), 'hops_py': int(len(pg)),
           'guard_hops_c': int(cg[:n].sum()), 'guard_hops_py': int(pg[:n].sum())}
    gm = np.nonzero(cg[:n] != pg[:n])[0]
    sm = np.nonzero(cs[:n] != ps[:n])[0]
    out['guard_mismatch_hops'] = int(len(gm))
    out['first_guard_mismatch_hop'] = int(gm[0]) if len(gm) else None
    out['serial_mismatch_hops'] = int(len(sm))
    out['first_serial_mismatch_hop'] = int(sm[0]) if len(sm) else None
    out['serial_final_c'] = int(cs[n - 1]) if n else None
    out['serial_final_py'] = int(ps[n - 1]) if n else None
    out['serial_changes_c'] = int(np.sum(cs[1:n] != cs[:n - 1]))
    out['serial_changes_py'] = int(np.sum(ps[1:n] != ps[:n - 1]))
    if len(sm):
        h = int(sm[0])
        out['first_serial_mismatch'] = {'hop': h, 'c': int(cs[h]), 'py': int(ps[h])}
    # delay events: current delay on every hop and the bad-relock reverts (hops where the reject count grows)
    cd, pd = c['dly'][:n], p['dly'][:n]
    dm = np.nonzero(cd[:, 0] != pd[:, 0])[0]
    out['delay_mismatch_hops'] = int(len(dm))
    out['first_delay_mismatch'] = ({'hop': int(dm[0]), 'c': int(cd[dm[0], 0]), 'py': int(pd[dm[0], 0])} if len(dm) else None)
    out['delay_changes_c'] = int(np.sum(cd[1:, 0] != cd[:-1, 0]))
    out['revert_hops_c'] = [int(h) for h in np.nonzero(np.diff(cd[:, 1]) > 0)[0] + 1]
    out['revert_hops_py'] = [int(h) for h in np.nonzero(np.diff(pd[:, 1]) > 0)[0] + 1]
    # why do the decisions differ? energy margin (float64, from each side's OWN dumped analysis spectra) at the mismatching hops,
    # and how often the margin is below float32 rounding scale at all (a quarter of the never-usable hops).
    Ec, Oc, uc = ctx_spectra(c['ctx'], K)
    Ep, Op, up = ctx_spectra(p['ctx'], K)
    mc, mp = guard_margin(Ec[:n], Oc[:n]), guard_margin(Ep[:n], Op[:n])
    unusable = (uc[:n] == 0) & (up[:n] == 0)
    out['unusable_hops'] = int(unusable.sum())
    amax = np.maximum(np.abs(mc), np.abs(mp))
    in_band = unusable & (amax > 0.0) & (amax <= GUARD_TIE_BAND)
    out['tie_band'] = GUARD_TIE_BAND
    out['tie_band_hops'] = int(in_band.sum())
    gm_in = gm[in_band[gm]] if len(gm) else gm
    gm_out = gm[~in_band[gm]] if len(gm) else gm
    out['guard_mismatch_in_band_hops'] = int(len(gm_in))
    out['guard_mismatch_outside_band_hops'] = int(len(gm_out))
    if len(gm_out):
        h = int(gm_out[0])
        out['first_guard_mismatch_outside_band'] = {'hop': h, 'c_guard': int(cg[h]), 'py_guard': int(pg[h]), 'usable_c': int(uc[h]),
                                                    'usable_py': int(up[h]), 'margin_c': r6(float(mc[h])),
                                                    'margin_py': r6(float(mp[h]))}
    if len(gm):
        h = int(gm[0])
        out['guard_mismatch_margin_max_abs_c'] = r6(float(np.max(np.abs(mc[gm]))))
        out['guard_mismatch_margin_max_abs_py'] = r6(float(np.max(np.abs(mp[gm]))))
        ec, ep = float(np.sum(np.abs(Ec[h]) ** 2)), float(np.sum(np.abs(Ep[h]) ** 2))
        out['first_guard_mismatch'] = {'hop': h, 'c_guard': int(cg[h]), 'py_guard': int(pg[h]), 'usable_c': int(uc[h]),
                                       'usable_py': int(up[h]), 'margin_c': r6(float(mc[h])), 'margin_py': r6(float(mp[h])),
                                       'error_energy_rel_diff_c_vs_py': r6(abs(ec - ep) / max(ep, 1e-30))}
    # where the analysis seam drifts (information): per-hop relative error of error_spec, C vs Python
    ph = np.linalg.norm(Ec[:n] - Ep[:n], axis=1) / (np.linalg.norm(Ep[:n], axis=1) + 1e-30)
    over = np.nonzero(ph > TOL['CTX_spec_rel_l2'])[0]
    out['error_spec_per_hop_rel_l2'] = {'median': r6(float(np.median(ph))), 'p99': r6(float(np.percentile(ph, 99))),
                                        'max': r6(float(np.max(ph))), 'hops_over_ctx_tol': int(len(over)),
                                        'first_hop_over_ctx_tol': int(over[0]) if len(over) else None}
    # rendering base and formed hop
    def rel(a, b):
        return float(np.linalg.norm((a - b).astype(np.complex128)) / (np.linalg.norm(b.astype(np.complex128)) + 1e-30))
    out['base_spec_rel_l2'] = r6(rel(cb[:n], pb[:n]))
    either = (cg[:n] == 1) | (pg[:n] == 1)
    out['base_spec_rel_l2_guard_hops'] = r6(rel(cb[:n][either], pb[:n][either])) if either.any() else None
    out['formed_hop_max_abs'] = r6(float(np.max(np.abs(c['formed'].astype(np.float64) - p['formed']))))
    bad = []
    if out['guard_mismatch_outside_band_hops'] or out['hops_c'] != out['hops_py']:
        bad.append('guard_active')
    if out['serial_mismatch_hops']:
        bad.append('post_reset_serial')
    if out['delay_mismatch_hops'] or out['revert_hops_c'] != out['revert_hops_py']:
        bad.append('delay_events')
    if len(out['revert_hops_py']) < min_reverts:
        bad.append('revert_not_reached(%d<%d)' % (len(out['revert_hops_py']), min_reverts))
    if worse(out['base_spec_rel_l2'], TOL['CTX_spec_rel_l2']):
        bad.append('base_spec')
    if worse(out['base_spec_rel_l2_guard_hops'], TOL['CTX_spec_rel_l2']):
        bad.append('base_spec(guard hops)')
    if worse(out['formed_hop_max_abs'], TOL['L1_raw_max_abs']):
        bad.append('formed_hop')
    out['fields_over_tolerance'] = bad
    return out


# ----------------------------------------------------------------------------------------------
# One scenario
# ----------------------------------------------------------------------------------------------
def render_bases(scn, c, p, K):
    """(C, Python) rendering base for the render-product res_gain gate: only the guard seam cells dump it."""
    if not scn.base:
        return None, None
    return base_records(c['base_raw'], K)[2], base_records(p['base_raw'], K)[2]


def eval_scenario(R, scn):
    try:
        return _eval_scenario(R, scn)
    except DumpError as e:
        return {'id': scn.id, 'hpf': scn.hpf, 'expected_red': list(scn.red), 'py_variants_run': 1,
                'status': 'FAIL (invalid dump, nothing compared: %s)' % e}


def _eval_scenario(R, scn):
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
        cc = ctx_compare(cstd['ctx'], py_best['ctx'], K, *render_bases(scn, cstd, py_best, K))
        res['context'] = cc
        ev['ctx_fields_over_tolerance'] = ctx_ok(cc)
    guard_bad = []
    if scn.base:
        gc = guard_compare(cstd, py_best, scn.fft // 2 + 1, scn.fft // 2, scn.reverts)
        res['guard'] = gc
        guard_bad = gc['fields_over_tolerance']
        ev['guard_fields_over_tolerance'] = guard_bad
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
        if audio_green and not ctx_bad and not guard_bad:
            status = 'PASS'
        elif audio_green and not ctx_bad:
            status = 'FAIL (guard seam: %s)' % ','.join(guard_bad)
        elif audio_green:
            status = 'FAIL (ctx: %s)' % ','.join(ctx_bad) + ('; guard seam: %s' % ','.join(guard_bad) if guard_bad else '')
        else:
            status = 'FAIL (layer 1 red, no marker)' + ('; guard seam: %s' % ','.join(guard_bad) if guard_bad else '')
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
    if scn.ties:
        # T-01: the hard seam gates must hold as run; every other gate must hold once the registered decisions are forced.
        K = scn.fft // 2 + 1
        forced = load_run(R.py_run(scn, 0, tag='py_ties'))
        fm = cell(cstd, forced)
        fbad = ctx_ok(ctx_compare(cstd['ctx'], forced['ctx'], K, *render_bases(scn, cstd, forced, K)))
        fg = guard_compare(cstd, forced, K, scn.fft // 2, scn.reverts)['fields_over_tolerance']
        hard_bad = [f for f in guard_bad if not (f.startswith('base_spec') or f == 'formed_hop')]
        forced_ok = l1_ok(fm) and not fbad and not fg
        ev['t01'] = {'forced_out_max_abs': fm['out']['max_abs'], 'forced_ctx_over': fbad, 'forced_guard_over': fg,
                     'hard_seam_over_as_run': hard_bad, 'forced_ok': bool(forced_ok)}
        if status == 'PASS':
            status = 'FAIL UNEXPECTED_GREEN(T-01): remove the marker'
        elif hard_bad or not forced_ok:
            status = ('FAIL T-01 shape lost (hard seam gates as run: %s; with the registered decisions forced: audio %s, ctx %s, '
                      'guard seam %s): treat as a real regression' % (hard_bad or 'green', 'green' if l1_ok(fm) else 'red',
                                                                      fbad or 'green', fg or 'green'))
        else:
            status = 'INFO(T-01): layer 1 %s; registered tie forced -> every gate green (out max_abs %s)' % (
                'within tolerance' if audio_green else 'over tolerance', fm['out']['max_abs'])
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
        (1 if 'py_guard_effect_out_max_abs' in ev else 0) + (1 if scn.ties else 0)
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
        try:
            m = cmp_metrics(P, C, 'hpf')
        except DumpError as e:
            cells.append({'id': 'hpf_fidelity.%dk' % (sr // 1000), 'sr': sr, 'py_f32_emulation_vs_c_std': {'max_abs': None, 'bitexact': False},
                          'c_ship_vs_c_std_bitexact': False, 'status': 'FAIL (invalid dump: %s)' % e})
            continue
        cells.append({'id': 'hpf_fidelity.%dk' % (sr // 1000), 'sr': sr, 'cutoff_hz': 80.0,
                      'py_f32_emulation_vs_c_std': {'max_abs': m['max_abs'], 'bitexact': m['bitexact']},
                      'c_ship_vs_c_std_bitexact': bool(np.array_equal(C, S)),
                      'status': 'PASS' if m['bitexact'] else 'FAIL (the float32 HPF emulation is not bit-exact to the C filter: '
                                                              'Layer-1 "HPF type: f32" labels would be wrong)'})
    return cells


def s03_cell(a):
    """Regenerate aec_balanced_config.h (and the test-only reference tables test/aec_ref_tables.h) from a scratch copy of
    the Python tree and diff them against the checked-in files -- the two must always be identical (S-03, fixed 2026-10-20: the header is
    regenerated alongside any change to the balanced config or the generator; make test-header-fresh
    covers this same check on an ordinary build without needing the Python<->C parity harness)."""
    root = a.aec_root
    d = os.path.join(a.work, 's03')
    if os.path.exists(d):
        shutil.rmtree(d)
    ign = shutil.ignore_patterns('__pycache__', '.pytest_cache', 'tests', 'output_*', '*.pyc')
    shutil.copytree(os.path.join(root, 'python'), os.path.join(d, 'python'), ignore=ign)
    os.makedirs(os.path.join(d, 'c_impl', 'include', 'internal'))
    gen = os.path.join(d, 'python', 'diag', 'gen_aec_balanced_config_h.py')
    p = run([sys.executable, gen], log=os.path.join(d, 'gen.log'), check=False)
    new_p = os.path.join(d, 'c_impl', 'include', 'internal', 'aec_balanced_config.h')
    old_p = os.path.join(root, 'c_impl', 'include', 'internal', 'aec_balanced_config.h')
    cell_ = {'id': 'static.s03_balanced_config_regen', 'generator': 'python/diag/gen_aec_balanced_config_h.py',
             'header': 'c_impl/include/internal/aec_balanced_config.h'}
    if p.returncode != 0 or not os.path.exists(new_p):
        cell_.update({'status': 'FAIL (generator did not run; see work/s03/gen.log)', 'rc': p.returncode})
        return cell_
    old = open(old_p).read().split('\n')
    new = open(new_p).read().split('\n')
    ref_new = os.path.join(d, 'c_impl', 'test', 'aec_ref_tables.h')
    ref_old = os.path.join(root, 'c_impl', 'test', 'aec_ref_tables.h')
    ref_stale = not (os.path.exists(ref_new) and os.path.exists(ref_old) and open(ref_new).read() == open(ref_old).read())

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
    if diff_names or other_lines or ref_stale:
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
    # source wavs (the recording and, for the guard cells, the blind-set clips)
    clip_wavs = sorted(set(rel for s in scn_list if s.kind in CLIPS for rel in CLIPS[s.kind][:2]))
    for rel in (REC_MIC, REC_REF) + tuple(clip_wavs):
        p = os.path.join(a.aec_root, rel)
        if not os.path.exists(p):
            bad.append('MISSING source wav %s%s' % (rel, ' (blind-set dataset: see wav/download_blind_test.py)' if rel in clip_wavs else ''))
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
                    '#   (rec8/rec48 = scipy resample_poly of the 16 kHz recording; echo/speechdt/dtb = seeded scenes.py;\n'
                    '#   <clip stem> = mic/far-end of a blind-set clip, truncated to the shorter channel).\n'
                    '# A mismatch means the audio changed (numpy/scipy drift or an edit): the baseline moved, so the run refuses.\n'
                    '# Regenerate (maintainers only): pyc_parity_aec.sh --write-pins\n')
            for rel in (REC_MIC, REC_REF) + tuple(clip_wavs):
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
        if 'layer1' not in r:
            print('%-31s %s' % (r['id'], r['status']))
            continue
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
                  'n01_c_builds_agree', 't01'):
            if k in ev:
                bits.append('%s=%s' % (k, ev[k]))
        if 'python_self_distance_1ulp_out_max_abs' in r:
            bits.append('py_1ulp_self_dist=%s' % r['python_self_distance_1ulp_out_max_abs'])
        for k in sorted(l2):
            if k.startswith('python_self_distance_at_'):
                bits.append('py_self_dist@%s=%s' % (k.split('_at_')[1].split('_')[0], l2[k]))
        if bits:
            print('%-31s     %s' % ('', '  '.join(bits)))
        g = r.get('guard')
        if g:
            print('%-31s     guard seam: hops=%d guard_hops c/py=%d/%d guard_mismatch=%d (first %s; inside tie band %d, OUTSIDE %d) '
                  'tie-band hops=%d (|m|<=%g) serial_mismatch=%d (first %s; changes c/py=%d/%d) base_rel_l2=%s (guard hops %s) '
                  'formed_max_abs=%s' % (
                      '', g['hops'], g['guard_hops_c'], g['guard_hops_py'], g['guard_mismatch_hops'], g['first_guard_mismatch_hop'],
                      g['guard_mismatch_in_band_hops'], g['guard_mismatch_outside_band_hops'], g['tie_band_hops'], g['tie_band'],
                      g['serial_mismatch_hops'], g['first_serial_mismatch_hop'], g['serial_changes_c'], g['serial_changes_py'],
                      g['base_spec_rel_l2'], g['base_spec_rel_l2_guard_hops'], g['formed_hop_max_abs']))
            if 'first_guard_mismatch_outside_band' in g:
                print('%-31s     first guard mismatch OUTSIDE the tie band: %s' % ('', g['first_guard_mismatch_outside_band']))
            print('%-31s     delay events: changes c=%d, mismatching hops %d (first %s), reverts at hops c=%s py=%s' % (
                '', g['delay_changes_c'], g['delay_mismatch_hops'], g['first_delay_mismatch'], g['revert_hops_c'],
                g['revert_hops_py']))
            if 'first_guard_mismatch' in g:
                m = g['first_guard_mismatch']
                print('%-31s     first guard divergence: hop %d c=%d py=%d usable c/py=%d/%d; energy margin (E-Y)/E c=%s py=%s '
                      '(max over all divergent hops |c|=%s |py|=%s; error-energy C-vs-Py rel diff there %s); unusable hops: %d' % (
                          '', m['hop'], m['c_guard'], m['py_guard'], m['usable_c'], m['usable_py'], m['margin_c'], m['margin_py'],
                          g['guard_mismatch_margin_max_abs_c'], g['guard_mismatch_margin_max_abs_py'],
                          m['error_energy_rel_diff_c_vs_py'], g['unusable_hops']))
            e = g['error_spec_per_hop_rel_l2']
            print('%-31s     error_spec per-hop rel_l2 C-vs-Py: median %s p99 %s max %s; hops over CTX_spec_rel_l2: %d (first %s)' % (
                '', e['median'], e['p99'], e['max'], e['hops_over_ctx_tol'], e['first_hop_over_ctx_tol']))
            if 'first_serial_mismatch' in g:
                m = g['first_serial_mismatch']
                print('%-31s     first serial divergence: hop %d c=%d py=%d' % ('', m['hop'], m['c'], m['py']))
        cx = r.get('context')
        if cx:
            g = cx['res_gain']
            rp = g.get('render')
            print('%-31s     ctx: res_gain wrel_l2=%s (gate CTX_gain_wrel_l2 %g) loud-bin max_abs=%s (gate %g, %d bins within %g dB '
                  'of the hop peak) render product per-hop rel_l2 max=%s p99=%s (gate %g, %s of %s hops within %g dB; whole %s) '
                  'max_abs=%s (printed only); over tolerance: %s' % (
                      '', g['wrel_l2'], TOL['CTX_gain_wrel_l2'], g['loud_max_abs'], TOL['CTX_gain_loud_max_abs'], g['loud_bins'],
                      LOUD_BIN_DB, rp['per_hop_max'] if rp else 'n/a', rp['per_hop_p99'] if rp else 'n/a',
                      TOL['CTX_render_hop_rel_l2'], rp['hops_gated'] if rp else '-', rp['hops'] if rp else '-', RENDER_HOP_FLOOR_DB,
                      rp['whole_rel_l2'] if rp else 'n/a', g['max_abs'], ','.join(ev.get('ctx_fields_over_tolerance', [])) or 'none'))
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
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--quick', action='store_true', help='older narrow subset (no aggressive scene, weak observer scene)')
    grp.add_argument('--full', action='store_true', help='every scenario (default: the lean suite)')
    ap.add_argument('--only', default=None, help='substring filter on scenario ids over the full set (comma separated = any)')
    ap.add_argument('--write-pins', action='store_true')
    ap.add_argument('--skip-static', action='store_true')
    a = ap.parse_args()
    a.aec_root = os.path.abspath(a.aec_root)
    a.work = os.path.abspath(a.work)
    os.makedirs(a.work, exist_ok=True)

    scn_list = build_scenarios()
    # --only and --write-pins address scenes outside the lean suite, so they select from the full set.
    suite = 'quick' if a.quick else 'full' if (a.full or a.only or a.write_pins) else 'lean'
    if suite == 'quick':
        scn_list = [s for s in scn_list if s.quick]
    elif suite == 'lean':
        scn_list = [s for s in scn_list if s.lean]
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
        'layer2_inconclusive_chaotic': sum(1 for r in results if r.get('layer2', {}).get('status') == 'INCONCLUSIVE_CHAOTIC'),
        'layer2_red': sum(1 for r in results if r.get('layer2', {}).get('status') == 'RED'),
        'python_variant_runs_beyond_v0': sum(r['py_variants_run'] - 1 for r in results),
        'failed_ids': fails,
    }
    doc = {'suite': suite, 'tolerances': TOL, 'env': env, 'expected_red_registry': EXPECTED_RED_INFO,
           'inputs': infos, 'hpf_fidelity': hpfc, 'scenarios': results, 'static': statics, 'summary': summ}
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write('\n')
    print()
    print('summary: %s' % json.dumps(summ, sort_keys=True))
    print('registry of known gaps and observations (S-ids are the appendix A ids of docs/c_python_function_map.md; marker => allowed to be red, must stay red):')
    for k in sorted(EXPECTED_RED_INFO):
        print('  %s(%s): %s' % ('INFO' if EXPECTED_RED_INFO[k].startswith('INFO') else 'EXPECTED_RED', k, EXPECTED_RED_INFO[k]))
    print('RESULT: %s' % ('FAIL' if fails else 'PASS'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
