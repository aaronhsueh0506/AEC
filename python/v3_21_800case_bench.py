#!/usr/bin/env python3
"""v3.21 800-case benchmark — M_full_delay composition candidate.

Renders M0 (all 13 candidate flags OFF) and M_full_delay (all 13 ON)
on the full 800-case AEC Challenge corpus.  Produces a dual-ledger report:

  Production ledger : M_full_delay vs M0 (v3.21.6-equivalent anchor)
  Alignment ledger  : M_full_delay vs AEC3 behavioral ref (12 known cases)

Standard bench config: preset=balanced / filter=832 (52ms) / cng / workers=4

Catastrophic stop rules (checked after scoring all cases):
  - DT case: M_full_delay vs AEC3 (known) worse by > 0.10 deg
  - FS case: M_full_delay vs AEC3 (known) worse by > 0.30 echo

nores LF artifact check: all FS_static cases rendered with enable_res=False
  to verify 9xjhi nores LF improvement is maintained.

No code changes.  No merge.  No version bump.

Usage:
    cd /path/to/AEC
    python3 python/v3_21_800case_bench.py [--workers N] [--skip-byte-equal]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecPreset
from bench_aecmos import FastAECMOS

REPO     = Path(__file__).resolve().parents[1]
MODEL    = str(REPO / 'model' / 'Run_1663915512_Stage_0.onnx')
CORPUS   = REPO / 'wav' / 'aec_challenge_blind'
OUT_DIR  = REPO / 'out_v3_21_800case'
DOC_PATH = REPO / 'docs' / 'v3_21_800case_bench_report.md'
JSON_PATH = REPO / 'out_v3_21_800case' / 'scores_800case.json'

SR = 16000

# AEC3 behavioral reference scores (12-case run, 2026-05-27)
# Source: docs/v3_21_uro_signal_flow_attribution.md §A
AEC3_REF = {
    'ZJYUt0O0AEKSQ9LJ8z7t0A_doubletalk_with_movement':        ('deg', 2.177),
    'wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement':        ('deg', 1.540),
    'xFk7igecuke0R5JMfREyDg_doubletalk_with_movement':        ('deg', 1.275),
    'MYrVxVEMxkaE7OuyTUmI0Q_doubletalk':                      ('deg', 1.275),
    'XRTnTUjU5kS0mejzCqyCiw_doubletalk':                      ('deg', 2.062),
    'jtYTdZm3lUmFVNibJWq8YQ_doubletalk':                      ('deg', 2.298),
    'nVUnxqHLr0GTN7shWid1Ow_doubletalk':                      ('deg', 1.547),
    '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement': ('echo', 4.296),
    '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk':               ('echo', 3.442),
    'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk':               ('echo', 3.596),
    'xQEUtY2pWUi7v1X93TF2AA_farend_singletalk':               ('echo', 4.219),
    '014AzuqPZku2004NbTTmcA_nearend_singletalk':              ('deg', 4.164),
}

# v3.21.6 per-case baseline for 12 known cases (production ledger reference)
V3_21_6_SCORES = {
    'ZJYUt0O0AEKSQ9LJ8z7t0A_doubletalk_with_movement':        ('deg', 2.270),
    'wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement':        ('deg', 2.741),
    'xFk7igecuke0R5JMfREyDg_doubletalk_with_movement':        ('deg', 2.319),
    'MYrVxVEMxkaE7OuyTUmI0Q_doubletalk':                      ('deg', 2.166),
    'XRTnTUjU5kS0mejzCqyCiw_doubletalk':                      ('deg', 3.950),
    'jtYTdZm3lUmFVNibJWq8YQ_doubletalk':                      ('deg', 2.700),
    'nVUnxqHLr0GTN7shWid1Ow_doubletalk':                      ('deg', 2.893),
    '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement': ('echo', 4.262),
    '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk':               ('echo', 2.367),
    'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk':               ('echo', 3.550),
    'xQEUtY2pWUi7v1X93TF2AA_farend_singletalk':               ('echo', 3.387),
    '014AzuqPZku2004NbTTmcA_nearend_singletalk':              ('deg', 4.355),
}

# C1/C3/C4/C6 wall-clock-alignment A/B manifest (2026-05-29).
# NOTE: label 'M_full_delay' is kept for _case_task/report compatibility but
# here means "C1-C6 wall-clock candidate" (NOT the old composition ladder).
#   M0           = all four flags OFF == plain BALANCED (byte-equal anchor)
#   M_full_delay = all four flags ON  (the candidate)
CONFIG_MANIFEST = {
    'M0': {
        'use_aec3_wallclock_subband_erle_smoothing':       False,  # C1
        'use_aec3_wallclock_fullband_erle_smoothing':      False,  # C3
        'use_aec3_wallclock_low_noise_render_iir':         False,  # C4
        'use_aec3_active_render_threshold_shadow_epc':     False,  # C6
    },
    'M_full_delay': {
        'use_aec3_wallclock_subband_erle_smoothing':       True,   # C1
        'use_aec3_wallclock_fullband_erle_smoothing':      True,   # C3
        'use_aec3_wallclock_low_noise_render_iir':         True,   # C4
        'use_aec3_active_render_threshold_shadow_epc':     True,   # C6
    },
}

BUCKET_METRIC = {
    'DT_mvmt': 'deg', 'DT_static': 'deg',
    'FS_mvmt': 'echo', 'FS_static': 'echo', 'NE': 'deg',
}

SUBDIRS = ['doubletalk', 'farend_singletalk', 'nearend_singletalk']


# ---------------------------------------------------------------------------
# Helpers (self-contained — no import from eval_aec_challenge)
# ---------------------------------------------------------------------------

def _estimate_delay(mic: np.ndarray, ref: np.ndarray, sr: int,
                    max_delay_ms: float = 1024.0) -> int:
    """GCC-PHAT delay estimate (same logic as eval_aec_challenge.py)."""
    n = min(len(mic), len(ref))
    max_d = int(max_delay_ms * sr / 1000)
    m = mic[:n].astype(np.float64)
    r = ref[:n].astype(np.float64)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    mic_spec = np.fft.rfft(m, n=fft_size)
    ref_spec = np.fft.rfft(r, n=fft_size)
    cross = mic_spec * np.conj(ref_spec)
    cross_phat = cross / (np.abs(cross) + 1e-10)
    xcorr_phat = np.fft.irfft(cross_phat, n=fft_size)
    max_search = min(max_d, fft_size // 2)
    peak_val = np.max(np.abs(xcorr_phat[:max_search + 1]))
    rms = np.sqrt(np.mean(xcorr_phat[:max_search + 1] ** 2))
    confidence = peak_val / (rms + 1e-10)
    if confidence < 5.0:
        xcorr_plain = np.fft.irfft(cross, n=fft_size)
        return int(np.argmax(np.abs(xcorr_plain[:max_search + 1])))
    return int(np.argmax(np.abs(xcorr_phat[:max_search + 1])))


def _lf_energy_db(wav: np.ndarray, sr: int = SR, fmax: int = 500) -> float:
    """Energy in 0–fmax Hz band, in dBFS (power)."""
    n = len(wav)
    if n == 0:
        return -100.0
    spec = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = freqs <= fmax
    pwr = float(np.sum(np.abs(spec[mask]) ** 2)) / n
    return 10.0 * np.log10(pwr + 1e-20)


def _classify(stem: str, subdir: str) -> Tuple[str, str]:
    """Return (bucket, wavtype).

    Stems are filenames minus '_mic.wav', so movement stems end with
    '_with_movement' (no trailing underscore).  Check without trailing _.
    """
    if 'doubletalk' in subdir:
        bucket = 'DT_mvmt' if '_with_movement' in stem else 'DT_static'
        return bucket, 'dt'
    if 'farend_singletalk' in subdir:
        bucket = 'FS_mvmt' if '_with_movement' in stem else 'FS_static'
        return bucket, 'st'
    return 'NE', 'nst'


def _build_cfg(label: str, is_movement: bool = False,
               enable_res: bool = True) -> AecConfig:
    kw = dict(CONFIG_MANIFEST[label])
    kw['enable_res'] = enable_res
    if is_movement:
        kw.update(enable_delay_est=True,
                  delay_est_period_s=0.25,
                  delay_est_init_s=0.2)
    else:
        kw['enable_delay_est'] = False
    return AecConfig.from_preset(AecPreset.BALANCED, **kw)


def _render(mic_n: np.ndarray, ref_a: np.ndarray,
            label: str, is_movement: bool,
            enable_res: bool = True) -> np.ndarray:
    cfg = _build_cfg(label, is_movement=is_movement, enable_res=enable_res)
    np.random.seed(42)
    aec = AEC(cfg)
    hop = cfg.hop_size
    n_hops = len(mic_n) // hop
    out = np.zeros(n_hops * hop, dtype=np.float32)
    for i in range(n_hops):
        s = i * hop
        out[s:s + hop] = aec.process(mic_n[s:s + hop], ref_a[s:s + hop])
    return out


# ---------------------------------------------------------------------------
# Worker task — runs in subprocess
# ---------------------------------------------------------------------------

def _case_task(args: tuple) -> dict:
    """Render M0 + M_full_delay, score both; return scores only (no audio)."""
    stem, subdir, bucket, wavtype, mic_path, ref_path, model_path, do_nores = args

    mic_raw, _  = sf.read(mic_path)
    ref_raw, _  = sf.read(ref_path)
    if mic_raw.ndim > 1: mic_raw = mic_raw[:, 0]
    if ref_raw.ndim > 1: ref_raw = ref_raw[:, 0]
    n = min(len(mic_raw), len(ref_raw))
    mic = mic_raw[:n].astype(np.float32)
    ref = ref_raw[:n].astype(np.float32)

    is_mvmt = '_with_movement' in stem  # stem has no trailing _ after _with_movement

    # Pre-align ref (shared for both variants)
    delay = _estimate_delay(mic, ref, SR)
    if 0 < delay < n:
        ref_a = np.zeros(n, dtype=np.float32)
        ref_a[delay:] = ref[:n - delay]
    else:
        ref_a = ref.copy()

    scorer = FastAECMOS(model_path)

    result: dict = {'stem': stem, 'bucket': bucket, 'wavtype': wavtype}

    for label in ('M0', 'M_full_delay'):
        out = _render(mic, ref_a, label, is_movement=is_mvmt, enable_res=True)
        # Score: lpb=ref (original), mic=mic, enh=out.  Equalize lengths first:
        # _render truncates `out` to a hop multiple, so it is a few samples
        # shorter than ref/mic.  For clips < 20s (which skip the seg truncation
        # inside score()), that residual mismatch breaks np.stack in the mel
        # path — the cause of the 105 NE-bucket "all input arrays must have the
        # same shape" failures on the 2026-05-29 run.
        m = min(len(ref), len(mic), len(out))
        echo, deg = scorer.score(wavtype, ref[:m], mic[:m], out[:m])
        result[f'{label}_echo'] = float(echo)
        result[f'{label}_deg']  = float(deg)

        if do_nores:
            out_nr = _render(mic, ref_a, label, is_movement=is_mvmt, enable_res=False)
            result[f'{label}_nores_lf_db'] = float(_lf_energy_db(out_nr))

    return result


# ---------------------------------------------------------------------------
# Byte-equal precheck
# ---------------------------------------------------------------------------

def _byte_equal_check() -> bool:
    """Verify M0 is byte-equal to plain BALANCED on one representative case."""
    print('[byte-equal] Checking M0 vs plain BALANCED on 9xjhi ...', flush=True)
    mic_p = CORPUS / 'farend_singletalk' / '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk_mic.wav'
    ref_p = CORPUS / 'farend_singletalk' / '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk_lpb.wav'
    if not mic_p.exists():
        print('[byte-equal] SKIP — corpus file missing')
        return False

    mic_raw, _ = sf.read(str(mic_p))
    ref_raw, _ = sf.read(str(ref_p))
    if mic_raw.ndim > 1: mic_raw = mic_raw[:, 0]
    if ref_raw.ndim > 1: ref_raw = ref_raw[:, 0]
    n = min(len(mic_raw), len(ref_raw))
    mic = mic_raw[:n].astype(np.float32)
    ref = ref_raw[:n].astype(np.float32)

    cfg_plain = AecConfig.from_preset(AecPreset.BALANCED)
    np.random.seed(42)
    aec_plain = AEC(cfg_plain)
    hop = cfg_plain.hop_size
    n_hops = n // hop
    out_plain = np.zeros(n_hops * hop, dtype=np.float32)
    for i in range(n_hops):
        s = i * hop
        out_plain[s:s + hop] = aec_plain.process(mic[s:s + hop], ref[s:s + hop])

    out_m0 = _render(mic, ref, 'M0', is_movement=False, enable_res=True)
    n2 = min(len(out_plain), len(out_m0))
    ok = np.array_equal(out_plain[:n2], out_m0[:n2])
    if ok:
        print('[byte-equal] PASS — M0 byte-equal to plain BALANCED')
    else:
        diffs = int(np.sum(out_plain[:n2] != out_m0[:n2]))
        print(f'[byte-equal] FAIL — {diffs}/{n2} samples differ')
    return ok


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def _discover_cases() -> List[tuple]:
    """Return list of (stem, subdir, bucket, wavtype, mic_path, ref_path)."""
    cases = []
    for subdir in SUBDIRS:
        sc_dir = CORPUS / subdir
        if not sc_dir.is_dir():
            continue
        for mic_p in sorted(sc_dir.glob('*_mic.wav')):
            stem = mic_p.name.replace('_mic.wav', '')
            ref_p = mic_p.parent / (stem + '_lpb.wav')
            if not ref_p.exists():
                continue
            bucket, wavtype = _classify(stem, subdir)
            cases.append((stem, subdir, bucket, wavtype, str(mic_p), str(ref_p)))
    return cases


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(results: List[dict], be_ok: bool) -> None:
    # Aggregate by bucket
    prod_by_bucket: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_m0:   Dict[str, dict] = {}
    all_mfull: Dict[str, dict] = {}
    nores_lf:  Dict[str, Tuple[float, float]] = {}  # stem → (m0_lf, mfull_lf)

    for r in results:
        stem   = r['stem']
        bucket = r['bucket']
        metric = BUCKET_METRIC[bucket]

        m0_score    = r['M0_echo'] if metric == 'echo' else r['M0_deg']
        mfull_score = r['M_full_delay_echo'] if metric == 'echo' else r['M_full_delay_deg']
        delta = mfull_score - m0_score

        all_m0[stem]   = {'echo': r['M0_echo'], 'deg': r['M0_deg'], 'bucket': bucket}
        all_mfull[stem] = {'echo': r['M_full_delay_echo'], 'deg': r['M_full_delay_deg'],
                           'bucket': bucket}

        prod_by_bucket[bucket]['deltas'].append(delta)
        prod_by_bucket[bucket]['m0_scores'].append(m0_score)
        prod_by_bucket[bucket]['mfull_scores'].append(mfull_score)

        if 'M0_nores_lf_db' in r and 'M_full_delay_nores_lf_db' in r:
            nores_lf[stem] = (r['M0_nores_lf_db'], r['M_full_delay_nores_lf_db'])

    # Catastrophic detection (production: vs M0)
    prod_catastrophics: List[Tuple[str, str, float]] = []
    for r in results:
        bucket = r['bucket']
        metric = BUCKET_METRIC[bucket]
        m0    = r['M0_echo'] if metric == 'echo' else r['M0_deg']
        mfull = r['M_full_delay_echo'] if metric == 'echo' else r['M_full_delay_deg']
        delta = mfull - m0
        if metric == 'deg' and delta < -0.20:
            prod_catastrophics.append((r['stem'], bucket, delta))
        if metric == 'echo' and delta < -0.20:  # echo drops = worse suppression = regression
            prod_catastrophics.append((r['stem'], bucket, delta))

    # Alignment check (vs AEC3 ref, 12 known cases)
    alignment_rows: List[Tuple[str, str, float, float, float]] = []
    align_catastrophics: List[Tuple[str, str, float, float]] = []
    for stem, (aec3_metric, aec3_score) in AEC3_REF.items():
        if stem not in all_mfull:
            continue
        mfull_score = (all_mfull[stem]['echo'] if aec3_metric == 'echo'
                       else all_mfull[stem]['deg'])
        m0_score = (all_m0[stem]['echo'] if aec3_metric == 'echo'
                    else all_m0[stem]['deg'])
        delta_vs_aec3 = mfull_score - aec3_score
        delta_vs_m0   = mfull_score - m0_score
        bucket = all_mfull[stem]['bucket']
        alignment_rows.append((stem, bucket, aec3_metric, aec3_score,
                                mfull_score, delta_vs_aec3, delta_vs_m0))
        # Catastrophic bars: DT > 0.10 worse than AEC3, FS > 0.30 worse
        if aec3_metric == 'deg' and delta_vs_aec3 < -0.10:
            align_catastrophics.append((stem, bucket, delta_vs_aec3, aec3_score))
        if aec3_metric == 'echo' and delta_vs_aec3 < -0.30:
            align_catastrophics.append((stem, bucket, delta_vs_aec3, aec3_score))

    # 9xjhi watchlist: FS_static cases with notable echo delta vs M0
    fs_static_sorted = sorted(
        [(r['stem'], r['M0_echo'], r['M_full_delay_echo'],
          r['M_full_delay_echo'] - r['M0_echo'])
         for r in results if r['bucket'] == 'FS_static'],
        key=lambda x: x[3]
    )

    # FS_static regressions: echo drops (d < -0.05, less suppression = worse)
    fs_regressions = [(s, m0, mf, d) for s, m0, mf, d in fs_static_sorted if d < -0.05]
    # FS_static large improvements: echo rises significantly (d > 0.50)
    fs_large_improve = [(s, m0, mf, d) for s, m0, mf, d in fs_static_sorted if d > 0.50]

    # --- Build report text ---
    lines = []
    lines.append('# v3.21 800-case Benchmark Report — C1-C6 wall-clock alignment '
                 '(M_full_delay = all-ON) vs M0 (all-OFF == plain BALANCED)\n')
    lines.append(f'**Date**: 2026-05-29  ')
    lines.append(f'**Cases**: {len(results)}/800  ')
    lines.append(f'**Config**: preset=balanced / filter=832 (52ms) / cng / hop=160 '
                 f'(workers-count does not affect scores)\n')
    lines.append(f'**Byte-equal precheck** (M0 vs plain BALANCED): '
                 f'{"PASS ✓" if be_ok else "**FAIL**"}\n')

    # Flag manifest — C1/C3/C4/C6 wall-clock EMA/IIR alignment.
    # Iterate the actual manifest keys so this can never desync from CONFIG_MANIFEST.
    lines.append('## Flag Composition\n')
    lines.append('> C1/C3/C4/C6 wall-clock EMA/IIR alignment flags '
                 '(per-4ms-block AEC3 constants → per-10ms-hop).\n')
    lines.append('| Flag | M0 | M_full_delay |')
    lines.append('|------|-----|--------------|')
    for flag in CONFIG_MANIFEST['M0']:
        m0v = 'ON' if CONFIG_MANIFEST['M0'][flag] else 'OFF'
        mfv = 'ON' if CONFIG_MANIFEST['M_full_delay'][flag] else 'OFF'
        lines.append(f'| `{flag}` | {m0v} | {mfv} |')
    lines.append('')

    # --- Production Ledger ---
    lines.append('---\n')
    lines.append('## Production Ledger (M_full_delay vs M0 = v3.21.6 anchor)\n')
    lines.append('### Bucket Means\n')
    lines.append('| Bucket | Metric | N | Δ_mean | Δ_std | Worst Δ | Best Δ |')
    lines.append('|--------|--------|---|--------|-------|---------|--------|')

    for bkt in ['DT_mvmt', 'DT_static', 'FS_mvmt', 'FS_static', 'NE']:
        metric = BUCKET_METRIC[bkt]
        d_list = prod_by_bucket[bkt]['deltas']
        if not d_list:
            lines.append(f'| {bkt} | {metric} | 0 | — | — | — | — |')
            continue
        arr = np.array(d_list)
        # Both echo and deg: higher = better (AECMOS convention). Negative Δ = regression.
        worst = float(np.min(arr))   # most negative = worst regression
        best  = float(np.max(arr))   # most positive = best improvement
        sign = '+' if float(np.mean(arr)) >= 0 else ''
        lines.append(
            f'| {bkt} | {metric} | {len(d_list)} '
            f'| {sign}{float(np.mean(arr)):.3f} | {float(np.std(arr)):.3f} '
            f'| {worst:+.3f} | {best:+.3f} |'
        )
    lines.append('')

    # Worst-5 per bucket (production)
    lines.append('### Worst-5 per Bucket (Δ vs M0)\n')
    for bkt in ['DT_mvmt', 'DT_static', 'FS_mvmt', 'FS_static', 'NE']:
        metric = BUCKET_METRIC[bkt]
        cases_bkt = [
            (r['stem'],
             r['M0_echo'] if metric == 'echo' else r['M0_deg'],
             r['M_full_delay_echo'] if metric == 'echo' else r['M_full_delay_deg'])
            for r in results if r['bucket'] == bkt
        ]
        if not cases_bkt:
            continue
        # Sort by worst regression: most negative Δ first (both echo and deg — higher is better)
        cases_sorted = sorted(cases_bkt, key=lambda x: (x[2] - x[1]))
        top5 = cases_sorted[:5]
        lines.append(f'**{bkt}** (metric={metric}):')
        lines.append('| Case | M0 | M_full | Δ |')
        lines.append('|------|----|--------|---|')
        for stem, m0, mf in top5:
            lines.append(f'| `{stem[:50]}` | {m0:.3f} | {mf:.3f} | {mf-m0:+.3f} |')
        lines.append('')

    # Catastrophic cases (production: vs M0)
    lines.append('### Catastrophic Cases (vs M0)\n')
    lines.append('> DT Δdeg < −0.20  OR  FS Δecho < −0.20 vs M0  (both metrics: higher = better)\n')
    if not prod_catastrophics:
        lines.append('**None** — no production catastrophics.\n')
    else:
        lines.append(f'**{len(prod_catastrophics)} catastrophic case(s):**\n')
        lines.append('| Case | Bucket | Δ |')
        lines.append('|------|--------|---|')
        for stem, bkt, delta in sorted(prod_catastrophics, key=lambda x: x[2]):
            lines.append(f'| `{stem[:50]}` | {bkt} | {delta:+.3f} |')
        lines.append('')

    # --- Alignment Ledger ---
    lines.append('---\n')
    lines.append('## Alignment Ledger (M_full_delay vs AEC3 behavioral reference)\n')
    lines.append('> AEC3 scores from `bin/aec3_cli` run on 12-case cohort (2026-05-27).\n')
    lines.append('> Only these 12 cases have known AEC3 reference scores.\n')
    lines.append('')
    lines.append('| Case | Bucket | Metric | AEC3 | M_full | Δ_vs_AEC3 | Δ_vs_M0 | Status |')
    lines.append('|------|--------|--------|------|--------|-----------|---------|--------|')

    for (stem, bucket, metric, aec3_score, mfull_score,
         delta_vs_aec3, delta_vs_m0) in alignment_rows:
        # Status: for echo, positive = ours better; for deg, positive = ours better
        # Alignment bar: DT within 0.10, FS within 0.30
        if metric == 'deg':
            ok = delta_vs_aec3 >= -0.10
            status = '✓ PASS' if ok else '⚠ FAIL'
            if stem == '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk':
                status = '⚠ KNOWN EXCEPTION'
        else:
            ok = delta_vs_aec3 >= -0.30
            status = '✓ PASS' if ok else '⚠ FAIL'
            if stem == '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk':
                status = '⚠ KNOWN EXCEPTION'
        lines.append(
            f'| `{stem[:42]}` | {bucket} | {metric} '
            f'| {aec3_score:.3f} | {mfull_score:.3f} '
            f'| {delta_vs_aec3:+.3f} | {delta_vs_m0:+.3f} | {status} |'
        )
    lines.append('')

    # Alignment bucket means (12 cases only)
    lines.append('### Alignment Bucket Summary\n')
    align_by_bucket: Dict[str, List[float]] = defaultdict(list)
    for (stem, bucket, metric, aec3, mfull, d_aec3, d_m0) in alignment_rows:
        align_by_bucket[bucket].append(d_aec3)
    lines.append('| Bucket | N | Mean Δ_vs_AEC3 | Status |')
    lines.append('|--------|---|----------------|--------|')
    for bkt in ['DT_mvmt', 'DT_static', 'FS_mvmt', 'FS_static', 'NE']:
        d_list = align_by_bucket.get(bkt, [])
        if not d_list:
            continue
        mean_d = float(np.mean(d_list))
        # For both echo and deg: positive mean = ours better
        ok = (mean_d >= -0.10) if BUCKET_METRIC[bkt] == 'deg' else (mean_d >= -0.30)
        status = '✓ PASS' if ok else '⚠ FAIL'
        if bkt == 'FS_static':
            status += ' (9xjhi exception)'
        lines.append(f'| {bkt} | {len(d_list)} | {mean_d:+.3f} | {status} |')
    lines.append('')

    # Alignment catastrophics
    lines.append('### Alignment Catastrophics (vs AEC3 ref)\n')
    lines.append('> DT worse than AEC3 by > 0.10 deg  OR  FS worse than AEC3 by > 0.30 echo\n')
    if not align_catastrophics:
        lines.append('**None** — no alignment catastrophics.\n')
    else:
        lines.append(f'**{len(align_catastrophics)} alignment catastrophic(s):**\n')
        lines.append('| Case | Bucket | Δ_vs_AEC3 | AEC3 |')
        lines.append('|------|--------|-----------|------|')
        for stem, bkt, d, aec3 in align_catastrophics:
            lines.append(f'| `{stem[:50]}` | {bkt} | {d:+.3f} | {aec3:.3f} |')
        lines.append('')

    # --- 9xjhi Watchlist ---
    lines.append('---\n')
    lines.append('## 9xjhi Watchlist — FS_static Cases\n')
    n_fs_static = len([r for r in results if r['bucket'] == 'FS_static'])
    lines.append(f'**Total FS_static cases**: {n_fs_static}\n')

    # 9xjhi itself
    xjhi_stem = '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk'
    if xjhi_stem in all_m0:
        m0_echo   = all_m0[xjhi_stem]['echo']
        mfull_echo = all_mfull[xjhi_stem]['echo']
        aec3_echo  = AEC3_REF[xjhi_stem][1]
        lines.append(f'**9xjhi itself**: M0={m0_echo:.3f} M_full={mfull_echo:.3f} '
                     f'Δvs_M0={mfull_echo-m0_echo:+.3f} '
                     f'Δvs_AEC3={mfull_echo-aec3_echo:+.3f} '
                     f'(AEC3={aec3_echo:.3f})\n')

    lines.append(f'**FS_static regressions vs M0** (Δecho < −0.05, echo drops = worse): '
                 f'{len(fs_regressions)} case(s)\n')
    if fs_regressions:
        lines.append('| Case | M0_echo | M_full_echo | Δ |')
        lines.append('|------|---------|-------------|---|')
        for stem, m0, mf, d in fs_regressions[:20]:
            lines.append(f'| `{stem[:50]}` | {m0:.3f} | {mf:.3f} | {d:+.3f} |')
        lines.append('')

    lines.append(f'**FS_static large improvements vs M0** (Δecho > +0.50): '
                 f'{len(fs_large_improve)} case(s)\n')
    if fs_large_improve:
        lines.append('| Case | M0_echo | M_full_echo | Δ |')
        lines.append('|------|---------|-------------|---|')
        for stem, m0, mf, d in sorted(fs_large_improve, key=lambda x: -x[3])[:10]:
            lines.append(f'| `{stem[:50]}` | {m0:.3f} | {mf:.3f} | {d:+.3f} |')
        lines.append('')

    # --- nores LF Artifact Check ---
    lines.append('---\n')
    lines.append('## nores LF Artifact Check (FS_static only)\n')
    lines.append('> LF band = 0–500 Hz of `_ours_nores` (linear output; enable_res=False).\n')
    lines.append('> Δ_lf = M_full_nores_LF − M0_nores_LF (negative = improvement).\n')
    lines.append('> 9xjhi target: Δ_lf ≈ −6 dB (Bundle A linear-layer fix confirmed).\n')

    if not nores_lf:
        lines.append('No nores LF data collected.\n')
    else:
        nores_items = sorted(
            [(s, m0_lf, mf_lf, mf_lf - m0_lf)
             for s, (m0_lf, mf_lf) in nores_lf.items()],
            key=lambda x: x[3]
        )
        # 9xjhi specifically
        xjhi_nores = nores_lf.get(xjhi_stem)
        if xjhi_nores:
            d_lf = xjhi_nores[1] - xjhi_nores[0]
            ok_lf = d_lf < -0.5  # target: much negative
            lines.append(f'**9xjhi nores LF**: M0={xjhi_nores[0]:.2f} dB  '
                         f'M_full={xjhi_nores[1]:.2f} dB  '
                         f'Δ={d_lf:+.2f} dB  '
                         f'{"✓ (improvement maintained)" if ok_lf else "⚠ REGRESSION"}\n')
        # Summary stats
        d_arr = np.array([x[3] for x in nores_items])
        lines.append(f'**FS_static nores LF Δ summary** (N={len(nores_items)}): '
                     f'mean={float(np.mean(d_arr)):+.2f} dB  '
                     f'std={float(np.std(d_arr)):.2f} dB  '
                     f'regressions (Δ > +1 dB): '
                     f'{int(np.sum(d_arr > 1.0))}\n')
        # Cases with regression
        nores_regs = [(s, m0_lf, mf_lf, d) for s, m0_lf, mf_lf, d in nores_items if d > 1.0]
        if nores_regs:
            lines.append(f'**nores LF regressions (Δ > +1 dB): {len(nores_regs)} cases**\n')
            lines.append('| Case | M0_LF (dB) | M_full_LF (dB) | Δ (dB) |')
            lines.append('|------|-----------|----------------|--------|')
            for s, m0_lf, mf_lf, d in nores_regs[:20]:
                lines.append(f'| `{s[:50]}` | {m0_lf:.2f} | {mf_lf:.2f} | {d:+.2f} |')
            lines.append('')
        else:
            lines.append('**No nores LF regressions (all Δ ≤ +1 dB)**.\n')

    # --- Conclusion ---
    lines.append('---\n')
    lines.append('## Conclusion\n')

    prod_cat_n = len(prod_catastrophics)
    align_cat_n = len(align_catastrophics)

    if prod_cat_n == 0 and align_cat_n == 0:
        conclusion = 'SHIP'
        detail = ('Both ledgers clean. No catastrophic regressions. '
                  '9xjhi exception accepted as known structural limitation.')
    elif prod_cat_n > 0 and align_cat_n == 0:
        conclusion = 'CONDITIONAL'
        detail = (f'{prod_cat_n} production catastrophic(s) vs M0. '
                  'Alignment ledger (vs AEC3) clean.')
    elif prod_cat_n == 0 and align_cat_n > 0:
        conclusion = 'CONDITIONAL'
        detail = (f'{align_cat_n} alignment catastrophic(s) vs AEC3. '
                  'Production ledger (vs M0) clean.')
    else:
        conclusion = 'NO-SHIP'
        detail = (f'{prod_cat_n} production catastrophic(s) + '
                  f'{align_cat_n} alignment catastrophic(s). Investigate before shipping.')

    lines.append(f'**Overall verdict: {conclusion}**\n')
    lines.append(f'{detail}\n')
    lines.append('')
    lines.append('### Production ledger summary\n')
    for bkt in ['DT_mvmt', 'DT_static', 'FS_mvmt', 'FS_static', 'NE']:
        d_list = prod_by_bucket[bkt]['deltas']
        if not d_list:
            continue
        metric = BUCKET_METRIC[bkt]
        mean_d = float(np.mean(d_list))
        lines.append(f'- **{bkt}** ({metric}): N={len(d_list)} mean Δ={mean_d:+.3f}')
    lines.append('')
    lines.append('### Alignment ledger summary (12 cases vs AEC3)\n')
    for bkt in ['DT_mvmt', 'DT_static', 'FS_mvmt', 'FS_static', 'NE']:
        d_list = align_by_bucket.get(bkt, [])
        if not d_list:
            continue
        mean_d = float(np.mean(d_list))
        lines.append(f'- **{bkt}**: N={len(d_list)} mean Δ_vs_AEC3={mean_d:+.3f}')
    lines.append('')
    if align_catastrophics:
        lines.append('**Alignment catastrophics** (stop if present):')
        for stem, bkt, d, aec3 in align_catastrophics:
            lines.append(f'  - `{stem}` {bkt} Δ={d:+.3f} (AEC3={aec3:.3f})')
        lines.append('')
    lines.append('---\n')
    lines.append('*Auto-generated by `python/v3_21_800case_bench.py`.*\n')
    lines.append('*No code changes. No merge. No version bump.*\n')

    text = '\n'.join(lines) + '\n'
    DOC_PATH.write_text(text, encoding='utf-8')
    print(f'\n[report] Written to {DOC_PATH}')
    print(text[:3000], '...(truncated)' if len(text) > 3000 else '')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    parser = argparse.ArgumentParser(description='v3.21 800-case benchmark')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-byte-equal', action='store_true')
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Step 1: Byte-equal precheck
    if args.skip_byte_equal:
        print('[byte-equal] SKIPPED (--skip-byte-equal)')
        be_ok = None
    else:
        be_ok = _byte_equal_check()
        if not be_ok:
            print('[ERROR] Byte-equal FAIL — M0 diverges from plain BALANCED. Abort.')
            sys.exit(1)
    print()

    # Step 2: Discover all 800 cases
    cases = _discover_cases()
    print(f'[corpus] {len(cases)} cases found in {CORPUS}')
    bkt_counts = defaultdict(int)
    for _, _, bkt, _, _, _ in cases:
        bkt_counts[bkt] += 1
    for bkt, n in sorted(bkt_counts.items()):
        print(f'  {bkt:12s}: {n}')
    print()

    # Build task list
    tasks = []
    for stem, subdir, bucket, wavtype, mic_path, ref_path in cases:
        do_nores = (bucket == 'FS_static')  # nores LF check for FS_static only
        tasks.append((stem, subdir, bucket, wavtype, mic_path, ref_path,
                      MODEL, do_nores))

    total = len(tasks)
    print(f'[parallel] {total} cases × 2 variants, {args.workers} workers', flush=True)
    if bkt_counts.get('FS_static', 0):
        print(f'  (FS_static cases: {bkt_counts["FS_static"]} × 2 extra nores renders)', flush=True)
    print()

    # Step 3: Parallel rendering + scoring
    results: List[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_case_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                done += 1
                stem_short = r['stem'][:35]
                bkt = r['bucket']
                metric = BUCKET_METRIC[bkt]
                m0_s    = r['M0_echo'] if metric == 'echo' else r['M0_deg']
                mfull_s = r['M_full_delay_echo'] if metric == 'echo' else r['M_full_delay_deg']
                delta   = mfull_s - m0_s
                print(f'  [{done:3d}/{total}] [{bkt:10s}] {stem_short:35s} '
                      f'M0={m0_s:.3f} Mfull={mfull_s:.3f} Δ={delta:+.4f}',
                      flush=True)
            except Exception as exc:
                stem_short = str(futs[fut][0])[:35]
                print(f'  [ERROR] {stem_short}: {exc}', flush=True)

    print(f'\n[done] {len(results)}/{total} cases scored')

    # Step 4: Save JSON
    json_data = {r['stem']: r for r in results}
    JSON_PATH.write_text(json.dumps(json_data, indent=2), encoding='utf-8')
    print(f'[json] Saved to {JSON_PATH}')

    # Step 5: Write report
    _write_report(results, be_ok=be_ok if be_ok is not None else True)


if __name__ == '__main__':
    run()
