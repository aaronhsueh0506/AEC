"""Per-case failure-mode analysis for the worst movement-DT cases.

For each worst-N case (already produced by diag_worst_movement_dt.py):
  - Reads state trace JSONL
  - Reads mic / ours / aec2 wav
  - Extracts:
      • frame-by-frame STFT energy (echo + final)
      • Convergence trajectory (% frames converged, EPC fire count, main_paused %)
      • Echo-leak windows: where ours_db > aec2_db significantly
      • DT presence pattern
  - Generates spectrogram PNGs (mic / ours / aec2) for visual inspection

Output:
  output_worst/<stem>.summary.txt   (textual analysis)
  output_worst/<stem>.spec.png      (3-panel spectrogram)
"""
import json
import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf

REPO = Path(__file__).parent.parent
WAV  = REPO / 'wav/aec_challenge_blind/doubletalk'
OUT  = Path(__file__).parent / 'output_worst'

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print('matplotlib not available — skipping PNGs', file=sys.stderr)


def _stft_db(x, sr, win=512, hop=256):
    n_frames = (len(x) - win) // hop + 1
    if n_frames <= 0:
        return np.zeros((win // 2 + 1, 0)), np.array([])
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, win),
        strides=(hop * x.strides[0], x.strides[0])
    )
    w = np.hanning(win).astype(np.float32)
    spec = np.fft.rfft(frames * w, axis=1).T
    db = 20 * np.log10(np.abs(spec) + 1e-10)
    t = np.arange(n_frames) * hop / sr
    return db, t


def _summarize_trace(stem, trace_path):
    if not trace_path.is_file():
        return ['(no trace)']
    rows = [json.loads(l) for l in open(trace_path)]
    n = len(rows)
    if n == 0: return ['(empty)']

    converged_n = sum(1 for r in rows if r['filter_converged'])
    once_conv_n = sum(1 for r in rows if r['filter_once_converged'])
    epc_n       = sum(1 for r in rows if r['epc_active'])
    paused_n    = sum(1 for r in rows if r['main_paused'])
    usable_n    = sum(1 for r in rows if r['usable_linear'])

    # First convergence frame
    first_conv = next((r['idx'] for r in rows if r['filter_converged']), -1)

    # EPC fire detection: epc_active False→True transitions
    epc_fires = 0
    prev = False
    for r in rows:
        if r['epc_active'] and not prev:
            epc_fires += 1
        prev = r['epc_active']

    # main_paused fire count (similar transition counter)
    paused_fires = 0
    prev = False
    for r in rows:
        if r['main_paused'] and not prev:
            paused_fires += 1
        prev = r['main_paused']

    # Mean DT signals in DT-active regions (dt_combined > 0.3)
    dt_rows = [r for r in rows if r['dt_combined'] > 0.3]
    n_dt = len(dt_rows)
    if n_dt > 0:
        dt_e = np.mean([r['dt_energy'] for r in dt_rows])
        dt_s = np.mean([r['dt_shadow'] for r in dt_rows])
        dt_c = np.mean([r['dt_coh']    for r in dt_rows])
    else:
        dt_e = dt_s = dt_c = 0.0

    # ERLE / divergence trajectory
    erle_post = [r['erle_inst'] for r in rows if r['filter_converged']]
    div_post  = [r['divergence'] for r in rows if r['filter_converged']]

    return [
        f'frames: {n}',
        f'first_converged_frame: {first_conv} ({first_conv*0.01:.2f}s @ 16k/160hop)' if first_conv >= 0 else 'NEVER CONVERGED',
        f'converged %: {100*converged_n/n:.1f}   once_converged %: {100*once_conv_n/n:.1f}',
        f'epc_active %: {100*epc_n/n:.1f}   epc_fires: {epc_fires}',
        f'main_paused %: {100*paused_n/n:.1f}  paused_fires: {paused_fires}',
        f'usable_linear %: {100*usable_n/n:.1f}',
        f'DT-active frames (dt_combined>0.3): {n_dt} ({100*n_dt/n:.1f}%)',
        f'  in DT regions: dt_energy={dt_e:.3f}  dt_shadow={dt_s:.3f}  dt_coh={dt_c:.3f}',
        f'post-conv ERLE: mean={np.mean(erle_post):.2f}dB  median={np.median(erle_post):.2f}dB' if erle_post else 'no post-conv frames',
        f'post-conv divergence: mean={np.mean(div_post):.3f}  max={np.max(div_post):.3f}' if div_post else '',
    ]


def _energy_per_window(x, sr, win=0.5):
    """Energy in dB per `win`-second window."""
    n = int(sr * win)
    out = []
    for i in range(0, len(x) - n + 1, n):
        e = np.mean(x[i:i+n].astype(np.float64) ** 2) + 1e-12
        out.append(10 * np.log10(e))
    return np.array(out)


def _spectrogram_png(stem, mic_path, ours_path, aec2_path, out_png):
    if not HAVE_MPL:
        return
    mic, sr = sf.read(mic_path, dtype='float32')
    ours, _ = sf.read(ours_path, dtype='float32')
    aec2, _ = sf.read(aec2_path, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if ours.ndim > 1: ours = ours[:, 0]
    if aec2.ndim > 1: aec2 = aec2[:, 0]

    n = min(len(mic), len(ours), len(aec2))
    mic, ours, aec2 = mic[:n], ours[:n], aec2[:n]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, sig, name in zip(axes, (mic, ours, aec2), ('MIC', 'OURS', 'AEC2')):
        S, t = _stft_db(sig, sr)
        ax.imshow(S, origin='lower', aspect='auto',
                  extent=[0, n / sr, 0, sr / 2], cmap='magma',
                  vmin=-80, vmax=0)
        ax.set_ylabel(f'{name}\nfreq (Hz)')
        ax.set_ylim(0, 4000)
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(stem)
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def main():
    rank_path = OUT / 'movement_dt_ranking.json'
    if not rank_path.is_file():
        print('run diag_worst_movement_dt.py first'); return 2
    ranking = json.load(open(rank_path))
    worst = ranking[:10]

    for r in worst:
        stem = r['stem']
        trace = OUT / f'{stem}.trace.jsonl'
        mic_p = WAV / f'{stem}_mic.wav'
        ours_p = OUT / f'{stem}_ours.wav'
        aec2_p = REPO / 'python/output_ref' / f'{stem}_aec2.wav'

        lines = [
            f'CASE {stem}',
            f'  AECMOS: ours_echo={r["ours_echo"]:.3f}  aec2_echo={r["aec2_echo"]:.3f}  '
            f'Δecho={r["d_echo"]:+.3f}  Δdeg={r["d_deg"]:+.3f}',
            f'  Energy: ours_db={r["ours_db"]:+.2f}dB  aec2_db={r["aec2_db"]:+.2f}dB  '
            f'Δdb={r["d_db"]:+.2f}dB  (positive = ours leaks more)',
            '',
            '— state-trace summary —',
        ]
        lines.extend('  ' + s for s in _summarize_trace(stem, trace))

        out_txt = OUT / f'{stem}.summary.txt'
        out_txt.write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        print(f'  → wrote {out_txt}')

        if HAVE_MPL and ours_p.is_file() and aec2_p.is_file():
            png = OUT / f'{stem}.spec.png'
            try:
                _spectrogram_png(stem, str(mic_p), str(ours_p), str(aec2_p), str(png))
                print(f'  → wrote {png}')
            except Exception as e:
                print(f'  spec failed: {e}')
        print()


if __name__ == '__main__':
    sys.exit(main() or 0)
