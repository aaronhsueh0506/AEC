#!/usr/bin/env python3
"""Phase 3 ResFilter parity baseline.

Drives Python ResFilter (gain_type="enr", echo_method="direct", BALANCED-like
config) with deterministic synthetic spectra. Captures per-frame output,
gain_smooth, echo_psd, error_psd, noise_psd to compare with C ResFilter.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aec import ResFilter

BLOCK = 512
FRAME = 320
HOP = 160
N_FREQS = BLOCK // 2 + 1
N_FRAMES = 200
SEED = 42
SR = 16000

# BALANCED-ish config
G_MIN_DB = -55.0
ENR_SCALE = 0.85
REVERB_DECAY = 0.65
REVERB_GAIN = 1.4
ALPHA_ECHO = 0.4
ALPHA_ERR  = 0.5
SPEC_FLOOR_DB = -38.0
NE_PROTECT_DB = -16.0

OUT_FILE   = Path(__file__).parent / 'parity_resfilter_python.npz'
INPUT_FILE = Path(__file__).parent / 'parity_resfilter_input.bin'


def main():
    rng = np.random.default_rng(SEED)

    # Synthetic time-domain error (post-linear-filter residual) per hop
    error_hop = (rng.standard_normal((N_FRAMES, HOP)) * 0.05).astype(np.float32)

    # Synthetic spectra (echo, far, near) per frame
    echo_re = (rng.standard_normal((N_FRAMES, N_FREQS)) * 0.2).astype(np.float32)
    echo_im = (rng.standard_normal((N_FRAMES, N_FREQS)) * 0.2).astype(np.float32)
    far_re  = (rng.standard_normal((N_FRAMES, N_FREQS)) * 0.4).astype(np.float32)
    far_im  = (rng.standard_normal((N_FRAMES, N_FREQS)) * 0.4).astype(np.float32)
    near_re = (rng.standard_normal((N_FRAMES, N_FREQS)) * 0.3).astype(np.float32)
    near_im = (rng.standard_normal((N_FRAMES, N_FREQS)) * 0.3).astype(np.float32)

    # Per-frame scalars
    far_power = (rng.uniform(0, 0.05, N_FRAMES)).astype(np.float32)
    erle_factor = (np.linspace(0, 1, N_FRAMES)).astype(np.float32)
    dt_indicator = (rng.uniform(0, 0.6, N_FRAMES)).astype(np.float32)
    over_sub = np.full(N_FRAMES, 5.0, dtype=np.float32)
    erl_estimate = np.full(N_FRAMES, 0.1, dtype=np.float32)

    # Booleans
    filter_converged = np.zeros(N_FRAMES, dtype=np.int32)
    filter_converged[60:] = 1
    is_stationary_dt = np.zeros(N_FRAMES, dtype=np.int32)
    epc_active = np.zeros(N_FRAMES, dtype=np.int32)
    saturation_level = np.zeros(N_FRAMES, dtype=np.float32)

    # Build Python ResFilter
    res = ResFilter(
        block_size=BLOCK, n_freqs=N_FREQS,
        g_min_db=G_MIN_DB, over_sub=5.0, alpha=0.5,
        enable_cng=False,
        max_drop_db_per_frame=6.0, max_rise_db_per_frame=6.0,
        enable_spectral_floor=True,
        spectral_floor_db=SPEC_FLOOR_DB, ne_protect_db=NE_PROTECT_DB,
        frame_size=FRAME, hop_size=HOP,
        echo_method="direct", gain_type="enr",
        enable_reverb=True, reverb_decay=REVERB_DECAY, reverb_gain=REVERB_GAIN,
        alpha_echo_psd=ALPHA_ECHO, alpha_error_psd=ALPHA_ERR,
        enr_scale=ENR_SCALE,
        sample_rate=SR,
    )

    out = np.zeros((N_FRAMES, HOP), dtype=np.float32)
    gain_states = np.zeros((N_FRAMES, N_FREQS), dtype=np.float32)
    echo_psd_states = np.zeros((N_FRAMES, N_FREQS), dtype=np.float32)
    error_psd_states = np.zeros((N_FRAMES, N_FREQS), dtype=np.float32)
    noise_psd_states = np.zeros((N_FRAMES, N_FREQS), dtype=np.float32)

    for f in range(N_FRAMES):
        echo_spec = (echo_re[f] + 1j * echo_im[f]).astype(np.complex64)
        far_spec  = (far_re[f]  + 1j * far_im[f] ).astype(np.complex64)
        near_spec = (near_re[f] + 1j * near_im[f]).astype(np.complex64)

        o = res.process(
            error_hop[f], echo_spec, float(far_power[f]), far_spec,
            filter_converged=bool(filter_converged[f]),
            erle_factor=float(erle_factor[f]),
            dt_indicator=float(dt_indicator[f]),
            near_spec=near_spec,
            divergence=0.0,
            is_stationary_dt=bool(is_stationary_dt[f]),
            saturation_level=float(saturation_level[f]),
            epc_active=bool(epc_active[f]),
            shadow_dt=0.0,
            erl_estimate=float(erl_estimate[f]),
        )
        out[f] = o
        gain_states[f] = res.gain_smooth.copy()
        echo_psd_states[f] = res.echo_psd.copy()
        error_psd_states[f] = res.error_psd.copy()
        noise_psd_states[f] = res.noise_psd.copy()

    np.savez(OUT_FILE,
             output=out.reshape(-1),
             gain=gain_states,
             echo_psd=echo_psd_states,
             error_psd=error_psd_states,
             noise_psd=noise_psd_states,
             config=np.array([BLOCK, FRAME, HOP, N_FREQS, N_FRAMES, SR],
                             dtype=np.int32))

    with open(INPUT_FILE, 'wb') as fp:
        fp.write(error_hop.tobytes())
        fp.write(echo_re.tobytes()); fp.write(echo_im.tobytes())
        fp.write(far_re.tobytes());  fp.write(far_im.tobytes())
        fp.write(near_re.tobytes()); fp.write(near_im.tobytes())
        fp.write(far_power.tobytes())
        fp.write(erle_factor.tobytes())
        fp.write(dt_indicator.tobytes())
        fp.write(over_sub.tobytes())
        fp.write(erl_estimate.tobytes())
        fp.write(filter_converged.tobytes())
        fp.write(is_stationary_dt.tobytes())
        fp.write(epc_active.tobytes())
        fp.write(saturation_level.tobytes())

    print(f"Saved Python baseline → {OUT_FILE}")
    print(f"  output energy:  {np.mean(out**2):.4e}")
    print(f"  gain[final] mean: {np.mean(gain_states[-1]):.4e}")
    print(f"  echo_psd[final] mean: {np.mean(echo_psd_states[-1]):.4e}")
    print(f"  error_psd[final] mean: {np.mean(error_psd_states[-1]):.4e}")


if __name__ == '__main__':
    main()
