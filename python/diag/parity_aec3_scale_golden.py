"""Golden reference for the C aec3_scale + freq_utils port (WS5 Phase 5.0).

Emits `key=%.17g` lines matching c_impl/test/parity_aec3_scale.c, so a bit-exact
`diff` proves the C conversion layer reproduces the Python reference. Run:

    python3 python/diag/parity_aec3_scale_golden.py > /tmp/py_scale.txt
    gcc -O2 -ffp-contract=off -std=c99 -Ic_impl/include \\
        c_impl/src/aec3_scale.c c_impl/src/freq_utils.c \\
        c_impl/test/parity_aec3_scale.c -lm -o /tmp/p_scale
    /tmp/p_scale > /tmp/c_scale.txt
    diff /tmp/py_scale.txt /tmp/c_scale.txt   # must be empty
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import modules.aec3_scale as s          # noqa: E402
from modules.freq_utils import hz_to_bin, bin_to_hz  # noqa: E402


def P(name, val):
    print(f"{name}={float(val):.17g}")


def main():
    P("PSD_SCALE", s.PSD_SCALE)
    P("PSD_SCALE_INV", s.PSD_SCALE_INV)
    P("NORMAL_RENDER_LIMIT_FLOAT", s.NORMAL_RENDER_LIMIT_FLOAT)
    P("LOW_RENDER_LIMIT_FLOAT", s.LOW_RENDER_LIMIT_FLOAT)
    P("FLOOR_POWER_FLOAT", s.FLOOR_POWER_FLOAT)
    P("FILTER_NOISE_GATE_POWER_FLOAT", s.FILTER_NOISE_GATE_POWER_FLOAT)
    P("STATIONARITY_MIN_NOISE_POWER_FLOAT", s.STATIONARITY_MIN_NOISE_POWER_FLOAT)
    P("MATCHED_FILTER_EXCITATION_LIMIT_FLOAT", s.MATCHED_FILTER_EXCITATION_LIMIT_FLOAT)
    P("MATCHED_FILTER_SATURATION_LIMIT_FLOAT", s.MATCHED_FILTER_SATURATION_LIMIT_FLOAT)
    P("H_ERROR_INIT_FLOAT", s.H_ERROR_INIT_FLOAT)
    P("H_ERROR_FLOOR_FLOAT", s.H_ERROR_FLOOR_FLOAT)
    P("H_ERROR_CEIL_FLOAT", s.H_ERROR_CEIL_FLOAT)
    P("MAX_ERLE_LF", s.MAX_ERLE_LF)
    P("MAX_ERLE_HF", s.MAX_ERLE_HF)
    P("MIN_ERLE", s.MIN_ERLE)
    P("RESIDUAL_NOISE_GATE_POWER", s.RESIDUAL_NOISE_GATE_POWER)

    P("psd(20075344)", s.psd_int16_to_float(20075344.0))
    P("psd(27509562)", s.psd_int16_to_float(27509562.0))
    P("psd(10)", s.psd_int16_to_float(10.0))

    P("b2h(40,160)", s.blocks_to_hops(40, 160, 16000))
    P("b2h(1000,160)", s.blocks_to_hops(1000, 160, 16000))
    P("b2h(7500,160)", s.blocks_to_hops(7500, 160, 16000))
    P("b2h(4,160)", s.blocks_to_hops(4, 160, 16000))
    P("b2h(50,160)", s.blocks_to_hops(50, 160, 16000))
    P("b2h(12,160)", s.blocks_to_hops(12, 160, 16000))
    P("b2h(13,160)", s.blocks_to_hops(13, 160, 16000))
    P("b2h(500,160)", s.blocks_to_hops(500, 160, 16000))
    P("b2h(40,320)", s.blocks_to_hops(40, 320, 16000))
    P("b2h(13,320)", s.blocks_to_hops(13, 320, 16000))

    P("ms2h(250,160)", s.ms_to_hops(250.0, 160, 16000))
    P("ms2h(200,160)", s.ms_to_hops(200.0, 160, 16000))

    P("pbr2ph(5e-5,160)", s.per_block_rate_to_per_hop(5e-5, 160, 16000))
    P("pbr2ph(5e-2,160)", s.per_block_rate_to_per_hop(5e-2, 160, 16000))
    P("pbr2ph(5e-3,160)", s.per_block_rate_to_per_hop(5e-3, 160, 16000))

    P("ema(0.05,160)", s.per_block_ema_alpha_to_per_hop(0.05, 160, 16000))
    P("ema(0.004,160)", s.per_block_ema_alpha_to_per_hop(0.004, 160, 16000))

    P("fds(64,512)", s.fft_density_scale(64.0, 512))
    P("fds(128,512)", s.fft_density_scale(128.0, 512))
    P("fds(1638400,512)", s.fft_density_scale(1638400.0, 512))

    P("pbpt(44015068,160)", s.per_bin_psd_threshold(44015068.0, 160))
    P("pbpt(44015068,320)", s.per_bin_psd_threshold(44015068.0, 320))
    P("pbpt(44015068,80)", s.per_bin_psd_threshold(44015068.0, 80))

    P("nl(160)", s.nl_r2_norm_power(160))
    P("nl(320)", s.nl_r2_norm_power(320))

    P("bes(160000,160)", s.block_energy_scale(160000.0, 160))

    P("pbg(1.0002,160)", s.per_block_growth_to_per_hop(1.0002, 160, 16000))

    P("hz2b(15.625,257)", hz_to_bin(15.625, 257, 16000))
    P("hz2b(46.875,257)", hz_to_bin(46.875, 257, 16000))
    P("hz2b(625,257)", hz_to_bin(625.0, 257, 16000))
    P("hz2b(1500,257)", hz_to_bin(1500.0, 257, 16000))
    P("hz2b(4000,257)", hz_to_bin(4000.0, 257, 16000))
    P("hz2b(2000,257)", hz_to_bin(2000.0, 257, 16000))

    P("b2hz(64,257)", bin_to_hz(64, 257, 16000))
    P("b2hz(48,257)", bin_to_hz(48, 257, 16000))


if __name__ == '__main__':
    main()
