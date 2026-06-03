/* parity_aec3_scale.c — emit aec3_scale + freq_utils values as `key=%.17g`
 * lines for bit-exact diff against the Python reference
 * (python/diag/parity_aec3_scale_golden.py). Phase 5.0 golden gate.
 *
 * Build (from c_impl/):
 *   gcc -Wall -O2 -ffp-contract=off -std=c99 -Iinclude \
 *       src/aec3_scale.c src/freq_utils.c test/parity_aec3_scale.c -lm -o /tmp/p_scale
 */
#include "aec3_scale.h"
#include "freq_utils.h"
#include <stdio.h>

#define P(name, val) printf("%s=%.17g\n", name, (double)(val))

int main(void) {
    /* constants */
    P("PSD_SCALE", AEC3_PSD_SCALE);
    P("PSD_SCALE_INV", AEC3_PSD_SCALE_INV);
    P("NORMAL_RENDER_LIMIT_FLOAT", AEC3_NORMAL_RENDER_LIMIT_FLOAT);
    P("LOW_RENDER_LIMIT_FLOAT", AEC3_LOW_RENDER_LIMIT_FLOAT);
    P("FLOOR_POWER_FLOAT", AEC3_FLOOR_POWER_FLOAT);
    P("FILTER_NOISE_GATE_POWER_FLOAT", AEC3_FILTER_NOISE_GATE_POWER_FLOAT);
    P("STATIONARITY_MIN_NOISE_POWER_FLOAT", AEC3_STATIONARITY_MIN_NOISE_POWER_FLOAT);
    P("MATCHED_FILTER_EXCITATION_LIMIT_FLOAT", AEC3_MATCHED_FILTER_EXCITATION_LIMIT_FLOAT);
    P("MATCHED_FILTER_SATURATION_LIMIT_FLOAT", AEC3_MATCHED_FILTER_SATURATION_LIMIT_FLOAT);
    P("H_ERROR_INIT_FLOAT", AEC3_H_ERROR_INIT_FLOAT);
    P("H_ERROR_FLOOR_FLOAT", AEC3_H_ERROR_FLOOR_FLOAT);
    P("H_ERROR_CEIL_FLOAT", AEC3_H_ERROR_CEIL_FLOAT);
    P("MAX_ERLE_LF", AEC3_MAX_ERLE_LF);
    P("MAX_ERLE_HF", AEC3_MAX_ERLE_HF);
    P("MIN_ERLE", AEC3_MIN_ERLE);
    P("RESIDUAL_NOISE_GATE_POWER", AEC3_RESIDUAL_NOISE_GATE_POWER);

    /* psd_int16_to_float */
    P("psd(20075344)", aec3_psd_int16_to_float(20075344.0));
    P("psd(27509562)", aec3_psd_int16_to_float(27509562.0));
    P("psd(10)", aec3_psd_int16_to_float(10.0));

    /* blocks_to_hops at hop=160 and hop=320 (rounding) */
    P("b2h(40,160)", aec3_blocks_to_hops(40, 160, 16000));
    P("b2h(1000,160)", aec3_blocks_to_hops(1000, 160, 16000));
    P("b2h(7500,160)", aec3_blocks_to_hops(7500, 160, 16000));
    P("b2h(4,160)", aec3_blocks_to_hops(4, 160, 16000));
    P("b2h(50,160)", aec3_blocks_to_hops(50, 160, 16000));
    P("b2h(12,160)", aec3_blocks_to_hops(12, 160, 16000));
    P("b2h(13,160)", aec3_blocks_to_hops(13, 160, 16000));
    P("b2h(500,160)", aec3_blocks_to_hops(500, 160, 16000));
    P("b2h(40,320)", aec3_blocks_to_hops(40, 320, 16000));
    P("b2h(13,320)", aec3_blocks_to_hops(13, 320, 16000));

    /* ms_to_hops */
    P("ms2h(250,160)", aec3_ms_to_hops(250.0, 160, 16000));
    P("ms2h(200,160)", aec3_ms_to_hops(200.0, 160, 16000));

    /* per_block_rate_to_per_hop */
    P("pbr2ph(5e-5,160)", aec3_per_block_rate_to_per_hop(5e-5, 160, 16000));
    P("pbr2ph(5e-2,160)", aec3_per_block_rate_to_per_hop(5e-2, 160, 16000));
    P("pbr2ph(5e-3,160)", aec3_per_block_rate_to_per_hop(5e-3, 160, 16000));

    /* per_block_ema_alpha_to_per_hop */
    P("ema(0.05,160)", aec3_per_block_ema_alpha_to_per_hop(0.05, 160, 16000));
    P("ema(0.004,160)", aec3_per_block_ema_alpha_to_per_hop(0.004, 160, 16000));

    /* fft_density_scale */
    P("fds(64,512)", aec3_fft_density_scale(64.0, 512));
    P("fds(128,512)", aec3_fft_density_scale(128.0, 512));
    P("fds(1638400,512)", aec3_fft_density_scale(1638400.0, 512));

    /* per_bin_psd_threshold */
    P("pbpt(44015068,160)", aec3_per_bin_psd_threshold(44015068.0, 160, 160));
    P("pbpt(44015068,320)", aec3_per_bin_psd_threshold(44015068.0, 320, 160));
    P("pbpt(44015068,80)", aec3_per_bin_psd_threshold(44015068.0, 80, 160));

    /* nl_r2_norm_power */
    P("nl(160)", aec3_nl_r2_norm_power(160, 160));
    P("nl(320)", aec3_nl_r2_norm_power(320, 160));

    /* block_energy_scale */
    P("bes(160000,160)", aec3_block_energy_scale(160000.0, 160));

    /* per_block_growth_to_per_hop */
    P("pbg(1.0002,160)", aec3_per_block_growth_to_per_hop(1.0002, 160, 16000));

    /* hz_to_bin — incl. round-half-to-even edges (15.625->0.5, 46.875->1.5) */
    P("hz2b(15.625,257)", hz_to_bin(15.625, 257, 16000));   /* 0.5 -> 0 (even) */
    P("hz2b(46.875,257)", hz_to_bin(46.875, 257, 16000));   /* 1.5 -> 2 (even) */
    P("hz2b(625,257)", hz_to_bin(625.0, 257, 16000));
    P("hz2b(1500,257)", hz_to_bin(1500.0, 257, 16000));
    P("hz2b(4000,257)", hz_to_bin(4000.0, 257, 16000));
    P("hz2b(2000,257)", hz_to_bin(2000.0, 257, 16000));

    /* bin_to_hz */
    P("b2hz(64,257)", bin_to_hz(64, 257, 16000));
    P("b2hz(48,257)", bin_to_hz(48, 257, 16000));
    return 0;
}
