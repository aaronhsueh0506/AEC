/* aec_config_build.c -- the configuration builders declared in aec_config_build.h: the suppression-gain config and tuning of a
 * resolved configuration, the run-length ENR/SU table expansion, the level retarget of a suppressor and the synthesis window.
 * The only translation unit that instantiates the generated rate table and the window half table (aec_balanced_config.h); the
 * rest of the library reads the rate rows through aec_balanced_rate_dims(). */
#include "aec_config_build.h"
#include "aec_internal.h"
#include "aec_res_internal.h"
#include "utility/mem_align.h"   /* ALIGN16 */

#include <assert.h>
#include <math.h>
#include <string.h>

const AecBalancedRateDims* aec_balanced_rate_dims(int sample_rate, int fft_size) {
    return aecb_rate_cfg(sample_rate, fft_size);
}

/* Synthesis window of length n (a power of two that divides AECB_SYNTH_HALF_N), rebuilt bit for bit from the half table. */
void aec_synth_window_build(float* w, int n) {
    const int stride = AECB_SYNTH_HALF_N / n;
    int i;
    assert(n >= 2 && AECB_SYNTH_HALF_N % n == 0);
    for (i = 0; i < n; ++i) w[i] = AECB_SYNTH_WINDOW_HALF[(i <= n / 2 ? i : n - i) * stride];
}

size_t aec_sg_tables_bytes(int n_bins) {
    return ck_mul_size(4, ck_align16_size((size_t)n_bins * sizeof(float)));
}

/* Expands the row's run-length ENR/SU tables into `mem`: four tables, each padded to 16 bytes and zero in the padding. */
void aec_sg_tables_expand(const AecBalancedRateDims* rd, void* mem, SuppressionGainTuning* tun) {
    const int n = rd->sg_table_len;
    const size_t stride = ALIGN16((size_t)n * sizeof(float)) / sizeof(float);
    float* const dst = (float*)mem;
    int t, r = 0, k;
    assert(rd->sg_runs >= 1 && rd->sg_run_end[rd->sg_runs - 1] >= n);
    for (k = 0; k < n; ++k) {
        while (k >= rd->sg_run_end[r]) ++r;
        for (t = 0; t < 4; ++t) dst[(size_t)t * stride + (size_t)k] = rd->sg_run_val[(size_t)t * (size_t)rd->sg_runs + (size_t)r];
    }
    for (t = 0; t < 4; ++t)
        for (k = n; (size_t)k < stride; ++k) dst[(size_t)t * stride + (size_t)k] = 0.0f;
    tun->nearend_enr_tr = dst;                tun->nearend_enr_su = dst + stride;
    tun->normal_enr_tr  = dst + 2 * stride;   tun->normal_enr_su  = dst + 3 * stride;
}

int aec_sg_tables_decode(const AecResolved* rs, void* mem, SuppressionGainTuning* tun) {
    const AecBalancedRateDims* rd;
    if (!rs || !mem || !tun) return -1;
    rd = aec_balanced_rate_dims(rs->cfg.sample_rate, rs->cfg.fft_size);
    if (!rd) return -1;
    aec_sg_tables_expand(rd, mem, tun);
    return 0;
}

/* Suppression-gain config and tuning for a resolved configuration; shared by
 * Aec and standalone RES instances. The ENR/SU tables of the tuning stay NULL (aec_sg_tables_decode() fills them) and the
 * EMR tables point at static data. Returns 0, or -1 without a signal grid or rate row. */
int aec_build_sg_config(const AecResolved* rs, SuppressionGainConfig* out_cfg,
                        SuppressionGainTuning* out_tun) {
    const AecConfig* cfg;
    const AecTuning* tn;
    AecSignalGrid grid;
    const AecBalancedRateDims* rd;
    SuppressionGainConfig scfg;
    SuppressionGainTuning stun;
    int hop, K;
    if (!rs || !out_cfg || !out_tun) return -1;
    cfg = &rs->cfg;
    tn = &rs->tun;
    if (aec_resolve_signal_grid(cfg->sample_rate, cfg->fft_size, &grid) != AEC_OK) return -1;
    rd = aec_balanced_rate_dims(cfg->sample_rate, cfg->fft_size);
    if (!rd) return -1;
    hop = grid.hop_size;
    K = grid.n_freqs;
    memset(&scfg, 0, sizeof(scfg));
    scfg.n_bins = K; scfg.sr = cfg->sample_rate; scfg.hop_size = hop;
    scfg.last_lf_smoothing_band = rd->sg_last_lf_smoothing_band;
    scfg.last_permanent_lf_smoothing_band = AECB_SG_LAST_PERMANENT;
    scfg.lf_smoothing_during_initial_phase = AECB_SG_LF_SMOOTHING_INITIAL;
    scfg.lf_clamp_bin = rd->sg_lf_clamp_bin;
    scfg.dne_lf_end = rd->sg_dne_lf_end;
    scfg.nearend_smoother_n = rd->sg_nearend_smoother_n;
    scfg.aud_lf_end_bin = rd->sg_aud_lf_end_bin; scfg.aud_mf_end_bin = rd->sg_aud_mf_end_bin;
    scfg.floor_power = rd->sg_floor_power;
    scfg.aud_thr_lf = AECB_SG_AUD_THR_LF; scfg.aud_thr_mf = AECB_SG_AUD_THR_MF;
    scfg.aud_thr_hf = AECB_SG_AUD_THR_HF;
    scfg.low_render_limit = rd->sg_low_render_limit;
    scfg.normal_render_limit = rd->sg_normal_render_limit;
    scfg.hf_lgb = rd->sg_hf_lgb; scfg.hf_biq = AECB_SG_HF_BIQ;
    scfg.conservative_hf = AECB_SG_CONSERVATIVE_HF;
    scfg.max_inc_normal = rd->sg_max_inc;
    scfg.max_inc_nearend = rd->sg_max_inc;
    scfg.max_dec_lf_normal = rd->sg_max_dec_lf;
    scfg.max_dec_lf_nearend = rd->sg_max_dec_lf;
    scfg.floor_first_increase = 0.00001f;
    scfg.low_render_threshold = rd->sg_low_render_threshold;
    /* From the level, so NO_FLOOR reaches the suppressor at construction. */
    scfg.split_floor_enabled = tn->min_gain_split_floor_enabled;
    scfg.split_floor_far_active =
        powf(10.0f, tn->min_gain_floor_far_active_db / 10.0f);
    scfg.split_floor_far_silent = AECB_SG_SPLIT_FLOOR_FAR_SILENT;
    scfg.split_floor_dt =
        powf(10.0f, tn->min_gain_floor_dt_db / 10.0f);
    scfg.split_floor_latch_power = AECB_SG_SPLIT_FLOOR_LATCH_POWER;
    scfg.soft_blend_enabled = AECB_SG_SOFT_BLEND_ENABLED;
    scfg.soft_blend_per_bin = AECB_SG_SOFT_BLEND_PER_BIN;
    scfg.soft_blend_enr_thr = (float)AECB_SG_SOFT_BLEND_ENR_THR;
    scfg.soft_blend_softness = (float)AECB_SG_SOFT_BLEND_SOFTNESS;
    scfg.dne_enr_threshold = AECB_SG_DNE_ENR_THRESHOLD;
    scfg.dne_enr_exit_threshold = AECB_SG_DNE_ENR_EXIT_THRESHOLD;
    scfg.dne_snr_threshold = AECB_SG_DNE_SNR_THRESHOLD;
    scfg.dne_use_during_initial_phase = AECB_SG_DNE_USE_DURING_INITIAL_PHASE;
    scfg.dne_use_unbounded_echo = AECB_SG_DNE_USE_UNBOUNDED_ECHO;
    scfg.dne_lf_endpoint_bin = rd->sg_dne_lf_endpoint_bin;
    scfg.dne_trigger_threshold_hops = rd->sg_trigger_threshold_hops;
    scfg.dne_hold_duration_hops = rd->sg_hold_duration_hops;
    scfg.stat_aware_ne_proxy_enabled = 0; scfg.stat_aware_ne_proxy_threshold = 0.10f;
    /* The ENR/SU tables are decoded into the pool (aec_sg_tables_decode()); only the EMR tables are static. */
    stun.nearend_emr_tr = rd->sg_nearend_emr_tr;
    stun.normal_emr_tr  = rd->sg_normal_emr_tr;
    stun.table_len      = rd->sg_table_len;
    *out_cfg = scfg;
    *out_tun = stun;
    return 0;
}

int aec_sg_apply_level(SuppressionGain* sg, AecLevel level) {
    float db, dt_db;
    int en;
    if (aec_level_floor_db(level, &db) != 0 || aec_level_dt_floor_db(level, &dt_db) != 0 ||
        aec_level_split_floor_enabled(level, &en) != 0) return -1;
    /* The suppressor validates and applies first and writes nothing on refusal. The ramp governs the far-active floor only; the
     * DT floor and the split-floor gate step immediately. */
    if (suppression_gain_set_split_floor_far_active_db(sg, db, AEC_LEVEL_RAMP_MS) != 0) return -1;
    if (suppression_gain_set_split_floor_dt_db(sg, dt_db) != 0) return -1;
    return suppression_gain_set_split_floor_enabled(sg, en) != 0 ? -1 : 0;
}
