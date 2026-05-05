/* aec.c — top-level AEC orchestration. */
#include "aec.h"
#include "aec_debug.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void aec_config_defaults(AecConfig* cfg, int sr) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->sample_rate = sr;
    cfg->filter_length_ms = 52.0f;
    cfg->mu = 0.3f;
    cfg->delta = 1e-8f;
    cfg->enable_cng = 0;             /* User Fix 2 */
    cfg->enable_delay_est = 1;
    cfg->enable_highpass = 1;
    cfg->enable_saturation = 1;
    cfg->enable_shadow_filter = 1;
    cfg->enable_residual_filter = 1;
    cfg->saturation_softclip_ref = 1;
    cfg->saturation_over_sub_boost = 3.0f;
    cfg->startup_dt_mu_min = 0.0f;
    cfg->shadow_copy_threshold = 0.65f;
    cfg->shadow_copy_hysteresis = 3;
    cfg->shadow_dtd_offset = 1.5f;
    cfg->shadow_dtd_advantage_scale = 3.0f;
    cfg->shadow_err_alpha = 0.80f;
    cfg->warmup_frames = 80;
    cfg->epc_hangover = 20;
    cfg->epc_total_rise = 1.5f;
    cfg->epc_delta_threshold = 0.3f;
    /* v3.10.4: max_delay_ms 250 → 512 → 1024; delay_buffer_ms 1024 → 2048. */
    cfg->max_delay_ms = 1024.0f;
    cfg->delay_buffer_ms = 2048.0f;
    cfg->delay_est_init_s = 0.3f;
    cfg->delay_est_period_s = 0.5f;
    cfg->highpass_cutoff_hz = 80.0f;
    cfg->saturation_threshold = 0.95f;
    cfg->kalman_q_high = 1e-3f;
    cfg->kalman_q_low  = 1e-6f;     /* Matches Python AecConfig.kalman_q_low default */
}

void aec_config_from_preset(AecConfig* cfg, AecPreset p, int sr) {
    aec_config_defaults(cfg, sr);
    switch (p) {
        case AEC_PRESET_MILD:
            /* v3.8.3: shifted one slot lighter — minimum-touch RES for
             * quiet/light-echo cases where NE intelligibility trumps echo
             * cleanup. Former v3.8.2 MILD values now live in SOFT. */
            cfg->res_alpha_echo_psd = 0.6f; cfg->res_alpha_error_psd = 0.6f;
            cfg->res_enr_scale = 1.15f;
            cfg->res_g_min_db = -25.0f;
            cfg->res_over_sub_base = 1.5f; cfg->res_over_sub_scale = 2.5f;
            cfg->res_dt_reduction = 4.5f;
            cfg->res_spectral_floor_db = -18.0f;
            cfg->res_ne_protect_db = -7.0f;
            cfg->res_reverb_decay = 0.45f; cfg->res_reverb_gain = 0.4f;
            cfg->enable_cng = 1;
            cfg->shadow_q_ratio = 3.0f; cfg->shadow_mu_min = 0.5f;
            cfg->kalman_q_high = 1.5e-3f;
            break;
        case AEC_PRESET_SOFT:
            /* = former v3.8.2 MILD. Shifted one slot to make room for an
             * even lighter MILD; preserved for users who liked the v3.8.2
             * MILD positioning (light RES with audible echo cleanup). */
            cfg->res_alpha_echo_psd = 0.5f; cfg->res_alpha_error_psd = 0.6f;
            cfg->res_enr_scale = 1.0f;
            cfg->res_g_min_db = -35.0f;
            cfg->res_over_sub_base = 2.5f; cfg->res_over_sub_scale = 4.0f;
            cfg->res_dt_reduction = 3.5f;
            cfg->res_spectral_floor_db = -25.0f;
            cfg->res_ne_protect_db = -10.0f;
            cfg->res_reverb_decay = 0.6f; cfg->res_reverb_gain = 0.8f;
            cfg->enable_cng = 1;
            cfg->shadow_q_ratio = 3.0f; cfg->shadow_mu_min = 0.5f;
            cfg->kalman_q_high = 1.5e-3f;
            break;
        case AEC_PRESET_BALANCED:
            cfg->res_alpha_echo_psd = 0.4f; cfg->res_alpha_error_psd = 0.5f;
            cfg->res_enr_scale = 0.85f;
            cfg->res_g_min_db = -55.0f;
            cfg->res_over_sub_base = 5.0f; cfg->res_over_sub_scale = 9.0f;
            cfg->res_dt_reduction = 2.5f;
            cfg->res_spectral_floor_db = -38.0f;
            cfg->res_ne_protect_db = -16.0f;
            cfg->res_reverb_decay = 0.85f;
            cfg->res_reverb_gain  = 1.6f;
            cfg->enable_cng = 1;
            cfg->shadow_q_ratio = 3.5f; cfg->shadow_mu_min = 0.6f;
            cfg->kalman_q_high = 1e-3f;
            break;
        case AEC_PRESET_AGGRESSIVE:
            cfg->res_alpha_echo_psd = 0.3f; cfg->res_alpha_error_psd = 0.4f;
            cfg->res_enr_scale = 0.7f;
            cfg->res_g_min_db = -65.0f;
            cfg->res_over_sub_base = 7.0f; cfg->res_over_sub_scale = 12.0f;
            cfg->res_dt_reduction = 1.5f;
            cfg->res_spectral_floor_db = -45.0f;
            cfg->res_ne_protect_db = -22.0f;
            cfg->res_reverb_decay = 0.7f; cfg->res_reverb_gain = 2.0f;
            cfg->enable_cng = 1;
            cfg->shadow_q_ratio = 4.0f; cfg->shadow_mu_min = 0.7f;
            cfg->kalman_q_high = 7e-4f;
            break;
        case AEC_PRESET_MAXIMUM:
            cfg->res_alpha_echo_psd = 0.2f; cfg->res_alpha_error_psd = 0.3f;
            cfg->res_enr_scale = 0.5f;
            cfg->res_g_min_db = -72.0f;
            cfg->res_over_sub_base = 10.0f; cfg->res_over_sub_scale = 15.0f;
            cfg->res_dt_reduction = 0.5f;
            cfg->res_spectral_floor_db = -52.0f;
            cfg->res_ne_protect_db = -28.0f;
            cfg->res_reverb_decay = 0.8f; cfg->res_reverb_gain = 3.0f;
            cfg->enable_cng = 1;
            cfg->shadow_q_ratio = 4.5f; cfg->shadow_mu_min = 0.75f;
            cfg->kalman_q_high = 5e-4f;
            break;
    }
}

static int next_pow2(int x) { int n=1; while(n<x) n<<=1; return n; }

/* Compute derived sizes shared by aec_create / aec_init / aec_get_mem_size. */
static void aec_derive_sizes(const AecConfig* cfg, int* out_hop, int* out_blk,
                              int* out_fft, int* out_K, int* out_n_parts,
                              int* out_ring_size) {
    int hop = (int)(0.010f * cfg->sample_rate);
    int blk = 2 * hop;
    int fft = next_pow2(blk);
    int K   = fft / 2 + 1;
    int n_parts = cfg->n_partitions;
    if (n_parts <= 0) {
        float filter_len_samp = cfg->filter_length_ms * cfg->sample_rate / 1000.0f;
        n_parts = (int)ceilf(filter_len_samp / (float)hop);
        if (n_parts < 1) n_parts = 1;
    }
    int ring = 0;
    if (cfg->enable_delay_est) {
        /* v3.10.0+: ring sized to delay_buffer_ms (default 2048 ms in v3.10.4),
         * separate from max_delay_ms. Fall back to legacy max_delay_samp+4096
         * if delay_buffer_ms unset (== 0). */
        if (cfg->delay_buffer_ms > 0.0f) {
            ring = (int)(cfg->delay_buffer_ms * cfg->sample_rate / 1000.0f);
        } else {
            int max_delay_samp = (int)(cfg->max_delay_ms * cfg->sample_rate / 1000.0f);
            ring = max_delay_samp + 4096;
        }
    }
    if (out_hop) *out_hop = hop;
    if (out_blk) *out_blk = blk;
    if (out_fft) *out_fft = fft;
    if (out_K) *out_K = K;
    if (out_n_parts) *out_n_parts = n_parts;
    if (out_ring_size) *out_ring_size = ring;
}

/* Build ResFilterConfig the way aec_create does. Used by both init paths. */
static void aec_build_res_cfg(const Aec* a, const AecConfig* cfg,
                               ResFilterConfig* rcfg) {
    memset(rcfg, 0, sizeof(*rcfg));
    rcfg->block_size = a->fft_size;
    rcfg->n_freqs    = a->n_freqs;
    rcfg->frame_size = a->block_size;
    rcfg->hop_size   = a->hop_size;
    rcfg->sample_rate = cfg->sample_rate;
    rcfg->g_min_db = cfg->res_g_min_db;
    rcfg->over_sub = cfg->res_over_sub_base;
    rcfg->alpha    = 0.8f;
    rcfg->enable_cng = cfg->enable_cng;
    rcfg->max_drop_db_per_frame = 6.0f;
    rcfg->max_rise_db_per_frame = 6.0f;
    rcfg->enable_spectral_floor = 1;
    rcfg->spectral_floor_db = cfg->res_spectral_floor_db;
    rcfg->ne_protect_db = cfg->res_ne_protect_db;
    rcfg->enable_reverb = 1;
    rcfg->reverb_decay = cfg->res_reverb_decay;
    rcfg->reverb_gain  = cfg->res_reverb_gain;
    rcfg->alpha_echo_psd  = cfg->res_alpha_echo_psd;
    rcfg->alpha_error_psd = cfg->res_alpha_error_psd;
    rcfg->enr_scale       = cfg->res_enr_scale;
    rcfg->startup_dt_min_ne_scale = 1.0f;
    rcfg->startup_dt_gain_floor = 1.0f;
    rcfg->startup_dt_noise_floor_scale = 1.0f;
    rcfg->render_min_ne_factor = 0.5f;
    rcfg->render_dt_gain_ceil  = 0.6f;
}

/* Set scalar runtime state. Called after buffers are populated. */
static void aec_init_scalar_state(Aec* a, const AecConfig* cfg) {
    a->saturation_level = 0.0;
    a->erl_estimate = 0.1;
    a->main_err_smooth = 0.0;
    a->shadow_err_smooth = 0.0;
    a->shadow_frame_count = 0;
    a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10;
    a->erle_window_err = 1e-10;
    a->erle_factor_prev = 0.0;
    a->inst_erle_smooth = 1.0;
    a->wn_err_baseline = 1e-8;
    a->stat_dt_hangover = 0;
    a->warmup_frames_remaining = cfg->warmup_frames;
    a->warmup_far_active = 0;
    a->simple_mu_ratio = 1.0;
    a->simple_mu_holdoff = 0;
    a->has_per_bin_mu = 0;
    a->limiter_gain = 1.0;
    a->alpha_pow = 0.95;
    a->near_power = 0.0; a->raw_error_power = 0.0; a->final_error_power = 0.0;
    a->near_power_sum = 0.0; a->raw_error_power_sum = 0.0;
    a->final_error_power_sum = 0.0;
    a->frame_count = 0;
}

int aec_create(Aec* a, const AecConfig* cfg) {
    memset(a, 0, sizeof(*a));
    a->cfg = *cfg;
    aec_derive_sizes(cfg, &a->hop_size, &a->block_size, &a->fft_size,
                     &a->n_freqs, NULL, NULL);

    int n_parts;
    aec_derive_sizes(cfg, NULL, NULL, NULL, NULL, &n_parts, NULL);

    if (cfg->enable_highpass) {
        hpf_init(&a->hp_mic, cfg->highpass_cutoff_hz, cfg->sample_rate);
        hpf_init(&a->hp_ref, cfg->highpass_cutoff_hz, cfg->sample_rate);
        a->has_hp = 1;
    }
    if (cfg->enable_saturation) {
        saturation_init(&a->sat_ref, cfg->saturation_threshold);
        saturation_init(&a->sat_mic, cfg->saturation_threshold);
        a->has_sat = 1;
    }
    if (cfg->enable_delay_est) {
        delay_est_init(&a->delay_est, cfg->sample_rate, cfg->max_delay_ms,
                          cfg->delay_est_init_s, cfg->delay_est_period_s);
        a->has_delay_est = 1;
        /* v3.10.0+: ring sized to delay_buffer_ms (v3.10.4: 2048 ms). */
        if (cfg->delay_buffer_ms > 0.0f) {
            a->ref_ring_size = (int)(cfg->delay_buffer_ms * cfg->sample_rate / 1000.0f);
        } else {
            int max_delay_samp = (int)(cfg->max_delay_ms * cfg->sample_rate / 1000.0f);
            a->ref_ring_size = max_delay_samp + 4096;
        }
        a->ref_ring = (float*)calloc((size_t)a->ref_ring_size, sizeof(float));
        a->current_delay = -1;
        a->pending_delay = -1;
        a->has_pending = 0;
        a->pending_delay_ttl = 0;
    } else {
        a->current_delay = 0;
    }

    pbfdkf_init(&a->main_filter, a->block_size, n_parts, cfg->mu, cfg->delta, a->hop_size);
    /* Apply preset Q_high + Q_low (matches Python aec.py:3138-3141) */
    for (int k = 0; k < a->n_freqs; ++k) {
        a->main_filter.Q_high[k] = cfg->kalman_q_high;
        a->main_filter.Q_low[k]  = cfg->kalman_q_low;
        a->main_filter.Q[k]      = cfg->kalman_q_high;
    }

    if (cfg->enable_shadow_filter) {
        pbfdkf_init(&a->shadow_filter, a->block_size, n_parts, cfg->mu, cfg->delta, a->hop_size);
        /* Python aec.py:3286-3288:
         *   shadow.Q_high = main.Q_high * shadow_q_ratio
         *   shadow.Q_low  = main.Q_low  * shadow_q_ratio
         *   shadow.Q      = shadow.Q_high
         */
        for (int k = 0; k < a->n_freqs; ++k) {
            a->shadow_filter.Q_high[k] = cfg->kalman_q_high * cfg->shadow_q_ratio;
            a->shadow_filter.Q_low[k]  = cfg->kalman_q_low  * cfg->shadow_q_ratio;
            a->shadow_filter.Q[k]      = a->shadow_filter.Q_high[k];
        }
        a->has_shadow = 1;
    }

    render_activity_init(&a->render_activity);
    filter_convergence_init(&a->convergence);
    doubletalk_init(&a->dt_analyzer, cfg->shadow_dtd_offset, cfg->shadow_dtd_advantage_scale);
    epc_init(&a->epc, cfg->epc_hangover, cfg->epc_total_rise, cfg->epc_delta_threshold);
    shadow_copy_init(&a->shadow_copy, SC_GATE_ENERGY,
                        cfg->shadow_copy_threshold, cfg->shadow_copy_hysteresis,
                        cfg->epc_hangover);
    filter_plateau_init(&a->plateau_detector);

    if (cfg->enable_residual_filter) {
        ResFilterConfig rcfg;
        aec_build_res_cfg(a, cfg, &rcfg);
        res_filter_init(&a->res, &rcfg);
        a->has_res = 1;
    }

    aec_init_scalar_state(a, cfg);
    a->per_bin_mu_scale = (float*)malloc((size_t)a->n_freqs * sizeof(float));
    a->near_hop = (float*)malloc((size_t)a->hop_size * sizeof(float));
    a->far_hop  = (float*)malloc((size_t)a->hop_size * sizeof(float));
    a->raw_output = (float*)malloc((size_t)a->hop_size * sizeof(float));
    a->res_output = (float*)malloc((size_t)a->hop_size * sizeof(float));
    a->is_static = 0;
    return 0;
}

size_t aec_get_mem_size(const AecConfig* cfg) {
    if (!cfg) return 0;
    int hop, blk, fft, K, n_parts, ring;
    aec_derive_sizes(cfg, &hop, &blk, &fft, &K, &n_parts, &ring);

    size_t total = 0;
    if (cfg->enable_delay_est) {
        total += ALIGN16((size_t)ring * sizeof(float));               /* ref_ring */
        total += delay_est_get_mem_size(cfg->sample_rate, cfg->max_delay_ms);
    }
    total += pbfdkf_get_mem_size(blk, n_parts, hop);                  /* main_filter */
    if (cfg->enable_shadow_filter) {
        total += pbfdkf_get_mem_size(blk, n_parts, hop);              /* shadow_filter */
    }
    if (cfg->enable_residual_filter) {
        Aec tmp = {0};
        tmp.hop_size = hop; tmp.block_size = blk;
        tmp.fft_size = fft; tmp.n_freqs = K;
        ResFilterConfig rcfg;
        aec_build_res_cfg(&tmp, cfg, &rcfg);
        total += res_filter_get_mem_size(&rcfg);
    }
    total += ALIGN16((size_t)K * sizeof(float));                      /* per_bin_mu_scale */
    total += ALIGN16((size_t)hop * sizeof(float));                    /* near_hop */
    total += ALIGN16((size_t)hop * sizeof(float));                    /* far_hop */
    total += ALIGN16((size_t)hop * sizeof(float));                    /* raw_output */
    total += ALIGN16((size_t)hop * sizeof(float));                    /* res_output */
    return total;
}

int aec_init(Aec* a, void* mem, size_t mem_size, const AecConfig* cfg) {
    if (!a || !mem || !cfg) return -1;
    size_t needed = aec_get_mem_size(cfg);
    if (mem_size < needed) return -1;
    AEC_DEBUG_LOG(1, "Init", "static-mem pool=%zu bytes (%.1f KB) sr=%d "
                              "hop=%d preset_q=%g cng=%d",
                  needed, (double)needed / 1024.0,
                  cfg->sample_rate, (int)(0.010f * cfg->sample_rate),
                  (double)cfg->kalman_q_high, cfg->enable_cng);

    memset(a, 0, sizeof(*a));
    a->cfg = *cfg;
    int hop, blk, fft, K, n_parts, ring;
    aec_derive_sizes(cfg, &hop, &blk, &fft, &K, &n_parts, &ring);
    a->hop_size = hop; a->block_size = blk;
    a->fft_size = fft; a->n_freqs = K;

    /* Value-typed sub-modules (no buffer needed) */
    if (cfg->enable_highpass) {
        hpf_init(&a->hp_mic, cfg->highpass_cutoff_hz, cfg->sample_rate);
        hpf_init(&a->hp_ref, cfg->highpass_cutoff_hz, cfg->sample_rate);
        a->has_hp = 1;
    }
    if (cfg->enable_saturation) {
        saturation_init(&a->sat_ref, cfg->saturation_threshold);
        saturation_init(&a->sat_mic, cfg->saturation_threshold);
        a->has_sat = 1;
    }
    render_activity_init(&a->render_activity);
    filter_convergence_init(&a->convergence);
    doubletalk_init(&a->dt_analyzer, cfg->shadow_dtd_offset, cfg->shadow_dtd_advantage_scale);
    epc_init(&a->epc, cfg->epc_hangover, cfg->epc_total_rise, cfg->epc_delta_threshold);
    shadow_copy_init(&a->shadow_copy, SC_GATE_ENERGY,
                     cfg->shadow_copy_threshold, cfg->shadow_copy_hysteresis,
                     cfg->epc_hangover);
    filter_plateau_init(&a->plateau_detector);

    /* Buffer-backed sub-modules — slice from caller's mem in declared order. */
    uint8_t* ptr = (uint8_t*)mem;

    if (cfg->enable_delay_est) {
        a->ref_ring = (float*)ptr;
        ptr += ALIGN16((size_t)ring * sizeof(float));
        memset(a->ref_ring, 0, (size_t)ring * sizeof(float));
        a->ref_ring_size = ring;
        size_t de_sz = delay_est_get_mem_size(cfg->sample_rate, cfg->max_delay_ms);
        delay_est_init_static(&a->delay_est, ptr, de_sz, cfg->sample_rate,
                               cfg->max_delay_ms, cfg->delay_est_init_s,
                               cfg->delay_est_period_s);
        ptr += de_sz;
        a->has_delay_est = 1;
        a->current_delay = -1;
        a->pending_delay = -1;
        a->has_pending = 0;
        a->pending_delay_ttl = 0;
    } else {
        a->current_delay = 0;
    }

    size_t mf_sz = pbfdkf_get_mem_size(blk, n_parts, hop);
    pbfdkf_init_static(&a->main_filter, ptr, mf_sz,
                        blk, n_parts, cfg->mu, cfg->delta, hop);
    ptr += mf_sz;
    for (int k = 0; k < K; ++k) {
        a->main_filter.Q_high[k] = cfg->kalman_q_high;
        a->main_filter.Q_low[k]  = cfg->kalman_q_low;
        a->main_filter.Q[k]      = cfg->kalman_q_high;
    }

    if (cfg->enable_shadow_filter) {
        size_t sf_sz = pbfdkf_get_mem_size(blk, n_parts, hop);
        pbfdkf_init_static(&a->shadow_filter, ptr, sf_sz,
                            blk, n_parts, cfg->mu, cfg->delta, hop);
        ptr += sf_sz;
        for (int k = 0; k < K; ++k) {
            a->shadow_filter.Q_high[k] = cfg->kalman_q_high * cfg->shadow_q_ratio;
            a->shadow_filter.Q_low[k]  = cfg->kalman_q_low  * cfg->shadow_q_ratio;
            a->shadow_filter.Q[k]      = a->shadow_filter.Q_high[k];
        }
        a->has_shadow = 1;
    }

    if (cfg->enable_residual_filter) {
        ResFilterConfig rcfg;
        aec_build_res_cfg(a, cfg, &rcfg);
        size_t rf_sz = res_filter_get_mem_size(&rcfg);
        res_filter_init_static(&a->res, ptr, rf_sz, &rcfg);
        ptr += rf_sz;
        a->has_res = 1;
    }

    aec_init_scalar_state(a, cfg);
    a->per_bin_mu_scale = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->near_hop  = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->far_hop   = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->raw_output = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->res_output = (float*)ptr; /* last */
    memset(a->per_bin_mu_scale, 0, (size_t)K   * sizeof(float));
    memset(a->near_hop,  0, (size_t)hop * sizeof(float));
    memset(a->far_hop,   0, (size_t)hop * sizeof(float));
    memset(a->raw_output, 0, (size_t)hop * sizeof(float));
    memset(a->res_output, 0, (size_t)hop * sizeof(float));
    a->is_static = 1;
    return 0;
}

void aec_destroy(Aec* a) {
    if (!a) return;
    if (a->is_static) {
        AEC_DEBUG_LOG(1, "Init", "destroy: static path (no free; caller owns pool)");
        if (a->has_res) res_filter_free(&a->res);
        pbfdkf_free(&a->main_filter);
        if (a->has_shadow) pbfdkf_free(&a->shadow_filter);
        if (a->has_delay_est) delay_est_free(&a->delay_est);
        return;
    }
    AEC_DEBUG_LOG(1, "Init", "destroy: heap path (freeing buffers)");
    if (a->has_res) res_filter_free(&a->res);
    pbfdkf_free(&a->main_filter);
    if (a->has_shadow) pbfdkf_free(&a->shadow_filter);
    if (a->has_delay_est) { delay_est_free(&a->delay_est); free(a->ref_ring); }
    free(a->per_bin_mu_scale);
    free(a->near_hop); free(a->far_hop);
    free(a->raw_output); free(a->res_output);
}

int aec_hop_size(const Aec* a) { return a->hop_size; }

void aec_reset(Aec* a) {
    if (a->has_hp) { hpf_reset(&a->hp_mic); hpf_reset(&a->hp_ref); }
    if (a->has_sat) { saturation_reset(&a->sat_ref); saturation_reset(&a->sat_mic); }
    if (a->has_delay_est) {
        delay_est_reset(&a->delay_est);
        memset(a->ref_ring, 0, (size_t)a->ref_ring_size * sizeof(float));
        a->ref_ring_write = 0; a->ref_ring_filled = 0;
        a->current_delay = -1; a->pending_delay = -1; a->has_pending = 0;
        a->pending_delay_ttl = 0;
    }
    filter_plateau_reset(&a->plateau_detector);
    pbfdkf_reset(&a->main_filter);
    if (a->has_shadow) pbfdkf_reset(&a->shadow_filter);
    render_activity_reset(&a->render_activity);
    filter_convergence_reset(&a->convergence);
    doubletalk_reset(&a->dt_analyzer);
    epc_reset(&a->epc);
    shadow_copy_reset(&a->shadow_copy);
    if (a->has_res) res_filter_reset(&a->res);
    a->saturation_level = 0.0;
    a->erl_estimate = 0.1;
    a->main_err_smooth = 0.0; a->shadow_err_smooth = 0.0;
    a->shadow_frame_count = 0;
    a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10; a->erle_window_err = 1e-10;
    a->erle_factor_prev = 0.0;
    a->inst_erle_smooth = 1.0;
    a->wn_err_baseline = 1e-8; a->stat_dt_hangover = 0;
    a->warmup_frames_remaining = a->cfg.warmup_frames;
    a->warmup_far_active = 0;
    a->simple_mu_ratio = 1.0; a->simple_mu_holdoff = 0; a->has_per_bin_mu = 0;
    a->limiter_gain = 1.0;
    a->near_power = 0.0; a->raw_error_power = 0.0; a->final_error_power = 0.0;
    a->near_power_sum = 0.0; a->raw_error_power_sum = 0.0; a->final_error_power_sum = 0.0;
    a->frame_count = 0;
}

/* ---- Helpers ------------------------------------------------------------ */

/* v3.10.2 — clear all filter-output-derived state. Called by:
 *   - 'plateau' (FilterPlateauDetector fire)
 *   - 'delay_first' (Path A: first delay acquisition)
 *   - 'delay_shift' (Path B: delay shift confirmed)
 *
 * Preserves input-side / temporal context (frame_count, _current_delay,
 * delay_est, ref_ring, plateau_detector, far/mic power EMAs, hp/sat
 * detectors, render activity, RES long-window EMA when preserve_render_ema).
 *
 * Re-arms warmup: filter Q boosted to Q_high, p_max_override = 1.0 / 30 frames,
 * warmup_frames = max(current, cfg.warmup_frames/2), warmup_far_active = 0,
 * pending_delay cleared.
 *
 * Mirrors python/aec.py:3961-4115 _reset_filter_derived_state.
 */
static void aec_reset_filter_derived_state(Aec* a, int preserve_render_ema) {
    /* Filter taps + shadow */
    pbfdkf_reset(&a->main_filter);
    if (a->has_shadow) pbfdkf_reset(&a->shadow_filter);

    /* Filter convergence + EPC */
    filter_convergence_reset(&a->convergence);
    epc_reset(&a->epc);

    /* Filter-output-derived: err smooths, error/near power EMAs, ERLE windows */
    a->main_err_smooth = 0.0;
    a->shadow_err_smooth = 0.0;
    a->raw_error_power = 0.0;
    a->final_error_power = 0.0;
    a->raw_error_power_sum = 0.0;
    a->final_error_power_sum = 0.0;
    a->near_power = 0.0;
    a->near_power_sum = 0.0;
    a->erle_window_near = 1e-10;
    a->erle_window_err  = 1e-10;
    a->erle_factor_prev = 0.0;
    a->inst_erle_smooth = 1.0;

    /* Mu-scale state */
    a->simple_mu_ratio = 1.0;
    a->simple_mu_holdoff = 0;
    a->has_per_bin_mu = 0;

    /* ERL + EPC-render-forced + DT analyzer */
    a->erl_estimate = 0.1;
    a->epc_render_forced_remaining = 0;
    doubletalk_reset(&a->dt_analyzer);
    a->stat_dt_hangover = 0;

    /* Shadow copy controller + frame count */
    a->shadow_frame_count = 0;
    shadow_copy_reset(&a->shadow_copy);

    /* RES — clears its echo_psd / error_psd / noise_psd / gain_smooth /
     * render state. Long-window far-PSD EMA preserved when
     * preserve_render_ema=1. */
    if (a->has_res) res_filter_reset_ex(&a->res, preserve_render_ema);

    /* Re-arm warmup with high mu + Q_high boost. */
    int K = a->main_filter.base.n_freqs;
    for (int k = 0; k < K; ++k) {
        a->main_filter.Q[k] = a->main_filter.Q_high[k];
        if (a->has_shadow) a->shadow_filter.Q[k] = a->shadow_filter.Q_high[k];
    }
    a->main_filter.p_max_override = 1.0f;
    a->main_filter.p_max_override_frames = 30;
    if (a->has_shadow) {
        a->shadow_filter.p_max_override = 1.0f;
        a->shadow_filter.p_max_override_frames = 30;
    }
    int half = a->cfg.warmup_frames / 2;
    if (a->warmup_frames_remaining < half) a->warmup_frames_remaining = half;
    a->warmup_far_active = 0;

    /* Clear pending delay shift state. */
    a->pending_delay = -1;
    a->has_pending = 0;
    a->pending_delay_ttl = 0;
}

/* Returns nearest mu_scale; if per-bin available, use it. */
static double get_simple_mu_scale_scalar(Aec* a, double mu_min) {
    if (a->warmup_frames_remaining > 0) {
        if (a->warmup_far_active) a->warmup_frames_remaining--;
        if (a->simple_mu_ratio < 0.2) {
            return (a->simple_mu_ratio > 0.2) ? a->simple_mu_ratio : 0.2;
        }
        double v = a->simple_mu_ratio + 0.2;
        if (v < 0.5) v = 0.5;
        if (v > 1.0) v = 1.0;
        return v;
    }
    /* mu_min adjustment based on convergence */
    double mm = mu_min;
    if (!a->convergence.converged) {
        if (mm < 0.3) mm = 0.3;
    } else {
        if (mm < 0.2) mm = 0.2;
    }
    return mm + (1.0 - mm) * a->simple_mu_ratio;
}

static double mean_sq(const float* x, int n);   /* forward decl */

static void update_simple_mu_ratio(Aec* a, const float* output,
                                   const float* far_end, int n) {
    /* error_power, far_power: Python np.mean(arr**2) — fp32 square then mean.
     * Use shared mean_sq helper for parity. */
    double error_power = mean_sq(output, n)  + 1e-10;
    double far_power   = mean_sq(far_end, n) + 1e-10;
    if (far_power < 1e-6 && error_power < 1e-6) return;
    if (far_power > 1e-4 && a->simple_mu_ratio < 0.1) {
        a->simple_mu_ratio = 0.8;
        a->simple_mu_holdoff = 0;
        return;
    }
    double ratio = far_power / error_power;
    if (ratio > 1.0) ratio = 1.0;
    /* echo_est_pwr / near_pwr ratio */
    int K = a->main_filter.base.n_freqs;
    double echo_est_pwr = 0.0, near_pwr = 0.0;
    for (int k = 0; k < K; ++k) {
        double er = a->main_filter.base.echo_spec[k].r;
        double ei = a->main_filter.base.echo_spec[k].i;
        echo_est_pwr += er * er + ei * ei;
        double nr = a->main_filter.base.near_spec[k].r;
        double ni = a->main_filter.base.near_spec[k].i;
        near_pwr += nr * nr + ni * ni;
    }
    echo_est_pwr += 1e-10;
    near_pwr     += 1e-10;
    if (near_pwr > 1e-8) {
        double r2 = echo_est_pwr / near_pwr;
        if (r2 < 0.0) r2 = 0.0; if (r2 > 1.0) r2 = 1.0;
        double r2_half = r2 * 0.5;
        if (r2_half > ratio) ratio = r2_half;
    }
    double alpha;
    if (ratio < a->simple_mu_ratio) {
        alpha = 0.3;
        a->simple_mu_holdoff = 20;
    } else if (a->simple_mu_holdoff > 0) {
        a->simple_mu_holdoff--;
        alpha = 0.99;
    } else {
        alpha = 0.95;
    }
    a->simple_mu_ratio = alpha * a->simple_mu_ratio + (1.0 - alpha) * ratio;
}

static double mean_sq(const float* x, int n) {
    /* Mirror Python: (x ** 2) squares in fp32, np.mean accumulates fp32
     * with pairwise summation, returns fp32, then float() promotes to fp64.
     * Use 8-block pairwise. */
    enum { BLK = 8 };
    int n_blocks = (n + BLK - 1) / BLK;
    float blocks[1024];
    for (int b = 0; b < n_blocks; ++b) {
        int s_idx = b * BLK; int e_idx = s_idx + BLK; if (e_idx > n) e_idx = n;
        float s = 0.0f;
        for (int i = s_idx; i < e_idx; ++i) {
            float sq = x[i] * x[i];
            s += sq;
        }
        blocks[b] = s;
    }
    int len = n_blocks;
    while (len > 1) {
        int half = len / 2;
        for (int i = 0; i < half; ++i) blocks[i] = blocks[2*i] + blocks[2*i+1];
        if (len & 1) { blocks[half] = blocks[len - 1]; half++; }
        len = half;
    }
    float mean_f32 = blocks[0] / (float)n;
    return (double)mean_f32;
}

void aec_process(Aec* a, const float* mic_in, const float* ref_in, float* out) {
    const int hop = a->hop_size;
    /* Copy inputs (Python copies near_end/far_end before HPF in-place modifies) */
    memcpy(a->near_hop, mic_in, (size_t)hop * sizeof(float));
    memcpy(a->far_hop,  ref_in, (size_t)hop * sizeof(float));

    /* HPF */
    if (a->has_hp) {
        hpf_process(&a->hp_mic, a->near_hop, a->near_hop, hop);
        hpf_process(&a->hp_ref, a->far_hop,  a->far_hop,  hop);
    }

    /* Saturation */
    if (a->has_sat) {
        double sat_ref = saturation_detect(&a->sat_ref, a->far_hop, hop);
        double sat_mic = saturation_detect(&a->sat_mic, a->near_hop, hop);
        a->saturation_level = (sat_ref > sat_mic * 0.5) ? sat_ref : sat_mic * 0.5;
        if (a->cfg.saturation_softclip_ref && sat_ref > 0.1) {
            saturation_soft_clip(a->far_hop, a->far_hop, hop, 0.8);
        }
    }

    /* Delay estimation + ring buffer */
    if (a->has_delay_est) {
        delay_est_accumulate(&a->delay_est, a->near_hop, a->far_hop, hop);
        int new_delay = a->delay_est.estimated_delay;
        int eligible = (new_delay >= 0 && a->delay_est.n_updates >= 3);

        /* Path A (v3.10.2/v3.10.3) — first acquisition. Demands solid PAR. */
        if (eligible && a->current_delay < 0 && delay_est_is_solid(&a->delay_est)) {
            a->current_delay = new_delay;
            aec_reset_filter_derived_state(a, /*preserve_render_ema=*/1);
            filter_convergence_mark_diverged(&a->convergence);
        }

        /* Pending TTL decrement — runs every estimation cycle. */
        if (a->pending_delay_ttl > 0) {
            a->pending_delay_ttl--;
            if (a->pending_delay_ttl <= 0) {
                a->has_pending = 0;
                a->pending_delay = -1;
            }
        }

        /* Path B — delay shift. Independent of Path A. Confidence >= 0.5
         * + |new - current| > 32. Pair with prior pending value (|<16|) → fire. */
        double conf = delay_est_confidence(&a->delay_est);
        if (eligible && a->current_delay >= 0 && conf >= 0.5
                && abs(new_delay - a->current_delay) > 32) {
            if (a->has_pending && abs(new_delay - a->pending_delay) < 16) {
                a->current_delay = new_delay;
                a->has_pending = 0; a->pending_delay = -1; a->pending_delay_ttl = 0;
                aec_reset_filter_derived_state(a, /*preserve_render_ema=*/1);
                epc_force_delay(&a->epc);
                /* Q + p_max boost (v3.10.3 M4) */
                for (int k = 0; k < a->main_filter.base.n_freqs; ++k) {
                    a->main_filter.Q[k] = a->main_filter.Q_high[k];
                    if (a->has_shadow) a->shadow_filter.Q[k] = a->shadow_filter.Q_high[k];
                }
                a->main_filter.p_max_override = 1.0f;
                a->main_filter.p_max_override_frames = 30;
                if (a->has_shadow) {
                    a->shadow_filter.p_max_override = 1.0f;
                    a->shadow_filter.p_max_override_frames = 30;
                }
                filter_convergence_mark_diverged(&a->convergence);
            } else {
                a->pending_delay = new_delay;
                a->has_pending = 1;
                a->pending_delay_ttl = 3;
            }
        }

        /* Write far_hop into ring */
        int w = a->ref_ring_write;
        int rs = a->ref_ring_size;
        if (w + hop <= rs) {
            memcpy(a->ref_ring + w, a->far_hop, (size_t)hop * sizeof(float));
        } else {
            int p1 = rs - w;
            memcpy(a->ref_ring + w, a->far_hop, (size_t)p1 * sizeof(float));
            memcpy(a->ref_ring, a->far_hop + p1, (size_t)(hop - p1) * sizeof(float));
        }
        a->ref_ring_write = (w + hop) % rs;
        a->ref_ring_filled += hop;

        /* Apply delay compensation */
        if (a->current_delay > 0 && a->ref_ring_filled >= a->current_delay + hop) {
            int d = a->current_delay;
            int read_pos = (a->ref_ring_write - hop - d) % rs;
            if (read_pos < 0) read_pos += rs;
            if (read_pos + hop <= rs) {
                memcpy(a->far_hop, a->ref_ring + read_pos, (size_t)hop * sizeof(float));
            } else {
                int p1 = rs - read_pos;
                memcpy(a->far_hop, a->ref_ring + read_pos, (size_t)p1 * sizeof(float));
                memcpy(a->far_hop + p1, a->ref_ring, (size_t)(hop - p1) * sizeof(float));
            }
        }
    }

    /* Render activity */
    RenderActivityResult ra = render_activity_update(&a->render_activity, a->far_hop, hop);
    a->warmup_far_active = ra.warmup_active;
    double far_pwr_global = ra.far_pwr;

    /* mu_scale (no DTD for BALANCED) */
    double mu_scalar = get_simple_mu_scale_scalar(a, a->cfg.shadow_mu_min);
    const float* mu_arr = NULL;
    if (a->has_per_bin_mu) {
        /* Apply mu_min floor element-wise */
        int K = a->main_filter.base.n_freqs;
        double mm = a->convergence.converged ? 0.2 : 0.3;
        if (a->cfg.shadow_mu_min > mm) mm = a->cfg.shadow_mu_min;
        for (int k = 0; k < K; ++k) {
            float v = a->per_bin_mu_scale[k];
            if (v < (float)mm) v = (float)mm;
            a->per_bin_mu_scale[k] = v;
        }
        mu_arr = a->per_bin_mu_scale;
    }

    /* startup_dt mu floor */
    double far_now = mean_sq(a->far_hop, hop);
    double last_eff_dt = a->has_res ? a->res.last_effective_dt : 0.0;
    if (a->cfg.startup_dt_mu_min > 0.0f && a->has_res &&
        last_eff_dt > 0.35 && far_now > 1e-4 &&
        !a->convergence.once_converged) {
        if (mu_arr) {
            int K = a->main_filter.base.n_freqs;
            for (int k = 0; k < K; ++k)
                if (a->per_bin_mu_scale[k] < a->cfg.startup_dt_mu_min)
                    a->per_bin_mu_scale[k] = a->cfg.startup_dt_mu_min;
        } else {
            if (mu_scalar < a->cfg.startup_dt_mu_min) mu_scalar = a->cfg.startup_dt_mu_min;
        }
    }
    /* Mic clipping emergency */
    if (a->has_sat) {
        if (a->sat_mic.saturation_level > 0.8) {
            mu_scalar = 0.0;
            mu_arr = NULL;
        }
    }
    /* Pause main when shadow_copy says */
    double main_mu = a->shadow_copy.main_paused ? 0.0 : mu_scalar;

    /* Main filter */
    pbfdkf_process(&a->main_filter, a->near_hop, a->far_hop,
                      mu_arr, (float)main_mu, a->raw_output);

    /* Shadow filter */
    int far_excited = (mean_sq(a->far_hop, hop) > 1e-4);
    int saturation_safe = (a->saturation_level < 0.5);
    float shadow_mu = (far_excited && saturation_safe) ? 1.0f : 0.1f;
    if (a->has_shadow) {
        a->shadow_frame_count++;
        float shadow_out[8192]; /* hop ≤ 8192 */
        pbfdkf_process(&a->shadow_filter, a->near_hop, a->far_hop,
                          NULL, shadow_mu, shadow_out);

        /* Error energies */
        int K = a->main_filter.base.n_freqs;
        double main_err = 0.0, shadow_err = 0.0;
        for (int k = 0; k < K; ++k) {
            double mr = a->main_filter.base.error_spec[k].r;
            double mi = a->main_filter.base.error_spec[k].i;
            main_err += mr * mr + mi * mi;
            double sr = a->shadow_filter.base.error_spec[k].r;
            double si = a->shadow_filter.base.error_spec[k].i;
            shadow_err += sr * sr + si * si;
        }
        double as = a->cfg.shadow_err_alpha, oas = 1.0 - as;
        a->main_err_smooth   = as * a->main_err_smooth   + oas * main_err;
        a->shadow_err_smooth = as * a->shadow_err_smooth + oas * shadow_err;

        doubletalk_update_shadow_dt(&a->dt_analyzer,
                                       a->shadow_frame_count, far_excited,
                                       a->main_err_smooth, a->shadow_err_smooth);

        /* v3.10.1: shadow-copy delay_reliable gate raised from any-confidence
         * to >= 0.5 (PAR halfway between par_low and par_solid). */
        int delay_reliable = a->has_delay_est
                          && (delay_est_confidence(&a->delay_est) >= 0.5);
        ShadowCopyDecision dec = shadow_copy_update(
            &a->shadow_copy, a->shadow_frame_count, far_now,
            a->main_err_smooth, a->shadow_err_smooth,
            a->epc.active, a->saturation_level,
            a->dt_analyzer.dt_from_energy, 0.0, delay_reliable);
        if (dec.boost_q) {
            int Kq = a->main_filter.base.n_freqs;
            for (int k = 0; k < Kq; ++k) a->main_filter.Q[k] = a->main_filter.Q_high[k];
            a->main_filter.p_max_override = 1.0f;
            a->main_filter.p_max_override_frames = 20;
        }
        if (dec.reverse_copy) {
            pbfdkf_copy_weights_from(&a->shadow_filter, &a->main_filter);
            a->shadow_err_smooth = a->main_err_smooth;
        }
    }

    /* EPC EPV trigger */
    EpcEvent epv = epc_update_epv(&a->epc, far_pwr_global,
                                       a->convergence.converged,
                                       a->shadow_copy.main_paused);
    if (epv.fired) {
        int K = a->main_filter.base.n_freqs;
        for (int k = 0; k < K; ++k) {
            a->main_filter.Q[k] = a->main_filter.Q_high[k];
            if (a->has_shadow) a->shadow_filter.Q[k] = a->shadow_filter.Q_high[k];
        }
        a->main_filter.p_max_override = 1.0f;
        a->main_filter.p_max_override_frames = 30;
        a->main_filter.p_floor_beta = 1.0f;
        a->main_filter.p_floor_beta_frames = 30;
        if (a->has_shadow) {
            a->shadow_filter.p_max_override = 1.0f;
            a->shadow_filter.p_max_override_frames = 30;
            a->shadow_filter.p_floor_beta = 1.0f;
            a->shadow_filter.p_floor_beta_frames = 30;
        }
        filter_convergence_mark_diverged(&a->convergence);
        a->epc_render_forced_remaining = a->cfg.epc_hangover;
        if (a->erl_estimate > 0.3) a->erl_estimate = 0.3;
    }

    /* EPC shadow_rise (only if shadow_filter and converged) */
    if (a->has_shadow && a->convergence.converged) {
        EpcEvent rise = epc_update_shadow_rise(&a->epc,
            a->main_err_smooth, a->shadow_err_smooth,
            a->render_activity.is_stationary);
        if (rise.fired) {
            int K = a->main_filter.base.n_freqs;
            for (int k = 0; k < K; ++k) {
                a->main_filter.Q[k] = a->main_filter.Q_high[k];
                a->shadow_filter.Q[k] = a->shadow_filter.Q_high[k];
            }
            filter_convergence_mark_diverged(&a->convergence);
            a->main_filter.p_max_override = 1.0f;
            a->main_filter.p_max_override_frames = 30;
            a->main_filter.p_floor_beta = 1.0f;
            a->main_filter.p_floor_beta_frames = 30;
            a->shadow_filter.p_max_override = 1.0f;
            a->shadow_filter.p_max_override_frames = 30;
            a->shadow_filter.p_floor_beta = 1.0f;
            a->shadow_filter.p_floor_beta_frames = 30;
            a->epc_render_forced_remaining = a->cfg.epc_hangover;
            if (a->erl_estimate > 0.3) a->erl_estimate = 0.3;
        } else {
            epc_tick_hangover(&a->epc);
        }
    }

    /* RES post-filter */
    float* final_out = a->raw_output;  /* default */
    double erle_factor = 0.0;
    int K = a->main_filter.base.n_freqs;
    /* State computation block: always runs (post-filter detectors,
     * per-frame ERLE / ERL / DT / divergence). RES-specific writes
     * (a->res.* and per-bin Kalman fine-tune from res.echo_psd) are
     * guarded individually below so the linear-only path
     * (enable_residual_filter=0) gets up-to-date AecResContext. */
    {
        double far_power = mean_sq(a->far_hop, hop);

        const double erle_decay = 0.999;
        a->erle_window_near = erle_decay * a->erle_window_near + a->near_power;
        a->erle_window_err  = erle_decay * a->erle_window_err  + a->raw_error_power;
        double erle_w = 10.0 * log10((a->erle_window_near + 1e-10)
                                   / (a->erle_window_err + 1e-10));
        double erle_inst;
        if (a->near_power < 1e-10 && a->raw_error_power < 1e-10) erle_inst = 0.0;
        else erle_inst = 10.0 * log10((a->near_power + 1e-10) / (a->raw_error_power + 1e-10));
        double erle_for_factor = erle_inst > erle_w ? erle_inst : erle_w;
        erle_factor = erle_for_factor / 10.0;
        if (erle_factor < 0.0) erle_factor = 0.0;
        if (erle_factor > 1.0) erle_factor = 1.0;
        a->erle_factor_prev = erle_factor;
        double base_over_sub = a->cfg.res_over_sub_base
                             + a->cfg.res_over_sub_scale * erle_factor;
        base_over_sub += a->saturation_level * a->cfg.saturation_over_sub_boost;

        double far_pwr = far_power + 1e-10;
        double mic_pwr = mean_sq(a->near_hop, hop) + 1e-10;
        double raw_err_pwr = mean_sq(a->raw_output, hop) + 1e-10;

        if (far_pwr > 1e-4) {
            double raw_dt_ratio = raw_err_pwr / (far_pwr + 1e-10);
            double inst_erl_raw = mic_pwr / far_pwr;
            if (raw_dt_ratio < 2.0 && inst_erl_raw < 1.5) {
                double inst_erl = inst_erl_raw;
                if (inst_erl < 0.001) inst_erl = 0.001;
                if (inst_erl > 1.0)   inst_erl = 1.0;
                double alpha_erl = a->convergence.converged ? 0.999 : 0.99;
                a->erl_estimate = alpha_erl * a->erl_estimate
                                + (1.0 - alpha_erl) * inst_erl;
            }
        }

        doubletalk_update_energy_dt(&a->dt_analyzer,
            a->render_activity.active_prev, far_pwr, mic_pwr, a->erl_estimate);

        double simple_dt = 1.0 - far_pwr / (mic_pwr + far_pwr);
        double raw_dt = a->dt_analyzer.dt_from_energy;
        double sd_half = simple_dt * 0.5;
        if (sd_half > raw_dt) raw_dt = sd_half;

        /* Stationary DT macro detection */
        int is_stationary_dt = 0;
        if (a->render_activity.is_stationary && a->convergence.converged) {
            int fft_size = a->main_filter.base.fft_size;
            double freq_per_bin = (double)a->cfg.sample_rate / (double)fft_size;
            int vb_start = (int)(100.0 / freq_per_bin); if (vb_start < 1) vb_start = 1;
            int vb_limit = (int)(3000.0 / freq_per_bin);
            if (vb_limit > K) vb_limit = K;
            double track_err_pwr = 0.0;
            for (int k = vb_start; k < vb_limit; ++k) {
                double er = a->main_filter.base.error_spec[k].r;
                double ei = a->main_filter.base.error_spec[k].i;
                track_err_pwr += er * er + ei * ei;
            }
            track_err_pwr += 1e-10;
            if (a->wn_err_baseline < 1e-6) a->wn_err_baseline = track_err_pwr;
            double jump_ratio = track_err_pwr / (a->wn_err_baseline + 1e-10);
            if (jump_ratio > 1.5) a->stat_dt_hangover = 80;
            if (a->stat_dt_hangover > 0) {
                is_stationary_dt = 1;
                a->stat_dt_hangover--;
                a->wn_err_baseline = 0.999 * a->wn_err_baseline + 0.001 * track_err_pwr;
            } else {
                a->wn_err_baseline = 0.95 * a->wn_err_baseline + 0.05 * track_err_pwr;
            }
        }
        if (a->convergence.converged && !a->render_activity.is_stationary
            && far_pwr > 1e-4 && a->wn_err_baseline > 1e-6) {
            a->wn_err_baseline = 0.995 * a->wn_err_baseline + 0.005 * raw_err_pwr;
        }

        /* inst_erle correction */
        double inst_erle_fast_raw = mic_pwr / raw_err_pwr;
        a->inst_erle_smooth = 0.7 * a->inst_erle_smooth + 0.3 * inst_erle_fast_raw;
        if (a->inst_erle_smooth > 2.0) {
            double erle_for_dt = a->inst_erle_smooth;
            if (erle_for_dt > 4.0) erle_for_dt = 4.0;
            raw_dt /= erle_for_dt;
        }
        if (a->epc.active) { raw_dt = 0.0; is_stationary_dt = 0; }
        double dt_indicator = raw_dt < 0.0 ? 0.0 : raw_dt;
        if (dt_indicator > 0.8) dt_indicator = 0.8;
        a->last_dt_indicator = dt_indicator;

        /* Divergence indicator EMA */
        filter_convergence_update_divergence(&a->convergence,
            a->near_power, a->raw_error_power);

        /* EPC render-forced (RES state-write only) */
        if (a->epc_render_forced_remaining > 0) {
            a->epc_render_forced_remaining--;
            if (a->has_res) a->res.residual_est.using_render_based = 1;
        }
        if (a->has_res) a->res.over_sub = (float)base_over_sub;

        /* dt_residual_scale on echo_spec */
        double dt_clip = dt_indicator < 0.0 ? 0.0 : dt_indicator;
        if (dt_clip > 0.8) dt_clip = 0.8;
        double dt_residual_scale = 1.0 - 0.5 * (dt_clip / 0.8);
        Complex eff_echo[8192];
        for (int k = 0; k < K; ++k) {
            eff_echo[k].r = (float)(a->main_filter.base.echo_spec[k].r * dt_residual_scale);
            eff_echo[k].i = (float)(a->main_filter.base.echo_spec[k].i * dt_residual_scale);
        }

        double shadow_dt_v = a->dt_analyzer.dt_from_energy;
        if (a->dt_analyzer.dt_from_shadow > shadow_dt_v) shadow_dt_v = a->dt_analyzer.dt_from_shadow;
        if (a->epc.active) shadow_dt_v *= 0.08;

        /* Stash always (visible via aec_get_res_context). */
        a->last_far_power         = far_power;
        a->last_shadow_dt         = shadow_dt_v;
        a->last_is_stationary_dt  = is_stationary_dt;

        if (a->has_res) {
            ResProcessInputs rin = {0};
            rin.error_hop = a->raw_output;
            rin.echo_spec = eff_echo;
            rin.far_spec  = a->main_filter.base.far_spec;
            rin.near_spec = a->main_filter.base.near_spec;
            /* Python AEC.process at aec.py:4061 does NOT pass error_spec_from_filter
             * to res.process — so ResFilter falls back to spec_synth (the windowed
             * OLA spec). Match Python: pass NULL to use spec_synth path. */
            rin.error_spec_from_filter = NULL;
            rin.far_power = far_power;
            rin.erle_factor = erle_factor;
            rin.dt_indicator = dt_indicator;
            rin.divergence = a->convergence.divergence;
            rin.is_stationary_dt = is_stationary_dt;
            rin.saturation_level = a->saturation_level;
            rin.epc_active = a->epc.active;
            rin.shadow_dt = shadow_dt_v;
            rin.erl_estimate = a->erl_estimate;
            rin.filter_converged = a->convergence.converged;
            rin.filter_once_converged = a->convergence.once_converged;
            res_filter_process(&a->res, &rin, a->res_output);
            final_out = a->res_output;
        }

        /* per-bin mu_scale update */
        if (a->has_res && a->convergence.converged) {
            /* Kalman fine-tune uses RES internal echo_psd / error_psd. */
            for (int k = 0; k < K; ++k) {
                double v = (double)a->res.echo_psd[k] /
                           ((double)a->res.error_psd[k] + 1e-10);
                if (v < 0.0) v = 0.0; if (v > 1.0) v = 1.0;
                double mu_min = a->cfg.shadow_mu_min;
                a->per_bin_mu_scale[k] = (float)(mu_min + (1.0 - mu_min) * v);
            }
            double mean_v = 0.0;
            for (int k = 0; k < K; ++k) mean_v += (double)a->per_bin_mu_scale[k];
            a->simple_mu_ratio = mean_v / (double)K;
            if (is_stationary_dt) {
                for (int k = 0; k < K; ++k) a->per_bin_mu_scale[k] = a->cfg.shadow_mu_min;
                a->simple_mu_ratio = a->cfg.shadow_mu_min;
            }
            a->has_per_bin_mu = 1;
        } else {
            a->has_per_bin_mu = 0;
            if (!a->convergence.converged) {
                update_simple_mu_ratio(a, a->raw_output, a->far_hop, hop);
            }
        }
    }

    /* Limiter */
    double near_peak = 0.0, out_peak = 0.0;
    for (int i = 0; i < hop; ++i) {
        double an = fabs((double)a->near_hop[i]);
        double ao = fabs((double)final_out[i]);
        if (an > near_peak) near_peak = an;
        if (ao > out_peak)  out_peak  = ao;
    }
    double target_gain = (out_peak > near_peak && near_peak > 1e-6)
                         ? near_peak / out_peak : 1.0;
    double alpha_lim = (target_gain < a->limiter_gain) ? 0.3 : 0.8;
    a->limiter_gain = alpha_lim * a->limiter_gain + (1.0 - alpha_lim) * target_gain;
    for (int i = 0; i < hop; ++i) out[i] = (float)(final_out[i] * a->limiter_gain);

    /* Power EMAs (Python: square fp32 first, then mix into fp64 EMA). */
    const double ap = a->alpha_pow, oap = 1.0 - a->alpha_pow;
    for (int i = 0; i < hop; ++i) {
        float n = a->near_hop[i];
        float r = a->raw_output[i];
        float f = out[i];
        float nsq = n * n, rsq = r * r, fsq = f * f;
        a->near_power        = ap * a->near_power        + oap * (double)nsq;
        a->raw_error_power   = ap * a->raw_error_power   + oap * (double)rsq;
        a->final_error_power = ap * a->final_error_power + oap * (double)fsq;
        a->near_power_sum        += (double)nsq;
        a->raw_error_power_sum   += (double)rsq;
        a->final_error_power_sum += (double)fsq;
    }

    /* Convergence detection: 10 frames ERLE > 5 dB while far-active */
    int just_converged = filter_convergence_update_convergence(
        &a->convergence, a->near_power, a->raw_error_power,
        far_now > 1e-4, a->warmup_frames_remaining <= 0);
    if (just_converged) {
        for (int k = 0; k < K; ++k) {
            a->main_filter.Q[k] = a->main_filter.Q_low[k];
            if (a->has_shadow) a->shadow_filter.Q[k] = a->shadow_filter.Q_low[k];
        }
    }

    /* v3.10.0 — filter plateau detection + one-shot recovery (post-convergence
     * tick). Uses windowed ERLE + DT signal pattern. */
    {
        double far_now_pwr = mean_sq(a->far_hop, hop);
        int    far_active_now = far_now_pwr > 1e-4;
        double erle_w_db = 10.0 * log10(
            (a->erle_window_near + 1e-10) / (a->erle_window_err + 1e-10));
        int    dt_signal_now = (a->dt_analyzer.dt_from_shadow > 0.3)
                            || (a->dt_analyzer.dt_from_energy > 0.3);
        if (filter_plateau_update(&a->plateau_detector,
                                     far_active_now, dt_signal_now,
                                     erle_w_db,
                                     a->convergence.once_converged)) {
            aec_reset_filter_derived_state(a, /*preserve_render_ema=*/1);
        }
    }

    a->frame_count++;
}

void aec_get_res_context(const Aec* a, AecResContext* ctx) {
    if (!a || !ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->n_freqs    = a->n_freqs;
    ctx->hop_size   = a->hop_size;
    ctx->linear_hop = a->raw_output;
    ctx->echo_spec  = a->main_filter.base.echo_spec;
    ctx->far_spec   = a->main_filter.base.far_spec;
    ctx->near_spec  = a->main_filter.base.near_spec;
    ctx->far_power           = (float)a->last_far_power;
    ctx->erle_factor         = (float)a->erle_factor_prev;
    ctx->dt_indicator        = (float)a->last_dt_indicator;
    ctx->divergence          = (float)a->convergence.divergence;
    ctx->saturation_level    = (float)a->saturation_level;
    ctx->erl_estimate        = (float)a->erl_estimate;
    ctx->shadow_dt           = (float)a->last_shadow_dt;
    ctx->is_stationary_dt    = a->last_is_stationary_dt;
    ctx->filter_converged    = a->convergence.converged;
    ctx->filter_once_converged = a->convergence.once_converged;
    ctx->epc_active          = a->epc.active;
}
