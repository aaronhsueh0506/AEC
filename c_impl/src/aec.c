/* aec.c — top-level v3.22 AEC orchestration (AEC3 post-filter chain).
 *
 * Faithful C port of python/modules/orchestrator.py AEC.process() audio path
 * for the BALANCED preset (mode=PBFDKF, enable_res=True,
 * enable_cng=True). The post-stage is driven by aec3_post_run() (bit-exact full
 * _aec3_post orchestration). See aec.h for scope.
 *
 * Built with -ffp-contract=off (Makefile). The construction of the AEC3 chain
 * mirrors test/parity_aec3_post_run.c, with the balanced sub-module config
 * baked from the live Python instance (include/aec3_balanced_config.h).
 */
#include "aec.h"
#include "aec_debug.h"
#include "aec3_balanced_config.h"
#include "aec3_scale.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* DelayAdjustment enum mirror (delay/delay_types.py). */
#define AEC_DA_NONE          0
#define AEC_DA_BUFFER_FLUSH  1
#define AEC_DA_NEW_DETECTED  2

/* Streaming render/capture call-jitter FIFO budget (milliseconds). Converted
 * to a whole number of hops at create time. 320 ms comfortably covers audio
 * callback scheduling jitter / block-size mismatches without wasting memory
 * (320 ms @ 16 kHz = 5120 samples ≈ 20 KB). NOT the echo delay (that is the
 * downstream ref_ring, delay_buffer_ms). */
#define AEC_STREAM_FIFO_MS 320

/* ───────────────────────── config ──────────────────────────────────────── */

void aec_config_defaults(AecConfig* cfg, int sr) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->sample_rate = sr;
    cfg->filter_length = 832;
    cfg->n_partitions = 0;
    cfg->mu = 0.3f;
    cfg->delta = 1e-8f;
    cfg->enable_cng = 1;
    cfg->enable_delay_est = 1;
    cfg->enable_highpass = 1;
    cfg->enable_saturation = 1;
    cfg->enable_shadow = 1;
    cfg->enable_res = 1;
    cfg->saturation_softclip_ref = 1;
    cfg->shadow_err_alpha = 0.80f;
    cfg->shadow_mu_min = 0.5f;
    cfg->shadow_mu_nlms = 0.5f;
    cfg->warmup_frames = 100;
    cfg->epc_hangover = 20;
    cfg->epc_total_rise = 1.5f;
    cfg->epc_delta_threshold = 0.3f;
    cfg->epc_mu_floor = 0.5f;
    cfg->max_delay_ms = 1024.0f;
    cfg->delay_buffer_ms = 2048.0f;
    cfg->delay_est_init_s = 0.3f;
    cfg->delay_est_period_s = 0.5f;
    cfg->highpass_cutoff_hz = 80.0f;
    cfg->saturation_threshold = 0.95f;
    cfg->kalman_q_high = 1e-3f;
    cfg->kalman_q_low = 1e-6f;
    cfg->delay_acquire_protect_converged = 1;
    cfg->min_gain_floor_far_active_db = -28.0f;   /* balanced */
    cfg->filter_misadjustment_stable_frames = 30;
    cfg->filter_misadjustment_hangover_frames = 100;
    cfg->filter_misadjustment_scale_min = 0.5f;
    cfg->filter_misadjustment_scale_max = 2.0f;
}

void aec_config_from_preset(AecConfig* cfg, AecPreset p, int sr) {
    aec_config_defaults(cfg, sr);
    /* gentle / balanced / aggressive differ ONLY in the far-active min-gain
     * floor (the SuppressionGain split-floor power axis). */
    switch (p) {
        case AEC_PRESET_GENTLE:     cfg->min_gain_floor_far_active_db = -20.0f; break;
        case AEC_PRESET_BALANCED:   cfg->min_gain_floor_far_active_db = -28.0f; break;
        case AEC_PRESET_AGGRESSIVE: cfg->min_gain_floor_far_active_db = -38.0f; break;
    }
}

/* ───────────────────────── helpers ─────────────────────────────────────── */

/* np.mean(x ** 2): square in fp32, sum via numpy-1.26 pairwise f32, divide
 * (np.float32 / int → float64). Uses the bit-exact pairwise from aec3_post. */
static double mean_sq(const float* x, int n) {
    float sq[8192];
    for (int i = 0; i < n; ++i) sq[i] = x[i] * x[i];
    float s = aec3_post_pairwise_sum_f32(sq, (size_t)n);
    return (double)(s / (float)n);
}

/* float(np.sum(near**2)) over hop floats in float64 (Python upcasts the
 * sample-loop EMA inputs; the sum_sum below is the EMA sum which is unused at
 * this layer, so we only need np.abs(c64)**2-style sums elsewhere). Not used. */

static float cmag2_c(float r, float i) {
    /* numpy complex64 |z|² via scaled-hypot FMA (cmag2_np). */
    float ar = r < 0.0f ? -r : r;
    float ai = i < 0.0f ? -i : i;
    float larger = ar > ai ? ar : ai;
    float smaller = ar > ai ? ai : ar;
    float m;
    if (larger == 0.0f) m = 0.0f;
    else { float ratio = smaller / larger; m = larger * sqrtf(fmaf(ratio, ratio, 1.0f)); }
    return m * m;
}

/* set Q on a filter to Q_high (Arc-M boost, no per-band scale in balanced). */
static void filter_q_high(PBFDKF* f) {
    int K = f->base.n_freqs;
    for (int k = 0; k < K; ++k) f->Q[k] = f->Q_high[k];
}

/* ── per-bin tuning array build (LF/HF interpolation, suppression_gain.cc) ─ */
/* (Baked arrays come from the generated header; no runtime build needed.) */

/* ── _reset_filter_derived_state (orchestrator 1210-1383). Used by delay
 *    Path-A (delay_first) + Path-B (delay_shift). preserve_render_ema is a
 *    no-op at this layer (render-side trackers live in the StationarityEstimator
 *    which is preserved per preserve_render_side=True). ───────────────────── */
static void aec3_post_chain_reset(Aec* a);   /* fwd */

static void aec_reset_filter_derived_state(Aec* a) {
    pbfdkf_reset(&a->main_filter);
    if (a->has_shadow) pbfdaf_reset(&a->shadow_filter);

    filter_convergence_reset(&a->convergence);
    epc_reset(&a->epc);

    a->main_err_smooth = 0.0;
    a->shadow_err_smooth = 0.0;
    a->raw_error_power = 0.0;
    a->near_power = 0.0;

    a->erle_window_near = 1e-10;
    a->erle_window_err = 1e-10;
    a->erle_factor_prev = 0.0;
    a->inst_erle_smooth = 1.0;

    a->simple_mu_ratio = 1.0;
    a->simple_mu_holdoff = 0;
    a->has_per_bin_mu = 0;          /* _per_bin_mu_scale = None */

    a->erl_estimate = 0.1;
    a->epc_render_forced_remaining = 0;
    doubletalk_reset(&a->dt_analyzer);
    a->stat_dt_hangover = 0;

    a->shadow_frame_count = 0;
    shadow_copy_reset(&a->regime);

    /* AEC3 post chain (filter-output-derived); render-side preserved. */
    aec3_post_chain_reset(a);

    /* Re-arm warmup with high Q. */
    filter_q_high(&a->main_filter);
    /* shadow is PBFDAF (no Q) — nothing to boost. */
    int half = a->cfg.warmup_frames / 2;
    if (a->warmup_frames_remaining < half) a->warmup_frames_remaining = half;
    a->warmup_far_active = 0;

    /* clear pending delay shift state */
    a->pending_delay = -1;
    a->has_pending = 0;
    a->pending_delay_ttl = 0;

    a->last_erle_windowed = 0.0;
}

/* _get_simple_mu_scale (orchestrator 1456-1476). Writes a per-bin array into
 * a->per_bin_mu_scale when one is active, returns scalar otherwise; sets
 * *out_is_array. */
static double get_simple_mu_scale(Aec* a, int* out_is_array) {
    double mu_min = a->cfg.shadow_mu_min;
    *out_is_array = 0;
    if (a->warmup_frames_remaining > 0) {
        if (a->warmup_far_active) a->warmup_frames_remaining--;
        if (a->simple_mu_ratio < 0.2) {
            return (a->simple_mu_ratio > 0.2) ? a->simple_mu_ratio : 0.2;  /* max(0.2,ratio) */
        }
        double v = a->simple_mu_ratio + 0.2;       /* min(1.0, max(0.5, .)) */
        if (v < 0.5) v = 0.5;
        if (v > 1.0) v = 1.0;
        return v;
    }
    if (!a->convergence.converged) { if (mu_min < 0.3) mu_min = 0.3; }
    else                          { if (mu_min < 0.2) mu_min = 0.2; }
    if (a->has_per_bin_mu) {
        /* np.maximum(_per_bin_mu_scale, mu_min) (f32 array vs f32 scalar). */
        int K = a->n_freqs;
        for (int k = 0; k < K; ++k) {
            float v = a->per_bin_mu_scale[k];
            if (v < (float)mu_min) v = (float)mu_min;
            a->per_bin_mu_scale[k] = v;
        }
        *out_is_array = 1;
        return 0.0;  /* unused */
    }
    return mu_min + (1.0 - mu_min) * a->simple_mu_ratio;
}

/* _update_simple_mu_ratio (orchestrator 1478-1522). */
static void update_simple_mu_ratio(Aec* a, const float* output,
                                   const float* far_end, int n) {
    double error_power = mean_sq(output, n) + 1e-10;
    double far_power   = mean_sq(far_end, n) + 1e-10;
    if (far_power < 1e-6 && error_power < 1e-6) return;
    if (far_power > 1e-4 && a->simple_mu_ratio < 0.1) {
        a->simple_mu_ratio = 0.8;
        a->simple_mu_holdoff = 0;
        return;
    }
    double ratio = far_power / error_power;
    if (ratio > 1.0) ratio = 1.0;
    int K = a->n_freqs;
    /* np.sum(np.abs(spec)**2): cmag2_np per bin → numpy-1.26 pairwise f32 sum;
     * + 1e-10 widens to f64. */
    float e2_echo[8192], e2_near[8192];
    for (int k = 0; k < K; ++k) {
        e2_echo[k] = cmag2_c(a->main_filter.base.echo_spec[k].r,
                             a->main_filter.base.echo_spec[k].i);
        e2_near[k] = cmag2_c(a->main_filter.base.near_spec[k].r,
                             a->main_filter.base.near_spec[k].i);
    }
    double echo_est_pwr = (double)aec3_post_pairwise_sum_f32(e2_echo, (size_t)K) + 1e-10;
    double near_pwr     = (double)aec3_post_pairwise_sum_f32(e2_near, (size_t)K) + 1e-10;
    if (near_pwr > 1e-8) {
        double r2 = echo_est_pwr / near_pwr;
        if (r2 < 0.0) r2 = 0.0; if (r2 > 1.0) r2 = 1.0;
        double r2_half = r2 * 0.5;
        if (r2_half > ratio) ratio = r2_half;
    }
    double alpha;
    if (ratio < a->simple_mu_ratio) { alpha = 0.3; a->simple_mu_holdoff = 20; }
    else if (a->simple_mu_holdoff > 0) { a->simple_mu_holdoff--; alpha = 0.99; }
    else alpha = 0.95;
    a->simple_mu_ratio = alpha * a->simple_mu_ratio + (1.0 - alpha) * ratio;
}

/* ───────────────────────── construction ────────────────────────────────── */

static int next_pow2(int x) { int n = 1; while (n < x) n <<= 1; return n; }

/* aec3_scale.blocks_to_hops(blocks, hop, sr) = round(blocks * 64/hop ... ).
 * For our hop=160/sr=16000 we bake the values the Python produces (the only
 * ones the orchestrator uses: poor_excitation 1000-block init). */
static long blocks_to_hops_round(int blocks, int hop, int sr) {
    /* AEC3 block = 64 samples; per-hop count = blocks * 64 / hop ... but the
     * Python helper uses round(blocks * kBlockSize / hop * (sr_ref/sr))? We use
     * the verified production value via the same formula as aec3_scale. The
     * only call here is 1000 → init poor-excitation counter. */
    double v = (double)blocks * (64.0) / (double)hop;
    (void)sr;
    return (long)floor(v + 0.5);
}

/* Clear + recreate the AEC3 post chain sub-objects (mirrors _reset_aec3_post
 * with preserve_render_side=True: StationarityEstimator + non_zero_render_seen
 * + active_hops are preserved; everything else re-init'd). */
static void aec3_post_chain_reset(Aec* a) {
    /* aec3_post_reset clears OLA / CNG / coherence EMAs / avg-reverb. */
    aec3_post_reset(&a->post);
    /* URO crossfade memory (_form_prev_output_time / _form_last_selection /
     * _refined_filter_output_last_selected) — _reset_aec3_post clears it
     * (orchestrator 1189-1190). */
    linear_filter_select_reset(&a->a3_lfs);
    /* AecState + REE + SuppressionGain are recreated in Python; here we
     * re-init in place (same backing storage). */
    {
        AecStateConfig acfg;
        aec_state_config_defaults(&acfg);
        acfg.n_bins = a->n_freqs;
        acfg.num_capture_channels = AEC3B_ST_NUM_CAPTURE_CHANNELS;
        acfg.hop_size = a->hop_size;
        acfg.enable_filter_analyzer = AEC3B_ST_ENABLE_FILTER_ANALYZER;
        acfg.erle_startup_hops = AEC3B_ST_ERLE_STARTUP_HOPS;
        acfg.erl_startup_hops = AEC3B_ST_ERL_STARTUP_HOPS;
        acfg.echo_can_saturate = AEC3B_ST_ECHO_CAN_SATURATE;
        acfg.use_linear_filter = AEC3B_ST_USE_LINEAR_FILTER;
        acfg.conservative_initial_phase = AEC3B_ST_CONSERVATIVE_INITIAL_PHASE;
        acfg.delay_headroom_samples = AEC3B_ST_DELAY_HEADROOM_SAMPLES;
        acfg.initial_state_seconds = AEC3B_ST_INITIAL_STATE_SECONDS;
        acfg.erle_min = AEC3B_ST_ERLE_MIN;
        acfg.erle_max_l = AEC3B_ST_ERLE_MAX_L;
        acfg.erle_max_h = AEC3B_ST_ERLE_MAX_H;
        acfg.filter_taps_size = AEC3B_FILTER_TAPS_SIZE;
        aec_state_init(&a->a3_state, &acfg, &a->a3_state_st);
    }
    {
        ReeEchoModelConfig em;
        memset(&em, 0, sizeof(em));
        em.min_noise_floor_power = AEC3B_REE_MIN_NOISE_FLOOR_POWER;
        em.noise_gate_power = AEC3B_REE_NOISE_GATE_POWER_LEGACY;
        em.noise_gate_slope = AEC3B_REE_NOISE_GATE_SLOPE;
        em.stationary_gate_slope = AEC3B_REE_STATIONARY_GATE_SLOPE;
        em.model_reverb_in_nonlinear_mode = AEC3B_REE_MODEL_REVERB_IN_NL;
        /* re-init reuses the same storage pointers already bound in a3_ree. */
        ree_init(&a->a3_ree, a->n_freqs, AEC3B_REE_HOP_SIZE, &em,
                 AEC3B_REE_DEFAULT_GAIN, AEC3B_REE_TM_GAIN, AEC3B_REE_ERLE_ONSET_COMP,
                 AEC3B_REE_REVERB_DECAY, AEC3B_REE_REVERB_MILD_SCALE,
                 AEC3B_REE_REVERB_ENABLED, AEC3B_REE_REVERB_TAIL_STRENGTH,
                 AEC3B_REE_USE_AEC3_RESIDUAL_NOISE_GATE, AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW,
                 AEC3B_REE_NL_R2_ENABLED, AEC3B_REE_NL_R2_ALPHA, AEC3B_REE_NL_NORM_POWER,
                 AEC3B_REE_RESIDUAL_NOISE_GATE_POWER, AEC3B_REE_NOISE_FLOOR_HOLD_HOPS,
                 AEC3B_REE_USE_FREQ_RESPONSE, AEC3B_REE_REVERB_USE_CONSERVATIVE,
                 AEC3B_REE_REVERB_SMOOTHING_BASE,
                 a->a3_ree.x2_noise_floor, a->a3_ree.x2_noise_floor_counter,
                 a->a3_ree.reverb_model.reverb,
                 a->a3_ree.reverb_freq_resp.tail_response,
                 a->a3_ree.render_history,
                 a->a3_ree.delay_render_buf, a->a3_ree.reverb_render_history,
                 a->a3_ree.last_r2_direct, a->a3_ree.last_r2_reverb,
                 a->a3_ree.scratch);
    }
    /* SuppressionGain recreate (preserves config; clears persistent state). */
    {
        SuppressionGain* sg = &a->a3_sg;
        SuppressionGainConfig scfg = sg->cfg;     /* keep config */
        SuppressionGainTuning stun = sg->tun;
        suppression_gain_init(sg, &scfg, &stun, sg->last_gain, sg->last_nearend,
                              sg->last_echo, sg->ma_buf, sg->nearend,
                              sg->weighted_residual, sg->min_gain, sg->max_gain,
                              sg->g_raw, sg->gain, sg->sum_scratch);
    }
}

int aec_create(Aec* a, const AecConfig* cfg) {
    memset(a, 0, sizeof(*a));
    a->cfg = *cfg;

    int hop = (int)(0.010f * cfg->sample_rate);
    int blk = 2 * hop;
    int fft = next_pow2(blk);
    int K   = fft / 2 + 1;
    int n_parts = cfg->n_partitions;
    if (n_parts <= 0) {
        n_parts = (cfg->filter_length + hop - 1) / hop;
        if (n_parts < 1) n_parts = 1;
    }
    a->hop_size = hop; a->block_size = blk; a->fft_size = fft;
    a->n_freqs = K; a->n_partitions = n_parts;

    /* HPF (mic only; ref HPF retired). */
    if (cfg->enable_highpass) {
        hpf_init(&a->hp_mic, cfg->highpass_cutoff_hz, cfg->sample_rate);
        a->has_hp = 1;
    }
    /* Saturation. */
    if (cfg->enable_saturation) {
        saturation_init(&a->sat_ref, cfg->saturation_threshold);
        saturation_init(&a->sat_mic, cfg->saturation_threshold);
        a->has_sat = 1;
    }
    /* Delay (AEC3 matched filter) + ring. */
    if (cfg->enable_delay_est) {
        delay_aec3_init(&a->delay);
        a->has_delay = 1;
        int max_delay_samp = (int)(cfg->max_delay_ms * cfg->sample_rate / 1000.0f);
        int buffer_samp = (int)(cfg->delay_buffer_ms * cfg->sample_rate / 1000.0f);
        if (buffer_samp < max_delay_samp + 4096) buffer_samp = max_delay_samp + 4096;
        a->ref_ring_size = buffer_samp;
        a->ref_ring = (float*)calloc((size_t)buffer_samp, sizeof(float));
        a->current_delay = -1;
        a->pending_delay = -1; a->has_pending = 0; a->pending_delay_ttl = 0;
    } else {
        a->current_delay = 0;
    }

    /* Streaming render-hop FIFO (async render/capture decoupling). Allocated
     * unconditionally; only exercised by the streaming API. Capacity is a
     * render/capture CALL-jitter budget in ms (independent of hop size and of
     * the echo delay, which the downstream ref_ring handles), converted to a
     * whole number of hops for the current hop_size / sample_rate. */
    {
        const int fifo_ms = AEC_STREAM_FIFO_MS;
        int cap = (fifo_ms * cfg->sample_rate / 1000 + a->hop_size - 1)
                  / a->hop_size;            /* ceil(fifo_ms / hop_ms) */
        if (cap < 2) cap = 2;               /* need ≥2 to buffer any skew */
        a->fifo_cap_hops = cap;
    }
    a->render_fifo = (float*)calloc((size_t)a->fifo_cap_hops * a->hop_size,
                                    sizeof(float));
    a->fifo_count = 0; a->fifo_read = 0; a->fifo_write = 0;
    a->render_call_count = 0; a->capture_call_count = 0;
    a->last_buffering_event = AEC_BUF_NONE;

    /* Main filter (PBFDKF). */
    pbfdkf_init(&a->main_filter, blk, n_parts, cfg->mu, cfg->delta, hop);
    for (int k = 0; k < K; ++k) {
        a->main_filter.Q_high[k] = cfg->kalman_q_high;
        a->main_filter.Q_low[k]  = cfg->kalman_q_low;
        a->main_filter.Q[k]      = cfg->kalman_q_high;
    }
    /* RSA-driven poor-excitation init (1000 blocks → hop-scaled). */
    long poor_init = blocks_to_hops_round(1000, hop, cfg->sample_rate);
    a->main_filter.base.poor_excitation_counter = poor_init;

    /* Shadow filter (PBFDAF NLMS). */
    if (cfg->enable_shadow) {
        pbfdaf_init(&a->shadow_filter, blk, n_parts, cfg->shadow_mu_nlms,
                    cfg->delta, hop);
        a->shadow_filter.poor_excitation_counter = poor_init;
        a->shadow_filter.saturated_capture = 0;
        a->has_shadow = 1;
    }

    /* detectors / EPC / regime / RSA. */
    render_activity_init(&a->render_activity);
    filter_convergence_init(&a->convergence);
    doubletalk_init(&a->dt_analyzer, 1.5, 3.0);   /* shadow_dtd_offset / advantage_scale */
    epc_init(&a->epc, cfg->epc_hangover, cfg->epc_total_rise, cfg->epc_delta_threshold);
    shadow_copy_init(&a->regime, SC_GATE_ENERGY, 0.65, 3, cfg->epc_hangover);
    a->rsa_counters = (int64_t*)calloc((size_t)(K - 2 > 0 ? K - 2 : 1), sizeof(int64_t));
    rsa_init(&a->rsa, a->rsa_counters, K, n_parts);  /* strong_peak_freeze_duration=n_partitions */

    /* ── AEC3 post chain ─────────────────────────────────────────────────── */
    a->post_fft = fft_create(fft);

    /* aec3_post driver. */
    {
        Aec3PostConfig pcfg;
        aec3_post_config_defaults(&pcfg);
        pcfg.n_bins = K; pcfg.fft_size = fft; pcfg.block_size = blk; pcfg.hop_size = hop;
        pcfg.erle_coh_gate_enabled = AEC3B_ERLE_COH_GATE_ENABLED;
        pcfg.erle_windowed_capture_psd = AEC3B_ERLE_WINDOWED_CAPTURE_PSD;
        pcfg.erle_render_x2_psd_scale = AEC3B_ERLE_RENDER_X2_PSD_SCALE;
        pcfg.output_capture_when_linear_unusable = AEC3B_OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE;
        pcfg.enable_cng = cfg->enable_cng;
        pcfg.cng_n2_update_onset_hops = AEC3B_CNG_N2_UPDATE_ONSET_HOPS;
        pcfg.cng_n2_initial_duration_hops = AEC3B_CNG_N2_INITIAL_DURATION_HOPS;
        pcfg.cng_y2_alpha = AEC3B_CNG_Y2_ALPHA;
        pcfg.cng_n2_track_freshness = AEC3B_CNG_N2_TRACK_FRESHNESS;
        pcfg.cng_n2_track_retention = AEC3B_CNG_N2_TRACK_RETENTION;
        pcfg.cng_n2_slow_up = AEC3B_CNG_N2_SLOW_UP;
        pcfg.cng_n2_initial_alpha = AEC3B_CNG_N2_INITIAL_ALPHA;
        pcfg.noise_floor_int16sq = AEC3B_NOISE_FLOOR_INT16SQ;
        pcfg.erle_coh_gate_alpha = AEC3B_ERLE_COH_GATE_ALPHA;
        pcfg.erle_coh_gate_threshold = AEC3B_ERLE_COH_GATE_THRESHOLD;

        /* driver backing storage */
        float *avg_rev   = (float*)malloc((size_t)K * sizeof(float));
        float *y2s       = (float*)malloc((size_t)K * sizeof(float));
        float *n2        = (float*)malloc((size_t)K * sizeof(float));
        float *n2i       = (float*)malloc((size_t)K * sizeof(float));
        float *sye_re    = (float*)malloc((size_t)K * sizeof(float));
        float *sye_im    = (float*)malloc((size_t)K * sizeof(float));
        double *syy      = (double*)malloc((size_t)K * sizeof(double));
        double *see      = (double*)malloc((size_t)K * sizeof(double));
        float *ola       = (float*)malloc((size_t)blk * sizeof(float));
        float *np_       = (float*)malloc((size_t)K * sizeof(float));
        float *fp_       = (float*)malloc((size_t)K * sizeof(float));
        float *ep_       = (float*)malloc((size_t)K * sizeof(float));
        float *erp_      = (float*)malloc((size_t)K * sizeof(float));
        float *cpe       = (float*)malloc((size_t)K * sizeof(float));
        float *x2r       = (float*)malloc((size_t)K * sizeof(float));
        float *cn        = (float*)malloc((size_t)K * sizeof(float));
        float *nf        = (float*)malloc((size_t)K * sizeof(float));
        unsigned char *cgm = (unsigned char*)malloc((size_t)K);
        Complex *eout    = (Complex*)malloc((size_t)K * sizeof(Complex));
        float *eout_full = (float*)malloc((size_t)fft * sizeof(float));
        aec3_post_init(&a->post, &pcfg, a->post_fft,
                       AEC3B_SYNTH_WINDOW, AEC3B_SQRT2_SIN_LUT, avg_rev,
                       y2s, n2, n2i, sye_re, sye_im, syy, see, ola,
                       np_, fp_, ep_, erp_, cpe, x2r, cgm, cn, nf,
                       eout, eout_full);
    }

    /* AecState (storage allocated then init). */
    {
        AecStateStorage* s = &a->a3_state_st;
        int ncc = AEC3B_ST_NUM_CAPTURE_CHANNELS;
        s->erle_max = (float*)malloc((size_t)K * sizeof(float));
        s->erle = (float*)malloc((size_t)K * sizeof(float));
        s->erle_oc = (float*)malloc((size_t)K * sizeof(float));
        s->erle_unb = (float*)malloc((size_t)K * sizeof(float));
        s->erle_during = (float*)malloc((size_t)K * sizeof(float));
        s->erle_coming_onset = (unsigned char*)malloc((size_t)K);
        s->erle_hold = (int32_t*)malloc((size_t)K * sizeof(int32_t));
        s->erle_y2_acc = (float*)malloc((size_t)K * sizeof(float));
        s->erle_e2_acc = (float*)malloc((size_t)K * sizeof(float));
        s->erle_low_render = (unsigned char*)malloc((size_t)K);
        s->erl = (float*)malloc((size_t)K * sizeof(float));
        s->erl_hold = (int*)malloc((size_t)(K - 2) * sizeof(int));
        s->filter_delays_blocks = (int*)malloc((size_t)ncc * sizeof(int));
        s->fa_h_highpass = (float*)malloc((size_t)AEC3B_FILTER_TAPS_SIZE * sizeof(float));

        AecStateConfig acfg;
        aec_state_config_defaults(&acfg);
        acfg.n_bins = K;
        acfg.num_capture_channels = ncc;
        acfg.hop_size = hop;
        acfg.enable_filter_analyzer = AEC3B_ST_ENABLE_FILTER_ANALYZER;
        acfg.erle_startup_hops = AEC3B_ST_ERLE_STARTUP_HOPS;
        acfg.erl_startup_hops = AEC3B_ST_ERL_STARTUP_HOPS;
        acfg.echo_can_saturate = AEC3B_ST_ECHO_CAN_SATURATE;
        acfg.use_linear_filter = AEC3B_ST_USE_LINEAR_FILTER;
        acfg.conservative_initial_phase = AEC3B_ST_CONSERVATIVE_INITIAL_PHASE;
        acfg.delay_headroom_samples = AEC3B_ST_DELAY_HEADROOM_SAMPLES;
        acfg.initial_state_seconds = AEC3B_ST_INITIAL_STATE_SECONDS;
        acfg.erle_min = AEC3B_ST_ERLE_MIN;
        acfg.erle_max_l = AEC3B_ST_ERLE_MAX_L;
        acfg.erle_max_h = AEC3B_ST_ERLE_MAX_H;
        acfg.filter_taps_size = AEC3B_FILTER_TAPS_SIZE;
        aec_state_init(&a->a3_state, &acfg, s);
    }

    /* ResidualEchoEstimator (storage + init). */
    {
        ResidualEchoEstimator* r = &a->a3_ree;
        int rh = (AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW ? 1 : 0) + 1 + 1;
        float *x2_nf   = (float*)malloc((size_t)K * sizeof(float));
        int   *x2_nf_c = (int*)malloc((size_t)K * sizeof(int));
        float *rm_st   = (float*)malloc((size_t)K * sizeof(float));
        float *rt_st   = (float*)malloc((size_t)K * sizeof(float));
        float *rh_st   = (float*)malloc((size_t)rh * (size_t)K * sizeof(float));
        float *drd_st  = (float*)malloc((size_t)REE_DELAY_BUF_SIZE * (size_t)K * sizeof(float));
        float *rrd_st  = (float*)malloc((size_t)REE_DELAY_BUF_SIZE * (size_t)K * sizeof(float));
        float *ld_st   = (float*)malloc((size_t)K * sizeof(float));
        float *lr_st   = (float*)malloc((size_t)K * sizeof(float));
        float *scr_st  = (float*)malloc((size_t)K * sizeof(float));
        ReeEchoModelConfig em;
        memset(&em, 0, sizeof(em));
        em.min_noise_floor_power = AEC3B_REE_MIN_NOISE_FLOOR_POWER;
        em.noise_gate_power = AEC3B_REE_NOISE_GATE_POWER_LEGACY;
        em.noise_gate_slope = AEC3B_REE_NOISE_GATE_SLOPE;
        em.stationary_gate_slope = AEC3B_REE_STATIONARY_GATE_SLOPE;
        em.model_reverb_in_nonlinear_mode = AEC3B_REE_MODEL_REVERB_IN_NL;
        ree_init(r, K, AEC3B_REE_HOP_SIZE, &em,
                 AEC3B_REE_DEFAULT_GAIN, AEC3B_REE_TM_GAIN, AEC3B_REE_ERLE_ONSET_COMP,
                 AEC3B_REE_REVERB_DECAY, AEC3B_REE_REVERB_MILD_SCALE,
                 AEC3B_REE_REVERB_ENABLED, AEC3B_REE_REVERB_TAIL_STRENGTH,
                 AEC3B_REE_USE_AEC3_RESIDUAL_NOISE_GATE, AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW,
                 AEC3B_REE_NL_R2_ENABLED, AEC3B_REE_NL_R2_ALPHA, AEC3B_REE_NL_NORM_POWER,
                 AEC3B_REE_RESIDUAL_NOISE_GATE_POWER, AEC3B_REE_NOISE_FLOOR_HOLD_HOPS,
                 AEC3B_REE_USE_FREQ_RESPONSE, AEC3B_REE_REVERB_USE_CONSERVATIVE,
                 AEC3B_REE_REVERB_SMOOTHING_BASE,
                 x2_nf, x2_nf_c, rm_st, rt_st, rh_st, drd_st, rrd_st,
                 ld_st, lr_st, scr_st);
    }

    /* SuppressionGain (config baked; tuning arrays = baked const tables). */
    {
        SuppressionGain* sg = &a->a3_sg;
        SuppressionGainConfig scfg;
        SuppressionGainTuning stun;
        memset(&scfg, 0, sizeof(scfg));
        scfg.n_bins = K; scfg.sr = cfg->sample_rate; scfg.hop_size = hop;
        scfg.last_lf_band = AEC3B_SG_LAST_LF_BAND;
        scfg.first_hf_band = AEC3B_SG_FIRST_HF_BAND;
        scfg.last_lf_smoothing_band = AEC3B_SG_LAST_LF_SMOOTHING_BAND;
        scfg.last_permanent_lf_smoothing_band = AEC3B_SG_LAST_PERMANENT;
        scfg.lf_smoothing_during_initial_phase = AEC3B_SG_LF_SMOOTHING_INITIAL;
        scfg.lf_clamp_bin = AEC3B_SG_LF_CLAMP_BIN;
        scfg.dne_lf_end = AEC3B_SG_DNE_LF_END;
        scfg.nearend_smoother_n = AEC3B_SG_NEAREND_SMOOTHER_N;
        scfg.aud_lf_end_bin = AEC3B_SG_AUD_LF_END_BIN;
        scfg.aud_mf_end_bin = AEC3B_SG_AUD_MF_END_BIN;
        scfg.floor_power = AEC3B_SG_FLOOR_POWER;
        scfg.aud_thr_lf = AEC3B_SG_AUD_THR_LF;
        scfg.aud_thr_mf = AEC3B_SG_AUD_THR_MF;
        scfg.aud_thr_hf = AEC3B_SG_AUD_THR_HF;
        scfg.low_render_limit = AEC3B_SG_LOW_RENDER_LIMIT;
        scfg.normal_render_limit = AEC3B_SG_NORMAL_RENDER_LIMIT;
        scfg.hf_lgb = AEC3B_SG_HF_LGB;
        scfg.hf_biq = AEC3B_SG_HF_BIQ;
        scfg.conservative_hf = AEC3B_SG_CONSERVATIVE_HF;
        scfg.max_inc_normal = (float)AEC3B_SG_MAX_INC;
        scfg.max_inc_nearend = (float)AEC3B_SG_MAX_INC;
        scfg.max_dec_lf_normal = (float)AEC3B_SG_MAX_DEC_LF;
        scfg.max_dec_lf_nearend = (float)AEC3B_SG_MAX_DEC_LF;
        scfg.floor_first_increase = 0.00001;
        scfg.low_render_threshold = AEC3B_SG_LOW_RENDER_THRESHOLD;
        scfg.split_floor_enabled = 1;
        /* preset axis: split_floor_far_active = 10^(db/10). */
        scfg.split_floor_far_active =
            pow(10.0, (double)cfg->min_gain_floor_far_active_db / 10.0);
        scfg.split_floor_far_silent = AEC3B_SG_SPLIT_FLOOR_FAR_SILENT;
        scfg.split_floor_latch_power = AEC3B_SG_SPLIT_FLOOR_LATCH_POWER;
        scfg.soft_blend_enabled = AEC3B_SG_SOFT_BLEND_ENABLED;
        scfg.soft_blend_per_bin = AEC3B_SG_SOFT_BLEND_PER_BIN;
        scfg.soft_blend_enr_thr = (float)AEC3B_SG_SOFT_BLEND_ENR_THR;
        scfg.soft_blend_softness = (float)AEC3B_SG_SOFT_BLEND_SOFTNESS;
        scfg.dne_enr_threshold = AEC3B_SG_DNE_ENR_THRESHOLD;
        scfg.dne_enr_exit_threshold = AEC3B_SG_DNE_ENR_EXIT_THRESHOLD;
        scfg.dne_snr_threshold = AEC3B_SG_DNE_SNR_THRESHOLD;
        scfg.dne_use_during_initial_phase = AEC3B_SG_DNE_USE_DURING_INITIAL_PHASE;
        scfg.dne_use_unbounded_echo = AEC3B_SG_DNE_USE_UNBOUNDED_ECHO;
        scfg.dne_lf_endpoint_bin = AEC3B_SG_DNE_LF_ENDPOINT_BIN;
        scfg.dne_trigger_threshold_hops = AEC3B_SG_TRIGGER_THRESHOLD_HOPS;
        scfg.dne_hold_duration_hops = AEC3B_SG_HOLD_DURATION_HOPS;
        scfg.stat_aware_ne_proxy_enabled = 0; scfg.stat_aware_ne_proxy_threshold = 0.10;
        stun.nearend_enr_tr = AEC3B_SG_NEAREND_ENR_TR;
        stun.nearend_enr_su = AEC3B_SG_NEAREND_ENR_SU;
        stun.nearend_emr_tr = AEC3B_SG_NEAREND_EMR_TR;
        stun.normal_enr_tr  = AEC3B_SG_NORMAL_ENR_TR;
        stun.normal_enr_su  = AEC3B_SG_NORMAL_ENR_SU;
        stun.normal_emr_tr  = AEC3B_SG_NORMAL_EMR_TR;

        float *last_gain = (float*)malloc((size_t)K * sizeof(float));
        float *last_ne   = (float*)malloc((size_t)K * sizeof(float));
        float *last_echo = (float*)malloc((size_t)K * sizeof(float));
        float *ma        = (float*)malloc((size_t)AEC3B_SG_NEAREND_SMOOTHER_N * (size_t)K * sizeof(float));
        float *ne        = (float*)malloc((size_t)K * sizeof(float));
        float *wr        = (float*)malloc((size_t)K * sizeof(float));
        float *ming      = (float*)malloc((size_t)K * sizeof(float));
        float *maxg      = (float*)malloc((size_t)K * sizeof(float));
        float *graw      = (float*)malloc((size_t)K * sizeof(float));
        float *gout      = (float*)malloc((size_t)K * sizeof(float));
        float *gsum      = (float*)malloc((size_t)K * sizeof(float));
        suppression_gain_init(sg, &scfg, &stun, last_gain, last_ne, last_echo,
                              ma, ne, wr, ming, maxg, graw, gout, gsum);
    }

    /* StationarityEstimator. */
    {
        float *stat_noise = (float*)malloc((size_t)K * sizeof(float));
        int32_t *stat_hang = (int32_t*)malloc((size_t)K * sizeof(int32_t));
        unsigned char *stat_flags = (unsigned char*)malloc((size_t)K);
        float *stat_hist = (float*)malloc((size_t)16 * (size_t)K * sizeof(float));
        stationarity_estimator_init(&a->a3_stat, K, hop, cfg->sample_rate,
                                    stat_noise, stat_hang, stat_flags, stat_hist);
    }

    /* LinearFilterSelect. */
    linear_filter_select_init(&a->a3_lfs, hop, blk, fft, K);

    /* run scratch. */
    {
        Aec3PostRunScratch* sc = &a->a3_sc;
        sc->sel_esw  = (Complex*)malloc((size_t)K * sizeof(Complex));
        sc->sel_echo = (Complex*)malloc((size_t)K * sizeof(Complex));
        sc->nsw_e1   = (Complex*)malloc((size_t)K * sizeof(Complex));
        sc->ybase    = (Complex*)malloc((size_t)K * sizeof(Complex));
        sc->abs_near = (float*)malloc((size_t)K * sizeof(float));
        sc->abs_far  = (float*)malloc((size_t)K * sizeof(float));
        sc->abs_sel_echo = (float*)malloc((size_t)K * sizeof(float));
        sc->abs_error    = (float*)malloc((size_t)K * sizeof(float));
        sc->abs_echo_coh = (float*)malloc((size_t)K * sizeof(float));
        sc->abs_nsw_e1   = (float*)malloc((size_t)K * sizeof(float));
        sc->abs_ybase    = (float*)malloc((size_t)K * sizeof(float));
        sc->x2_at_delay  = (float*)malloc((size_t)K * sizeof(float));
        sc->x2_past      = (float*)malloc((size_t)K * sizeof(float));
        sc->w_mag2 = (float*)malloc((size_t)n_parts * (size_t)K * sizeof(float));
        sc->render_block_scaled = (float*)malloc((size_t)hop * sizeof(float));
        sc->bridge_taps = (float*)malloc((size_t)fft * sizeof(float));
        sc->r2       = (float*)malloc((size_t)K * sizeof(float));
        sc->r2_unb   = (float*)malloc((size_t)K * sizeof(float));
        sc->nearend_pwr = (float*)malloc((size_t)K * sizeof(float));
        sc->stat_mask = (unsigned char*)malloc((size_t)K);
    }

    /* pending EPV mirror + stationarity read-state. */
    a->pending_gain_change = 0;
    a->pending_delay_change = -1;
    a->stationarity_active_hops = 0;
    a->non_zero_render_seen = 0;
    a->render_peak_floor = 10.0f / 32768.0f;
    a->block_stationary_next = 0;
    a->stationarity_converge_hops = AEC3B_STATIONARITY_CONVERGE_HOPS;

    /* scalar state. */
    a->saturation_level = 0.0;
    a->erl_estimate = 0.1;
    a->main_err_smooth = 0.0; a->shadow_err_smooth = 0.0;
    a->shadow_frame_count = 0;
    a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10; a->erle_window_err = 1e-10;
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
    a->has_limiter_lag = 0;
    a->near_power = 0.0; a->raw_error_power = 0.0;
    a->alpha_pow = 0.95;
    a->frame_count = 0;
    a->poor_coarse_counter = 0;
    a->coarse_reset_hangover = 0;
    a->misadj_e2_acum = 0.0; a->misadj_y2_acum = 0.0; a->misadj_n_acum = 0;
    a->misadj_inv = 0.0; a->misadj_overhang = 0;
    a->misadj_stable_count = 0; a->misadj_hangover_remaining = 0;
    a->last_erle_windowed = 0.0;

    /* hop scratch. */
    a->per_bin_mu_scale = (float*)malloc((size_t)K * sizeof(float));
    a->limiter_near_lag = (float*)malloc((size_t)hop * sizeof(float));
    a->near_hop   = (float*)malloc((size_t)hop * sizeof(float));
    a->far_hop    = (float*)malloc((size_t)hop * sizeof(float));
    a->raw_output = (float*)malloc((size_t)hop * sizeof(float));
    a->shadow_out = (float*)malloc((size_t)hop * sizeof(float));
    a->final_out  = (float*)malloc((size_t)hop * sizeof(float));
    a->filter_taps_full = (float*)malloc((size_t)n_parts * (size_t)hop * sizeof(float));
    a->W_all      = (Complex*)malloc((size_t)n_parts * (size_t)K * sizeof(Complex));
    a->X_buf_all  = (Complex*)malloc((size_t)n_parts * (size_t)K * sizeof(Complex));
    return 0;
}

void aec_destroy(Aec* a) {
    if (!a) return;
    if (a->has_delay) free(a->ref_ring);
    free(a->render_fifo);
    pbfdkf_free(&a->main_filter);
    if (a->has_shadow) pbfdaf_free(&a->shadow_filter);
    free(a->rsa_counters);
    /* sub-module storage frees omitted (process owns no extra; OS reclaims on
     * exit). The malloc'd backing arrays are reachable only through the
     * structs; freeing them individually is a maintenance burden with no
     * functional effect for the CLI/golden lifecycle. */
    free(a->per_bin_mu_scale); free(a->limiter_near_lag);
    free(a->near_hop); free(a->far_hop); free(a->raw_output);
    free(a->shadow_out); free(a->final_out);
    free(a->filter_taps_full); free(a->W_all); free(a->X_buf_all);
}

int aec_hop_size(const Aec* a) { return a->hop_size; }

void aec_reset(Aec* a) {
    if (a->has_hp) hpf_reset(&a->hp_mic);
    if (a->has_sat) { saturation_reset(&a->sat_ref); saturation_reset(&a->sat_mic); }
    if (a->has_delay) {
        delay_aec3_reset(&a->delay);
        memset(a->ref_ring, 0, (size_t)a->ref_ring_size * sizeof(float));
        a->ref_ring_write = 0; a->ref_ring_filled = 0;
        a->current_delay = -1; a->pending_delay = -1; a->has_pending = 0;
        a->pending_delay_ttl = 0;
    }
    if (a->render_fifo)
        memset(a->render_fifo, 0,
               (size_t)a->fifo_cap_hops * a->hop_size * sizeof(float));
    a->fifo_count = 0; a->fifo_read = 0; a->fifo_write = 0;
    a->render_call_count = 0; a->capture_call_count = 0;
    a->last_buffering_event = AEC_BUF_NONE;
    pbfdkf_reset(&a->main_filter);
    if (a->has_shadow) pbfdaf_reset(&a->shadow_filter);
    render_activity_reset(&a->render_activity);
    filter_convergence_reset(&a->convergence);
    doubletalk_reset(&a->dt_analyzer);
    epc_reset(&a->epc);
    shadow_copy_reset(&a->regime);
    rsa_reset(&a->rsa);
    stationarity_estimator_reset(&a->a3_stat);
    a->stationarity_active_hops = 0;
    a->non_zero_render_seen = 0;
    a->block_stationary_next = 0;
    aec3_post_chain_reset(a);
    a->pending_gain_change = 0;
    a->pending_delay_change = -1;
    a->saturation_level = 0.0;
    a->erl_estimate = 0.1;
    a->main_err_smooth = 0.0; a->shadow_err_smooth = 0.0;
    a->shadow_frame_count = 0;
    a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10; a->erle_window_err = 1e-10;
    a->erle_factor_prev = 0.0; a->inst_erle_smooth = 1.0;
    a->wn_err_baseline = 1e-8; a->stat_dt_hangover = 0;
    a->warmup_frames_remaining = a->cfg.warmup_frames;
    a->warmup_far_active = 0;
    a->simple_mu_ratio = 1.0; a->simple_mu_holdoff = 0; a->has_per_bin_mu = 0;
    a->limiter_gain = 1.0; a->has_limiter_lag = 0;
    a->near_power = 0.0; a->raw_error_power = 0.0;
    a->frame_count = 0;
    a->poor_coarse_counter = 0; a->coarse_reset_hangover = 0;
    a->misadj_e2_acum = 0.0; a->misadj_y2_acum = 0.0; a->misadj_n_acum = 0;
    a->misadj_inv = 0.0; a->misadj_overhang = 0;
    a->misadj_stable_count = 0; a->misadj_hangover_remaining = 0;
    a->last_erle_windowed = 0.0;
}

/* ───────────────────────── misadjustment estimator ─────────────────────── */

/* _update_misadjustment_estimator (orchestrator 100-148). */
static void misadj_update(Aec* a, const float* near_hpf, const float* raw_out, int hop) {
    /* e2/y2 block = float(np.sum(arr.astype(f64)**2)) over hop. */
    double e2_block = 0.0, y2_block = 0.0;
    for (int i = 0; i < hop; ++i) {
        double r = (double)raw_out[i]; e2_block += r * r;
        double n = (double)near_hpf[i]; y2_block += n * n;
    }
    a->misadj_e2_acum += e2_block;
    a->misadj_y2_acum += y2_block;
    a->misadj_n_acum += 1;
    int n_hops_target = 2;
    if (a->misadj_n_acum < n_hops_target) return;
    double total_samples = (double)(n_hops_target * hop);
    double int16_sq = 32768.0 * 32768.0;
    double y2_threshold = (200.0 * 200.0) * total_samples / int16_sq;
    double e2_overhang_threshold = (7500.0 * 7500.0) * total_samples / int16_sq;
    if (a->misadj_y2_acum > y2_threshold) {
        double denom = a->misadj_y2_acum > 1e-20 ? a->misadj_y2_acum : 1e-20;
        double update = a->misadj_e2_acum / denom;
        if (a->misadj_e2_acum > e2_overhang_threshold) a->misadj_overhang = 4;
        else { a->misadj_overhang -= 1; if (a->misadj_overhang < 0) a->misadj_overhang = 0; }
        if ((update < a->misadj_inv) || (a->misadj_overhang > 0))
            a->misadj_inv += 0.1 * (update - a->misadj_inv);
    }
    a->misadj_e2_acum = 0.0; a->misadj_y2_acum = 0.0; a->misadj_n_acum = 0;
}

/* _fire_aec3_misadj_scale (orchestrator 150-191). */
static void misadj_fire(Aec* a) {
    if (a->misadj_hangover_remaining > 0) { a->misadj_hangover_remaining--; return; }
    int stable = a->convergence.converged && !a->epc.active && !a->regime.main_paused;
    if (!stable) { a->misadj_stable_count = 0; return; }
    a->misadj_stable_count++;
    if (a->misadj_stable_count < a->cfg.filter_misadjustment_stable_frames) return;
    if (a->misadj_inv <= 10.0) return;
    double base = a->misadj_inv > 1e-6 ? a->misadj_inv : 1e-6;
    double scale_raw = 2.0 / sqrt(base);
    double scale = scale_raw;
    if (scale < a->cfg.filter_misadjustment_scale_min) scale = a->cfg.filter_misadjustment_scale_min;
    if (scale > a->cfg.filter_misadjustment_scale_max) scale = a->cfg.filter_misadjustment_scale_max;
    pbfdkf_scale_filter(&a->main_filter, (float)scale);
    a->misadj_inv = 0.0; a->misadj_overhang = 0;
    a->misadj_hangover_remaining = a->cfg.filter_misadjustment_hangover_frames;
}

/* ───────────────────────── process ─────────────────────────────────────── */

void aec_process(Aec* a, const float* mic_in, const float* ref_in, float* out) {
    const int hop = a->hop_size, K = a->n_freqs, N = a->n_partitions;
    memcpy(a->near_hop, mic_in, (size_t)hop * sizeof(float));
    memcpy(a->far_hop,  ref_in, (size_t)hop * sizeof(float));

    /* 1. mic HPF (ref HPF OFF). */
    if (a->has_hp) hpf_process(&a->hp_mic, a->near_hop, a->near_hop, hop);

    /* 2. saturation. */
    double sat_ref = 0.0;
    if (a->has_sat) {
        sat_ref = saturation_detect(&a->sat_ref, a->far_hop, hop);
        double sat_mic = saturation_detect(&a->sat_mic, a->near_hop, hop);
        a->saturation_level = (sat_ref > sat_mic * 0.5) ? sat_ref : sat_mic * 0.5;
        if (a->cfg.saturation_softclip_ref && sat_ref > 0.1)
            saturation_soft_clip(a->far_hop, a->far_hop, hop, 0.8);
    }

    /* 3. delay estimation + ring-buffer alignment. */
    if (a->has_delay) {
        delay_aec3_accumulate(&a->delay, a->near_hop, a->far_hop, hop);
        int new_delay = delay_aec3_estimated_delay(&a->delay);
        int eligible = (new_delay >= 0 && delay_aec3_n_updates(&a->delay) >= 3);

        /* Path A — first acquisition. */
        int already_cancelling = a->cfg.delay_acquire_protect_converged
                                 && (a->last_erle_windowed > 2.5);
        if (eligible && a->current_delay < 0 && delay_aec3_is_solid(&a->delay)
                && !already_cancelling) {
            a->current_delay = new_delay;
            aec_reset_filter_derived_state(a);
            filter_convergence_mark_diverged(&a->convergence);
            pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
            /* shadow is PBFDAF: handle_echo_path_change has no PBFDAF variant;
             * the Python shadow.handle_echo_path_change resets the coarse
             * counter. Mirror via the PBFDAF counters directly. */
            if (a->has_shadow) {
                a->shadow_filter.poor_excitation_counter = AEC3_POOR_EXC_COUNTER_INITIAL_HOPS;
                a->shadow_filter.call_counter = 0;
            }
            a->pending_delay_change = AEC_DA_NEW_DETECTED;
        }

        /* pending TTL aging (once per estimation cycle / per hop). */
        if (a->pending_delay_ttl > 0) {
            a->pending_delay_ttl--;
            if (a->pending_delay_ttl <= 0) { a->has_pending = 0; a->pending_delay = -1; }
        }

        /* Path B — delay shift. */
        double conf = delay_aec3_confidence(&a->delay);
        if (eligible && a->current_delay >= 0 && conf >= 0.5
                && abs(new_delay - a->current_delay) > 32) {
            if (a->has_pending && abs(new_delay - a->pending_delay) < 16) {
                a->current_delay = new_delay;
                a->has_pending = 0; a->pending_delay = -1; a->pending_delay_ttl = 0;
                aec_reset_filter_derived_state(a);
                epc_force_delay(&a->epc);
                filter_q_high(&a->main_filter);
                if (a->has_shadow) {
                    a->shadow_filter.poor_excitation_counter = AEC3_POOR_EXC_COUNTER_INITIAL_HOPS;
                    a->shadow_filter.call_counter = 0;
                }
                filter_convergence_mark_diverged(&a->convergence);
                pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
                a->pending_delay_change = AEC_DA_NEW_DETECTED;
            } else {
                a->pending_delay = new_delay; a->has_pending = 1; a->pending_delay_ttl = 3;
            }
        }

        /* ring write. */
        int w = a->ref_ring_write, rs = a->ref_ring_size;
        if (w + hop <= rs) memcpy(a->ref_ring + w, a->far_hop, (size_t)hop * sizeof(float));
        else {
            int p1 = rs - w;
            memcpy(a->ref_ring + w, a->far_hop, (size_t)p1 * sizeof(float));
            memcpy(a->ref_ring, a->far_hop + p1, (size_t)(hop - p1) * sizeof(float));
        }
        a->ref_ring_write = (w + hop) % rs;
        a->ref_ring_filled += hop;

        /* delay compensation read. */
        if (a->current_delay > 0 && a->ref_ring_filled >= a->current_delay + hop) {
            int d = a->current_delay;
            int read_pos = (a->ref_ring_write - hop - d) % rs;
            if (read_pos < 0) read_pos += rs;
            if (read_pos + hop <= rs)
                memcpy(a->far_hop, a->ref_ring + read_pos, (size_t)hop * sizeof(float));
            else {
                int p1 = rs - read_pos;
                memcpy(a->far_hop, a->ref_ring + read_pos, (size_t)p1 * sizeof(float));
                memcpy(a->far_hop + p1, a->ref_ring, (size_t)(hop - p1) * sizeof(float));
            }
        }
    }

    /* 4. render activity. */
    RenderActivityResult ra = render_activity_update(&a->render_activity, a->far_hop, hop);
    a->warmup_far_active = ra.warmup_active;
    double far_pwr_global = ra.far_pwr;

    /* 5. mu_scale (simple variable mu, Valin 2007 RER-inspired). */
    int mu_is_array = 0;
    double mu_scalar = get_simple_mu_scale(a, &mu_is_array);

    /* 6. mic-clip emergency. */
    if (a->has_sat && a->sat_mic.saturation_level > 0.8) {
        mu_scalar = 0.0;
        mu_is_array = 0;
    }

    /* 7. RSA update + poor-excitation counters + saturated_capture. The PSD is
     *    |filter.far_spec|² from the PREVIOUS hop (far_spec set inside step-9
     *    pbfdkf_process). First hop: far_spec is zero. */
    {
        float rsa_psd[8192];
        for (int k = 0; k < K; ++k)
            rsa_psd[k] = cmag2_c(a->main_filter.base.far_spec[k].r,
                                 a->main_filter.base.far_spec[k].i);
        rsa_update(&a->rsa, rsa_psd, a->far_hop, hop);
        int poor = rsa_poor_signal_excitation(&a->rsa);
        if (poor) a->main_filter.base.poor_excitation_counter = 0;
        else      a->main_filter.base.poor_excitation_counter += 1;
        a->main_filter.base.saturated_capture = (a->saturation_level > 0.5);
        if (a->has_shadow) {
            if (poor) a->shadow_filter.poor_excitation_counter = 0;
            else      a->shadow_filter.poor_excitation_counter += 1;
            a->shadow_filter.saturated_capture = (a->saturation_level > 0.5);
        }
    }

    /* 8. push PREVIOUS hop's _block_stationary_for_next_hop latch onto BOTH
     *    filters (orchestrator 1787-1790). Main consumes it at step 9, shadow
     *    at step 11 — so it MUST be applied here, before step 10 recomputes the
     *    next-hop latch. */
    a->main_filter.base.block_stationary = a->block_stationary_next;
    if (a->has_shadow) a->shadow_filter.block_stationary = a->block_stationary_next;

    /* The RSA narrowband mask is applied INSIDE Python's _update_weights for
     * BOTH main and shadow (filters.py:338-341 + 627-629); the C pbfdkf/pbfdaf
     * omit it. Compute it ONCE here and apply to main (step 9) + shadow (11). */
    float rsa_mask[8192];
    for (int k = 0; k < K; ++k) rsa_mask[k] = 1.0f;
    rsa_mask_regions_around_narrow_bands(&a->rsa, rsa_mask);
    int rsa_mask_active = 0;
    for (int k = 0; k < K; ++k) if (rsa_mask[k] < 1.0f) { rsa_mask_active = 1; break; }

    /* 8.5 E15: shadow runs BEFORE main so main's H_error refresh reads
     *     same-hop e2_coarse (mirrors orchestrator.py L1576-1590 E15 fix).
     *     far_excited hoisted here so step 13 (doubletalk_update_shadow_dt) can use it. */
    int far_excited = 0;
    if (a->has_shadow) {
        a->shadow_frame_count++;
        double far_mean_pre = mean_sq(a->far_hop, hop);
        far_excited = (far_mean_pre > 1e-4);
        int saturation_safe_pre = (a->saturation_level < 0.5);
        float shadow_mu_pre = (far_excited && saturation_safe_pre) ? 1.0f : 0.1f;
        if (rsa_mask_active) {
            float mu_buf_pre[8192];
            for (int k = 0; k < K; ++k) mu_buf_pre[k] = shadow_mu_pre * rsa_mask[k];
            pbfdaf_process(&a->shadow_filter, a->near_hop, a->far_hop, mu_buf_pre, 0.0f, a->shadow_out);
        } else {
            pbfdaf_process(&a->shadow_filter, a->near_hop, a->far_hop, NULL,
                           shadow_mu_pre, a->shadow_out);
        }
        /* Publish current-hop e2_coarse to main_filter BEFORE main runs. */
        {
            float e2coa_pre[8192];
            for (int k = 0; k < K; ++k)
                e2coa_pre[k] = cmag2_c(a->shadow_filter.error_spec[k].r,
                                        a->shadow_filter.error_spec[k].i);
            a->main_filter.e2_coarse_for_refresh =
                (float)aec3_post_pairwise_sum_f32(e2coa_pre, (size_t)K);
            for (int k = 0; k < K; ++k)
                a->main_filter.e2_coarse_per_bin[k] = e2coa_pre[k];
            a->main_filter.e2_coarse_per_bin_valid = 1;
        }
    }

    /* 9. MAIN filter. */
    double main_mu_scalar = a->regime.main_paused ? 0.0 : mu_scalar;
    {
        float mu_buf[8192];
        int use_array = mu_is_array;
        if (mu_is_array) {
            float pause = a->regime.main_paused ? 0.0f : 1.0f;
            for (int k = 0; k < K; ++k) mu_buf[k] = a->per_bin_mu_scale[k] * pause;
        }
        if (rsa_mask_active && !use_array) {
            /* Python: scalar → full(scalar) → × mask. */
            for (int k = 0; k < K; ++k) mu_buf[k] = (float)main_mu_scalar * rsa_mask[k];
            use_array = 1;
        } else if (rsa_mask_active && use_array) {
            for (int k = 0; k < K; ++k) mu_buf[k] = mu_buf[k] * rsa_mask[k];
        }
        if (use_array)
            pbfdkf_process(&a->main_filter, a->near_hop, a->far_hop, mu_buf, 0.0f, a->raw_output);
        else
            pbfdkf_process(&a->main_filter, a->near_hop, a->far_hop, NULL,
                           (float)main_mu_scalar, a->raw_output);
    }

    /* 10. stationarity refresh for NEXT hop (StationarityEstimator on
     *     |filter.far_spec|²; first-render latch; block-stationary push). */
    {
        float far_max = 0.0f;
        for (int i = 0; i < hop; ++i) { float v = fabsf(a->far_hop[i]); if (v > far_max) far_max = v; }
        if (!a->non_zero_render_seen && far_max >= a->render_peak_floor)
            a->non_zero_render_seen = 1;
        if (a->non_zero_render_seen) {
            float far_psd[8192];
            for (int k = 0; k < K; ++k)
                far_psd[k] = cmag2_c(a->main_filter.base.far_spec[k].r,
                                     a->main_filter.base.far_spec[k].i);
            stationarity_estimator_update_noise_estimator(&a->a3_stat, far_psd);
            /* E16: pass avg_reverb from previous hop's aec3_post_compute_x2_reverb —
             * same state Python reads from _aec3_avg_render_reverb.reverb (one-hop stale,
             * aec3_post_run fires at step 18, AFTER this stationarity refresh). */
            stationarity_estimator_update_stationarity_flags(&a->a3_stat, far_psd,
                                                              a->post.avg_reverb.reverb);
            a->stationarity_active_hops += 1;
        }
        /* latch flag for the NEXT hop (applied at step 8 of hop+1). Do NOT
         * overwrite this hop's filter.block_stationary — step 11's shadow
         * still reads the value pushed at step 8. */
        int converged_enough = (a->stationarity_active_hops >= a->stationarity_converge_hops);
        a->block_stationary_next =
            converged_enough && stationarity_estimator_is_block_stationary(&a->a3_stat);
    }

    /* 11. SHADOW filter — already ran pre-main in step 8.5 (E15 fix). */

    /* 12. e2_coarse + erl publish + poor-coarse rescue. */
    if (a->has_shadow) {
        /* _e2_ref = Σ|filter.error_spec|² (cmag2_np, pairwise f32). */
        float e2ref_arr[8192], e2coa_arr[8192], erl_arr[8192];
        for (int k = 0; k < K; ++k)
            e2ref_arr[k] = cmag2_c(a->main_filter.base.error_spec[k].r,
                                   a->main_filter.base.error_spec[k].i);
        for (int k = 0; k < K; ++k)
            e2coa_arr[k] = cmag2_c(a->shadow_filter.error_spec[k].r,
                                   a->shadow_filter.error_spec[k].i);
        double e2_ref = (double)aec3_post_pairwise_sum_f32(e2ref_arr, (size_t)K);
        double e2_coa = (double)aec3_post_pairwise_sum_f32(e2coa_arr, (size_t)K);
        /* erl[k] = Σ_p |W_p[k]|². */
        for (int k = 0; k < K; ++k) erl_arr[k] = 0.0f;
        for (int part = 0; part < N; ++part) {
            const Complex* Wp = a->main_filter.base.W + (size_t)part * K;
            for (int k = 0; k < K; ++k) erl_arr[k] += cmag2_c(Wp[k].r, Wp[k].i);
        }
        /* publish to the refined filter. */
        a->main_filter.e2_coarse_for_refresh = (float)e2_coa;
        for (int k = 0; k < K; ++k) a->main_filter.e2_coarse_per_bin[k] = e2coa_arr[k];
        a->main_filter.e2_coarse_per_bin_valid = 1;
        for (int k = 0; k < K; ++k) a->main_filter.erl_per_bin[k] = erl_arr[k];

        /* rescue: cond_fire = e2_ref < 0.5*e2_coa; threshold_hops=blocks_to_hops(5)=2. */
        int cond_fire = (e2_ref < 0.5 * e2_coa);
        int threshold_hops = 2;
        if (cond_fire) a->poor_coarse_counter += 1; else a->poor_coarse_counter = 0;
        if (a->poor_coarse_counter >= threshold_hops) {
            pbfdaf_copy_weights_from(&a->shadow_filter, &a->main_filter.base);
            a->coarse_reset_hangover = 10;  /* blocks_to_hops(25)=10 */
            a->poor_coarse_counter = 0;
            ree_reset(&a->a3_ree);          /* REE reset on rescue rising edge. */
        }
        if (a->coarse_reset_hangover > 0) {
            a->coarse_reset_hangover--;
            a->main_filter.disallow_leakage_diverged = 1;
        } else {
            a->main_filter.disallow_leakage_diverged = 0;
        }
    }

    /* 13. smoothed errs + DT analyzer + regime handler. */
    if (a->has_shadow) {
        double main_err   = pbfdkf_get_error_energy(&a->main_filter);
        double shadow_err = pbfdaf_get_error_energy(&a->shadow_filter);
        double as = a->cfg.shadow_err_alpha, oas = 1.0 - as;
        a->main_err_smooth   = as * a->main_err_smooth   + oas * main_err;
        a->shadow_err_smooth = as * a->shadow_err_smooth + oas * shadow_err;
        doubletalk_update_shadow_dt(&a->dt_analyzer, a->shadow_frame_count,
                                    far_excited, a->main_err_smooth, a->shadow_err_smooth);
        int delay_reliable = a->has_delay && (delay_aec3_confidence(&a->delay) >= 0.5);
        ShadowCopyDecision dec = shadow_copy_update(
            &a->regime, a->shadow_frame_count, mean_sq(a->far_hop, hop),
            a->main_err_smooth, a->shadow_err_smooth,
            a->epc.active, a->saturation_level,
            a->dt_analyzer.dt_from_energy, 0.0, delay_reliable);
        if (dec.boost_q) filter_q_high(&a->main_filter);
        /* reverse_copy: no-op on NLMS shadow (Python never invokes it). */
    }

    /* 14. EPV trigger. */
    EpcEvent epv = epc_update_epv(&a->epc, far_pwr_global,
                                  a->convergence.converged, a->regime.main_paused);
    if (epv.fired) {
        a->pending_gain_change = 1;
        pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
        filter_q_high(&a->main_filter);
        filter_convergence_mark_diverged(&a->convergence);
        a->epc_render_forced_remaining = a->cfg.epc_hangover;
        if (a->erl_estimate > 0.3) a->erl_estimate = 0.3;
    }

    /* 15. shadow_rise (only if shadow + converged); else hangover tick. */
    if (a->has_shadow && a->convergence.converged) {
        EpcEvent rise = epc_update_shadow_rise(&a->epc, a->main_err_smooth,
                                               a->shadow_err_smooth,
                                               a->render_activity.is_stationary);
        if (rise.fired) {
            a->pending_gain_change = 1;
            pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
            filter_q_high(&a->main_filter);
            filter_convergence_mark_diverged(&a->convergence);
            a->epc_render_forced_remaining = a->cfg.epc_hangover;
            if (a->erl_estimate > 0.3) a->erl_estimate = 0.3;
        } else {
            epc_tick_hangover(&a->epc);
        }
    }

    /* final_output starts from raw_output. */
    memcpy(a->final_out, a->raw_output, (size_t)hop * sizeof(float));

    /* 16. misadjustment estimator update + fire. */
    misadj_update(a, a->near_hop, a->raw_output, hop);
    misadj_fire(a);

    /* 17. RES / post block (enable_res). */
    int is_stationary_dt = 0;
    double dt_indicator = 0.0;
    double erle_windowed = 0.0;
    double far_power = 0.0;
    if (a->cfg.enable_res) {
        far_power = mean_sq(a->far_hop, hop);
        /* erle_windowed (step 13a). */
        const double erle_decay = 0.999;
        a->erle_window_near = erle_decay * a->erle_window_near + a->near_power;
        a->erle_window_err  = erle_decay * a->erle_window_err  + a->raw_error_power;
        erle_windowed = 10.0 * log10((a->erle_window_near + 1e-10)
                                     / (a->erle_window_err + 1e-10));
        /* erle_for_factor = max(get_erle_instant(), erle_windowed). */
        double erle_inst;
        if (a->near_power < 1e-10 && a->raw_error_power < 1e-10) erle_inst = 0.0;
        else erle_inst = 10.0 * log10((a->near_power + 1e-10) / (a->raw_error_power + 1e-10));
        double erle_for_factor = erle_inst > erle_windowed ? erle_inst : erle_windowed;
        double erle_factor = erle_for_factor / 10.0;
        if (erle_factor < 0.0) erle_factor = 0.0;
        if (erle_factor > 1.0) erle_factor = 1.0;
        a->erle_factor_prev = erle_factor;

        double far_pwr = far_power + 1e-10;
        double mic_pwr = mean_sq(a->near_hop, hop) + 1e-10;
        double raw_err_pwr = mean_sq(a->raw_output, hop) + 1e-10;

        /* erl tracking (B4). */
        if (far_pwr > 1e-4) {
            double raw_dt_ratio = raw_err_pwr / (far_pwr + 1e-10);
            double inst_erl_raw = mic_pwr / far_pwr;
            if (raw_dt_ratio < 2.0 && inst_erl_raw < 1.5) {
                double inst_erl = inst_erl_raw;
                if (inst_erl < 0.001) inst_erl = 0.001; if (inst_erl > 1.0) inst_erl = 1.0;
                double alpha_erl = a->convergence.converged ? 0.999 : 0.99;
                a->erl_estimate = alpha_erl * a->erl_estimate + (1.0 - alpha_erl) * inst_erl;
            }
        }

        doubletalk_update_energy_dt(&a->dt_analyzer, ra.is_active,
                                    far_pwr, mic_pwr, a->erl_estimate);

        /* base DT confidence (simple energy-ratio estimate). */
        double simple_dt = 1.0 - far_pwr / (mic_pwr + far_pwr);
        double raw_dt = a->dt_analyzer.dt_from_energy;
        double sd_half = simple_dt * 0.5;
        if (sd_half > raw_dt) raw_dt = sd_half;

        /* stationary-DT macro detection. */
        if (a->render_activity.is_stationary && a->convergence.converged) {
            int fft_size = a->main_filter.base.fft_size;
            double freq_per_bin = (double)a->cfg.sample_rate / (double)fft_size;
            int vb_start = (int)(100.0 / freq_per_bin); if (vb_start < 1) vb_start = 1;
            int vb_limit = (int)(3000.0 / freq_per_bin); if (vb_limit > K) vb_limit = K;
            double track_err_pwr = 0.0;
            for (int k = vb_start; k < vb_limit; ++k)
                track_err_pwr += (double)cmag2_c(a->main_filter.base.error_spec[k].r,
                                                 a->main_filter.base.error_spec[k].i);
            track_err_pwr += 1e-10;
            if (a->wn_err_baseline < 1e-6) a->wn_err_baseline = track_err_pwr;
            double jump_ratio = track_err_pwr / (a->wn_err_baseline + 1e-10);
            if (jump_ratio > 1.5) a->stat_dt_hangover = 80;
            if (a->stat_dt_hangover > 0) {
                is_stationary_dt = 1;
                a->stat_dt_hangover--;
                a->wn_err_baseline = 0.999 * a->wn_err_baseline + 0.001 * track_err_pwr;
            } else {
                is_stationary_dt = 0;
                a->wn_err_baseline = 0.95 * a->wn_err_baseline + 0.05 * track_err_pwr;
            }
        }
        /* D4 baseline slow-track. */
        if (a->convergence.converged && !a->render_activity.is_stationary
                && far_pwr > 1e-4 && a->wn_err_baseline > 1e-6) {
            a->wn_err_baseline = 0.995 * a->wn_err_baseline + 0.005 * raw_err_pwr;
        }

        /* inst_erle correction. */
        double inst_erle_fast_raw = mic_pwr / raw_err_pwr;
        a->inst_erle_smooth = 0.7 * a->inst_erle_smooth + 0.3 * inst_erle_fast_raw;
        if (a->inst_erle_smooth > 2.0) {
            double erle_for_dt = a->inst_erle_smooth;
            if (erle_for_dt > 4.0) erle_for_dt = 4.0;
            raw_dt /= erle_for_dt;
        }
        /* EPC physical gate. */
        if (a->epc.active) { raw_dt = 0.0; is_stationary_dt = 0; }
        dt_indicator = raw_dt; if (dt_indicator < 0.0) dt_indicator = 0.0;
        if (dt_indicator > 0.8) dt_indicator = 0.8;

        /* divergence indicator EMA. */
        filter_convergence_update_divergence(&a->convergence, a->near_power, a->raw_error_power);

        /* EPC render-forced countdown (Python decrements + sets RES flag;
         * the RES flag is consumed inside REE which aec3_post_run drives —
         * the orchestrator no longer wires using_render_based to REE, so this
         * is a counter decrement only). */
        if (a->epc_render_forced_remaining > 0) a->epc_render_forced_remaining--;

        /* shadow_dt stash for RES context. */
        double shadow_dt_v = a->dt_analyzer.dt_from_energy;
        if (a->dt_analyzer.dt_from_shadow > shadow_dt_v) shadow_dt_v = a->dt_analyzer.dt_from_shadow;
        if (a->epc.active) shadow_dt_v *= 0.08;
        a->last_far_power = far_power;
        a->last_shadow_dt = shadow_dt_v;
        a->last_is_stationary_dt = is_stationary_dt;
        a->last_dt_indicator = dt_indicator;

        /* ── aec3_post_run → final_output ── */
        {
            /* Snapshot W + X_buf flat for the run driver. */
            memcpy(a->W_all, a->main_filter.base.W, (size_t)N * (size_t)K * sizeof(Complex));
            memcpy(a->X_buf_all, a->main_filter.base.X_buf, (size_t)N * (size_t)K * sizeof(Complex));
            pbfdaf_get_time_domain_filter(&a->main_filter.base, a->filter_taps_full);

            Aec3PostRunIn in;
            memset(&in, 0, sizeof(in));
            in.near_spec = a->main_filter.base.near_spec;
            in.far_spec  = a->main_filter.base.far_spec;
            in.echo_spec = a->main_filter.base.echo_spec;
            in.error_spec_windowed = a->main_filter.base.error_spec_windowed;
            in.W0 = a->main_filter.base.W;             /* W[0] */
            in.W_all = a->W_all;
            in.X_buf = a->X_buf_all;
            in.sqrt_hann = a->main_filter.base.sqrt_hann;
            in.kalman_P = NULL; in.kalman_P_len = 0;   /* divergence_indicator dead */
            in.partition_idx = a->main_filter.base.partition_idx;
            in.n_partitions = N;
            in.filter_taps_full = a->filter_taps_full;
            in.filter_taps_full_len = N * hop;
            in.shadow_present = a->has_shadow;
            in.shadow_error_spec = a->has_shadow ? a->shadow_filter.error_spec : NULL;
            in.last_shadow_output_time = a->has_shadow ? a->shadow_out : NULL;
            in.s_ref_max = (double)a->main_filter.base.last_s_max_abs * 32768.0;
            in.s_coa_max = a->has_shadow ? (double)a->shadow_filter.last_s_max_abs * 32768.0 : 0.0;
            in.main_error_energy = pbfdkf_get_error_energy(&a->main_filter);
            in.shadow_error_energy = a->has_shadow ? pbfdaf_get_error_energy(&a->shadow_filter) : 0.0;
            in.raw_output = a->raw_output;
            in.near_end = a->near_hop;
            in.far_end = a->far_hop;
            in.current_delay = a->current_delay;
            in.delay_active = a->has_delay;
            in.saturation_level = a->saturation_level;
            in.pending_gain_change = a->pending_gain_change;
            in.pending_delay_change = a->pending_delay_change;
            in.stationarity_active_hops = a->stationarity_active_hops;
            in.stationarity_converge_hops = a->stationarity_converge_hops;
            in.erle_coh_gate_enabled = AEC3B_ERLE_COH_GATE_ENABLED;
            in.use_stationarity_properties = AEC3B_USE_STATIONARITY_PROPERTIES;
            in.active_render_threshold = AEC3B_ACTIVE_RENDER_THRESHOLD;

            Aec3PostRunObj obj;
            obj.state = &a->a3_state; obj.ree = &a->a3_ree; obj.sg = &a->a3_sg;
            obj.stationarity = &a->a3_stat; obj.lfs = &a->a3_lfs; obj.fft = a->post_fft;

            int pgc = a->pending_gain_change, pdc = a->pending_delay_change;
            aec3_post_run(&a->post, &in, &obj, &a->a3_sc, a->final_out, &pgc, &pdc);
            a->pending_gain_change = pgc;
            a->pending_delay_change = pdc;
        }

        /* per-bin mu_scale update AFTER RES (orchestrator 2291-2307). */
        if (a->convergence.converged) {
            float mu_min = a->cfg.shadow_mu_min;
            for (int k = 0; k < K; ++k) a->per_bin_mu_scale[k] = mu_min;
            a->simple_mu_ratio = 0.0;
            if (is_stationary_dt) {
                for (int k = 0; k < K; ++k) a->per_bin_mu_scale[k] = mu_min;
                a->simple_mu_ratio = mu_min;
            }
            a->has_per_bin_mu = 1;
        } else {
            a->has_per_bin_mu = 0;
            update_simple_mu_ratio(a, a->raw_output, a->far_hop, hop);
        }
    }
    /* enable_res==False C-parity fallback (orchestrator 2313-2316) is unused
     * here since balanced always has enable_res=True. */

    /* 18. OLA-lagged output limiter. Python computes near_peak/out_peak via
     *     np.max(np.abs(.)) (float32) and target_gain = near_peak/out_peak
     *     (float32 / float32 → float32); the EMA mix is float64 (scalar
     *     promotion). The comparison `out_peak > near_peak > 1e-6` is float. */
    {
        if (!a->has_limiter_lag) { memset(a->limiter_near_lag, 0, (size_t)hop * sizeof(float)); a->has_limiter_lag = 1; }
        float near_peak = 0.0f, out_peak = 0.0f;
        for (int i = 0; i < hop; ++i) {
            float an = fabsf(a->limiter_near_lag[i]);
            float ao = fabsf(a->final_out[i]);
            if (an > near_peak) near_peak = an;
            if (ao > out_peak)  out_peak  = ao;
        }
        memcpy(a->limiter_near_lag, a->near_hop, (size_t)hop * sizeof(float));
        /* `out_peak > near_peak` is f32>f32; `near_peak > 1e-6` is f32>pyfloat
         * (near_peak promoted to f64). target_gain = near_peak/out_peak is
         * f32/f32 → f32 (then widened in the EMA mix). */
        double target_gain;
        if (out_peak > near_peak && (double)near_peak > 1e-6)
            target_gain = (double)(near_peak / out_peak);
        else
            target_gain = 1.0;
        double alpha_lim = (target_gain < a->limiter_gain) ? 0.3 : 0.8;
        a->limiter_gain = alpha_lim * a->limiter_gain + (1.0 - alpha_lim) * target_gain;
        /* final_output *= _limiter_gain: float32_arr *= python_float → all-f32
         * (numpy<2 scalar value-cast rule). */
        float lg = (float)a->limiter_gain;
        for (int i = 0; i < hop; ++i)
            a->final_out[i] = a->final_out[i] * lg;
    }

    /* 19. power EMAs (sample loop; read by NEXT frame's step 17). */
    {
        const double ap = a->alpha_pow, oap = 1.0 - a->alpha_pow;
        for (int i = 0; i < hop; ++i) {
            float n = a->near_hop[i], r = a->raw_output[i];
            a->near_power      = ap * a->near_power      + oap * (double)(n * n);
            a->raw_error_power = ap * a->raw_error_power + oap * (double)(r * r);
        }
    }

    /* 20. convergence detection. */
    {
        int far_active = (mean_sq(a->far_hop, hop) > 1e-4);
        int just_converged = filter_convergence_update_convergence(
            &a->convergence, a->near_power, a->raw_error_power,
            far_active, a->warmup_frames_remaining <= 0);
        if (just_converged) {
            int Kk = a->main_filter.base.n_freqs;
            for (int k = 0; k < Kk; ++k) a->main_filter.Q[k] = a->main_filter.Q_low[k];
            /* shadow is PBFDAF (no Q). */
        }
    }

    /* cache erle_windowed for next frame's Path-A guard. */
    if (a->cfg.enable_res) a->last_erle_windowed = erle_windowed;

    /* ── per-frame structured trace ("logr"). Audio-passive, read-only: only
     *    runs when --debug-trace set a CSV file. Zero hot-path cost otherwise
     *    (single NULL test). Reads the post-filter internals — the three not
     *    otherwise persisted (aec3_converged / far_active / gain_mean) were
     *    stashed on a->post.trace during aec3_post_run; everything else is
     *    re-read here from the AEC3 sub-module accessors. */
    if (a->cfg.enable_res && aec_debug_trace_active()) {
        int Kk = a->n_freqs;
        AecDebugTraceRow tr;
        double esum = 0.0, r2sum = 0.0, cnsum = 0.0;
        const float *erle = aec_state_erle(&a->a3_state, /*onset=*/1);
        int kk;
        for (kk = 0; kk < Kk; ++kk) {
            esum  += (double)erle[kk];
            r2sum += (double)a->a3_sc.r2[kk];
            cnsum += (double)a->post.comfort_noise[kk];
        }
        tr.delay            = aec_state_min_direct_path_filter_delay(&a->a3_state);
        tr.far_active       = a->post.trace.far_active;
        tr.saturated_echo   = aec_state_saturated_echo(&a->a3_state);
        tr.usable_linear    = aec_state_usable_linear_estimate(&a->a3_state);
        tr.dominant_nearend = suppression_gain_is_dominant_nearend(&a->a3_sg);
        tr.filter_converged = a->post.trace.aec3_converged;
        tr.fullband_erle    = aec_state_fullband_erle_log2(&a->a3_state);
        tr.erle_mean        = (Kk > 0) ? esum  / Kk : 0.0;
        tr.r2_mean          = (Kk > 0) ? r2sum / Kk : 0.0;
        tr.gain_mean        = a->post.trace.gain_mean;
        tr.comfort_noise_mean = (Kk > 0) ? cnsum / Kk : 0.0;
        tr.near_pwr         = a->near_power;
        tr.raw_err_pwr      = a->raw_error_power;
        tr.limiter_gain     = a->limiter_gain;
        aec_debug_trace_row(&tr);
    }

    /* 21. emit. */
    memcpy(out, a->final_out, (size_t)hop * sizeof(float));
    a->frame_count++;
}

/* ── Streaming API ────────────────────────────────────────────────────────
 * Render-hop FIFO wrapper over the bit-exact aec_process() engine. In lockstep
 * (one analyze_render then one process_capture) the FIFO is pass-through
 * (count 1→0, no event) so the engine sees exactly the same (mic,ref) it would
 * via aec_process() → byte-identical output. Only real async jitter exercises
 * the underrun/overrun paths. */
AecBufferingEvent aec_analyze_render(Aec* a, const float* ref) {
    const int hop = a->hop_size;
    a->render_call_count++;
    AecBufferingEvent ev = AEC_BUF_NONE;
    if (a->fifo_count >= a->fifo_cap_hops) {
        /* Overrun: render piled up past capacity — drop the oldest hop. */
        a->fifo_read = (a->fifo_read + 1) % a->fifo_cap_hops;
        a->fifo_count--;
        ev = AEC_BUF_RENDER_OVERRUN;
    }
    memcpy(a->render_fifo + (size_t)a->fifo_write * hop, ref,
           (size_t)hop * sizeof(float));
    a->fifo_write = (a->fifo_write + 1) % a->fifo_cap_hops;
    a->fifo_count++;
    return ev;
}

AecBufferingEvent aec_process_capture(Aec* a, const float* mic, float* out) {
    const int hop = a->hop_size;
    a->capture_call_count++;
    AecBufferingEvent ev = AEC_BUF_NONE;
    const float* ref;
    if (a->fifo_count > 0) {
        ref = a->render_fifo + (size_t)a->fifo_read * hop;
        a->fifo_read = (a->fifo_read + 1) % a->fifo_cap_hops;
        a->fifo_count--;
    } else {
        /* Underrun: capture ran ahead of render. Process with a silent render
         * hop (no echo to cancel this hop) and signal the caller. Reuse the
         * (unconsumed) read slot as scratch — fifo_read is NOT advanced. */
        float* slot = a->render_fifo + (size_t)a->fifo_read * hop;
        memset(slot, 0, (size_t)hop * sizeof(float));
        ref = slot;
        ev = AEC_BUF_RENDER_UNDERRUN;
    }
    aec_process(a, mic, ref, out);
    a->last_buffering_event = (int)ev;
    return ev;
}

AecBufferingEvent aec_last_buffering_event(const Aec* a) {
    return (AecBufferingEvent)a->last_buffering_event;
}

void aec_get_res_context(const Aec* a, AecResContext* ctx) {
    if (!a || !ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->n_freqs = a->n_freqs;
    ctx->hop_size = a->hop_size;
    ctx->linear_hop = a->raw_output;
    ctx->echo_spec  = a->main_filter.base.echo_spec;
    ctx->far_spec   = a->main_filter.base.far_spec;
    ctx->near_spec  = a->main_filter.base.near_spec;
    ctx->far_power = (float)a->last_far_power;
    ctx->erle_factor = (float)a->erle_factor_prev;
    ctx->dt_indicator = (float)a->last_dt_indicator;
    ctx->divergence = (float)a->convergence.divergence;
    ctx->saturation_level = (float)a->saturation_level;
    ctx->erl_estimate = (float)a->erl_estimate;
    ctx->shadow_dt = (float)a->last_shadow_dt;
    ctx->is_stationary_dt = a->last_is_stationary_dt;
    ctx->filter_converged = a->convergence.converged;
    ctx->filter_once_converged = a->convergence.once_converged;
    ctx->epc_active = a->epc.active;
}
