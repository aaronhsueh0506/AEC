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
    /* Warm tap-transfer (v3.24.1, default ON) + its gate threshold; option-A
     * (block realign) default-OFF. Mirrors config.py:266/271/286. */
    cfg->delay_acquire_warm_transfer = 1;
    cfg->delay_acquire_inst_erle_db = 4.0f;
    cfg->delay_acquire_protect_inst_erle = 0;
    /* DT-deg recovery stack (default ON, mirrors Python 16285fd). */
    cfg->dt_aware_recovery_soft = 1;
    cfg->dt_aware_res_floor_enabled = 1;
    /* -16 (re-tuned from -20 alongside constraint_round_robin): round-robin's
     * deeper linear convergence lifts FS echo, freeing headroom to raise this
     * DT-only floor and neutralise round-robin's DT-deg cost. */
    cfg->min_gain_floor_dt_db = -16.0f;
    cfg->ne_recent_threshold = 0.3f;   /* float32-by-design (Python bit-exact parity retired) */
    cfg->ne_recent_hold = 150;
    cfg->ne_recent_sustain = 3;
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

/* np.mean(x ** 2): square in fp32, sum via numpy-1.26 pairwise f32, divide.
 * Runs entirely in float32-by-design (f64 widening retired — see aec.c/aec.h
 * Stage-2 conversion note). Uses the bit-exact pairwise from aec3_post.
 * `scratch` must hold >= n floats (n is always hop_size at every call site;
 * callers pass a->scr_sq, sized hop_size -- see aec.h struct comment). */
static float mean_sq(const float* x, int n, float* scratch) {
    for (int i = 0; i < n; ++i) scratch[i] = x[i] * x[i];
    float s = aec3_post_pairwise_sum_f32(scratch, (size_t)n);
    return s / (float)n;
}

/* (historical note: this used to widen to float64 to match a Python
 * reference; the reference comparison is retired, kept float32.) Not used. */

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

    a->main_err_smooth = 0.0f;
    a->shadow_err_smooth = 0.0f;
    a->raw_error_power = 0.0f;
    a->near_power = 0.0f;

    a->erle_window_near = 1e-10f;
    a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f;
    a->inst_erle_smooth = 1.0f;
    a->erle_slope_len = 0;     /* mirror _erle_slope_buf.clear() (orch 1176) */
    a->erle_slope_head = 0;

    a->simple_mu_ratio = 1.0f;
    a->simple_mu_holdoff = 0;
    a->has_per_bin_mu = 0;          /* _per_bin_mu_scale = None */

    a->erl_estimate = 0.1f;
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

    a->last_erle_windowed = 0.0f;
}

/* _get_simple_mu_scale (orchestrator 1456-1476). Writes a per-bin array into
 * a->per_bin_mu_scale when one is active, returns scalar otherwise; sets
 * *out_is_array. */
static float get_simple_mu_scale(Aec* a, int* out_is_array) {
    float mu_min = a->cfg.shadow_mu_min;
    *out_is_array = 0;
    if (a->warmup_frames_remaining > 0) {
        if (a->warmup_far_active) a->warmup_frames_remaining--;
        if (a->simple_mu_ratio < 0.2f) {
            return (a->simple_mu_ratio > 0.2f) ? a->simple_mu_ratio : 0.2f;  /* max(0.2,ratio) */
        }
        float v = a->simple_mu_ratio + 0.2f;       /* min(1.0, max(0.5, .)) */
        if (v < 0.5f) v = 0.5f;
        if (v > 1.0f) v = 1.0f;
        return v;
    }
    if (!a->convergence.converged) { if (mu_min < 0.3f) mu_min = 0.3f; }
    else                          { if (mu_min < 0.2f) mu_min = 0.2f; }
    if (a->has_per_bin_mu) {
        /* np.maximum(_per_bin_mu_scale, mu_min) (f32 array vs f32 scalar). */
        int K = a->n_freqs;
        for (int k = 0; k < K; ++k) {
            float v = a->per_bin_mu_scale[k];
            if (v < mu_min) v = mu_min;
            a->per_bin_mu_scale[k] = v;
        }
        *out_is_array = 1;
        return 0.0f;  /* unused */
    }
    return mu_min + (1.0f - mu_min) * a->simple_mu_ratio;
}

/* _update_simple_mu_ratio (orchestrator 1478-1522). */
static void update_simple_mu_ratio(Aec* a, const float* output,
                                   const float* far_end, int n) {
    float error_power = mean_sq(output, n, a->scr_sq) + 1e-10f;
    float far_power   = mean_sq(far_end, n, a->scr_sq) + 1e-10f;
    if (far_power < 1e-6f && error_power < 1e-6f) return;
    if (far_power > 1e-4f && a->simple_mu_ratio < 0.1f) {
        a->simple_mu_ratio = 0.8f;
        a->simple_mu_holdoff = 0;
        return;
    }
    float ratio = far_power / error_power;
    if (ratio > 1.0f) ratio = 1.0f;
    int K = a->n_freqs;
    /* np.sum(np.abs(spec)**2): cmag2_np per bin → numpy-1.26 pairwise f32 sum;
     * float32 throughout (f64 widening retired). */
    float *e2_echo = a->scr_e2_echo, *e2_near = a->scr_e2_near;
    for (int k = 0; k < K; ++k) {
        e2_echo[k] = cmag2_c(a->main_filter.base.echo_spec[k].r,
                             a->main_filter.base.echo_spec[k].i);
        e2_near[k] = cmag2_c(a->main_filter.base.near_spec[k].r,
                             a->main_filter.base.near_spec[k].i);
    }
    float echo_est_pwr = aec3_post_pairwise_sum_f32(e2_echo, (size_t)K) + 1e-10f;
    float near_pwr     = aec3_post_pairwise_sum_f32(e2_near, (size_t)K) + 1e-10f;
    if (near_pwr > 1e-8f) {
        float r2 = echo_est_pwr / near_pwr;
        if (r2 < 0.0f) r2 = 0.0f; if (r2 > 1.0f) r2 = 1.0f;
        float r2_half = r2 * 0.5f;
        if (r2_half > ratio) ratio = r2_half;
    }
    float alpha;
    if (ratio < a->simple_mu_ratio) { alpha = 0.3f; a->simple_mu_holdoff = 20; }
    else if (a->simple_mu_holdoff > 0) { a->simple_mu_holdoff--; alpha = 0.99f; }
    else alpha = 0.95f;
    a->simple_mu_ratio = alpha * a->simple_mu_ratio + (1.0f - alpha) * ratio;
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
     * only call here is 1000 → init poor-excitation counter. Init-only,
     * float32-by-design. */
    float v = (float)blocks * 64.0f / (float)hop;
    (void)sr;
    return (long)floorf(v + 0.5f);
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
                 AEC3B_REE_USE_AEC3_RESIDUAL_NOISE_GATE,
                 AEC3B_USE_STATIONARITY_PROPERTIES,
                 AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW,
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

    /* inst-ERLE slope ring cap = Python _slope_n = max(2, int(0.5*sr/hop)),
     * clamped to the static array size (orchestrator.py:649). */
    {
        int _sn = (int)(0.5f * (float)cfg->sample_rate / (float)(hop > 0 ? hop : 1));
        if (_sn < 2) _sn = 2;
        if (_sn > (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0])))
            _sn = (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0]));
        a->erle_slope_cap = _sn;
        a->erle_slope_len = 0;
        a->erle_slope_head = 0;
    }

    /* HPF (mic only; ref HPF retired). */
    if (cfg->enable_highpass) {
        hpf_f64_init(&a->hp_mic, cfg->highpass_cutoff_hz, cfg->sample_rate);
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
    a->duty_last_delay = -1;   /* duty-cycle change detect (rest zeroed by memset) */

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
    /* AEC3 round-robin TD constraint (production default ON) — main + shadow. */
    a->main_filter.base.constraint_round_robin = 1;

    /* Shadow filter (PBFDAF NLMS). */
    if (cfg->enable_shadow) {
        pbfdaf_init(&a->shadow_filter, blk, n_parts, cfg->shadow_mu_nlms,
                    cfg->delta, hop);
        a->shadow_filter.poor_excitation_counter = poor_init;
        a->shadow_filter.saturated_capture = 0;
        /* FFT dedup: the shadow's near_spec / error_spec_windowed are never read
         * — skip computing them (byte-equal, 2 fewer FFTs/hop). */
        a->shadow_filter.lightweight = 1;
        a->shadow_filter.constraint_round_robin = 1;
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
                 AEC3B_REE_USE_AEC3_RESIDUAL_NOISE_GATE,
                 AEC3B_USE_STATIONARITY_PROPERTIES,
                 AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW,
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
            powf(10.0f, cfg->min_gain_floor_far_active_db / 10.0f);
        scfg.split_floor_far_silent = AEC3B_SG_SPLIT_FLOOR_FAR_SILENT;
        /* DT-gated floor: 10^(min_gain_floor_dt_db/10) (mirrors Python
         * split_floor_dt_db -> _split_floor_dt). */
        scfg.split_floor_dt =
            powf(10.0f, cfg->min_gain_floor_dt_db / 10.0f);
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
    a->saturation_level = 0.0f;
    a->erl_estimate = 0.1f;
    a->main_err_smooth = 0.0f; a->shadow_err_smooth = 0.0f;
    a->shadow_frame_count = 0;
    a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10f; a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f;
    a->inst_erle_smooth = 1.0f;
    a->wn_err_baseline = 1e-8f;
    a->stat_dt_hangover = 0;
    a->warmup_frames_remaining = cfg->warmup_frames;
    a->warmup_far_active = 0;
    a->simple_mu_ratio = 1.0f;
    a->simple_mu_holdoff = 0;
    a->has_per_bin_mu = 0;
    a->limiter_gain = 1.0f;
    a->has_limiter_lag = 0;
    a->near_power = 0.0f; a->raw_error_power = 0.0f;
    a->alpha_pow = 0.95f;
    a->frame_count = 0;
    a->poor_coarse_counter = 0;
    a->coarse_reset_hangover = 0;
    a->leakage_div_sustained_counter = 0;
    a->misadj_e2_acum = 0.0f; a->misadj_y2_acum = 0.0f; a->misadj_n_acum = 0;
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_stable_count = 0; a->misadj_hangover_remaining = 0;
    a->last_erle_windowed = 0.0f;

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

    /* per-hop freq-bin scratch (see aec.h struct comment). */
    a->scr_sq        = (float*)malloc((size_t)hop * sizeof(float));
    a->scr_e2_echo   = (float*)malloc((size_t)K   * sizeof(float));
    a->scr_e2_near   = (float*)malloc((size_t)K   * sizeof(float));
    a->scr_rsa_psd   = (float*)malloc((size_t)K   * sizeof(float));
    a->scr_rsa_mask  = (float*)malloc((size_t)K   * sizeof(float));
    a->scr_mu_buf_pre= (float*)malloc((size_t)K   * sizeof(float));
    a->scr_e2coa_pre = (float*)malloc((size_t)K   * sizeof(float));
    a->scr_mu_buf    = (float*)malloc((size_t)K   * sizeof(float));
    a->scr_far_psd   = (float*)malloc((size_t)K   * sizeof(float));
    a->scr_e2ref_arr = (float*)malloc((size_t)K   * sizeof(float));
    a->scr_e2coa_arr = (float*)malloc((size_t)K   * sizeof(float));
    a->scr_erl_arr   = (float*)malloc((size_t)K   * sizeof(float));

    a->is_static = 0;
    return 0;
}

/* ── static-memory helper ────────────────────────────────────────────────── */

/* Derive all frame-dimension parameters from cfg — same logic as aec_create(). */
static void aec_derive_dims(const AecConfig* cfg,
                            int* o_hop, int* o_blk, int* o_fft, int* o_K,
                            int* o_nparts, int* o_buf_samp, int* o_fifo_cap) {
    int hop = (int)(0.010f * cfg->sample_rate);
    int blk = 2 * hop;
    int fft = next_pow2(blk);
    *o_hop = hop; *o_blk = blk; *o_fft = fft; *o_K = fft / 2 + 1;
    int n = cfg->n_partitions;
    if (n <= 0) { n = (cfg->filter_length + hop - 1) / hop; if (n < 1) n = 1; }
    *o_nparts = n;
    int max_d = (int)(cfg->max_delay_ms * cfg->sample_rate / 1000.0f);
    int buf   = (int)(cfg->delay_buffer_ms * cfg->sample_rate / 1000.0f);
    if (buf < max_d + 4096) buf = max_d + 4096;
    *o_buf_samp = buf;
    int cap = (AEC_STREAM_FIFO_MS * cfg->sample_rate / 1000 + hop - 1) / hop;
    if (cap < 2) cap = 2;
    *o_fifo_cap = cap;
}

size_t aec_get_mem_size(const AecConfig* cfg) {
    if (!cfg) return 0;
    int hop, blk, fft, K, np, buf, fcap;
    aec_derive_dims(cfg, &hop, &blk, &fft, &K, &np, &buf, &fcap);

    const int ncc  = AEC3B_ST_NUM_CAPTURE_CHANNELS;
    const int rh   = (AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW ? 1 : 0) + 2;
    const int sg_n = AEC3B_SG_NEAREND_SMOOTHER_N;

    size_t t = 0;
    t += ALIGN16(sizeof(Aec));
    if (cfg->enable_delay_est) t += ALIGN16((size_t)buf * sizeof(float));
    t += ALIGN16((size_t)fcap * hop * sizeof(float));
    t += ALIGN16((size_t)(K - 2 > 0 ? K - 2 : 1) * sizeof(int64_t));
    t += pbfdkf_get_mem_size(blk, np, hop);
    if (cfg->enable_shadow) t += pbfdaf_get_mem_size(blk, np, hop);
    t += fft_get_mem_size(fft);
    /* aec3_post (21) */
    t += ALIGN16(K * sizeof(float)) * 6;           /* avg_rev y2s n2 n2i sye_re sye_im */
    t += ALIGN16(K * sizeof(double)) * 2;           /* syy see */
    t += ALIGN16((size_t)blk * sizeof(float));      /* ola */
    t += ALIGN16(K * sizeof(float)) * 8;            /* np_ fp_ ep_ erp_ cpe x2r cn nf */
    t += ALIGN16(K * sizeof(unsigned char));        /* cgm */
    t += ALIGN16(K * sizeof(Complex));              /* eout */
    t += ALIGN16((size_t)fft * sizeof(float));      /* eout_full */
    /* AecStateStorage (14) */
    t += ALIGN16(K * sizeof(float)) * 5;            /* erle_max/erle/oc/unb/during */
    t += ALIGN16(K * sizeof(unsigned char));        /* erle_coming_onset */
    t += ALIGN16(K * sizeof(int32_t));              /* erle_hold */
    t += ALIGN16(K * sizeof(float)) * 2;            /* erle_y2_acc/e2_acc */
    t += ALIGN16(K * sizeof(unsigned char));        /* erle_low_render */
    t += ALIGN16(K * sizeof(float));                /* erl */
    t += ALIGN16((size_t)(K - 2) * sizeof(int));    /* erl_hold */
    t += ALIGN16((size_t)ncc * sizeof(int));         /* filter_delays_blocks */
    t += ALIGN16(AEC3B_FILTER_TAPS_SIZE * sizeof(float)); /* fa_h_highpass */
    /* ResidualEchoEstimator (10) */
    t += ALIGN16(K * sizeof(float));                /* x2_nf */
    t += ALIGN16(K * sizeof(int));                  /* x2_nf_c */
    t += ALIGN16(K * sizeof(float)) * 2;            /* rm_st rt_st */
    t += ALIGN16((size_t)rh * K * sizeof(float));   /* rh_st */
    t += ALIGN16((size_t)REE_DELAY_BUF_SIZE * K * sizeof(float)) * 2; /* drd rrd */
    t += ALIGN16(K * sizeof(float)) * 3;            /* ld_st lr_st scr_st */
    /* SuppressionGain (11) */
    t += ALIGN16(K * sizeof(float)) * 3;            /* last_gain/ne/echo */
    t += ALIGN16((size_t)sg_n * K * sizeof(float)); /* ma */
    t += ALIGN16(K * sizeof(float)) * 7;            /* ne wr ming maxg graw gout gsum */
    /* StationarityEstimator (4) */
    t += ALIGN16(K * sizeof(float));                /* stat_noise */
    t += ALIGN16(K * sizeof(int32_t));              /* stat_hang */
    t += ALIGN16(K * sizeof(unsigned char));        /* stat_flags */
    t += ALIGN16(16 * K * sizeof(float));           /* stat_hist */
    /* LinearFilterSelect (4) */
    t += ALIGN16((size_t)hop * sizeof(float)) * 2;  /* prev_output_time e_form */
    t += ALIGN16((size_t)blk * sizeof(float));      /* block_win */
    t += ALIGN16(K * sizeof(Complex));              /* lfs sel_esw */
    /* Aec3PostRunScratch (20) */
    t += ALIGN16(K * sizeof(Complex)) * 4;          /* sel_esw sel_echo nsw_e1 ybase */
    t += ALIGN16(K * sizeof(float)) * 9;            /* abs_near..x2_past */
    t += ALIGN16((size_t)np * K * sizeof(float));   /* w_mag2 */
    t += ALIGN16((size_t)hop * sizeof(float));      /* render_block_scaled */
    t += ALIGN16((size_t)fft * sizeof(float));      /* bridge_taps */
    t += ALIGN16(K * sizeof(float)) * 3;            /* r2 r2_unb nearend_pwr */
    t += ALIGN16(K * sizeof(unsigned char));        /* stat_mask */
    /* hop scratch (10) */
    t += ALIGN16(K * sizeof(float));                /* per_bin_mu_scale */
    t += ALIGN16((size_t)hop * sizeof(float)) * 6;  /* limiter_near_lag near_hop far_hop raw shadow final */
    t += ALIGN16((size_t)np * hop * sizeof(float)); /* filter_taps_full */
    t += ALIGN16((size_t)np * K * sizeof(Complex)) * 2; /* W_all X_buf_all */
    /* per-hop freq-bin scratch (12; see aec.h struct comment) */
    t += ALIGN16((size_t)hop * sizeof(float));      /* scr_sq */
    t += ALIGN16((size_t)K * sizeof(float)) * 11;   /* scr_e2_echo .. scr_erl_arr */
    return t;
}

/* Place Aec + all backing arrays in the provided pool; no malloc called.
 * Returns (Aec*)mem on success, NULL on invalid inputs or undersized pool. */
Aec* aec_init(void* mem, size_t mem_size, const AecConfig* cfg) {
    if (!mem || !cfg || mem_size < aec_get_mem_size(cfg)) return NULL;

    int hop, blk, fft, K, np, buf_samp, fcap;
    aec_derive_dims(cfg, &hop, &blk, &fft, &K, &np, &buf_samp, &fcap);

    Aec* a = (Aec*)mem;
    uint8_t* ptr = (uint8_t*)mem + ALIGN16(sizeof(Aec));
    memset(a, 0, sizeof(Aec));
    a->cfg = *cfg;
    a->hop_size = hop; a->block_size = blk; a->fft_size = fft;
    a->n_freqs = K; a->n_partitions = np;

    /* inst-ERLE slope ring cap = Python _slope_n = max(2, int(0.5*sr/hop)),
     * clamped to the static array size. (orchestrator.py:649) */
    {
        int _sn = (int)(0.5f * (float)cfg->sample_rate / (float)(hop > 0 ? hop : 1));
        if (_sn < 2) _sn = 2;
        if (_sn > (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0])))
            _sn = (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0]));
        a->erle_slope_cap = _sn;
        a->erle_slope_len = 0;
        a->erle_slope_head = 0;
    }

    /* HPF */
    if (cfg->enable_highpass) { hpf_f64_init(&a->hp_mic, cfg->highpass_cutoff_hz, cfg->sample_rate); a->has_hp = 1; }
    /* Saturation */
    if (cfg->enable_saturation) {
        saturation_init(&a->sat_ref, cfg->saturation_threshold);
        saturation_init(&a->sat_mic, cfg->saturation_threshold);
        a->has_sat = 1;
    }
    /* Delay ring */
    if (cfg->enable_delay_est) {
        delay_aec3_init(&a->delay);
        a->has_delay = 1;
        a->ref_ring_size = buf_samp;
        a->ref_ring = (float*)ptr; ptr += ALIGN16((size_t)buf_samp * sizeof(float));
        memset(a->ref_ring, 0, (size_t)buf_samp * sizeof(float));
        a->current_delay = -1;
        a->pending_delay = -1; a->has_pending = 0; a->pending_delay_ttl = 0;
    } else {
        a->current_delay = 0;
    }
    a->duty_last_delay = -1;   /* duty-cycle change detect (rest zeroed by memset) */
    /* Streaming FIFO */
    a->fifo_cap_hops = fcap;
    a->render_fifo = (float*)ptr; ptr += ALIGN16((size_t)fcap * hop * sizeof(float));
    memset(a->render_fifo, 0, (size_t)fcap * hop * sizeof(float));
    a->fifo_count = 0; a->fifo_read = 0; a->fifo_write = 0;
    a->render_call_count = 0; a->capture_call_count = 0;
    a->last_buffering_event = AEC_BUF_NONE;

    /* RSA counters */
    a->rsa_counters = (int64_t*)ptr;
    size_t rsa_sz = ALIGN16((size_t)(K - 2 > 0 ? K - 2 : 1) * sizeof(int64_t));
    memset(ptr, 0, rsa_sz); ptr += rsa_sz;

    /* Main filter (PBFDKF static) */
    size_t kf_sz = pbfdkf_get_mem_size(blk, np, hop);
    pbfdkf_init_static(&a->main_filter, ptr, kf_sz, blk, np, cfg->mu, cfg->delta, hop);
    for (int k = 0; k < K; ++k) {
        a->main_filter.Q_high[k] = cfg->kalman_q_high;
        a->main_filter.Q_low[k]  = cfg->kalman_q_low;
        a->main_filter.Q[k]      = cfg->kalman_q_high;
    }
    a->main_filter.base.poor_excitation_counter = blocks_to_hops_round(1000, hop, cfg->sample_rate);
    a->main_filter.base.constraint_round_robin = 1;  /* AEC3 round-robin (main) */
    ptr += kf_sz;

    /* Shadow filter (PBFDAF static) */
    if (cfg->enable_shadow) {
        size_t af_sz = pbfdaf_get_mem_size(blk, np, hop);
        pbfdaf_init_static(&a->shadow_filter, ptr, af_sz, blk, np, cfg->shadow_mu_nlms, cfg->delta, hop);
        a->shadow_filter.poor_excitation_counter = a->main_filter.base.poor_excitation_counter;
        a->shadow_filter.saturated_capture = 0;
        /* FFT dedup: skip the shadow's unread near_spec / error_spec_windowed. */
        a->shadow_filter.lightweight = 1;
        a->shadow_filter.constraint_round_robin = 1;  /* AEC3 round-robin (shadow) */
        a->has_shadow = 1;
        ptr += af_sz;
    }

    /* Detectors / EPC / regime / RSA */
    render_activity_init(&a->render_activity);
    filter_convergence_init(&a->convergence);
    doubletalk_init(&a->dt_analyzer, 1.5, 3.0);
    epc_init(&a->epc, cfg->epc_hangover, cfg->epc_total_rise, cfg->epc_delta_threshold);
    shadow_copy_init(&a->regime, SC_GATE_ENERGY, 0.65, 3, cfg->epc_hangover);
    rsa_init(&a->rsa, a->rsa_counters, K, np);

    /* FFT handle */
    size_t fft_sz = fft_get_mem_size(fft);
    a->post_fft = fft_init(ptr, fft_sz, fft);
    ptr += fft_sz;

    /* aec3_post backing */
    {
        Aec3PostConfig pcfg;
        aec3_post_config_defaults(&pcfg);
        pcfg.n_bins = K; pcfg.fft_size = fft; pcfg.block_size = blk; pcfg.hop_size = hop;
        pcfg.erle_coh_gate_enabled       = AEC3B_ERLE_COH_GATE_ENABLED;
        pcfg.erle_windowed_capture_psd   = AEC3B_ERLE_WINDOWED_CAPTURE_PSD;
        pcfg.erle_render_x2_psd_scale    = AEC3B_ERLE_RENDER_X2_PSD_SCALE;
        pcfg.output_capture_when_linear_unusable = AEC3B_OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE;
        pcfg.enable_cng                  = cfg->enable_cng;
        pcfg.cng_n2_update_onset_hops    = AEC3B_CNG_N2_UPDATE_ONSET_HOPS;
        pcfg.cng_n2_initial_duration_hops = AEC3B_CNG_N2_INITIAL_DURATION_HOPS;
        pcfg.cng_y2_alpha                = AEC3B_CNG_Y2_ALPHA;
        pcfg.cng_n2_track_freshness      = AEC3B_CNG_N2_TRACK_FRESHNESS;
        pcfg.cng_n2_track_retention      = AEC3B_CNG_N2_TRACK_RETENTION;
        pcfg.cng_n2_slow_up              = AEC3B_CNG_N2_SLOW_UP;
        pcfg.cng_n2_initial_alpha        = AEC3B_CNG_N2_INITIAL_ALPHA;
        pcfg.noise_floor_int16sq         = AEC3B_NOISE_FLOOR_INT16SQ;
        pcfg.erle_coh_gate_alpha         = AEC3B_ERLE_COH_GATE_ALPHA;
        pcfg.erle_coh_gate_threshold     = AEC3B_ERLE_COH_GATE_THRESHOLD;

#define P_FSLICE(n) ((float*)(ptr));    ptr += ALIGN16(K * sizeof(float));    (void)(n)
#define P_DSLICE(n) ((double*)(ptr));   ptr += ALIGN16(K * sizeof(double));   (void)(n)
#define P_CSLICE(n) ((Complex*)(ptr));  ptr += ALIGN16(K * sizeof(Complex));  (void)(n)
#define P_BSLICE(n) ((unsigned char*)(ptr)); ptr += ALIGN16(K * sizeof(unsigned char)); (void)(n)
        float        *avg_rev  = P_FSLICE(0); float *y2s    = P_FSLICE(0);
        float        *n2       = P_FSLICE(0); float *n2i    = P_FSLICE(0);
        float        *sye_re   = P_FSLICE(0); float *sye_im = P_FSLICE(0);
        double       *syy      = P_DSLICE(0); double *see   = P_DSLICE(0);
        float        *ola      = (float*)ptr;  ptr += ALIGN16((size_t)blk * sizeof(float));
        float        *np_      = P_FSLICE(0); float *fp_    = P_FSLICE(0);
        float        *ep_      = P_FSLICE(0); float *erp_   = P_FSLICE(0);
        float        *cpe      = P_FSLICE(0); float *x2r    = P_FSLICE(0);
        float        *cn       = P_FSLICE(0); float *nf_    = P_FSLICE(0);
        unsigned char *cgm     = P_BSLICE(0);
        Complex      *eout     = P_CSLICE(0);
        float        *eout_full = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
        memset(syy, 0, K * sizeof(double)); memset(see, 0, K * sizeof(double));
        aec3_post_init(&a->post, &pcfg, a->post_fft,
                       AEC3B_SYNTH_WINDOW, AEC3B_SQRT2_SIN_LUT, avg_rev,
                       y2s, n2, n2i, sye_re, sye_im, syy, see, ola,
                       np_, fp_, ep_, erp_, cpe, x2r, cgm, cn, nf_,
                       eout, eout_full);
    }

    /* AecStateStorage */
    {
        AecStateStorage* s = &a->a3_state_st;
        s->erle_max           = P_FSLICE(0); s->erle            = P_FSLICE(0);
        s->erle_oc            = P_FSLICE(0); s->erle_unb        = P_FSLICE(0);
        s->erle_during        = P_FSLICE(0);
        s->erle_coming_onset  = P_BSLICE(0);
        s->erle_hold          = (int32_t*)ptr; ptr += ALIGN16(K * sizeof(int32_t));
        s->erle_y2_acc        = P_FSLICE(0); s->erle_e2_acc     = P_FSLICE(0);
        s->erle_low_render    = P_BSLICE(0);
        s->erl                = P_FSLICE(0);
        s->erl_hold           = (int*)ptr; ptr += ALIGN16((size_t)(K - 2) * sizeof(int));
        s->filter_delays_blocks = (int*)ptr; ptr += ALIGN16((size_t)AEC3B_ST_NUM_CAPTURE_CHANNELS * sizeof(int));
        s->fa_h_highpass      = (float*)ptr; ptr += ALIGN16(AEC3B_FILTER_TAPS_SIZE * sizeof(float));
        {
            AecStateConfig acfg; aec_state_config_defaults(&acfg);
            acfg.n_bins = K; acfg.num_capture_channels = AEC3B_ST_NUM_CAPTURE_CHANNELS;
            acfg.hop_size = hop; acfg.enable_filter_analyzer = AEC3B_ST_ENABLE_FILTER_ANALYZER;
            acfg.erle_startup_hops = AEC3B_ST_ERLE_STARTUP_HOPS;
            acfg.erl_startup_hops  = AEC3B_ST_ERL_STARTUP_HOPS;
            acfg.echo_can_saturate = AEC3B_ST_ECHO_CAN_SATURATE;
            acfg.use_linear_filter = AEC3B_ST_USE_LINEAR_FILTER;
            acfg.conservative_initial_phase = AEC3B_ST_CONSERVATIVE_INITIAL_PHASE;
            acfg.delay_headroom_samples = AEC3B_ST_DELAY_HEADROOM_SAMPLES;
            acfg.initial_state_seconds  = AEC3B_ST_INITIAL_STATE_SECONDS;
            acfg.erle_min = AEC3B_ST_ERLE_MIN; acfg.erle_max_l = AEC3B_ST_ERLE_MAX_L;
            acfg.erle_max_h = AEC3B_ST_ERLE_MAX_H;
            acfg.filter_taps_size = AEC3B_FILTER_TAPS_SIZE;
            aec_state_init(&a->a3_state, &acfg, s);
        }
    }

    /* ResidualEchoEstimator */
    {
        int rh = (AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW ? 1 : 0) + 2;
        float *x2_nf  = P_FSLICE(0);
        int   *x2_nfc = (int*)ptr;   ptr += ALIGN16(K * sizeof(int));
        float *rm_st  = P_FSLICE(0); float *rt_st = P_FSLICE(0);
        float *rh_st  = (float*)ptr; ptr += ALIGN16((size_t)rh * K * sizeof(float));
        float *drd_st = (float*)ptr; ptr += ALIGN16((size_t)REE_DELAY_BUF_SIZE * K * sizeof(float));
        float *rrd_st = (float*)ptr; ptr += ALIGN16((size_t)REE_DELAY_BUF_SIZE * K * sizeof(float));
        float *ld_st  = P_FSLICE(0); float *lr_st = P_FSLICE(0); float *scr_st = P_FSLICE(0);
        ReeEchoModelConfig em; memset(&em, 0, sizeof(em));
        em.min_noise_floor_power = AEC3B_REE_MIN_NOISE_FLOOR_POWER;
        em.noise_gate_power      = AEC3B_REE_NOISE_GATE_POWER_LEGACY;
        em.noise_gate_slope      = AEC3B_REE_NOISE_GATE_SLOPE;
        em.stationary_gate_slope = AEC3B_REE_STATIONARY_GATE_SLOPE;
        em.model_reverb_in_nonlinear_mode = AEC3B_REE_MODEL_REVERB_IN_NL;
        ree_init(&a->a3_ree, K, AEC3B_REE_HOP_SIZE, &em,
                 AEC3B_REE_DEFAULT_GAIN, AEC3B_REE_TM_GAIN, AEC3B_REE_ERLE_ONSET_COMP,
                 AEC3B_REE_REVERB_DECAY, AEC3B_REE_REVERB_MILD_SCALE,
                 AEC3B_REE_REVERB_ENABLED, AEC3B_REE_REVERB_TAIL_STRENGTH,
                 AEC3B_REE_USE_AEC3_RESIDUAL_NOISE_GATE,
                 AEC3B_USE_STATIONARITY_PROPERTIES,
                 AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW,
                 AEC3B_REE_NL_R2_ENABLED, AEC3B_REE_NL_R2_ALPHA, AEC3B_REE_NL_NORM_POWER,
                 AEC3B_REE_RESIDUAL_NOISE_GATE_POWER, AEC3B_REE_NOISE_FLOOR_HOLD_HOPS,
                 AEC3B_REE_USE_FREQ_RESPONSE, AEC3B_REE_REVERB_USE_CONSERVATIVE,
                 AEC3B_REE_REVERB_SMOOTHING_BASE,
                 x2_nf, x2_nfc, rm_st, rt_st, rh_st, drd_st, rrd_st,
                 ld_st, lr_st, scr_st);
    }

    /* SuppressionGain */
    {
        SuppressionGain* sg = &a->a3_sg;
        SuppressionGainConfig scfg; SuppressionGainTuning stun;
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
        scfg.aud_lf_end_bin = AEC3B_SG_AUD_LF_END_BIN; scfg.aud_mf_end_bin = AEC3B_SG_AUD_MF_END_BIN;
        scfg.floor_power = AEC3B_SG_FLOOR_POWER;
        scfg.aud_thr_lf = AEC3B_SG_AUD_THR_LF; scfg.aud_thr_mf = AEC3B_SG_AUD_THR_MF;
        scfg.aud_thr_hf = AEC3B_SG_AUD_THR_HF;
        scfg.low_render_limit = AEC3B_SG_LOW_RENDER_LIMIT;
        scfg.normal_render_limit = AEC3B_SG_NORMAL_RENDER_LIMIT;
        scfg.hf_lgb = AEC3B_SG_HF_LGB; scfg.hf_biq = AEC3B_SG_HF_BIQ;
        scfg.conservative_hf = AEC3B_SG_CONSERVATIVE_HF;
        scfg.max_inc_normal = (float)AEC3B_SG_MAX_INC;
        scfg.max_inc_nearend = (float)AEC3B_SG_MAX_INC;
        scfg.max_dec_lf_normal = (float)AEC3B_SG_MAX_DEC_LF;
        scfg.max_dec_lf_nearend = (float)AEC3B_SG_MAX_DEC_LF;
        scfg.floor_first_increase = 0.00001;
        scfg.low_render_threshold = AEC3B_SG_LOW_RENDER_THRESHOLD;
        scfg.split_floor_enabled = 1;
        scfg.split_floor_far_active =
            powf(10.0f, cfg->min_gain_floor_far_active_db / 10.0f);
        scfg.split_floor_far_silent = AEC3B_SG_SPLIT_FLOOR_FAR_SILENT;
        /* DT-gated floor: 10^(min_gain_floor_dt_db/10) (mirrors aec_create). */
        scfg.split_floor_dt =
            powf(10.0f, cfg->min_gain_floor_dt_db / 10.0f);
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
        stun.nearend_enr_tr = AEC3B_SG_NEAREND_ENR_TR; stun.nearend_enr_su = AEC3B_SG_NEAREND_ENR_SU;
        stun.nearend_emr_tr = AEC3B_SG_NEAREND_EMR_TR;
        stun.normal_enr_tr  = AEC3B_SG_NORMAL_ENR_TR;  stun.normal_enr_su  = AEC3B_SG_NORMAL_ENR_SU;
        stun.normal_emr_tr  = AEC3B_SG_NORMAL_EMR_TR;
        float *last_gain = P_FSLICE(0); float *last_ne = P_FSLICE(0); float *last_echo = P_FSLICE(0);
        float *ma   = (float*)ptr; ptr += ALIGN16((size_t)AEC3B_SG_NEAREND_SMOOTHER_N * K * sizeof(float));
        float *ne_s = P_FSLICE(0); float *wr_s = P_FSLICE(0);
        float *ming = P_FSLICE(0); float *maxg = P_FSLICE(0);
        float *graw = P_FSLICE(0); float *gout = P_FSLICE(0); float *gsum = P_FSLICE(0);
        suppression_gain_init(sg, &scfg, &stun, last_gain, last_ne, last_echo,
                              ma, ne_s, wr_s, ming, maxg, graw, gout, gsum);
    }

    /* StationarityEstimator */
    {
        float        *st_noise = P_FSLICE(0);
        int32_t      *st_hang  = (int32_t*)ptr; ptr += ALIGN16(K * sizeof(int32_t));
        unsigned char *st_flg  = P_BSLICE(0);
        float        *st_hist  = (float*)ptr;   ptr += ALIGN16(16 * K * sizeof(float));
        stationarity_estimator_init(&a->a3_stat, K, hop, cfg->sample_rate,
                                    st_noise, st_hang, st_flg, st_hist);
    }

    /* LinearFilterSelect (manual slice — no static API) */
    {
        memset(&a->a3_lfs, 0, sizeof(a->a3_lfs));
        a->a3_lfs.hop        = hop;
        a->a3_lfs.block_size = blk;
        a->a3_lfs.fft_size   = fft;
        a->a3_lfs.n_freqs    = K;
        a->a3_lfs.prev_output_time = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.e_form           = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.block_win        = (float*)ptr; ptr += ALIGN16((size_t)blk * sizeof(float));
        a->a3_lfs.sel_esw          = (Complex*)ptr; ptr += ALIGN16(K * sizeof(Complex));
        linear_filter_select_reset(&a->a3_lfs);
    }

    /* Aec3PostRunScratch */
    {
        Aec3PostRunScratch* sc = &a->a3_sc;
        sc->sel_esw  = P_CSLICE(0); sc->sel_echo = P_CSLICE(0);
        sc->nsw_e1   = P_CSLICE(0); sc->ybase    = P_CSLICE(0);
        sc->abs_near     = P_FSLICE(0); sc->abs_far      = P_FSLICE(0);
        sc->abs_sel_echo = P_FSLICE(0); sc->abs_error    = P_FSLICE(0);
        sc->abs_echo_coh = P_FSLICE(0); sc->abs_nsw_e1   = P_FSLICE(0);
        sc->abs_ybase    = P_FSLICE(0); sc->x2_at_delay  = P_FSLICE(0);
        sc->x2_past      = P_FSLICE(0);
        sc->w_mag2 = (float*)ptr; ptr += ALIGN16((size_t)np * K * sizeof(float));
        sc->render_block_scaled = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        sc->bridge_taps         = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
        sc->r2          = P_FSLICE(0); sc->r2_unb   = P_FSLICE(0);
        sc->nearend_pwr = P_FSLICE(0);
        sc->stat_mask   = P_BSLICE(0);
    }
#undef P_FSLICE
#undef P_DSLICE
#undef P_CSLICE
#undef P_BSLICE

    /* hop scratch */
    a->per_bin_mu_scale = (float*)ptr; ptr += ALIGN16(K * sizeof(float));
    a->limiter_near_lag = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->near_hop   = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->far_hop    = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->raw_output = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->shadow_out = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->final_out  = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->filter_taps_full = (float*)ptr; ptr += ALIGN16((size_t)np * hop * sizeof(float));
    a->W_all     = (Complex*)ptr; ptr += ALIGN16((size_t)np * K * sizeof(Complex));
    a->X_buf_all = (Complex*)ptr; ptr += ALIGN16((size_t)np * K * sizeof(Complex));

    /* per-hop freq-bin scratch (see aec.h struct comment). */
    a->scr_sq        = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->scr_e2_echo   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2_near   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_rsa_psd   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_rsa_mask  = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_mu_buf_pre= (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2coa_pre = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_mu_buf    = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_far_psd   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2ref_arr = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2coa_arr = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_erl_arr   = (float*)ptr; /* last — no advance needed */

    /* scalar state (mirrors aec_create tail) */
    a->pending_gain_change = 0; a->pending_delay_change = -1;
    a->stationarity_active_hops = 0; a->non_zero_render_seen = 0;
    a->render_peak_floor = 10.0f / 32768.0f;
    a->block_stationary_next = 0;
    a->stationarity_converge_hops = AEC3B_STATIONARITY_CONVERGE_HOPS;
    a->saturation_level = 0.0f; a->erl_estimate = 0.1f;
    a->main_err_smooth = 0.0f; a->shadow_err_smooth = 0.0f;
    a->shadow_frame_count = 0; a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10f; a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f; a->inst_erle_smooth = 1.0f;
    a->wn_err_baseline = 1e-8f; a->stat_dt_hangover = 0;
    a->warmup_frames_remaining = cfg->warmup_frames; a->warmup_far_active = 0;
    a->simple_mu_ratio = 1.0f; a->simple_mu_holdoff = 0; a->has_per_bin_mu = 0;
    a->limiter_gain = 1.0f; a->has_limiter_lag = 0;
    a->near_power = 0.0f; a->raw_error_power = 0.0f; a->alpha_pow = 0.95f;
    a->frame_count = 0; a->poor_coarse_counter = 0; a->coarse_reset_hangover = 0;
    a->misadj_e2_acum = 0.0f; a->misadj_y2_acum = 0.0f; a->misadj_n_acum = 0;
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_stable_count = 0; a->misadj_hangover_remaining = 0;
    a->last_erle_windowed = 0.0f;
    filter_q_high(&a->main_filter);
    a->is_static = 1;
    return a;
}

void aec_destroy(Aec* a) {
    if (!a) return;
    /* Release library-internal FFT/filter allocations on BOTH the heap and
     * static-pool paths. pbfdkf_free/pbfdaf_free already guard their own
     * pool-owned buffers internally (is_static branch frees nothing from the
     * pool) but they still call fft_destroy() on their nested FFT handle,
     * which is the only thing that releases a NE10 backend's R2C twiddle
     * config — an allocation that lives outside the caller's pool and is
     * therefore NOT reclaimed by free(pool). Same story for post_fft below.
     * fft_destroy() itself is a no-op for a NULL handle and (KISS backend)
     * for a pool-owned handle, so this is safe pre- or post- pool-teardown. */
    fft_destroy(a->post_fft);
    pbfdkf_free(&a->main_filter);
    if (a->has_shadow) pbfdaf_free(&a->shadow_filter);

    if (a->is_static) return;   /* caller owns the pool: nothing else to free */

    if (a->has_delay) free(a->ref_ring);
    free(a->render_fifo);
    free(a->rsa_counters);
    /* sub-module storage frees omitted (process owns no extra; OS reclaims on
     * exit). The malloc'd backing arrays are reachable only through the
     * structs; freeing them individually is a maintenance burden with no
     * functional effect for the CLI/golden lifecycle. */
    free(a->per_bin_mu_scale); free(a->limiter_near_lag);
    free(a->near_hop); free(a->far_hop); free(a->raw_output);
    free(a->shadow_out); free(a->final_out);
    free(a->filter_taps_full); free(a->W_all); free(a->X_buf_all);
    free(a->scr_sq); free(a->scr_e2_echo); free(a->scr_e2_near);
    free(a->scr_rsa_psd); free(a->scr_rsa_mask); free(a->scr_mu_buf_pre);
    free(a->scr_e2coa_pre); free(a->scr_mu_buf); free(a->scr_far_psd);
    free(a->scr_e2ref_arr); free(a->scr_e2coa_arr); free(a->scr_erl_arr);
}

int aec_hop_size(const Aec* a) { return a->hop_size; }

void aec_reset(Aec* a) {
    if (a->has_hp) hpf_f64_reset(&a->hp_mic);
    if (a->has_sat) { saturation_reset(&a->sat_ref); saturation_reset(&a->sat_mic); }
    if (a->has_delay) {
        delay_aec3_reset(&a->delay);
        memset(a->ref_ring, 0, (size_t)a->ref_ring_size * sizeof(float));
        a->ref_ring_write = 0; a->ref_ring_filled = 0;
        a->current_delay = -1; a->pending_delay = -1; a->has_pending = 0;
        a->pending_delay_ttl = 0;
    }
    a->duty_active = 0; a->duty_stable_hops = 0; a->duty_pos = 0;
    a->duty_last_delay = -1; a->duty_erle_peak = 0.0f;
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
    a->saturation_level = 0.0f;
    a->erl_estimate = 0.1f;
    a->main_err_smooth = 0.0f; a->shadow_err_smooth = 0.0f;
    a->shadow_frame_count = 0;
    a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10f; a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f; a->inst_erle_smooth = 1.0f;
    a->wn_err_baseline = 1e-8f; a->stat_dt_hangover = 0;
    a->warmup_frames_remaining = a->cfg.warmup_frames;
    a->warmup_far_active = 0;
    a->simple_mu_ratio = 1.0f; a->simple_mu_holdoff = 0; a->has_per_bin_mu = 0;
    a->limiter_gain = 1.0f; a->has_limiter_lag = 0;
    a->near_power = 0.0f; a->raw_error_power = 0.0f;
    a->frame_count = 0;
    a->ne_above = 0; a->ne_recent_frames = 0;
    a->poor_coarse_counter = 0; a->coarse_reset_hangover = 0;
    a->leakage_div_sustained_counter = 0;
    a->misadj_e2_acum = 0.0f; a->misadj_y2_acum = 0.0f; a->misadj_n_acum = 0;
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_stable_count = 0; a->misadj_hangover_remaining = 0;
    a->last_erle_windowed = 0.0f;
}

/* ───────────────────────── misadjustment estimator ─────────────────────── */

/* _update_misadjustment_estimator (orchestrator 100-148). float32-by-design
 * (Stage-2 conversion; block-energy sums accumulate in int16²-domain
 * magnitudes, same domain as the power EMAs above). */
static void misadj_update(Aec* a, const float* near_hpf, const float* raw_out, int hop) {
    float e2_block = 0.0f, y2_block = 0.0f;
    for (int i = 0; i < hop; ++i) {
        float r = raw_out[i]; e2_block += r * r;
        float n = near_hpf[i]; y2_block += n * n;
    }
    a->misadj_e2_acum += e2_block;
    a->misadj_y2_acum += y2_block;
    a->misadj_n_acum += 1;
    int n_hops_target = 2;
    if (a->misadj_n_acum < n_hops_target) return;
    float total_samples = (float)(n_hops_target * hop);
    float int16_sq = 32768.0f * 32768.0f;
    float y2_threshold = (200.0f * 200.0f) * total_samples / int16_sq;
    float e2_overhang_threshold = (7500.0f * 7500.0f) * total_samples / int16_sq;
    if (a->misadj_y2_acum > y2_threshold) {
        float denom = a->misadj_y2_acum > 1e-20f ? a->misadj_y2_acum : 1e-20f;
        float update = a->misadj_e2_acum / denom;
        if (a->misadj_e2_acum > e2_overhang_threshold) a->misadj_overhang = 4;
        else { a->misadj_overhang -= 1; if (a->misadj_overhang < 0) a->misadj_overhang = 0; }
        if ((update < a->misadj_inv) || (a->misadj_overhang > 0))
            a->misadj_inv += 0.1f * (update - a->misadj_inv);
    }
    a->misadj_e2_acum = 0.0f; a->misadj_y2_acum = 0.0f; a->misadj_n_acum = 0;
}

/* _fire_aec3_misadj_scale (orchestrator 150-191). float32-by-design. */
static void misadj_fire(Aec* a) {
    if (a->misadj_hangover_remaining > 0) { a->misadj_hangover_remaining--; return; }
    int stable = a->convergence.converged && !a->epc.active && !a->regime.main_paused;
    if (!stable) { a->misadj_stable_count = 0; return; }
    a->misadj_stable_count++;
    if (a->misadj_stable_count < a->cfg.filter_misadjustment_stable_frames) return;
    if (a->misadj_inv <= 10.0f) return;
    float base = a->misadj_inv > 1e-6f ? a->misadj_inv : 1e-6f;
    float scale_raw = 2.0f / sqrtf(base);
    float scale = scale_raw;
    if (scale < a->cfg.filter_misadjustment_scale_min) scale = a->cfg.filter_misadjustment_scale_min;
    if (scale > a->cfg.filter_misadjustment_scale_max) scale = a->cfg.filter_misadjustment_scale_max;
    pbfdkf_scale_filter(&a->main_filter, scale);
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_hangover_remaining = a->cfg.filter_misadjustment_hangover_frames;
}

/* max() of the last 15 inst-ERLE ring entries — the warm-transfer gate input.
 * Mirrors Python `max(list(self._erle_slope_buf)[-15:] or [0.0])`
 * (orchestrator.py:1409): 0.0 when the ring is empty, else the true max
 * (which may be negative) of the most recent <=15 appends. */
static float aec_erle_ring_max_last15(const Aec* a) {
    int n = a->erle_slope_len < 15 ? a->erle_slope_len : 15;
    if (n <= 0) return 0.0f;
    int idx = a->erle_slope_head;   /* head = next write = one past newest */
    float m = 0.0f; int got = 0;
    for (int i = 0; i < n; i++) {
        idx = (idx - 1 + a->erle_slope_cap) % a->erle_slope_cap;
        float v = a->erle_slope_buf[idx];
        if (!got || v > m) { m = v; got = 1; }
    }
    return m;
}

/* ───────────────────────── process ─────────────────────────────────────── */

void aec_process(Aec* a, const float* mic_in, const float* ref_in, float* out) {
    const int hop = a->hop_size, K = a->n_freqs, N = a->n_partitions;
    memcpy(a->near_hop, mic_in, (size_t)hop * sizeof(float));
    memcpy(a->far_hop,  ref_in, (size_t)hop * sizeof(float));

    /* 1. mic HPF (ref HPF OFF). */
    if (a->has_hp) hpf_f64_process(&a->hp_mic, a->near_hop, a->near_hop, hop);

    /* Held "near-end seen recently" gate for DT-aware soft recovery (mirrors
     * Python orchestrator 16285fd). Reads the PREVIOUS frame's dt_from_energy
     * (doubletalk_update_energy_dt runs later in this frame, in the RES block),
     * so the value here is the prior hop's — exactly like Python's _dt_from_energy
     * property read at the top of process(). dt_from_energy ONLY: it is ~0 in
     * far-end single-talk (mic ≈ echo), so FS never arms the gate (FS echo depth
     * preserved). Require `sustain` consecutive frames above threshold before
     * arming, then hold for `hold` frames. */
    {
        float ne_ind = (float)a->dt_analyzer.dt_from_energy;  /* receive: DoubleTalk (Stage-3) is double */
        if (ne_ind > a->cfg.ne_recent_threshold)
            a->ne_above += 1;
        else
            a->ne_above = 0;
        if (a->ne_above >= a->cfg.ne_recent_sustain)
            a->ne_recent_frames = a->cfg.ne_recent_hold;
        else if (a->ne_recent_frames > 0)
            a->ne_recent_frames -= 1;
        else
            a->ne_recent_frames = 0;
    }

    /* 2. saturation. */
    float sat_ref = 0.0f;  /* receive: Saturation (Stage-3) getter returns double */
    if (a->has_sat) {
        sat_ref = (float)saturation_detect(&a->sat_ref, a->far_hop, hop);
        float sat_mic = (float)saturation_detect(&a->sat_mic, a->near_hop, hop);
        a->saturation_level = (sat_ref > sat_mic * 0.5f) ? sat_ref : sat_mic * 0.5f;
        if (a->cfg.saturation_softclip_ref && sat_ref > 0.1f)
            saturation_soft_clip(a->far_hop, a->far_hop, hop, 0.8);
    }

    /* 3. delay estimation + ring-buffer alignment. Duty-cycled analysis is
     * ALWAYS ON (baked in — see the duty_* field doc in aec.h): the estimator
     * is FED every hop regardless (accumulate_ex keeps decimators + ring
     * gapless); only the matched-filter analysis is decimated to 1-in-K once
     * the estimate has been solid+unchanged for delay_est_init_s. This is an
     * intentional, sampled-cost-free divergence from the Python reference
     * (which always analyses every hop). */
    if (a->has_delay) {
        float hop_s = (float)hop / (float)a->cfg.sample_rate;
        int hold_hops = (int)lrintf(a->cfg.delay_est_init_s / hop_s);
        int K = (int)lrintf(a->cfg.delay_est_period_s / hop_s) / 5;
        int run_filter = 1;
        if (hold_hops < 1) hold_hops = 1;
        if (K < 2) K = 2;
        if (a->duty_active) {
            a->duty_pos += 1;
            if (a->duty_pos >= K) a->duty_pos = 0;
            run_filter = (a->duty_pos == 0);
        }
        delay_aec3_accumulate_ex(&a->delay, a->near_hop, a->far_hop, hop,
                                 run_filter);
        {
            int cur = delay_aec3_estimated_delay(&a->delay);
            int solid = delay_aec3_is_solid(&a->delay);
            if (!solid || cur < 0 || cur != a->duty_last_delay) {
                /* estimate moved / lost confidence → full rate + re-arm */
                a->duty_active = 0;
                a->duty_stable_hops = 0;
            } else if (!a->duty_active) {
                a->duty_stable_hops += 1;
                if (a->duty_stable_hops >= hold_hops) {
                    a->duty_active = 1;
                    a->duty_pos = 0;
                    a->duty_erle_peak = a->last_erle_windowed;
                }
            }
            a->duty_last_delay = cur;
        }
        if (a->duty_active) {
            /* ERLE watchdog: leaky peak (~0.1 dB/s at 10 ms hops); resume
             * full-rate analysis on a >6 dB collapse from that peak. Armed
             * only once the peak exceeds 6 dB so it cannot fire before the
             * filter ever converged. */
            if (a->last_erle_windowed > a->duty_erle_peak)
                a->duty_erle_peak = a->last_erle_windowed;
            else
                a->duty_erle_peak -= 0.001f;
            if (a->duty_erle_peak > 6.0f &&
                a->last_erle_windowed < a->duty_erle_peak - 6.0f) {
                a->duty_active = 0;
                a->duty_stable_hops = 0;
                a->duty_erle_peak = 0.0f;
            }
        }
        int new_delay = delay_aec3_estimated_delay(&a->delay);
        int eligible = (new_delay >= 0 && delay_aec3_n_updates(&a->delay) >= 3);

        /* Path A — first acquisition. */
        int already_cancelling = a->cfg.delay_acquire_protect_converged
                                 && (a->last_erle_windowed > 2.5f);
        /* option-A (default-OFF): also protect via inst-ERLE peak (orch 1387). */
        if (a->cfg.delay_acquire_protect_inst_erle
                && aec_erle_ring_max_last15(a) > a->cfg.delay_acquire_inst_erle_db)
            already_cancelling = 1;
        if (eligible && a->current_delay < 0 && delay_aec3_is_solid(&a->delay)
                && !already_cancelling) {
            a->current_delay = new_delay;
            /* Warm tap-transfer (orch 1407-1422): if the filter is already
             * cancelling (inst-ERLE peak > thresh) AND the delay fits the tap
             * reach, shift the learned IR by the delay instead of zeroing — the
             * cold-start cancellation survives the realign (the "line" fix).
             * Outside that, fall through to soft / hard reset unchanged. */
            int warm_ok = 0;
            if (a->cfg.delay_acquire_warm_transfer) {
                float wpk = aec_erle_ring_max_last15(a);
                int reach = a->main_filter.base.n_partitions
                            * a->main_filter.base.hop_size;
                warm_ok = (wpk > a->cfg.delay_acquire_inst_erle_db)
                          && (new_delay > 0) && (new_delay < reach);
            }
            if (warm_ok) {
                pbfdaf_warm_shift_ir(&a->main_filter.base, new_delay);
                if (a->has_shadow) pbfdaf_warm_shift_ir(&a->shadow_filter, new_delay);
            } else if (a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
                /* Soft acquisition (mirrors Python 16285fd): apply the ring
                 * alignment (current_delay set above) + reset filter excitation
                 * counters ONLY; keep the converged taps + let them re-adapt at
                 * normal step size. No tap wipe, no mark_diverged, no AecState
                 * full reset (no pending_delay_change). Avoids the aggressive-
                 * recovery tail overfitting near-end that arrives after this
                 * (far-only) acquisition. */
                pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
                if (a->has_shadow) {
                    a->shadow_filter.poor_excitation_counter = AEC3_POOR_EXC_COUNTER_INITIAL_HOPS;
                    a->shadow_filter.call_counter = 0;
                }
            } else {
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
        }

        /* pending TTL aging (once per estimation cycle / per hop). */
        if (a->pending_delay_ttl > 0) {
            a->pending_delay_ttl--;
            if (a->pending_delay_ttl <= 0) { a->has_pending = 0; a->pending_delay = -1; }
        }

        /* Path B — delay shift. */
        float conf = delay_aec3_confidence(&a->delay);
        if (eligible && a->current_delay >= 0 && conf >= 0.5
                && abs(new_delay - a->current_delay) > 32) {
            if (a->has_pending && abs(new_delay - a->pending_delay) < 16) {
                a->current_delay = new_delay;
                a->has_pending = 0; a->pending_delay = -1; a->pending_delay_ttl = 0;
                if (a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
                    /* Soft realign (mirrors Python 16285fd): the ring read
                     * offset is already updated (current_delay above). Keep the
                     * converged filter, re-adapt at normal step size — no tap
                     * wipe, no Kalman P-override, no q-boost, no mark_diverged,
                     * no epc_force_delay, no pending_delay_change. Avoids the
                     * over-cancel + near-end overfit when movement re-locks
                     * through double-talk. (Python only sets the dead
                     * _epc_reset_fired_this_frame FQA flag here — no audio
                     * effect, not modelled in C.) */
                } else {
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
                }
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
    float far_pwr_global = (float)ra.far_pwr;  /* receive: RenderActivity (Stage-3) is double */

    /* 5. mu_scale (simple variable mu, Valin 2007 RER-inspired). */
    int mu_is_array = 0;
    float mu_scalar = get_simple_mu_scale(a, &mu_is_array);

    /* 6. mic-clip emergency. */
    if (a->has_sat && a->sat_mic.saturation_level > 0.8) {
        mu_scalar = 0.0f;
        mu_is_array = 0;
    }

    /* 7. RSA update + poor-excitation counters + saturated_capture. The PSD is
     *    |filter.far_spec|² from the PREVIOUS hop (far_spec set inside step-9
     *    pbfdkf_process). First hop: far_spec is zero. */
    {
        float *rsa_psd = a->scr_rsa_psd;
        for (int k = 0; k < K; ++k)
            rsa_psd[k] = cmag2_c(a->main_filter.base.far_spec[k].r,
                                 a->main_filter.base.far_spec[k].i);
        rsa_update(&a->rsa, rsa_psd, a->far_hop, hop);
        int poor = rsa_poor_signal_excitation(&a->rsa);
        if (poor) a->main_filter.base.poor_excitation_counter = 0;
        else      a->main_filter.base.poor_excitation_counter += 1;
        a->main_filter.base.saturated_capture = (a->saturation_level > 0.5f);
        if (a->has_shadow) {
            if (poor) a->shadow_filter.poor_excitation_counter = 0;
            else      a->shadow_filter.poor_excitation_counter += 1;
            a->shadow_filter.saturated_capture = (a->saturation_level > 0.5f);
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
    float *rsa_mask = a->scr_rsa_mask;
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
        float far_mean_pre = mean_sq(a->far_hop, hop, a->scr_sq);
        far_excited = (far_mean_pre > 1e-4f);
        int saturation_safe_pre = (a->saturation_level < 0.5f);
        float shadow_mu_pre = (far_excited && saturation_safe_pre) ? 1.0f : 0.1f;
        if (rsa_mask_active) {
            float *mu_buf_pre = a->scr_mu_buf_pre;
            for (int k = 0; k < K; ++k) mu_buf_pre[k] = shadow_mu_pre * rsa_mask[k];
            pbfdaf_process(&a->shadow_filter, a->near_hop, a->far_hop, mu_buf_pre, 0.0f, a->shadow_out);
        } else {
            pbfdaf_process(&a->shadow_filter, a->near_hop, a->far_hop, NULL,
                           shadow_mu_pre, a->shadow_out);
        }
        /* Publish current-hop e2_coarse to main_filter BEFORE main runs. */
        {
            float *e2coa_pre = a->scr_e2coa_pre;
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
    /* FFT dedup: the shadow ran pre-main on the SAME far_hop with an identical
     * far_buffer (lockstep shift + paired reset), so reuse its far_spec instead
     * of recomputing — byte-equal, saves 1 FFT/hop. One-shot (frontend clears). */
    a->main_filter.base.precomputed_far_spec =
        a->has_shadow ? a->shadow_filter.far_spec : NULL;
    float main_mu_scalar = a->regime.main_paused ? 0.0f : mu_scalar;
    {
        float *mu_buf = a->scr_mu_buf;
        int use_array = mu_is_array;
        if (mu_is_array) {
            float pause = a->regime.main_paused ? 0.0f : 1.0f;
            for (int k = 0; k < K; ++k) mu_buf[k] = a->per_bin_mu_scale[k] * pause;
        }
        if (rsa_mask_active && !use_array) {
            /* Python: scalar → full(scalar) → × mask. */
            for (int k = 0; k < K; ++k) mu_buf[k] = main_mu_scalar * rsa_mask[k];
            use_array = 1;
        } else if (rsa_mask_active && use_array) {
            for (int k = 0; k < K; ++k) mu_buf[k] = mu_buf[k] * rsa_mask[k];
        }
        if (use_array)
            pbfdkf_process(&a->main_filter, a->near_hop, a->far_hop, mu_buf, 0.0f, a->raw_output);
        else
            pbfdkf_process(&a->main_filter, a->near_hop, a->far_hop, NULL,
                           main_mu_scalar, a->raw_output);
    }

    /* 10. stationarity refresh for NEXT hop (StationarityEstimator on
     *     |filter.far_spec|²; first-render latch; block-stationary push). */
    {
        float far_max = 0.0f;
        for (int i = 0; i < hop; ++i) { float v = fabsf(a->far_hop[i]); if (v > far_max) far_max = v; }
        if (!a->non_zero_render_seen && far_max >= a->render_peak_floor)
            a->non_zero_render_seen = 1;
        if (a->non_zero_render_seen) {
            float *far_psd = a->scr_far_psd;
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
        float *e2ref_arr = a->scr_e2ref_arr, *e2coa_arr = a->scr_e2coa_arr,
              *erl_arr = a->scr_erl_arr;
        for (int k = 0; k < K; ++k)
            e2ref_arr[k] = cmag2_c(a->main_filter.base.error_spec[k].r,
                                   a->main_filter.base.error_spec[k].i);
        for (int k = 0; k < K; ++k)
            e2coa_arr[k] = cmag2_c(a->shadow_filter.error_spec[k].r,
                                   a->shadow_filter.error_spec[k].i);
        float e2_ref = aec3_post_pairwise_sum_f32(e2ref_arr, (size_t)K);
        float e2_coa = aec3_post_pairwise_sum_f32(e2coa_arr, (size_t)K);
        /* erl[k] = Σ_p |W_p[k]|². */
        for (int k = 0; k < K; ++k) erl_arr[k] = 0.0f;
        for (int part = 0; part < N; ++part) {
            const Complex* Wp = a->main_filter.base.W + (size_t)part * K;
            for (int k = 0; k < K; ++k) erl_arr[k] += cmag2_c(Wp[k].r, Wp[k].i);
        }
        /* publish to the refined filter. */
        a->main_filter.e2_coarse_for_refresh = e2_coa;
        for (int k = 0; k < K; ++k) a->main_filter.e2_coarse_per_bin[k] = e2coa_arr[k];
        a->main_filter.e2_coarse_per_bin_valid = 1;
        for (int k = 0; k < K; ++k) a->main_filter.erl_per_bin[k] = erl_arr[k];

        /* rescue: cond_fire = e2_ref < 0.5*e2_coa; threshold_hops=blocks_to_hops(5)=2. */
        int cond_fire = (e2_ref < 0.5f * e2_coa);
        int threshold_hops = 2;
        if (cond_fire) a->poor_coarse_counter += 1; else a->poor_coarse_counter = 0;
        if (a->poor_coarse_counter >= threshold_hops) {
            pbfdaf_copy_weights_from(&a->shadow_filter, &a->main_filter.base);
            a->coarse_reset_hangover = 10;  /* blocks_to_hops(25)=10 */
            a->poor_coarse_counter = 0;
            ree_reset(&a->a3_ree);          /* REE reset on rescue rising edge. */
        }
        /* Track F: sustained leakage_div gate. Fires after 5 consecutive hops
         * (50 ms) with >50% bins on the diverged-leakage branch — covers the
         * DT-onset window before coarse_reset_hangover kicks in. Mirrors
         * orchestrator.py _leakage_div_sustained_counter / _dt_leakage_gate. */
        float ld_frac = a->main_filter.last_leakage_div_frac;
        if (ld_frac > 0.5f) {
            if (a->leakage_div_sustained_counter < 10)
                a->leakage_div_sustained_counter++;
        } else {
            if (a->leakage_div_sustained_counter > 0)
                a->leakage_div_sustained_counter--;
        }
        int dt_leakage_gate = (a->leakage_div_sustained_counter >= 5);
        if (a->coarse_reset_hangover > 0) {
            a->coarse_reset_hangover--;
            a->main_filter.disallow_leakage_diverged = 1;
        } else if (dt_leakage_gate) {
            a->main_filter.disallow_leakage_diverged = 1;
        } else {
            a->main_filter.disallow_leakage_diverged = 0;
        }
    }

    /* 13. smoothed errs + DT analyzer + regime handler. */
    if (a->has_shadow) {
        float main_err   = (float)pbfdkf_get_error_energy(&a->main_filter);   /* receive: PBFDKF (Stage-3) is double */
        float shadow_err = (float)pbfdaf_get_error_energy(&a->shadow_filter); /* receive: PBFDAF (Stage-3) is double */
        float as = a->cfg.shadow_err_alpha, oas = 1.0f - as;
        a->main_err_smooth   = as * a->main_err_smooth   + oas * main_err;
        a->shadow_err_smooth = as * a->shadow_err_smooth + oas * shadow_err;
        doubletalk_update_shadow_dt(&a->dt_analyzer, a->shadow_frame_count,
                                    far_excited, a->main_err_smooth, a->shadow_err_smooth);
        int delay_reliable = a->has_delay && (delay_aec3_confidence(&a->delay) >= 0.5);
        ShadowCopyDecision dec = shadow_copy_update(
            &a->regime, a->shadow_frame_count, mean_sq(a->far_hop, hop, a->scr_sq),
            a->main_err_smooth, a->shadow_err_smooth,
            a->epc.active, a->saturation_level,
            a->dt_analyzer.dt_from_energy, 0.0, delay_reliable);
        if (dec.boost_q) filter_q_high(&a->main_filter);
        /* reverse_copy: no-op on NLMS shadow (Python never invokes it). */
    }

    /* 14. EPV trigger. */
    EpcEvent epv = epc_update_epv(&a->epc, far_pwr_global,
                                  a->convergence.converged, a->regime.main_paused);
    /* (Python's _epv_suppressed weak-filter damping is env-gated, default OFF,
     * so epv.fired here == fired-and-not-suppressed in the balanced config.) */
    if (epv.fired && a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
        /* Soft (mirrors Python 16285fd): EPV gain-change recovery is
         * double-talk-blind (far-power EMA swing only). While near recently
         * present, skip the aggressive Kalman re-adapt + mark_diverged + ERLE
         * reset; the converged filter tracks moderate gain change at its normal
         * step size. (Python only sets the dead _epc_reset_fired_this_frame FQA
         * flag — no audio effect, not modelled in C.) */
    } else if (epv.fired) {
        a->pending_gain_change = 1;
        pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
        filter_q_high(&a->main_filter);
        filter_convergence_mark_diverged(&a->convergence);
        a->epc_render_forced_remaining = a->cfg.epc_hangover;
        if (a->erl_estimate > 0.3f) a->erl_estimate = 0.3f;
    }

    /* 15. shadow_rise (only if shadow + converged); else hangover tick. */
    if (a->has_shadow && a->convergence.converged) {
        EpcEvent rise = epc_update_shadow_rise(&a->epc, a->main_err_smooth,
                                               a->shadow_err_smooth,
                                               a->render_activity.is_stationary);
        if (rise.fired && a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
            /* Soft (mirrors Python 16285fd): shadow_rise recovery is
             * double-talk-blind (only a far-stationarity guard). While near
             * recently present, skip the aggressive re-adapt + mark_diverged +
             * ERLE reset; its tail overfits near-end that arrives after the
             * (far-only) trigger. No hangover tick (Python ticks only when rise
             * did NOT fire). (Python only sets the dead _epc_reset_fired_this_frame
             * FQA flag here — no audio effect, not modelled in C.) */
        } else if (rise.fired) {
            a->pending_gain_change = 1;
            pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
            filter_q_high(&a->main_filter);
            filter_convergence_mark_diverged(&a->convergence);
            a->epc_render_forced_remaining = a->cfg.epc_hangover;
            if (a->erl_estimate > 0.3f) a->erl_estimate = 0.3f;
        } else {
            epc_tick_hangover(&a->epc);
        }
    }

    /* final_output starts from raw_output. */
    memcpy(a->final_out, a->raw_output, (size_t)hop * sizeof(float));

    /* 16. misadjustment estimator update + fire. */
    misadj_update(a, a->near_hop, a->raw_output, hop);
    misadj_fire(a);

    /* inst-ERLE slope ring: append get_erle_instant() (orchestrator.py:2382)
     * UNCONDITIONALLY every freq-path hop — Python appends OUTSIDE the
     * enable_res sub-block, so the warm-transfer gate works even with
     * enable_res=0 / return_res_context=0. Value = the SAME formula as the
     * erle_for_factor compute inside the block below (get_erle_instant uses
     * error_power == raw_error_power, alias set at orch 2563). near_power /
     * raw_error_power here are the prior-hop EMAs (updated later, step ~1958),
     * matching Python's read-before-update ordering. Read at Path-A (step 3)
     * is therefore 1-hop delayed, matching Python (read@1412 / append@2382). */
    {
        float _ei;
        if (a->near_power < 1e-10f && a->raw_error_power < 1e-10f) _ei = 0.0f;
        else _ei = 10.0f * log10f((a->near_power + 1e-10f) / (a->raw_error_power + 1e-10f));
        int _h = a->erle_slope_head;
        a->erle_slope_buf[_h] = _ei;
        a->erle_slope_head = (_h + 1) % a->erle_slope_cap;
        if (a->erle_slope_len < a->erle_slope_cap) a->erle_slope_len++;
    }

    /* 17. RES / post block (enable_res). */
    int is_stationary_dt = 0;
    float dt_indicator = 0.0f;
    float erle_windowed = 0.0f;
    float far_power = 0.0f;
    /* Run the AEC3 post block when EITHER the internal RES is on OR the caller
     * asked for the res-context seam (mirrors Python orchestrator.py:2021,
     * `enable_res or return_res_context`). With enable_res=1 this is unchanged
     * (return_res_context is irrelevant) → production cascade byte-exact. */
    if (a->cfg.enable_res || a->cfg.return_res_context) {
        far_power = mean_sq(a->far_hop, hop, a->scr_sq);
        /* erle_windowed (step 13a). */
        const float erle_decay = 0.999f;
        a->erle_window_near = erle_decay * a->erle_window_near + a->near_power;
        a->erle_window_err  = erle_decay * a->erle_window_err  + a->raw_error_power;
        erle_windowed = 10.0f * log10f((a->erle_window_near + 1e-10f)
                                     / (a->erle_window_err + 1e-10f));
        /* erle_for_factor = max(get_erle_instant(), erle_windowed). */
        float erle_inst;
        if (a->near_power < 1e-10f && a->raw_error_power < 1e-10f) erle_inst = 0.0f;
        else erle_inst = 10.0f * log10f((a->near_power + 1e-10f) / (a->raw_error_power + 1e-10f));
        float erle_for_factor = erle_inst > erle_windowed ? erle_inst : erle_windowed;
        float erle_factor = erle_for_factor / 10.0f;
        if (erle_factor < 0.0f) erle_factor = 0.0f;
        if (erle_factor > 1.0f) erle_factor = 1.0f;
        a->erle_factor_prev = erle_factor;

        float far_pwr = far_power + 1e-10f;
        float mic_pwr = mean_sq(a->near_hop, hop, a->scr_sq) + 1e-10f;
        float raw_err_pwr = mean_sq(a->raw_output, hop, a->scr_sq) + 1e-10f;

        /* erl tracking (B4). */
        if (far_pwr > 1e-4f) {
            float raw_dt_ratio = raw_err_pwr / (far_pwr + 1e-10f);
            float inst_erl_raw = mic_pwr / far_pwr;
            if (raw_dt_ratio < 2.0f && inst_erl_raw < 1.5f) {
                float inst_erl = inst_erl_raw;
                if (inst_erl < 0.001f) inst_erl = 0.001f; if (inst_erl > 1.0f) inst_erl = 1.0f;
                float alpha_erl = a->convergence.converged ? 0.999f : 0.99f;
                a->erl_estimate = alpha_erl * a->erl_estimate + (1.0f - alpha_erl) * inst_erl;
            }
        }

        doubletalk_update_energy_dt(&a->dt_analyzer, ra.is_active,
                                    far_pwr, mic_pwr, a->erl_estimate);

        /* base DT confidence (simple energy-ratio estimate). */
        float simple_dt = 1.0f - far_pwr / (mic_pwr + far_pwr);
        float raw_dt = (float)a->dt_analyzer.dt_from_energy;  /* receive: DoubleTalk (Stage-3) is double */
        float sd_half = simple_dt * 0.5f;
        if (sd_half > raw_dt) raw_dt = sd_half;

        /* stationary-DT macro detection. */
        if (a->render_activity.is_stationary && a->convergence.converged) {
            int fft_size = a->main_filter.base.fft_size;
            float freq_per_bin = (float)a->cfg.sample_rate / (float)fft_size;
            int vb_start = (int)(100.0f / freq_per_bin); if (vb_start < 1) vb_start = 1;
            int vb_limit = (int)(3000.0f / freq_per_bin); if (vb_limit > K) vb_limit = K;
            float track_err_pwr = 0.0f;
            for (int k = vb_start; k < vb_limit; ++k)
                track_err_pwr += cmag2_c(a->main_filter.base.error_spec[k].r,
                                                 a->main_filter.base.error_spec[k].i);
            track_err_pwr += 1e-10f;
            if (a->wn_err_baseline < 1e-6f) a->wn_err_baseline = track_err_pwr;
            float jump_ratio = track_err_pwr / (a->wn_err_baseline + 1e-10f);
            if (jump_ratio > 1.5f) a->stat_dt_hangover = 80;
            if (a->stat_dt_hangover > 0) {
                is_stationary_dt = 1;
                a->stat_dt_hangover--;
                a->wn_err_baseline = 0.999f * a->wn_err_baseline + 0.001f * track_err_pwr;
            } else {
                is_stationary_dt = 0;
                a->wn_err_baseline = 0.95f * a->wn_err_baseline + 0.05f * track_err_pwr;
            }
        }
        /* D4 baseline slow-track. */
        if (a->convergence.converged && !a->render_activity.is_stationary
                && far_pwr > 1e-4f && a->wn_err_baseline > 1e-6f) {
            a->wn_err_baseline = 0.995f * a->wn_err_baseline + 0.005f * raw_err_pwr;
        }

        /* inst_erle correction. */
        float inst_erle_fast_raw = mic_pwr / raw_err_pwr;
        a->inst_erle_smooth = 0.7f * a->inst_erle_smooth + 0.3f * inst_erle_fast_raw;
        if (a->inst_erle_smooth > 2.0f) {
            float erle_for_dt = a->inst_erle_smooth;
            if (erle_for_dt > 4.0f) erle_for_dt = 4.0f;
            raw_dt /= erle_for_dt;
        }
        /* EPC physical gate. */
        if (a->epc.active) { raw_dt = 0.0f; is_stationary_dt = 0; }
        dt_indicator = raw_dt; if (dt_indicator < 0.0f) dt_indicator = 0.0f;
        if (dt_indicator > 0.8f) dt_indicator = 0.8f;

        /* divergence indicator EMA. */
        filter_convergence_update_divergence(&a->convergence, a->near_power, a->raw_error_power);

        /* EPC render-forced countdown (Python decrements + sets RES flag;
         * the RES flag is consumed inside REE which aec3_post_run drives —
         * the orchestrator no longer wires using_render_based to REE, so this
         * is a counter decrement only). */
        if (a->epc_render_forced_remaining > 0) a->epc_render_forced_remaining--;

        /* shadow_dt stash for RES context. Both operands received from
         * DoubleTalk (Stage-3, double) — cast at the receive/compare site. */
        float shadow_dt_v = (float)a->dt_analyzer.dt_from_energy;
        if ((float)a->dt_analyzer.dt_from_shadow > shadow_dt_v) shadow_dt_v = (float)a->dt_analyzer.dt_from_shadow;
        if (a->epc.active) shadow_dt_v *= 0.08f;
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

            /* DT-gated min-gain floor lift (mirrors Python 16285fd): during
             * double-talk (near recently present) protect near-end by lifting
             * the RES floor; FS (no near) keeps the aggressive far_active floor.
             * Set on the SG just before get_gain (driven inside aec3_post_run). */
            a->a3_sg.dt_protect_active =
                (a->cfg.dt_aware_res_floor_enabled && a->ne_recent_frames > 0) ? 1 : 0;

            int pgc = a->pending_gain_change, pdc = a->pending_delay_change;
            aec3_post_run(&a->post, &in, &obj, &a->a3_sc, a->final_out, &pgc, &pdc);
            a->pending_gain_change = pgc;
            a->pending_delay_change = pdc;

            /* Context-only seam (return_res_context && !enable_res): the AEC3
             * post ran so res_gain / R² / comfort_noise are valid for the
             * caller's external RES, but its suppression must NOT be applied —
             * emit the linear residual (raw_output) so the external RES is the
             * sole suppressor (mirrors Python orchestrator.py:3485-3486). */
            if (!a->cfg.enable_res)
                memcpy(a->final_out, a->raw_output, (size_t)hop * sizeof(float));
        }

        /* per-bin mu_scale update AFTER RES (orchestrator 2291-2307). */
        if (a->convergence.converged) {
            float mu_min = a->cfg.shadow_mu_min;
            for (int k = 0; k < K; ++k) a->per_bin_mu_scale[k] = mu_min;
            a->simple_mu_ratio = 0.0f;
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

    /* 18. OLA-lagged output limiter. near_peak/out_peak via np.max(np.abs(.))
     *     and target_gain = near_peak/out_peak, the EMA mix and the limiter
     *     gain itself all run float32-by-design (Stage-2 conversion; the
     *     historical Python float64 EMA-mix reference is retired). */
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
        float target_gain;
        if (out_peak > near_peak && near_peak > 1e-6f)
            target_gain = near_peak / out_peak;
        else
            target_gain = 1.0f;
        float alpha_lim = (target_gain < a->limiter_gain) ? 0.3f : 0.8f;
        a->limiter_gain = alpha_lim * a->limiter_gain + (1.0f - alpha_lim) * target_gain;
        float lg = a->limiter_gain;
        for (int i = 0; i < hop; ++i)
            a->final_out[i] = a->final_out[i] * lg;
    }

    /* 19. power EMAs (sample loop; read by NEXT frame's step 17). */
    {
        const float ap = a->alpha_pow, oap = 1.0f - a->alpha_pow;
        for (int i = 0; i < hop; ++i) {
            float n = a->near_hop[i], r = a->raw_output[i];
            a->near_power      = ap * a->near_power      + oap * (n * n);
            a->raw_error_power = ap * a->raw_error_power + oap * (r * r);
        }
    }

    /* 20. convergence detection. */
    {
        int far_active = (mean_sq(a->far_hop, hop, a->scr_sq) > 1e-4f);
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
        float esum = 0.0f, r2sum = 0.0f, cnsum = 0.0f;
        const float *erle = aec_state_erle(&a->a3_state, /*onset=*/1);
        int kk;
        for (kk = 0; kk < Kk; ++kk) {
            esum  += erle[kk];
            r2sum += a->a3_sc.r2[kk];
            cnsum += a->post.comfort_noise[kk];
        }
        tr.delay            = aec_state_min_direct_path_filter_delay(&a->a3_state);
        tr.far_active       = a->post.trace.far_active;
        tr.saturated_echo   = aec_state_saturated_echo(&a->a3_state);
        tr.usable_linear    = aec_state_usable_linear_estimate(&a->a3_state);
        tr.dominant_nearend = suppression_gain_is_dominant_nearend(&a->a3_sg);
        tr.filter_converged = a->post.trace.aec3_converged;
        tr.fullband_erle    = aec_state_fullband_erle_log2(&a->a3_state);
        tr.erle_mean        = (Kk > 0) ? esum  / Kk : 0.0f;
        tr.r2_mean          = (Kk > 0) ? r2sum / Kk : 0.0f;
        tr.gain_mean        = a->post.trace.gain_mean;
        tr.comfort_noise_mean = (Kk > 0) ? cnsum / Kk : 0.0f;
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
    /* Freq-domain seam (valid when the AEC3 post block ran this hop, i.e.
     * enable_res || return_res_context). These alias the internal per-hop
     * buffers populated by aec3_post_run: error_spec_windowed (audio scale,
     * == Python ctx.error_spec), the SuppressionGain output (a3_sg.gain), the
     * residual-echo PSD (a3_sc.r2, int16²) and the CNG N² (post.comfort_noise,
     * int16²). Left NULL when neither RES nor the context seam is enabled. */
    if (a->cfg.enable_res || a->cfg.return_res_context) {
        ctx->error_spec    = a->main_filter.base.error_spec_windowed;
        ctx->res_gain      = a->a3_sg.gain;
        ctx->r2            = a->a3_sc.r2;
        ctx->comfort_noise = a->post.comfort_noise;
    }
    ctx->far_power = a->last_far_power;
    ctx->erle_factor = a->erle_factor_prev;
    ctx->dt_indicator = a->last_dt_indicator;
    ctx->divergence = (float)a->convergence.divergence;  /* receive: FilterConvergence (Stage-3) is double */
    ctx->saturation_level = a->saturation_level;
    ctx->erl_estimate = a->erl_estimate;
    ctx->shadow_dt = a->last_shadow_dt;
    ctx->is_stationary_dt = a->last_is_stationary_dt;
    ctx->filter_converged = a->convergence.converged;
    ctx->filter_once_converged = a->convergence.once_converged;
    ctx->epc_active = a->epc.active;
}

/* Read-only status query — see AecDebugStatus (aec.h). No state mutated, no
 * per-frame cost added: every field read here is already maintained by the
 * engine on the hot path regardless of whether this is ever called. */
void aec_debug_status(const Aec* a, AecDebugStatus* out) {
    if (!a || !out) return;
    memset(out, 0, sizeof(*out));

    out->delay_samples    = a->current_delay;
    out->delay_confidence = delay_aec3_confidence(&a->delay);
    out->delay_updates    = delay_aec3_n_updates(&a->delay);

    out->erle_windowed_db = a->last_erle_windowed;
    out->usable_linear    = aec_state_usable_linear_estimate(&a->a3_state);
    out->filter_converged = a->convergence.converged;

    out->near_power = a->near_power;
    out->out_power  = a->raw_error_power;
}
