/* aec.h — top-level v3.22 AEC orchestration (AEC3 post-filter chain).
 *
 * Wires the bit-exact v3.22 modules into one frame-level aec_process() that
 * mirrors python/modules/orchestrator.py AEC.process for the BALANCED preset
 * (mode=PBFDKF, simple variable-mu, enable_res=True,
 * enable_cng per preset). The post-stage is driven entirely by aec3_post_run
 * (the bit-exact full _aec3_post orchestration); this file ports the AUDIO
 * PATH of process() around it.
 *
 * Out of scope for this cutover:
 *   - AecMode != PBFDKF (NLMS / buffered FDAF queue)
 *   - get_stats / debug logger / capture_stages / pure-diagnostic _diag stashes
 */
#ifndef AEC_AEC_H
#define AEC_AEC_H

#include "fft_wrapper.h"
#include "hpf.h"
#include "saturation.h"
#include "delay_aec3.h"
#include "detectors.h"
#include "pbfdkf.h"
#include "epc_shadow.h"
#include "render_signal_analyzer.h"
#include "aec3_post.h"   /* Aec3Post + Aec3PostRunObj/In/Scratch + sub-modules */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { AEC_PRESET_GENTLE = 0, AEC_PRESET_BALANCED, AEC_PRESET_AGGRESSIVE } AecPreset;

typedef struct AecConfig {
    int    sample_rate;        /* 16000 */
    int    filter_length;      /* 832 samples (52 ms) */
    int    n_partitions;       /* derived if 0 */
    float  mu;                 /* 0.3 */
    float  delta;              /* 1e-8 */

    /* toggles */
    int    enable_cng;             /* 1 */
    int    enable_delay_est;       /* 1 */
    int    enable_highpass;        /* 1 (mic only) */
    int    enable_saturation;      /* 1 */
    int    enable_shadow;          /* 1 */
    int    enable_res;             /* 1 */
    int    saturation_softclip_ref;/* 1 */
    /* Linear-AEC + external-RES seam (mirrors Python AecConfig.return_res_context,
     * config.py:332). When set, the AEC3 post block RUNS (computes res_gain / R² /
     * comfort_noise / windowed error spectrum and exposes them via AecResContext)
     * but, if enable_res==0, its suppression is NOT applied — the time output stays
     * the linear residual (raw_output). Lets a caller run AEC(linear) → NR → RES
     * externally. Default 0 (memset) → production cascade untouched. */
    int    return_res_context;     /* 0 */

    /* scalar tunables (balanced base) */
    float  shadow_err_alpha;       /* 0.80 */
    float  shadow_mu_min;          /* 0.5  */
    float  shadow_mu_nlms;         /* 0.5  */
    int    warmup_frames;          /* 100  */
    int    epc_hangover;           /* 20   */
    float  epc_total_rise;         /* 1.5  */
    float  epc_delta_threshold;    /* 0.3  */
    float  epc_mu_floor;           /* 0.5  */
    float  max_delay_ms;           /* 1024 */
    float  delay_buffer_ms;        /* 2048 */
    float  delay_est_init_s;       /* 0.3  */
    float  delay_est_period_s;     /* 0.5  */
    /* Delay-estimator duty cycling (default 0 = analyse every hop, exactly
     * the Python reference → byte-identical). When 1, a DOCUMENTED DIVERGENCE
     * from the reference (pending a quality bench): once the delay estimate
     * is solid (confidence 1.0) AND unchanged for delay_est_init_s seconds,
     * the matched-filter analysis drops to 1 hop in every K, where
     * K = max(2, round(delay_est_period_s/hop_s)/5) — 10 with the default
     * 0.5 s period @10 ms hop, cutting ~90% of the matched-filter cost. The
     * estimator's decimators + render ring stay FED on every hop (only the
     * analysis decimates), so the matched filter never sees gapped audio.
     * Full-rate analysis resumes immediately when (a) the estimate changes
     * or loses solidity, or (b) the ERLE watchdog fires: erle_windowed drops
     * >6 dB below its running peak (peak leaks ~0.1 dB/s; armed only once
     * the peak exceeds 6 dB, so it cannot fire before first convergence). */
    int    delay_est_duty_cycle;   /* 0 */
    float  highpass_cutoff_hz;     /* 80   */
    float  saturation_threshold;   /* 0.95 */
    float  kalman_q_high;          /* 1e-3 */
    float  kalman_q_low;           /* 1e-6 */
    int    delay_acquire_protect_converged;   /* 1 */
    /* Warm tap-transfer on first delay acquisition (v3.24.1, default ON):
     * shift the learned IR by the acquired delay instead of zeroing, so the
     * cold-start cancellation survives the realign (the "1.14s vertical line"
     * fix). Gated on inst-ERLE peak > _inst_erle_db AND delay < tap reach.
     * _protect_inst_erle is the A/B-rejected option-A (block the realign). */
    int    delay_acquire_warm_transfer;       /* 1 */
    double delay_acquire_inst_erle_db;        /* 4.0 */
    int    delay_acquire_protect_inst_erle;   /* 0 */

    /* DT-deg recovery stack (default ON, mirrors Python 16285fd). The
     * energy-based held near-end gate (_ne_recent_frames) arms two levers
     * during double-talk only (FS never raises dt_from_energy): */
    int    dt_aware_recovery_soft;            /* 1 — soft (non-destructive)
                                               * Path A/B/EPV/shadow_rise realign
                                               * while near recently present */
    int    dt_aware_res_floor_enabled;        /* 1 — DT-gated RES min-gain floor */
    float  min_gain_floor_dt_db;              /* -16.0 — DT floor (dB) */
    /* ne_recent gate parameters (mirror AecConfig.ne_recent_*). threshold is
     * f64 to match Python's float(0.3) exactly (0.3f rounds differently). */
    double ne_recent_threshold;               /* 0.3 */
    int    ne_recent_hold;                    /* 150 */
    int    ne_recent_sustain;                 /* 3 */

    /* preset strength axis (the ONLY field gentle/balanced/aggressive differ
     * in): SuppressionGain far-active min-gain floor in dB
     * (gentle -20 / balanced -28 / aggressive -38). */
    float  min_gain_floor_far_active_db;

    /* FilterMisadjustmentEstimator (AEC3 ScaleFilter) */
    int    filter_misadjustment_stable_frames;   /* 30  */
    int    filter_misadjustment_hangover_frames;  /* 100 */
    float  filter_misadjustment_scale_min;        /* 0.5 */
    float  filter_misadjustment_scale_max;        /* 2.0 */
} AecConfig;

void aec_config_defaults(AecConfig* cfg, int sample_rate);
void aec_config_from_preset(AecConfig* cfg, AecPreset preset, int sample_rate);

/* ── opaque-ish context ────────────────────────────────────────────────── */
/* Streaming render/capture buffering events (mirror AEC3
 * render_delay_buffer.h:28 BufferingEvent). Returned by the streaming API. */
typedef enum AecBufferingEvent {
    AEC_BUF_NONE = 0,
    AEC_BUF_RENDER_UNDERRUN,   /* capture ran ahead — no buffered render */
    AEC_BUF_RENDER_OVERRUN,    /* render piled up past the FIFO — oldest dropped */
    AEC_BUF_API_SKEW           /* render/capture call alternation anomaly (diag) */
} AecBufferingEvent;

typedef struct Aec {
    AecConfig cfg;

    int hop_size, block_size, fft_size, n_freqs, n_partitions;

    /* preprocessing */
    Hpf hp_mic;            int has_hp;
    Saturation sat_ref, sat_mic;  int has_sat;

    /* delay (AEC3 matched filter) + ring */
    DelayAec3 delay;       int has_delay;
    float* ref_ring;       int ref_ring_size, ref_ring_write, ref_ring_filled;
    int    current_delay;  /* -1 until first acquisition */
    int    pending_delay;  int has_pending; int pending_delay_ttl;
    /* duty-cycle state (only driven when cfg.delay_est_duty_cycle != 0) */
    int    duty_active;        /* 1 = decimated analysis in effect */
    int    duty_stable_hops;   /* consecutive solid+unchanged hops (arming) */
    int    duty_pos;           /* position in the 1-in-K analysis cycle */
    int    duty_last_delay;    /* estimate on the previous hop (change detect) */
    double duty_erle_peak;     /* leaky running peak of erle_windowed (dB) */

    /* linear filters */
    PBFDKF main_filter;
    PBFDAF shadow_filter;  int has_shadow;

    /* detectors / EPC / regime / RSA */
    RenderActivity    render_activity;
    FilterConvergence convergence;
    DoubleTalk        dt_analyzer;
    EpcDetector       epc;
    ShadowCopy        regime;
    RenderSignalAnalyzer rsa;   int64_t* rsa_counters;

    /* AEC3 post chain (driver + sub-module objects, all caller-owned storage) */
    FftHandle*  post_fft;
    Aec3Post    post;
    AecState              a3_state;   AecStateStorage a3_state_st;
    ResidualEchoEstimator a3_ree;
    SuppressionGain       a3_sg;
    StationarityEstimator a3_stat;
    LinearFilterSelect    a3_lfs;
    Aec3PostRunScratch    a3_sc;
    /* tuning arrays for SuppressionGain (pointers into baked const tables) */

    /* AEC3 pending EPV mirror + stationarity read-state */
    int    pending_gain_change;   /* bool */
    int    pending_delay_change;  /* -1 == None, else DelayAdjustment */
    int    stationarity_active_hops;
    int    stationarity_converge_hops;
    int    non_zero_render_seen;
    float  render_peak_floor;
    int    block_stationary_next;   /* _block_stationary_for_next_hop latch */

    /* per-frame scalars (mirror orchestrator instance fields) */
    double saturation_level;
    double erl_estimate;
    double main_err_smooth, shadow_err_smooth;
    long   shadow_frame_count;
    int    epc_render_forced_remaining;
    double erle_window_near, erle_window_err;
    double erle_factor_prev;
    double inst_erle_smooth;
    /* inst-ERLE slope ring (mirrors Python orchestrator _erle_slope_buf, a
     * deque maxlen ~500ms/hop). Holds get_erle_instant() dB values; the warm
     * tap-transfer gate reads max() of the last 15. Filled every freq-path hop
     * OUTSIDE the enable_res gate (Python appends unconditionally). */
    float  erle_slope_buf[64];   /* cap >= ceil(500ms/hop)=50 @hop160/16k */
    int    erle_slope_cap, erle_slope_len, erle_slope_head;
    double wn_err_baseline;
    int    stat_dt_hangover;
    int    warmup_frames_remaining;
    int    warmup_far_active;
    double simple_mu_ratio;
    int    simple_mu_holdoff;
    float* per_bin_mu_scale;   int has_per_bin_mu;
    double limiter_gain;
    float* limiter_near_lag;   int has_limiter_lag;
    /* power EMAs (alpha=0.95 sample loop) */
    double near_power, raw_error_power;
    double alpha_pow;
    long   frame_count;

    /* DT-deg recovery: held "near-end seen recently" gate (mirrors Python
     * _ne_above consecutive-above counter + _ne_recent_frames hold countdown). */
    int    ne_above;
    int    ne_recent_frames;

    /* poor-coarse rescue (process loop) */
    int    poor_coarse_counter;
    int    coarse_reset_hangover;
    /* Track F: sustained leakage-diverged gate counter (clamped 0..10).
     * Mirrors orchestrator._leakage_div_sustained_counter. */
    int    leakage_div_sustained_counter;

    /* FilterMisadjustmentEstimator accumulators */
    double misadj_e2_acum, misadj_y2_acum;
    int    misadj_n_acum;
    double misadj_inv;
    int    misadj_overhang;
    int    misadj_stable_count;
    int    misadj_hangover_remaining;

    /* erle_windowed cache for the delay Path-A _already_cancelling guard
     * (= last frame's erle_windowed dB). */
    double last_erle_windowed;

    /* RES-context stash (last frame) */
    double last_far_power, last_shadow_dt, last_dt_indicator;
    int    last_is_stationary_dt;

    /* hop scratch */
    float* near_hop;
    float* far_hop;
    float* raw_output;
    float* shadow_out;
    float* final_out;
    float* filter_taps_full;   /* [n_partitions × hop] */
    Complex* W_all;            /* [n_partitions × n_freqs] flat snapshot */
    Complex* X_buf_all;        /* [n_partitions × n_freqs] flat snapshot */

    /* per-hop freq-bin scratch (Tier-1 stack-safety fix): these used to be
     * `float x[8192]` locals inside aec_process() / mean_sq(), sized far
     * beyond the actual n_freqs (K) they ever hold and with no assertion
     * tying 8192 to fft_size — a larger FFT config would silently smash the
     * stack, and some scopes stacked 2-3 of them at once (~96 KB worst case
     * in one block), which is unsafe headroom for an embedded RTOS task
     * stack (often 16-64 KB total). Moved into the Aec instance (malloc'd in
     * aec_create, pool-carved in aec_init, sized exactly K — or hop for
     * scr_sq, which mean_sq() only ever runs over hop-length buffers) so
     * stack usage no longer depends on n_freqs at all. Bit-exact: same
     * values, same order, only storage location changed. */
    float* scr_sq;          /* mean_sq() scratch, size hop_size (not K) */
    float* scr_e2_echo;
    float* scr_e2_near;
    float* scr_rsa_psd;
    float* scr_rsa_mask;
    float* scr_mu_buf_pre;
    float* scr_e2coa_pre;
    float* scr_mu_buf;
    float* scr_far_psd;
    float* scr_e2ref_arr;
    float* scr_e2coa_arr;
    float* scr_erl_arr;

    /* ── streaming render-hop FIFO (async render/capture decoupling) ──
     * Only exercised by the streaming API (aec_analyze_render /
     * aec_process_capture). The lockstep aec_process() bypasses it entirely,
     * so offline byte-exact parity with Python is untouched. In lockstep use
     * (one analyze_render then one process_capture) the FIFO is pass-through
     * (count 1→0, no event) → identical output to aec_process(). */
    float* render_fifo;        /* [fifo_cap_hops × hop_size] ring of render hops */
    int    fifo_cap_hops;      /* capacity in hops */
    int    fifo_count;         /* buffered render hops not yet consumed */
    int    fifo_read, fifo_write;
    long   render_call_count, capture_call_count;
    int    last_buffering_event;   /* AecBufferingEvent of the last capture step */

} Aec;

int  aec_create(Aec* a, const AecConfig* cfg);
void aec_destroy(Aec* a);
void aec_reset(Aec* a);

/* Process exactly hop_size samples — OFFLINE / lockstep path. Byte-exact to
 * Python aec.py. Render and capture supplied together. */
void aec_process(Aec* a, const float* mic, const float* ref, float* out);

/* ── Streaming API (real-time async render/capture) ───────────────────────
 * For deployments where the far-end (render) and mic (capture) arrive on
 * separate calls / threads and not necessarily 1:1. aec_analyze_render()
 * buffers a render hop; aec_process_capture() consumes the delay-aligned
 * render and produces output. Both return the buffering event observed.
 *
 * Lockstep equivalence: calling aec_analyze_render(ref) immediately followed
 * by aec_process_capture(mic, out) yields byte-identical output to
 * aec_process(mic, ref, out) — the FIFO is pass-through and no event fires.
 *
 * Underrun (capture with empty FIFO): processed with a silent render hop and
 * AEC_BUF_RENDER_UNDERRUN returned. Overrun (render past FIFO capacity): the
 * oldest buffered hop is dropped and AEC_BUF_RENDER_OVERRUN returned. */
AecBufferingEvent aec_analyze_render(Aec* a, const float* ref);
AecBufferingEvent aec_process_capture(Aec* a, const float* mic, float* out);
/* Last buffering event from the most recent aec_process_capture(). */
AecBufferingEvent aec_last_buffering_event(const Aec* a);

int  aec_hop_size(const Aec* a);

/* Linear-AEC + external-RES integration hook (research / NN surface). */
typedef struct AecResContext {
    int            n_freqs;
    int            hop_size;
    const float*   linear_hop;
    const Complex* echo_spec;
    const Complex* far_spec;
    const Complex* near_spec;
    /* Freq-domain seam for an external post-NR residual suppressor (mirrors the
     * Python AecResContext freq fields). Valid only when return_res_context=1;
     * NULL otherwise. error_spec = windowed linear error E(f) in AUDIO scale
     * (== Python ctx.error_spec, |E|²·32768² = error_psd). res_gain = the AEC3
     * SuppressionGain G_res(f), amplitude [0..1]. r2 = residual-echo PSD R²(f)
     * and comfort_noise = CNG N²(f), BOTH int16²-scaled (divide by 32768² to
     * reach the |E|² audio-power scale, as run_res does). All length n_freqs.
     * The pointers alias the AEC's internal per-hop buffers — read before the
     * next aec_process() call. */
    const Complex* error_spec;     /* windowed linear error E(f), audio scale */
    const float*   res_gain;       /* AEC3 G_res(f), amplitude [0..1]          */
    const float*   r2;             /* residual-echo PSD R²(f), int16²-scaled   */
    const float*   comfort_noise;  /* CNG N²(f), int16²-scaled                 */
    float          far_power;
    float          erle_factor;
    float          dt_indicator;
    float          divergence;
    float          saturation_level;
    float          erl_estimate;
    float          shadow_dt;
    int            is_stationary_dt;
    int            filter_converged;
    int            filter_once_converged;
    int            epc_active;
} AecResContext;

void aec_get_res_context(const Aec* a, AecResContext* ctx);

/* ── Runtime debug/status introspection ───────────────────────────────────
 * Read-only status query for an integrator to poll (e.g. once per second)
 * to tell whether the pipeline is behaving on an embedded target. Fills a
 * plain struct from fields the engine already maintains — no logging inside
 * the library, no extra per-frame state, no added hot-path cost; the caller
 * (e.g. example/aec_wav.c --debug) is responsible for printing. */
typedef struct AecDebugStatus {
    /* delay estimation (EchoPathDelayEstimator / AEC3 matched-filter) */
    int    delay_samples;      /* current applied delay, samples; -1 = not yet
                                 * acquired (mirrors a->current_delay)         */
    double delay_confidence;   /* 0.0 / 0.5 / 1.0 (delay_aec3_confidence)      */
    int    delay_updates;      /* matched-filter update count
                                 * (delay_aec3_n_updates)                      */

    /* linear filter health */
    double erle_windowed_db;   /* windowed ERLE (dB) from the last hop that
                                 * ran the AEC3 post chain (a->last_erle_windowed),
                                 * 0.0 before the post chain has run once      */
    int    usable_linear;      /* AEC3 usable-linear-estimate gate state
                                 * (aec_state_usable_linear_estimate)          */
    int    filter_converged;   /* FilterConvergenceAnalyzer latch state
                                 * (a->convergence.converged, same value
                                 * AecResContext.filter_converged exposes)     */

    /* levels — EMA (alpha=0.95), NOT an instantaneous last-hop mean. Reuses
     * the power-tracking EMAs the ERLE machinery already updates every hop
     * (two multiply-accumulate loops that run unconditionally), so reading
     * them here adds zero per-frame cost. */
    double near_power;         /* EMA of mic/near-hop power (a->near_power)    */
    double out_power;          /* EMA of the linear, pre-RES-gain output power
                                 * (a->raw_error_power)                        */
} AecDebugStatus;

void aec_debug_status(const Aec* a, AecDebugStatus* out);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_H */
