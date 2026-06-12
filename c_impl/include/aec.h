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
    float  highpass_cutoff_hz;     /* 80   */
    float  saturation_threshold;   /* 0.95 */
    float  kalman_q_high;          /* 1e-3 */
    float  kalman_q_low;           /* 1e-6 */
    int    delay_acquire_protect_converged;   /* 1 */

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

    /* poor-coarse rescue (process loop) */
    int    poor_coarse_counter;
    int    coarse_reset_hangover;

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

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_H */
