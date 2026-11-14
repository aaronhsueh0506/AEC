/* aec_impl.h -- the body of struct Aec and the types only it needs (library-internal; aec.c, aec_array.c, aec_array_internal.h and
 * the white-box tests via test/aec_white_box.h). The public header declares `typedef struct Aec Aec;` and nothing more. */
#ifndef AEC_AEC_IMPL_H
#define AEC_AEC_IMPL_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"
#include "aec_tuning.h"
#include "hpf.h"
#include "saturation.h"
#include "delay_aec.h"
#include "detectors.h"
#include "pbfdkf.h"
#include "epc_shadow.h"
#include "render_signal_analyzer.h"
#include "aec_post.h"   /* AecPost and its sub-modules */

/* Bad-relock judgment window (hops): a shifted delay is reverted if it cancels worse after this long. */
#define AEC_DELAY_SHIFT_JUDGMENT_HOPS 100
/* Capacity of Aec::delay_reject_values. */
#define AEC_DELAY_REJECT_CAP 8

/* Matched-filter duty-cycle gate: full-rate analysis until the estimate has been solid and unchanged for hold_hops, then one
 * hop in period_hops, back to full rate on a windowed-ERLE collapse. Internal; shared by the single-channel instance and the
 * array flow (aec_duty_* in aec_array_internal.h). */
typedef struct AecDutyGate {
    int    active;
    int    stable_hops;
    int    pos;
    int    last_delay;
    float  erle_peak;
    float  erle_leak_db;
    int    hold_hops;
    int    period_hops;
    unsigned long long hops_run;
} AecDutyGate;

/* Instance state: library-internal (aec.c, aec_array.c and white-box tests); the public API is aec.h. */
struct Aec {
    AecConfig cfg;
    AecTuning tun;            /* the tuning as built, with the fields authored on the 10 ms grid retimed to this one */

    int hop_size, block_size, fft_size, n_freqs, n_partitions;

    Hpf  *hp_mic;             /* NULL when highpass disabled */
    Saturation sat_ref, sat_mic;  int has_sat;

    /* delay: matched-filter state, valid iff has_delay (delay_mode ==
     * MATCHED). */
    DelayAec delay;       int has_delay;
    float* ref_ring;       int ref_ring_size, ref_ring_write, ref_ring_filled;
    int    current_delay;
    int    pending_delay;  int has_pending; int pending_delay_ttl;
    int    delay_shift_judgment_active;
    int    delay_shift_judgment_pre_delay;
    float  delay_shift_judgment_pre_erle;
    int    delay_shift_judgment_hops_left;
    Complex* delay_shift_judgment_pre_W;
    Complex* delay_shift_judgment_pre_shadow_W;
    float*   delay_shift_judgment_pre_H_error;   /* main filter H_error_per_bin [n_freqs] */
    int    delay_reject_values[AEC_DELAY_REJECT_CAP];
    int    delay_reject_count;
    unsigned int delay_generation;
    unsigned int delay_gen_hop_start;
    int    far_hop_aligned;
    AecDutyGate duty;
    unsigned long long duty_hops_total;

    PBFDKF main_filter;
    PBFDAF shadow_filter;  int has_shadow;
    PBFDAF observer_filter;  int has_observer;

    RenderActivity    render_activity;
    float*            ra_pairwise_scratch;
    FilterConvergence convergence;
    DoubleTalk        dt_analyzer;
    EpcDetector       epc;
    ShadowCopy        regime;
    RenderSignalAnalyzer rsa;   int64_t* rsa_counters;

    FftHandle*  post_fft;
    AecPost    post;
    AecState              a3_state;   AecStateStorage a3_state_st;
    ResidualEchoEstimator a3_ree;
    SuppressionGain       a3_sg;
    StationarityEstimator a3_stat;
    LinearFilterSelect    a3_lfs;
    AecPostRunScratch    a3_sc;
    int    pending_gain_change;   /* bool */
    int    pending_delay_change;
    int    stationarity_active_hops;
    int    stationarity_converge_hops;
    int    non_zero_render_seen;
    float  render_peak_floor;
    int    block_stationary_next;

    float  saturation_level;
    float  erl_estimate;
    float  main_err_smooth, shadow_err_smooth;
    long   shadow_frame_count;     /* saturates at LONG_MAX */
    int    epc_render_forced_remaining;
    float  erle_window_near, erle_window_err;
    float  erle_factor_prev;
    float  inst_erle_smooth;
    float  erle_slope_buf[64];
    int    erle_slope_cap, erle_slope_len, erle_slope_head;
    float  wn_err_baseline;
    int    stat_dt_hangover;
    int    warmup_frames_remaining;
    int    warmup_far_active;
    float  simple_mu_ratio;
    int    simple_mu_holdoff;
    float* per_bin_mu_scale;   int has_per_bin_mu;
    float  near_power, raw_error_power;
    float  alpha_pow;
    long   frame_count;            /* saturates at LONG_MAX */

    int    ne_above;
    int    ne_recent_frames;

    int    poor_coarse_counter;
    int    coarse_reset_hangover;
    int    leakage_div_sustained_counter;
    int    poor_coarse_threshold_hops;
    int    coarse_reset_hangover_hops;
    int    leakage_div_sustain_hops;
    int    stat_dt_hangover_hops;

    float  misadj_e2_acum, misadj_y2_acum;
    int    misadj_n_acum;
    float  misadj_inv;
    int    misadj_overhang;
    int    misadj_stable_count;
    int    misadj_hangover_remaining;

    float  last_erle_windowed;

    float  last_far_power, last_shadow_dt, last_dt_indicator;
    float  alpha_erl_tracking, alpha_erl_converged;
    int    last_is_stationary_dt;

    float* near_hop;
    float* far_hop;
    float* raw_output;
    float* shadow_out;
    float* final_out;
    float* observer_far_hop;
    int    observer_far_hop_valid;
    int    observer_shift_hops;
    float* filter_taps_full;

    float* scr_sq;
    float* scr_e2_echo;
    float* scr_e2_near;
    float* scr_rsa_mask;
    float* scr_mu_buf_pre;
    float* scr_mu_buf;
    float* scr_e2ref_arr;

    unsigned char is_static;    /* the caller owns the memory */
    unsigned char ready;        /* set by a successful create / init, cleared by aec_destroy() before it frees; a handle that is not ready is refused */

    /* Two-stage hop bookkeeping (aec_process_linear / aec_process_residual): 0 idle, 1 = stage 1 done and the residual
     * stage pending. hop_carry holds the stage-1 locals the residual stage needs. Internal. */
    int hop_phase;
    struct AecHopCarry {
        float far_hop_mean_sq;
        float erle_inst;            /* inst-ERLE of this hop (misadjustment estimator) */
        float erle_windowed;
        int   render_active;
        int   stationarity_block_for_post;
        uint32_t t_stage;           /* clock stamp that closed the steering window */
    } hop_carry;
    int residual_valid;             /* a residual stage has completed since construction or aec_reset() */

    /* Array instance (AecConfig.n_mics == AEC_ARRAY_MICS): the flow state in src/aec_array.c; NULL for a single-channel
     * instance. On an array instance every field above is unused except cfg and the dimensions. */
    struct AecArray* array;

    void*  heap_arena;

    AecStageTiming last_timing;
};

/* The public names of internal values agree with them. */
_Static_assert(AEC_DELAY_NUM_FILTERS_MAX == DA_NUM_FILTERS, "AEC_DELAY_NUM_FILTERS_MAX is the delay estimator's bank limit");
_Static_assert(AEC_DELAY_FILTER_SIZE == DA_FILTER_SIZE, "AEC_DELAY_FILTER_SIZE is the matched filter length");
_Static_assert(AEC_DELAY_FILTER_INTRA_SHIFT == DA_FILTER_INTRA_SHIFT, "AEC_DELAY_FILTER_INTRA_SHIFT is the filter spacing");
_Static_assert(AEC_DELAY_DOWN_SAMPLING_FACTOR == DA_DOWN_SAMPLING_FACTOR, "AEC_DELAY_DOWN_SAMPLING_FACTOR is the estimator decimation");
/* A pool is 16-byte aligned; the Aec header carved at its start needs no more. */
_Static_assert(_Alignof(struct Aec) <= 16, "struct Aec must not need more than the pool alignment");

#endif /* AEC_AEC_IMPL_H */
