/* aec_impl.h -- the body of struct Aec and the types only it needs (library-internal; aec.c, aec_array.c, aec_array_internal.h and
 * the white-box tests via test/aec_white_box.h). The public header declares `typedef struct Aec Aec;` and nothing more. */
#ifndef AEC_AEC_IMPL_H
#define AEC_AEC_IMPL_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
 * hop in period_hops, back to full rate on a windowed-ERLE collapse. Shared by the single-channel instance and the array flow. */
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

/* The analysis period is the configured period divided by AEC_DUTY_PERIOD_DIVISOR; a windowed-ERLE drop of
 * AEC_DUTY_ERLE_COLLAPSE_DB below the leaky peak resumes full rate. The caller configures the gate once and keeps its own census. */
#define AEC_DUTY_PERIOD_DIVISOR    5
#define AEC_DUTY_ERLE_COLLAPSE_DB  6.0f

/* Per-hop leak of the duty watchdog the grid implies: 0.001 dB at the 10 ms reference hop, scaled with the hop duration. */
float aec_duty_leak_db(int hop_size, int sample_rate);
/* Seconds -> hops (round, floor 1) against the grid: hold_hops from hold_s, period_hops from period_s / AEC_DUTY_PERIOD_DIVISOR,
 * and the grid's erle_leak_db. */
void  aec_duty_configure(AecDutyGate* g, float hold_s, float period_s, int hop_size, int sample_rate);
/* Clears the gate's state (not its hold / period / leak). */
void  aec_duty_reset(AecDutyGate* g);
/* Before the estimator is fed: advances the decimation phase and returns 1 when this hop's analysis runs (counts it). */
int   aec_duty_begin(AecDutyGate* g);
/* After the estimator is fed: the estimate (cur_delay, solid) and the windowed ERLE read at this hop arm, hold or drop the gate. */
void  aec_duty_end(AecDutyGate* g, int cur_delay, int solid, float erle);

/* Shared-delay change admission (Path B of the single channel and the array's shared alignment): a change to the alignment in
 * force is admitted only when the movement exceeds AEC_DELAY_CHANGE_MIN_SAMPLES and is offered again, within
 * AEC_DELAY_CHANGE_CONFIRM_SAMPLES, before the held candidate ages out (TTL in hops). */
#define AEC_DELAY_CHANGE_MIN_SAMPLES      32
#define AEC_DELAY_CHANGE_CONFIRM_SAMPLES  16
#define AEC_DELAY_CHANGE_CANDIDATE_TTL     3

typedef struct AecDelayAdmission {
    int candidate;
    int ttl;
} AecDelayAdmission;

/* One hop of life spent, whether or not this hop had a usable estimate. Call once per hop, before offer(). */
static inline void aec_delay_admission_age(AecDelayAdmission* admission) {
    if (!admission || admission->ttl <= 0) return;
    admission->ttl -= 1;
    if (admission->ttl <= 0) {
        admission->ttl = 0;
        admission->candidate = -1;
    }
}

/* Offers this hop's eligible estimate against the alignment in force. 1: the movement is admitted (the caller applies it and
 * realigns); 0: absorbed as too small, or held / replaced as a candidate. */
static inline int aec_delay_admission_offer(AecDelayAdmission* admission, int accepted_delay, int estimated) {
    if (!admission) return 0;
    /* Too small to be worth four IR shifts: absorbed and left to age. */
    if (abs(estimated - accepted_delay) <= AEC_DELAY_CHANGE_MIN_SAMPLES) return 0;
    if (admission->ttl > 0 &&
        abs(estimated - admission->candidate) < AEC_DELAY_CHANGE_CONFIRM_SAMPLES) {
        admission->candidate = -1;
        admission->ttl = 0;
        return 1;
    }
    admission->candidate = estimated;
    admission->ttl = AEC_DELAY_CHANGE_CANDIDATE_TTL;
    return 0;
}

/* Drops the held candidate (-1 holds none; only candidate with ttl > 0 is live). */
static inline void aec_delay_admission_clear(AecDelayAdmission* admission) {
    admission->candidate = -1;
    admission->ttl = 0;
}

/* Reference-ring copies of n <= ring_size samples at index `at`: at most one wrap, so one or two bounded runs. */
static inline void aec_ring_write(float* ring, int ring_size, int at, const float* src, int n) {
    int first = ring_size - at;
    if (first > n) first = n;
    memcpy(ring + at, src, (size_t)first * sizeof(float));
    if (first < n) memcpy(ring, src + first, (size_t)(n - first) * sizeof(float));
}
static inline void aec_ring_read(const float* ring, int ring_size, int at, float* dst, int n) {
    int first = ring_size - at;
    if (first > n) first = n;
    memcpy(dst, ring + at, (size_t)first * sizeof(float));
    if (first < n) memcpy(dst + first, ring, (size_t)(n - first) * sizeof(float));
}

/* Stage timing is on by default; -DAEC_STAGE_TIMING=0 folds the stamps to 0. A 32-bit microsecond monotonic stamp; a
 * target without CLOCK_MONOTONIC defines -DAEC_NOW_US=<fn> (no-argument function returning monotonic uint32_t microseconds). */
#if AEC_STAGE_TIMING
#ifndef AEC_NOW_US
#include <time.h>
static inline uint32_t aec_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull);
}
#else
static inline uint32_t aec_now_us(void) { return AEC_NOW_US(); }
#endif
#else
#define aec_now_us() 0u
#endif

/* Far-only analysis shared across array lanes: lane 0 publishes its far-only results each hop and lanes 1-3 reuse them. hop
 * (stages that read the far hop alone), fft (the donor's far spectrum and what derives from it), analysis (tonal counters, noise
 * floor; valid while the far history has equalled the donor's since the last reset), post (the donor's far PSD). */
typedef struct AecFarShare {
    const Aec* donor;
    int hop, fft, analysis, post;
    float sat_ref;
    RenderActivityResult ra;
    float far_mean_sq;
    int far_floor_hit;
    int rsa_poor;
    int rsa_mask_active;
} AecFarShare;

/* Instance state: library-internal (aec.c, aec_array.c and white-box tests); the public API is aec.h. */
struct Aec {
    AecConfig cfg;
    AecTuning tun;            /* the tuning as built, with the fields authored on the 10 ms grid retimed to this one */

    int hop_size, block_size, n_freqs, n_partitions;

    Hpf  *hp_mic;             /* NULL when highpass disabled */
    Saturation sat_ref, sat_mic;  int has_sat;

    /* delay: matched-filter state, valid iff has_delay (delay_mode ==
     * MATCHED). */
    DelayAec delay;       int has_delay;
    float* ref_ring;       int ref_ring_size, ref_ring_write, ref_ring_filled;
    int    current_delay;
    AecDelayAdmission delay_admission;   /* Path B: a shifted delay is admitted when offered twice (aec_delay_admission_offer()) */
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

/* A NULL, never-built, failed-create or destroyed handle: every entry point refuses it. */
static inline int aec_unusable(const Aec* a) { return !a || !a->ready; }

/* A hop whose results are not finite, either instance kind (aec.c). aec_cvec_finite: 1 when `count` bins are finite, 0 for
 * NULL or count < 0. aec_refuse_hop: resets the instance as aec_reset() (the linear view then reads zeros; an array's is lane 0's) and
 * returns AEC_ERR_DSP. */
int aec_cvec_finite(const Complex* v, int count);
int aec_refuse_hop(Aec* a);
/* Reference ring capacity in samples for a resolved configuration: 0 for EXTERNAL_ALIGNED, an estimator that cannot be sized or an
 * int overflow (aec.c; the single channel and the array size the ring through it). */
int aec_ref_ring_samples(const AecResolved* rs, int hop);

#endif /* AEC_AEC_IMPL_H */
