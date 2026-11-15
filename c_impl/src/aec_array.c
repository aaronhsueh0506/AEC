/* aec_array.c -- the four-microphone array instance of the AEC (AecConfig.n_mics == AEC_ARRAY_MICS): one shared delay
 * estimator, one aligned far end, four lanes (each an ordinary single-channel Aec in EXTERNAL_ALIGNED mode, built inside the
 * same pool), the fusion of the lanes by the caller's beamformer weights and, in AEC_RESIDUAL_FULL, the one residual gain on
 * the fused spectra. Contract: the "Array instance" section of aec.h. The beamformer (SRP-PHAT, BF, VAD) is the caller's;
 * synthesis is not in the library.
 *
 * Per hop:
 *     aec_array_impl_process_linear()   shared delay -> aligned far -> four lanes (each lane: its linear stage and its lane
 *                                  residual stage; lanes use ESTIMATE when AecConfig.residual is ESTIMATE / FULL and STATE
 *                                  when it is STATE)
 *     aec_array_impl_process_residual() the lane contexts projected with the caller's weights: near, R^2 (coherently), comfort
 *                                  noise (power sum), the rendering base; FULL adds the gain on the fused spectra
 * The lanes' EXTERNAL_ALIGNED delay mode and residual mode are the contract the flow stamps on each lane config
 * (array_derive_lane()). A lane is built by aec_lane_init() (aec_array_internal.h): it borrows the array's one FFT handle and
 * omits the buffers that contract never runs. */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>

#include "aec.h"
#include "aec_impl.h"
#include "aec_array_internal.h"
#include "aec_config_build.h"           /* aec_synth_window_build */
#include "aec_tuning.h"                /* AecResolved, aec_resolve_tuned, aec_apply_level */
#include "aec_res_internal.h"
#include "delay_aec.h"
#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"    /* skn_all_finite_f32 */

/* Matched-filter duty cycle: full-rate analysis until the estimate has been solid and unchanged for DUTY_HOLD_S, then one hop in
 * K = round(delay_est_period_s / hop_s) / AEC_DUTY_PERIOD_DIVISOR. The period floor is 1, so 0.0 means true full rate. */
#define DUTY_HOLD_S               0.3f

typedef struct ArraySnapshot {
    /* Borrowed views into a lane's buffers (lifetime: bind_lane_view()). */
    const Complex* error_spec;
    const Complex* echo_spec;
    const Complex* far_spec;
    const Complex* near_spec;
    const float* r2;
    const float* comfort_noise;
    const float* formed_hop;
    /* Rendering base: error_spec, or the capture where the lane's guard fired. */
    const Complex* base_spec;
    int guard_active;
    uint32_t reset_serial;

    float far_power;
    float dt_indicator;
    float saturation_level;
    int filter_converged;
    int usable_linear;
} ArraySnapshot;

/* The fused hop (aec_process_residual()). */
typedef struct ArrayFrame {
    const Complex* e;          /* the beamformed error E(f) */
    const Complex* b;          /* rendering base: e when guard_lanes == 0 */
    const Complex* y;          /* projected capture */
    const Complex* x;          /* lane 0's far spectrum */
    const float* r2;
    const float* cn;
    const float* gain;         /* FULL only */
    const float* far_hop;
    int guard_lanes;
    uint32_t serial;
    int all_converged, usable_linear, clock_drift;
    float max_dt, max_sat;
} ArrayFrame;

struct AecArray {
    int hop_size;
    int n_freqs;
    int delay_mode;
    int capture_proxy;
    int fixed_delay;           /* FIXED: the caller-measured delay in samples */
    /* Flags and masks are bytes to keep the control block at its size. */
    unsigned char quarantine_enabled;
    unsigned char post;        /* residual != STATE: the fusion buffers exist */
    unsigned char full;        /* residual == FULL: the gain stage exists */
    /* Far-only analysis hoist (see far_resync()): bit ch of far_sync = lane ch holds the same far history as lane 0 since the
     * last reset, so lane 0's far-only results are its own; bit ch of far_stale = that lane has skipped its own analysis
     * update (lane 0's state stands in for it). far_hoist_off is a test switch that forces every lane local. */
    unsigned char far_sync;
    unsigned char far_stale;
    unsigned char far_hoist_off;

    Aec* lanes[AEC_ARRAY_MICS];
    FftHandle* fft;            /* the lanes' one transform handle; lanes run strictly one after another */
    DelayAec delay;
    AecRes* gain_stage;

    ArraySnapshot snapshots[AEC_ARRAY_MICS];

    float* synth_window;       /* fft_size floats, rebuilt at init; the one a borrower gets */
    float* mic_lane;
    float* aligned_ref;

    float* delay_ring;
    int delay_ring_size;
    uint64_t delay_samples_seen;
    int accepted_delay;
    /* Backward-jump quarantine: _left = hops before a held earlier estimate is adopted anyway (-1 = disarmed, the only state
     * that may re-arm; 0 = armed and expired), _hops = the window converted once at init. */
    int delay_quarantine_left;
    int delay_quarantine_hops;
    /* Two-step admission of a shared-delay change; cleared by acceptance, expiry and reset. */
    AecArrayAdmission delay_admission;
    long realign_warm_lanes;
    long realign_soft_lanes;
    uint64_t delay_calls;
    AecDelayState last_delay;
    unsigned int generation;

    Complex* fused_error;
    Complex* fused_near;
    Complex* residual_work;
    Complex* residual_proj;    /* coherent residual vector, then its power in fused_r2 */
    float* fused_r2;
    float* fused_comfort;
    Complex* render_base;      /* sum_ch W_ch * B_ch on a hop where a contributing lane's guard fired */

    int front_valid;           /* a linear stage ran since the last reset */
    int fused_valid;           /* the residual stage ran on it */
    uint32_t front_serial;     /* the lanes' reset counters summed, as of that linear stage */
    ArrayFrame frame;

    uint32_t cng_seed;         /* the comfort-noise generator; reseeded when the summed serial moves */
    uint32_t handled_serial;

    AecArrayTiming timing;

    /* This hop's evidence that all four lanes consumed one far spectrum (far-FFT counters); 0 (also after any bail-out) costs
     * the full per-bin comparison in fuse_contexts(). */
    int far_spec_shared_hop;
    /* This hop routed a mixed warm/soft realign: the lanes' far_spec legitimately diverge, so fuse_contexts() skips the
     * agreement predicate. */
    int far_spec_mixed_hop;

    /* Matched-filter duty cycle (see update_shared_delay()). */
    AecDutyGate duty;
};

/* ---- configuration ------------------------------------------------------------------------------------------------- */

/* Resolved geometry and pool sizes of an array cfg. */
typedef struct ArrayDims {
    AecResolved lane;
    int hop, fft, n, ring;
    int post, full, matched;
    size_t ctl_bytes, buf_bytes, delay_bytes, res_bytes, fft_bytes, lane_each_bytes, total;   /* total: beyond the Aec header */
} ArrayDims;

/* The flow's contract stamped on a lane config: this grid, the shared aligned far (EXTERNAL_ALIGNED), the lane's residual
 * stage at ESTIMATE (STATE when the array is STATE), the flow's estimator duty cycle. Every other field (level values, filter
 * length, ...) is the array config's. */
static void array_derive_lane(const AecResolved* in, AecResolved* lane) {
    *lane = *in;
    lane->cfg.n_mics = 1;
    lane->cfg.capture_proxy_channel = 0;
    lane->cfg.delay_backward_quarantine_enabled = 0;
    lane->cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    lane->cfg.delay_num_filters = AEC_DELAY_NUM_FILTERS_MAX;
    lane->cfg.fixed_delay_samples = -1;
    lane->cfg.residual = (in->cfg.residual == AEC_RESIDUAL_STATE) ? AEC_RESIDUAL_STATE : AEC_RESIDUAL_ESTIMATE;
    /* The lanes never estimate delay; the period is the flow's estimator duty cycle (already in the tuning, which every lane
     * shares with the array). */
}

static size_t flow_buffer_size(int hop, int fft, int n, int ring, int post) {
    size_t total = ck_field_size(0, (size_t)fft, sizeof(float));   /* synth_window */
    total = ck_field_size_reps(total, (size_t)hop, sizeof(float), 2);
    if (post) {
        total = ck_field_size_reps(total, (size_t)n, sizeof(Complex), 5);
        total = ck_field_size_reps(total, (size_t)n, sizeof(float), 2);
    }
    if (ring > 0) total = ck_field_size(total, (size_t)ring, sizeof(float));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

/* Fills *d for a validated array cfg; 0 when a size cannot be formed. */
static int array_dims(const AecResolved* rs, ArrayDims* d) {
    const AecConfig* cfg = &rs->cfg;
    AecSignalGrid g;
    size_t total;
    memset(d, 0, sizeof(*d));
    if (aec_resolve_signal_grid(cfg->sample_rate, cfg->fft_size, &g) != AEC_OK) return 0;
    d->hop = g.hop_size; d->fft = g.fft_size; d->n = g.n_freqs;
    d->post = cfg->residual != AEC_RESIDUAL_STATE;
    d->full = cfg->residual == AEC_RESIDUAL_FULL;
    d->matched = cfg->delay_mode == AEC_DELAY_MATCHED;
    array_derive_lane(rs, &d->lane);

    if (d->matched) {
        /* As the single channel: the estimator's largest reportable delay + hop (eligible up to ring - hop). */
        int reach = delay_aec_max_delay_samples(cfg->sample_rate, cfg->delay_num_filters);
        d->ring = reach > 0 ? reach + d->hop : 0;
        d->delay_bytes = delay_aec_get_mem_size(cfg->sample_rate, d->hop, cfg->delay_num_filters);
        /* An estimator that cannot be sized makes the config invalid. */
        if (d->ring <= 0 || d->delay_bytes == 0) return 0;
    } else if (cfg->delay_mode == AEC_DELAY_FIXED) {
        d->ring = cfg->fixed_delay_samples + d->hop;
        if (d->ring <= 0) return 0;
    }
    d->ctl_bytes = ck_align16_size(sizeof(struct AecArray));
    d->buf_bytes = flow_buffer_size(d->hop, d->fft, d->n, d->ring, d->post);
    if (d->buf_bytes == 0) return 0;
    if (d->full) {
        d->res_bytes = aec_res_get_mem_size_resolved(&d->lane);
        if (d->res_bytes == 0) return 0;
    }
    d->fft_bytes = fft_get_mem_size(d->fft);
    if (d->fft_bytes == 0) return 0;
    d->lane_each_bytes = aec_lane_get_mem_size(&d->lane);
    if (d->lane_each_bytes == 0) return 0;

    total = d->ctl_bytes;
    total = ck_add_size(total, d->buf_bytes);
    if (d->matched) total = ck_add_size(total, ck_align16_size(d->delay_bytes));
    if (d->full) total = ck_add_size(total, ck_align16_size(d->res_bytes));
    total = ck_add_size(total, ck_align16_size(d->fft_bytes));
    total = ck_add_size(total, ck_mul_size((size_t)AEC_ARRAY_MICS, ck_align16_size(d->lane_each_bytes)));
    if (MEM_SIZE_INVALID(total)) return 0;
    d->total = total;
    return 1;
}

int aec_array_impl_config_ok(const AecResolved* rs) {
    const AecConfig* cfg = &rs->cfg;
    const AecTuning* tn = &rs->tun;
    int n = cfg->n_mics;
    if (n != 1 && n != AEC_ARRAY_MICS) return 0;
    if (n == 1) {
        return cfg->capture_proxy_channel == 0 && cfg->delay_backward_quarantine_enabled == 0;
    }
    {
        ArrayDims d;
        AecResolved lane_resolved;
        if (cfg->residual == AEC_RESIDUAL_OFF) return 0;
        if (cfg->capture_proxy_channel < 0 || cfg->capture_proxy_channel >= AEC_ARRAY_MICS) return 0;
        /* The flow's three product grids only. */
        if (!((cfg->sample_rate == 16000 && (cfg->fft_size == 256 || cfg->fft_size == 512)) ||
              (cfg->sample_rate == 48000 && cfg->fft_size == 1024))) return 0;
        if (!isfinite(tn->max_delay_ms) || tn->max_delay_ms < 0.0f || tn->max_delay_ms > 4096.0f) return 0;
        if (!array_dims(rs, &d)) return 0;
        return aec_resolve_tuned(&d.lane.cfg, &d.lane.tun, &lane_resolved);
    }
}

size_t aec_array_impl_get_mem_size(const AecResolved* rs) {
    ArrayDims d;
    size_t total;
    if (!array_dims(rs, &d)) return 0;
    total = ck_add_size(ck_align16_size(sizeof(Aec)), d.total);
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

int aec_array_impl_get_mem_breakdown(const AecResolved* rs, AecMemBreakdown* out) {
    ArrayDims d;
    size_t total, lanes, ring, est;
    if (!out) return AEC_ERR_ARG;
    memset(out, 0, sizeof(*out));
    if (!array_dims(rs, &d)) return AEC_ERR_ARG;
    total = aec_array_impl_get_mem_size(rs);
    if (total == 0) return AEC_ERR_ARG;
    lanes = ck_mul_size((size_t)AEC_ARRAY_MICS, ck_align16_size(d.lane_each_bytes));
    ring = d.ring > 0 ? ck_field_size(0, (size_t)d.ring, sizeof(float)) : 0;
    est = d.matched ? d.delay_bytes : 0;
    out->total_bytes = total;
    out->estimator_bytes = est;
    out->ring_bytes = ring;
    out->lane_bytes = lanes;
    /* control block, the buffers without the ring, the gain stage, the shared transform handle (the Aec header is not a part of
     * the array pool) */
    out->fusion_bytes = d.total - lanes - ring - (d.matched ? ck_align16_size(d.delay_bytes) : 0);
    return AEC_OK;
}

/* ---- construction -------------------------------------------------------------------------------------------------- */

static void array_clear_state(AecArray* p);

/* Builds the array in `base` (>= d.total bytes, 16-byte aligned), with the Aec handle `a` already zeroed. 0, or -1. */
static int array_setup(Aec* a, uint8_t* base, size_t avail, const AecResolved* rs) {
    const AecConfig* cfg = &rs->cfg;
    const AecTuning* tn = &rs->tun;
    ArrayDims d;
    AecArray* p;
    AcPoolCursor cursor;
    int ch, q_hops;
    void* region;

    if (!array_dims(rs, &d) || avail < d.total) return -1;
    memset(base, 0, d.ctl_bytes + d.buf_bytes);           /* control block and the flow buffers; lane pools are built by aec_init */
    p = (AecArray*)base;
    cursor.ptr = base + d.ctl_bytes;
    cursor.remaining = d.total - d.ctl_bytes;             /* a larger caller pool leaves its tail unused */

    p->hop_size = d.hop; p->n_freqs = d.n;
    p->delay_mode = (int)cfg->delay_mode;
    p->capture_proxy = cfg->capture_proxy_channel;
    p->fixed_delay = cfg->fixed_delay_samples;
    p->quarantine_enabled = cfg->delay_backward_quarantine_enabled;
    p->post = d.post; p->full = d.full;
    p->delay_ring_size = d.ring;

    p->synth_window = (float*)ac_pool_carve(&cursor, (size_t)d.fft, sizeof(float));
    if (!p->synth_window) return -1;
    aec_synth_window_build(p->synth_window, d.fft);
    p->mic_lane = (float*)ac_pool_carve(&cursor, (size_t)d.hop, sizeof(float));
    p->aligned_ref = (float*)ac_pool_carve(&cursor, (size_t)d.hop, sizeof(float));
    if (!p->mic_lane || !p->aligned_ref) return -1;
    if (d.post) {
        p->fused_error = (Complex*)ac_pool_carve(&cursor, (size_t)d.n, sizeof(Complex));
        p->fused_near = (Complex*)ac_pool_carve(&cursor, (size_t)d.n, sizeof(Complex));
        p->residual_work = (Complex*)ac_pool_carve(&cursor, (size_t)d.n, sizeof(Complex));
        p->residual_proj = (Complex*)ac_pool_carve(&cursor, (size_t)d.n, sizeof(Complex));
        p->fused_r2 = (float*)ac_pool_carve(&cursor, (size_t)d.n, sizeof(float));
        p->fused_comfort = (float*)ac_pool_carve(&cursor, (size_t)d.n, sizeof(float));
        p->render_base = (Complex*)ac_pool_carve(&cursor, (size_t)d.n, sizeof(Complex));
        if (!p->fused_error || !p->fused_near || !p->residual_work || !p->residual_proj || !p->fused_r2 ||
            !p->fused_comfort || !p->render_base) return -1;
    }
    if (d.ring > 0) {
        p->delay_ring = (float*)ac_pool_carve(&cursor, (size_t)d.ring, sizeof(float));
        if (!p->delay_ring) return -1;
    }
    if (d.matched) {
        region = ac_pool_carve(&cursor, d.delay_bytes, 1);
        if (!region || delay_aec_init(&p->delay, region, d.delay_bytes, cfg->sample_rate, d.hop,
                                      cfg->delay_num_filters) != 0) return -1;
    }
    if (d.full) {
        region = ac_pool_carve(&cursor, d.res_bytes, 1);
        p->gain_stage = region ? aec_res_init_resolved(region, d.res_bytes, &d.lane) : NULL;
        if (!p->gain_stage) return -1;
    }
    region = ac_pool_carve(&cursor, d.fft_bytes, 1);
    p->fft = region ? fft_init(region, d.fft_bytes, d.fft) : NULL;
    if (!p->fft) return -1;
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
        region = ac_pool_carve(&cursor, d.lane_each_bytes, 1);
        p->lanes[ch] = region ? aec_lane_init(region, d.lane_each_bytes, &d.lane, p->fft) : NULL;
        if (!p->lanes[ch]) return -1;      /* every lane lives in the caller's pool: a failed build holds nothing to release */
    }
    if (cursor.remaining != 0) return -1;

    /* Seconds -> hops once against the resolved grid, with the single-channel floor of 1. The duty floor is 1, not 2: 2 would
     * cap delay_est_period_s = 0.0 at 50 % engagement. */
    q_hops = (int)lrintf(cfg->delay_backward_quarantine_s / ((float)d.hop / (float)cfg->sample_rate));
    p->delay_quarantine_hops = q_hops < 1 ? 1 : q_hops;
    aec_duty_configure(&p->duty, DUTY_HOLD_S, tn->delay_est_period_s, d.hop, cfg->sample_rate);
    array_clear_state(p);

    a->cfg = *cfg;
    a->tun = *tn;
    a->hop_size = d.hop; a->block_size = d.fft; a->n_freqs = d.n;
    a->n_partitions = p->lanes[0]->n_partitions;
    a->is_static = 1;
    a->array = p;
    return 0;
}

Aec* aec_array_impl_init(void* mem, size_t mem_size, const AecResolved* rs) {
    Aec* a;
    size_t need = aec_array_impl_get_mem_size(rs);
    size_t header = ALIGN16(sizeof(Aec));
    if (!mem || need == 0 || mem_size < need) return NULL;
    a = (Aec*)mem;
    memset(a, 0, sizeof(Aec));
    if (array_setup(a, (uint8_t*)mem + header, mem_size - header, rs) != 0) {
        memset(a, 0, sizeof(Aec));
        return NULL;
    }
    a->ready = 1;
    return a;
}

void aec_array_impl_destroy(Aec* a) {
    AecArray* p = a->array;
    void* heap;
    int ch;
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) aec_destroy(p->lanes[ch]);
    fft_destroy(p->fft);
    heap = a->is_static ? NULL : a->heap_arena;
    a->ready = 0;
    if (heap) free(heap);       /* aec_create(): the block that holds *a, the control block and the lanes */
}

/* ---- shared alignment ----------------------------------------------------------------------------------------------- */

static void array_clear_state(AecArray* p) {
    p->delay_samples_seen = 0;
    p->accepted_delay = p->delay_mode == (int)AEC_DELAY_FIXED ? p->fixed_delay : 0;
    p->delay_quarantine_left = -1;
    p->delay_admission.candidate = 0;
    p->delay_admission.ttl = 0;
    p->realign_warm_lanes = 0;
    p->realign_soft_lanes = 0;
    p->delay_calls = 0;
    p->far_spec_shared_hop = 0;
    p->far_spec_mixed_hop = 0;
    p->far_sync = (unsigned char)((1u << AEC_ARRAY_MICS) - 2u);      /* lanes 1..n-1 */
    p->far_stale = 0u;
    aec_duty_reset(&p->duty);
    memset(&p->last_delay, 0, sizeof(p->last_delay));
    p->front_valid = 0;
    p->fused_valid = 0;
    p->cng_seed = AEC_CNG_SEED;
    p->handled_serial = 0u;
}

/* Duty watchdog signal: the best windowed ERLE of the four lanes (one quiet lane is no evidence that the shared alignment
 * collapsed); last hop's readings, 0 until the ERLE machinery ran. */
static float max_lane_erle_windowed(const AecArray* p) {
    float best = 0.0f;
    int ch;
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
        if (p->lanes[ch]->last_erle_windowed > best) best = p->lanes[ch]->last_erle_windowed;
    return best;
}

static AecDelayState update_shared_delay(AecArray* p, const float* capture, const float* render) {
    AecDelayState state;
    int estimated, eligible, was_usable, now_usable, accepted, run_filter;

    memset(&state, 0, sizeof(state));
    if (p->delay_mode == (int)AEC_DELAY_EXTERNAL_ALIGNED) {
        state.delay_samples = 0;
        state.confidence = 1.0f;
        state.solid = 1;
        p->accepted_delay = 0;
        p->last_delay = state;
        return state;
    }
    if (p->delay_mode == (int)AEC_DELAY_FIXED) {
        state.delay_samples = p->fixed_delay;
        state.confidence = 1.0f;
        /* Runs right before align_render(): readable once the stored samples cover the delay (the single-channel
         * `filled >= delay + hop`), so LOCKED lands on the hop the audio switches. */
        state.solid = p->delay_samples_seen >= (uint64_t)p->fixed_delay;
        p->accepted_delay = p->fixed_delay;
        p->last_delay = state;
        return state;
    }

    /* The estimator is fed every hop (decimators and ring stay gapless); only the analysis is decimated once the estimate held
     * still, so a movement is seen within duty.period_hops. */
    run_filter = aec_duty_begin(&p->duty);
    delay_aec_accumulate_ex(&p->delay, capture, render, p->hop_size, run_filter);
    p->delay_calls += 1;
    aec_duty_end(&p->duty, delay_aec_estimated_delay(&p->delay), delay_aec_is_solid(&p->delay), max_lane_erle_windowed(p));

    estimated = delay_aec_estimated_delay(&p->delay);
    /* An estimate past the reference ring cannot be served (align_render() would refuse it): not eligible. */
    eligible = estimated >= 0 && estimated <= p->delay_ring_size - p->hop_size && delay_aec_is_solid(&p->delay) &&
               delay_aec_n_updates(&p->delay) >= 3;

    /* Aged ahead of the quarantine and admission, as the single-channel AEC ages its pending delay. */
    aec_array_admission_age(&p->delay_admission);

    /* Backward-jump quarantine: judged on the capture-proxy lane's cancellation read before this hop's lanes run; armed once
     * per backward episode, adopted at 0. */
    if (eligible && p->quarantine_enabled && p->last_delay.solid && estimated < p->accepted_delay &&
        aec_linear_is_cancelling(p->lanes[p->capture_proxy])) {
        if (p->delay_quarantine_left < 0) p->delay_quarantine_left = p->delay_quarantine_hops;
        if (p->delay_quarantine_left > 0) {
            p->delay_quarantine_left--;
            eligible = 0;
        }
    } else {
        p->delay_quarantine_left = -1;
    }

    /* `solid` is sticky once an accepted generation exists; `changed` marks a new usable generation (a value compare would
     * miss an acquisition or relock at delay 0). First acquisition is immediate (the single-channel Path A); later moves go
     * through the admission, since each costs an IR shift on all four lanes. accepted_delay moves only on acceptance, so the
     * served alignment and the lanes' realign land on the same hop. */
    was_usable = p->last_delay.solid;
    now_usable = was_usable || eligible;
    accepted = 0;
    if (eligible) {
        if (!was_usable) {
            accepted = 1;
            p->delay_admission.candidate = 0;
            p->delay_admission.ttl = 0;
        } else {
            accepted = aec_array_admission_offer(&p->delay_admission, p->accepted_delay, estimated);
        }
    }
    state.changed = accepted;
    if (accepted) p->accepted_delay = estimated;
    state.delay_samples = p->accepted_delay;
    state.confidence = delay_aec_confidence(&p->delay);
    state.solid = now_usable;
    state.estimator_calls = p->delay_calls;
    state.estimator_updates = delay_aec_n_updates(&p->delay);
    p->last_delay = state;
    return state;
}

static int align_render(AecArray* p, const float* render, int delay_samples) {
    int hop, ring, write_at, read_at;
    uint64_t start;
    if (p->delay_mode == (int)AEC_DELAY_EXTERNAL_ALIGNED) {
        memcpy(p->aligned_ref, render, (size_t)p->hop_size * sizeof(float));
        p->delay_samples_seen += (uint64_t)p->hop_size;
        return 1;
    }
    if (delay_samples < 0 || delay_samples > p->delay_ring_size - p->hop_size) return 0;

    hop = p->hop_size;
    ring = p->delay_ring_size;
    start = p->delay_samples_seen;
    /* `start` is reduced once (no per-sample 64-bit divide). */
    write_at = (int)(start % (uint64_t)ring);
    aec_ring_write(p->delay_ring, ring, write_at, render, hop);
    /* Whole-hop decision on the predicate of FIXED's `solid`: until the samples seen cover the offset the far is the raw hop,
     * never silence or a splice. */
    if (start >= (uint64_t)delay_samples) {
        /* write_at - delay_samples + ring lies in (0, 2*ring): one conditional subtract. */
        read_at = write_at - delay_samples;
        if (read_at < 0) read_at += ring;
        aec_ring_read(p->delay_ring, ring, read_at, p->aligned_ref, hop);
    } else {
        memcpy(p->aligned_ref, render, (size_t)hop * sizeof(float));
    }
    p->delay_samples_seen += (uint64_t)hop;
    return 1;
}

/* Legal at any time. The lanes' buffers are not cleared (stale views, not a fault). */
void aec_array_impl_reset(Aec* a) {
    AecArray* p = a->array;
    int ch;
    if (p->delay_mode == (int)AEC_DELAY_MATCHED) delay_aec_reset(&p->delay);
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) aec_reset(p->lanes[ch]);
    if (p->gain_stage) aec_res_reset(p->gain_stage);
    if (p->delay_ring) memset(p->delay_ring, 0, (size_t)p->delay_ring_size * sizeof(float));
    memset(p->aligned_ref, 0, (size_t)p->hop_size * sizeof(float));
    array_clear_state(p);
    if (p->generation != 0xFFFFFFFFu) p->generation++;
}

/* ---- linear stage --------------------------------------------------------------------------------------------------- */

/* Points dst at the lane's per-hop buffers instead of copying: a lane runs again only inside aec_array_impl_process_linear() or
 * after a reset, and the array's views are valid until then. A reset does not clear that memory; the fusion refuses a stale
 * view. */
static int bind_lane_view(ArraySnapshot* dst, const Aec* lane, int n_freqs, int residual_half) {
    AecLinearView lv;
    AecResidualView rv;
    memset(&lv, 0, sizeof(lv)); lv.struct_size = sizeof(lv);
    memset(&rv, 0, sizeof(rv)); rv.struct_size = sizeof(rv);
    if (aec_get_linear(lane, &lv) != AEC_OK || aec_get_residual(lane, &rv) != AEC_OK) return 0;
    if (rv.n_freqs != n_freqs || lv.n_freqs != n_freqs || !rv.error_spec || !rv.echo_spec || !rv.far_spec ||
        !rv.near_spec || !rv.base_spec || !rv.formed_hop) return 0;
    if (residual_half && (!rv.r2 || !rv.comfort_noise)) return 0;

    dst->error_spec = rv.error_spec;
    dst->echo_spec = rv.echo_spec;
    dst->far_spec = rv.far_spec;
    dst->near_spec = rv.near_spec;
    dst->r2 = rv.r2;
    dst->comfort_noise = rv.comfort_noise;
    dst->formed_hop = rv.formed_hop;
    dst->base_spec = rv.base_spec;
    dst->guard_active = rv.guard_active;
    dst->reset_serial = rv.post_reset_serial;

    dst->far_power = lv.far_power;
    dst->dt_indicator = lv.dt_indicator;
    dst->saturation_level = lv.saturation_level;
    dst->filter_converged = lv.filter_converged;
    dst->usable_linear = rv.usable_linear;
    return 1;
}

/* A realign can leave a lane with a different far history from lane 0's (a soft one clears it, a warm one keeps it). Compared
 * on the buffers every far-only stage derives from: a lane whose history differs follows its own analysis from here on, until
 * the next reset. Invariant: only an external realign and aec_reset change one lane's far history separately from the others;
 * any new lane-local operation that clears far_buffer or the RSA / noise state must call this afterwards. */
static void far_resync(AecArray* p) {
    int ch;
    for (ch = 1; ch < AEC_ARRAY_MICS; ++ch) {
        if (((p->far_sync >> ch) & 1u) && !aec_lane_far_history_equal(p->lanes[0], p->lanes[ch]))
            p->far_sync = (unsigned char)(p->far_sync & ~(1u << ch));
    }
}

/* Lanes that reuse lane 0's far-only analysis this hop. Any other lane that was skipping its own update takes lane 0's state
 * now, before lane 0 runs, and updates locally from there. */
static unsigned far_analysis_lanes(AecArray* p) {
    unsigned borrow = 0u;
    int ch;
    for (ch = 1; ch < AEC_ARRAY_MICS; ++ch) {
        if (!p->far_hoist_off && !p->far_spec_mixed_hop && ((p->far_sync >> ch) & 1u)) {
            borrow |= 1u << ch;
        } else if ((p->far_stale >> ch) & 1u) {
            aec_lane_adopt_far_analysis(p->lanes[ch], p->lanes[0]);
            p->far_stale = (unsigned char)(p->far_stale & ~(1u << ch));
        }
    }
    return borrow;
}

int aec_array_impl_process_linear(Aec* a, const float* mics, const float* ref) {
    AecArray* p = a->array;
    AecDelayState delay;
    int hop, ch, i, old_align, lanes_borrowed;
    uint32_t t0;
    const Complex* shared_far_spec = NULL;
    long far_fft_before[AEC_ARRAY_MICS];
    AecFarShare share;
    unsigned borrow_analysis;

    if (!mics || !ref) return AEC_ERR_ARG;
    hop = p->hop_size;
    if (!skn_all_finite_f32(mics, (size_t)hop * AEC_ARRAY_MICS) || !skn_all_finite_f32(ref, (size_t)hop))
        return AEC_ERR_ARG;
    /* Cleared once the call is accepted, so a hop that bails out reports zero for stages it never reached; the lane figures
     * accumulate over the four lanes; fuse_us is cleared in aec_array_impl_process_residual(). */
    p->timing.delay_us = 0;
    p->timing.frontend_us = 0;
    p->timing.linear_us = 0;
    p->timing.lane_res_us = 0;
    p->far_spec_shared_hop = 0;
    p->far_spec_mixed_hop = 0;
    p->front_valid = 0;
    p->fused_valid = 0;

    /* Only lane 0 with proxy 0 can reuse this extraction: earlier lanes overwrite mic_lane before a later proxy lane runs. */
    for (i = 0; i < hop; ++i) p->mic_lane[i] = mics[i * AEC_ARRAY_MICS + p->capture_proxy];
    /* The alignment the lanes were served before this hop's decision (raw 0 until the estimate first turns solid). */
    old_align = p->last_delay.solid ? p->accepted_delay : 0;
    t0 = aec_now_us();
    delay = update_shared_delay(p, p->mic_lane, ref);
    p->timing.delay_us = aec_now_us() - t0;
    if (!align_render(p, ref, delay.delay_samples)) return aec_refuse_hop(a);
    if (delay.changed) {
        /* Realign instead of reset: each converged filter is shifted by the alignment delta (warm when the evidence gate holds,
         * filter-only reset otherwise), so cancellation survives and the WOLA sequences continue. */
        int hop_warm = 0, hop_soft = 0;
        for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
            int outcome = aec_apply_external_realign(p->lanes[ch], delay.delay_samples - old_align);
            if (outcome == 1) {
                p->realign_warm_lanes = aec_array_count_add_sat(p->realign_warm_lanes, 1);
                hop_warm += 1;
            } else if (outcome == 0) {
                p->realign_soft_lanes = aec_array_count_add_sat(p->realign_soft_lanes, 1);
                hop_soft += 1;
            }
        }
        /* A soft outcome zeroes that lane's far OLA history, so a mixed hop leaves the lanes out of lockstep: no far-FFT
         * borrowing and no far agreement requirement on this hop. */
        p->far_spec_mixed_hop = (hop_warm > 0 && hop_soft > 0);
        if (p->generation != 0xFFFFFFFFu) p->generation++;
        far_resync(p);
    }
    borrow_analysis = far_analysis_lanes(p);

    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) far_fft_before[ch] = aec_far_fft_real_compute_count(p->lanes[ch]);
    lanes_borrowed = 1;
    memset(&share, 0, sizeof(share));
    share.donor = p->lanes[0];

    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
        int rc;
        if (ch != 0 || p->capture_proxy != 0) {
            for (i = 0; i < hop; ++i) p->mic_lane[i] = mics[i * AEC_ARRAY_MICS + ch];
        }
        /* Lane 0 produces this hop's far spectrum; lanes 1-3 copy it after their prior-hop RSA read. On a mixed warm/soft hop
         * all lanes compute locally. A lane completes its hop (linear stage, then its residual stage) before the next starts. */
        share.hop = ch != 0 && !p->far_hoist_off;
        share.fft = ch != 0 && !p->far_spec_mixed_hop;
        share.post = share.hop && share.fft;
        share.analysis = (int)((borrow_analysis >> ch) & 1u);
        rc = aec_lane_process_linear(p->lanes[ch], p->mic_lane, p->aligned_ref, &share, ch == 0);
        if (share.analysis) p->far_stale = (unsigned char)(p->far_stale | (1u << ch));
        if (rc == AEC_OK) rc = aec_lane_process_residual(p->lanes[ch], ch == 0 ? NULL : &share);
        if (rc != AEC_OK || !bind_lane_view(&p->snapshots[ch], p->lanes[ch], p->n_freqs, p->post))
            return aec_refuse_hop(a);
        if (ch == 0) shared_far_spec = p->snapshots[0].far_spec;
        {
            AecStageTiming lane_time;
            aec_get_last_timing(p->lanes[ch], &lane_time);
            /* Lane-internal delay work is structurally zero (EXTERNAL_ALIGNED) and summed anyway. */
            p->timing.delay_us += lane_time.delay_us;
            p->timing.frontend_us += lane_time.frontend_us;
            p->timing.linear_us += lane_time.linear_us;
            p->timing.lane_res_us += lane_time.res_us;
        }
        /* Borrow-chain signature: the computing lane ran one far transform, a borrowing lane none; anything else sends the
         * fusion to the per-bin comparison. */
        {
            long after = aec_far_fft_real_compute_count(p->lanes[ch]);
            if (!aec_array_far_fft_delta_matches(far_fft_before[ch], after, ch == 0 ? 1 : 0)) lanes_borrowed = 0;
        }
    }
    p->far_spec_shared_hop = lanes_borrowed && shared_far_spec != NULL;

    p->front_valid = 1;
    p->front_serial = 0u;
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) p->front_serial += p->snapshots[ch].reset_serial;   /* wraps modulo 2^32 */

    /* The single-channel timing record, for aec_get_last_timing(). */
    a->last_timing.delay_us = p->timing.delay_us;
    a->last_timing.frontend_us = p->timing.frontend_us;
    a->last_timing.linear_us = p->timing.linear_us;
    a->last_timing.res_us = p->timing.lane_res_us;
    return AEC_OK;
}

/* ---- fusion ---------------------------------------------------------------------------------------------------------- */

/* A lane zeroed in every bin contributes nothing (bypassed beamformer: [1,0,0,0]). */
static int lane_contributes(const Complex* lane_weights, int n) {
    int k;
    for (k = 0; k < n; ++k)
        if (lane_weights[k].r != 0.0f || lane_weights[k].i != 0.0f) return 1;
    return 0;
}

/* Rendering base of the beam: the beam E itself when no contributing lane's guard fired, else sum_ch W_ch * B_ch into
 * render_base, lane order 0..3 as the error rebuild. 0 when that projection is not finite. */
static int project_render_base(AecArray* p, const Complex* weights, const Complex* analysis,
                               const Complex** base, int* guard_lanes) {
    int n = p->n_freqs;
    int mask = 0;
    int ch;
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
        if (p->snapshots[ch].guard_active && lane_contributes(weights + (size_t)ch * (size_t)n, n)) mask |= 1 << ch;
    *guard_lanes = mask;
    if (mask == 0) {
        *base = analysis;
        return 1;
    }
    memset(p->render_base, 0, (size_t)n * sizeof(Complex));
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
        sk_cmac_f32(p->render_base, weights + (size_t)ch * (size_t)n, p->snapshots[ch].base_spec, n);
    *base = p->render_base;
    return aec_cvec_finite(p->render_base, n);
}

static int fuse_contexts(AecArray* p, const Complex* weights, const Complex* trusted_beamformed_error,
                         int* all_converged, int* usable_linear, float* max_dt, float* max_saturation) {
    int n = p->n_freqs;
    int ch;
    float base_far_power = p->snapshots[0].far_power;

    *all_converged = 1;
    *usable_linear = 1;
    *max_dt = 0.0f;
    *max_saturation = 0.0f;

    /* A caller-supplied beamformed_error is used in place (validated by the caller of this function); fused_error is rebuilt
     * only without one. */
    if (!trusted_beamformed_error) memset(p->fused_error, 0, (size_t)n * sizeof(Complex));
    memset(p->fused_near, 0, (size_t)n * sizeof(Complex));
    memset(p->residual_proj, 0, (size_t)n * sizeof(Complex));
    memset(p->fused_comfort, 0, (size_t)n * sizeof(float));

    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
        const Complex* lane_weights = weights + (size_t)ch * (size_t)n;
        /* A lane that contributes nothing must not vote on the scalars that steer the gain stage. */
        int contributes = lane_contributes(lane_weights, n);
        float delta = fabsf(p->snapshots[ch].far_power - base_far_power);
        float scale = 1.0f + fabsf(base_far_power);
        if (delta > 1e-5f * scale) return 0;
        if (contributes) {
            if (!p->snapshots[ch].filter_converged) *all_converged = 0;
            if (!p->snapshots[ch].usable_linear) *usable_linear = 0;
            if (p->snapshots[ch].dt_indicator > *max_dt) *max_dt = p->snapshots[ch].dt_indicator;
            if (p->snapshots[ch].saturation_level > *max_saturation) *max_saturation = p->snapshots[ch].saturation_level;
        }

        /* Skipped on a mixed warm/soft-realign hop; the fused X is lane 0's far_spec either way. */
        if (!p->far_spec_mixed_hop &&
            !aec_array_far_spec_agrees(p->far_spec_shared_hop, p->snapshots[ch].far_spec, p->snapshots[0].far_spec, n))
            return 0;

        if (!trusted_beamformed_error)
            sk_cmac_f32(p->fused_error, lane_weights, p->snapshots[ch].error_spec, n);
        sk_cmac_f32(p->fused_near, lane_weights, p->snapshots[ch].near_spec, n);
        aec_array_residual_vector(p->residual_work, p->snapshots[ch].echo_spec, p->snapshots[ch].r2, n);
        sk_cmac_f32(p->residual_proj, lane_weights, p->residual_work, n);
        aec_array_comfort_accumulate(p->fused_comfort, lane_weights, p->snapshots[ch].comfort_noise, n);
    }

    sk_cmag_f32(p->fused_r2, p->residual_proj, n, 0.0f);   /* no epsilon: a sum of two squares is never -0.0 */
    /* Finite weights can still overflow the projections (|w| near 1e19 squares past FLT_MAX). */
    return aec_cvec_finite(p->fused_near, n) &&
           (trusted_beamformed_error || aec_cvec_finite(p->fused_error, n)) &&
           skn_all_finite_f32(p->fused_r2, (size_t)n) && skn_all_finite_f32(p->fused_comfort, (size_t)n);
}

int aec_array_impl_process_residual(Aec* a, const AecBeam* beam) {
    AecArray* p = a->array;
    int all_converged, usable_linear, guard_lanes, clock_drift;
    float max_dt, max_saturation;
    const Complex* analysis;
    const Complex* render;
    const float* gain = NULL;
    uint32_t t0;

    if (!p->post) return AEC_ERR_MODE;                       /* STATE: the lanes only, nothing to fuse */
    if (!beam || !beam->weights) return AEC_ERR_ARG;
    if (!p->front_valid) return AEC_ERR_SEQUENCE;
    /* One residual stage per linear stage once a gain stage exists (it advances with every call). Without one (ESTIMATE) the
     * fusion is stateless and may be repeated on the same hop with other weights. */
    if (p->fused_valid && p->full) return AEC_ERR_SEQUENCE;
    if (!aec_array_weights_valid(beam->weights, (size_t)AEC_ARRAY_MICS * (size_t)p->n_freqs)) return AEC_ERR_ARG;
    if (beam->beamformed_error && !aec_cvec_finite(beam->beamformed_error, p->n_freqs)) return AEC_ERR_ARG;

    p->timing.fuse_us = 0;
    t0 = aec_now_us();
    analysis = beam->beamformed_error ? beam->beamformed_error : p->fused_error;
    if (!fuse_contexts(p, beam->weights, beam->beamformed_error, &all_converged, &usable_linear, &max_dt,
                       &max_saturation) ||
        !project_render_base(p, beam->weights, analysis, &render, &guard_lanes))
        return aec_refuse_hop(a);

    clock_drift = p->delay_mode == (int)AEC_DELAY_MATCHED ? delay_aec_has_clockdrift(&p->delay) : 0;
    if (p->full) {
        AecResInput in;
        memset(&in, 0, sizeof(in));
        in.error_spec = analysis;
        in.near_spec = p->fused_near;
        in.r2 = p->fused_r2;
        in.comfort_noise = p->fused_comfort;
        in.render_hop = p->aligned_ref;
        in.all_converged = all_converged;
        in.max_dt = max_dt;
        in.max_saturation = max_saturation;
        in.clock_drift = clock_drift;
        gain = aec_res_process(p->gain_stage, &in);
        if (!gain) return aec_refuse_hop(a);
    }
    p->timing.fuse_us = aec_now_us() - t0;

    memset(&p->frame, 0, sizeof(p->frame));
    p->frame.e = analysis;
    p->frame.b = render;
    p->frame.y = p->fused_near;
    p->frame.x = p->snapshots[0].far_spec;
    p->frame.r2 = p->fused_r2;
    p->frame.cn = p->fused_comfort;
    p->frame.gain = gain;
    p->frame.far_hop = p->aligned_ref;
    p->frame.guard_lanes = guard_lanes;
    p->frame.serial = p->front_serial;
    p->frame.all_converged = all_converged;
    p->frame.usable_linear = usable_linear;
    p->frame.clock_drift = clock_drift;
    p->frame.max_dt = max_dt;
    p->frame.max_sat = max_saturation;
    p->fused_valid = 1;

    /* The comfort-noise generator restarts at the seed when a lane hard-reset since the last hop, before any draw of this hop. */
    if (p->frame.serial != p->handled_serial) {
        p->cng_seed = AEC_CNG_SEED;
        p->handled_serial = p->frame.serial;
    }
    return AEC_OK;
}

/* ---- views and getters ------------------------------------------------------------------------------------------------ */

int aec_array_impl_get_residual(const Aec* a, AecResidualView* out) {
    const AecArray* p = a->array;
    const ArrayFrame* f;
    if (!p->post) return AEC_ERR_MODE;
    if (!p->fused_valid) return AEC_ERR_SEQUENCE;
    f = &p->frame;
    out->n_freqs = p->n_freqs;
    out->hop_size = p->hop_size;
    out->formed_hop = NULL;                 /* one hop per lane: AecArrayView.lane_formed_hop */
    out->error_spec = f->e;
    out->echo_spec = NULL;
    out->near_spec = f->y;
    out->far_spec = f->x;
    out->base_spec = f->b;
    out->guard_active = f->guard_lanes != 0;
    out->res_gain = f->gain;
    out->r2 = f->r2;
    out->comfort_noise = f->cn;
    out->res_floor_protect = f->max_dt > AEC_RES_DT_PROTECT_THRESHOLD;
    out->usable_linear = f->usable_linear;
    out->post_reset_serial = f->serial;
    out->all_converged = f->all_converged;
    out->max_dt = f->max_dt;
    out->max_saturation = f->max_sat;
    out->clock_drift = f->clock_drift;
    out->guard_lanes = (unsigned int)f->guard_lanes;
    out->far_hop = f->far_hop;
    return AEC_OK;
}

static int array_gain_ready(const Aec* a) {
    const AecArray* p = a->array;
    if (!p->full) return AEC_ERR_MODE;
    if (!p->fused_valid) return AEC_ERR_SEQUENCE;
    return AEC_OK;
}

int aec_array_impl_apply_gain(const Aec* a, const float* extra_gain, Complex* out_spec) {
    const AecArray* p = a->array;
    int rc = array_gain_ready(a);
    if (rc != AEC_OK) return rc;
    if (!out_spec) return AEC_ERR_ARG;
    aec_apply_gain_spec(out_spec, p->frame.b, p->frame.gain, extra_gain, p->n_freqs);
    return AEC_OK;
}

int aec_array_impl_apply_cng(Aec* a, const float* extra_gain, Complex* spec) {
    AecArray* p = a->array;
    int rc = array_gain_ready(a);
    if (rc != AEC_OK) return rc;
    if (!spec) return AEC_ERR_ARG;
    aec_apply_cng_spec(&p->cng_seed, spec, p->frame.cn, p->frame.gain, extra_gain, p->n_freqs);
    return AEC_OK;
}

uint32_t aec_array_impl_post_reset_serial(const Aec* a) {
    const AecArray* p = a->array;
    uint32_t sum = 0u;
    int ch;
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) sum += aec_post_reset_serial(p->lanes[ch]);   /* wraps modulo 2^32 */
    return sum;
}

int aec_array_impl_synthesis_window(const Aec* a, const float** window, int* length) {
    const AecArray* p = a->array;
    *window = p->synth_window;
    *length = a->block_size;
    return 0;
}

int aec_array_impl_set_level(Aec* a, AecLevel level) {
    AecArray* p = a->array;
    /* Each store validates the level before writing; the gain stage goes first and refuses without writing. */
    if (p->gain_stage && aec_res_set_level(p->gain_stage, level) != 0) return -1;
    return aec_apply_level(&a->cfg, &a->tun, level);
}

void aec_array_impl_debug_status(const Aec* a, AecDebugStatus* out) {
    const AecArray* p = a->array;
    int solid;
    aec_debug_status(p->lanes[0], out);
    solid = p->last_delay.solid;
    out->delay_samples = (p->delay_mode == (int)AEC_DELAY_MATCHED && !solid) ? -1 : p->accepted_delay;
    out->delay_confidence = p->delay_mode == (int)AEC_DELAY_MATCHED ? p->last_delay.confidence : 1.0f;
    out->delay_updates = p->delay_mode == (int)AEC_DELAY_MATCHED ? delay_aec_n_updates(&p->delay) : 0;
    out->erle_windowed_db = max_lane_erle_windowed(p);
    out->duty_hops_total = p->delay_mode == (int)AEC_DELAY_MATCHED ? p->delay_calls : 0ull;
    out->duty_hops_run = p->duty.hops_run;
    out->n_partitions = a->n_partitions;
    out->delay_period_hops = p->duty.period_hops;
}

long aec_array_impl_far_fft_count(const Aec* a) {
    const AecArray* p = a->array;
    long total = 0;
    int ch;
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
        total = aec_array_count_add_sat(total, aec_far_fft_real_compute_count(p->lanes[ch]));
    return total;
}

int aec_array_impl_is_cancelling(const Aec* a) {
    const AecArray* p = a->array;
    return aec_linear_is_cancelling(p->lanes[p->capture_proxy]);
}

/* The aggregate linear view of an array: lane 0 stands for the four lanes (they share one far end, so far_power and the
 * lane-independent status agree; the fusion checks it); the alignment fields are the shared ones. */
int aec_array_impl_get_linear(const Aec* a, AecLinearView* out) {
    const AecArray* p = a->array;
    const AecDelayState* d = &p->last_delay;
    int rc;
    rc = aec_get_linear(p->lanes[0], out);
    if (rc != AEC_OK) return rc;
    out->aligned_far_hop = p->aligned_ref;
    out->generation = p->generation;
    if (p->delay_mode == (int)AEC_DELAY_EXTERNAL_ALIGNED) {
        out->delay_samples = 0;
        out->delay_confidence = 1.0f;
        out->delay_state = AEC_LINEAR_DELAY_LOCKED;
    } else if (p->delay_mode == (int)AEC_DELAY_FIXED) {
        out->delay_samples = p->fixed_delay;
        out->delay_confidence = 1.0f;
        out->delay_state = d->solid ? AEC_LINEAR_DELAY_LOCKED : AEC_LINEAR_DELAY_UNLOCKED;
    } else {
        out->delay_samples = d->solid ? d->delay_samples : -1;
        out->delay_confidence = d->confidence;
        out->delay_state = !d->solid ? AEC_LINEAR_DELAY_UNLOCKED
                         : (d->changed ? AEC_LINEAR_DELAY_CHANGED : AEC_LINEAR_DELAY_LOCKED);
    }
    return AEC_OK;
}

/* ---- aec_get_array: both kinds ---------------------------------------------------------------------------------------- */

int aec_get_array(const Aec* a, AecArrayView* out) {
    int ch;
    if (!out || out->struct_size < sizeof(*out)) return AEC_ERR_ARG;
    memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(*out);
    if (aec_unusable(a)) return AEC_ERR_ARG;
    if (a->array) {
        const AecArray* p = a->array;
        if (!p->front_valid) return AEC_ERR_SEQUENCE;
        out->n_freqs = p->n_freqs;
        for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
            out->lane_error_spec[ch] = p->snapshots[ch].error_spec;
            out->lane_base_spec[ch] = p->snapshots[ch].base_spec;
            out->lane_formed_hop[ch] = p->snapshots[ch].formed_hop;
            out->lane_guard_active[ch] = p->snapshots[ch].guard_active;
        }
        out->aligned_ref = p->aligned_ref;
        out->delay = p->last_delay;
        out->post_reset_serial = p->front_serial;
        return AEC_OK;
    }
    {
        /* A single-channel instance: lane 0 only, available once its residual stage ran. */
        AecResidualView rv;
        AecLinearView lv;
        AecDebugStatus ds;
        int rc;
        memset(&rv, 0, sizeof(rv)); rv.struct_size = sizeof(rv);
        memset(&lv, 0, sizeof(lv)); lv.struct_size = sizeof(lv);
        rc = aec_get_residual(a, &rv);
        if (rc != AEC_OK) return rc;
        if (aec_get_linear(a, &lv) != AEC_OK) return AEC_ERR_ARG;
        aec_debug_status(a, &ds);
        out->n_freqs = rv.n_freqs;
        out->lane_error_spec[0] = rv.error_spec;
        out->lane_base_spec[0] = rv.base_spec;
        out->lane_formed_hop[0] = rv.formed_hop;
        out->lane_guard_active[0] = rv.guard_active;
        out->aligned_ref = lv.aligned_far_hop;
        out->delay.delay_samples = lv.delay_samples;
        out->delay.confidence = lv.delay_confidence;
        out->delay.solid = lv.delay_state != AEC_LINEAR_DELAY_UNLOCKED;
        out->delay.changed = lv.delay_state == AEC_LINEAR_DELAY_CHANGED;
        out->delay.estimator_calls = ds.duty_hops_total;
        out->delay.estimator_updates = ds.delay_updates;
        out->post_reset_serial = rv.post_reset_serial;
        return AEC_OK;
    }
}

/* ---- diagnostics ------------------------------------------------------------------------------------------------------- */

/* The array of a live array handle, else NULL (single-channel, NULL or destroyed). */
static const AecArray* live_array(const Aec* a) {
    return aec_unusable(a) ? NULL : a->array;
}

long aec_array_realign_warm_lane_count(const Aec* a) {
    const AecArray* p = live_array(a);
    return p ? p->realign_warm_lanes : 0;
}

long aec_array_realign_soft_lane_count(const Aec* a) {
    const AecArray* p = live_array(a);
    return p ? p->realign_soft_lanes : 0;
}

int aec_array_matched_filter_count(const Aec* a) {
    return !aec_unusable(a) && a->cfg.delay_mode == AEC_DELAY_MATCHED ? 1 : 0;
}

int aec_array_linear_aec_count(const Aec* a) {
    if (aec_unusable(a)) return 0;
    return a->array ? AEC_ARRAY_MICS : 1;
}

void aec_array_get_timing(const Aec* a, AecArrayTiming* out) {
    const AecArray* p = live_array(a);
    if (!out) return;
    if (p) { *out = p->timing; return; }
    memset(out, 0, sizeof(*out));
    if (aec_unusable(a)) return;
    out->delay_us = a->last_timing.delay_us;
    out->frontend_us = a->last_timing.frontend_us;
    out->linear_us = a->last_timing.linear_us;
    out->lane_res_us = a->last_timing.res_us;
}

/* ---- test-only hooks (aec_test_hooks.h; compiled only under -DAEC_TESTING) ------------------------------------------------ */
#ifdef AEC_TESTING
int aec_testing_array_pending_delay_candidate(const Aec* a) {
    const AecArray* p = live_array(a);
    if (!p || p->delay_admission.ttl <= 0) return -1;
    return p->delay_admission.candidate;
}

/* A period of one hop is "analyse every hop" at the decision site, while arming, hold and watchdog keep running and stay
 * measurable through the census. */
void aec_testing_array_pin_duty_full_rate(Aec* a) {
    if (live_array(a)) a->array->duty.period_hops = 1;
}

float aec_testing_array_duty_erle_leak_db(const Aec* a) {
    return live_array(a) ? a->array->duty.erle_leak_db : 0.0f;
}

float aec_testing_array_duty_erle_peak(const Aec* a) {
    return live_array(a) ? a->array->duty.erle_peak : 0.0f;
}

int aec_testing_array_far_spec_provenance(const Aec* a) {
    return live_array(a) ? a->array->far_spec_shared_hop : 0;
}

void aec_testing_array_set_far_hoist(Aec* a, int enabled) {
    if (live_array(a)) a->array->far_hoist_off = enabled ? 0 : 1;
}

unsigned aec_testing_array_far_sync_lanes(const Aec* a) {
    return live_array(a) ? a->array->far_sync : 0u;
}

unsigned aec_testing_array_far_stale_lanes(const Aec* a) {
    return live_array(a) ? a->array->far_stale : 0u;
}

const Aec* aec_testing_array_lane(const Aec* a, int ch) {
    return live_array(a) && ch >= 0 && ch < AEC_ARRAY_MICS ? a->array->lanes[ch] : NULL;
}
#endif
