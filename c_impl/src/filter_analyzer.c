/* filter_analyzer.c -- see filter_analyzer.h. Float32 throughout. */
#include "filter_analyzer.h"
#include "aec_scale.h"

#include "utility/simd_aec.h"

#include <limits.h>
#include <math.h>
#include <string.h>

/* Minimum-phase high-pass taps, cutoff ~600 Hz. */
static const float FA_HPF_COEFFS[3] = {
    0.7929742f, -0.36072128f, -0.47047766f
};
#define FA_HPF_LEN 3

/* ConsistentFilterDetector thresholds. */
#define FA_FLOOR_RATIO        10.0f
#define FA_SECONDARY_RATIO     2.0f
#define FA_FLOOR_LOW_SAMPLES_16K    64
#define FA_FLOOR_HIGH_SAMPLES_16K  128

static int fa_floor_scale(int sample_rate) {
    /* Floor window scale: x3 on the 48 kHz grid, x1 otherwise. */
    return sample_rate == 48000 ? 3 : 1;
}

/* ConsistentFilterDetector */
static void cd_reset(FaConsistentDetector *c) {
    c->significant_peak = 0;
    c->floor_accum = 0.0f;
    c->secondary_peak = 0.0f;
    c->floor_low_limit = 0;
    c->floor_high_limit = 0;
    c->counter = 0;
    c->delay_ref = -10;
}

/* Accumulate the pairwise sum of |taps| and the max over [lo, hi). */
static void cd_accumulate_slice(FaConsistentDetector *c, const float *taps,
                                int lo, int hi /* exclusive */, float *scratch) {
    int i, n = hi - lo;
    float seg_max;
    if (n <= 0) {
        return;
    }
    scratch[0] = fabsf(taps[lo]);
    seg_max = scratch[0];
    for (i = 1; i < n; ++i) {
        scratch[i] = fabsf(taps[lo + i]);
        if (scratch[i] > seg_max) seg_max = scratch[i];
    }
    c->floor_accum += sk_pairwise_sum_tailfold_b_f32(scratch, (size_t)n);
    if (seg_max > c->secondary_peak) {
        c->secondary_peak = seg_max;
    }
}

/* Returns 1 when the peak is consistent for more than consistent_hold_hops hops. */
static int cd_detect(FaConsistentDetector *c, const float *h, int size,
                     int region_start, int region_end,
                     const float *render_block, int render_block_len,
                     int peak_index, int delay_blocks,
                     int sample_rate,
                     float *scratch) {
    int hi_lower, lo_upper;

    if (region_start == 0) {
        int scale = fa_floor_scale(sample_rate);
        int floor_low_offset = FA_FLOOR_LOW_SAMPLES_16K * scale;
        int floor_high_offset = FA_FLOOR_HIGH_SAMPLES_16K * scale;
        c->floor_accum = 0.0f;
        c->secondary_peak = 0.0f;
        c->floor_low_limit =
            (peak_index >= floor_low_offset) ? peak_index - floor_low_offset : 0;
        c->floor_high_limit =
            (peak_index > size - (floor_high_offset + 1))
                ? 0
                : peak_index + floor_high_offset;
    }

    /* Lower sweep: region_start .. min(region_end + 1, floor_low_limit). */
    hi_lower = region_end + 1;
    if (c->floor_low_limit < hi_lower) hi_lower = c->floor_low_limit;
    if (region_start < hi_lower) {
        cd_accumulate_slice(c, h, region_start, hi_lower, scratch);
    }

    /* Upper sweep: max(floor_high_limit, region_start) .. region_end. */
    lo_upper = (c->floor_high_limit > region_start) ? c->floor_high_limit : region_start;
    if (lo_upper <= region_end) {
        cd_accumulate_slice(c, h, lo_upper, region_end + 1, scratch);
    }

    /* End-of-sweep significance check. */
    if (region_end == size - 1) {
        int denom_i = c->floor_low_limit + size - c->floor_high_limit;
        float denom = (denom_i > 1) ? (float)denom_i : 1.0f;
        float floor_v = c->floor_accum / denom;
        float abs_peak = fabsf(h[peak_index]);
        c->significant_peak =
            (abs_peak > FA_FLOOR_RATIO * floor_v
             && abs_peak > FA_SECONDARY_RATIO * c->secondary_peak);
    }

    if (c->significant_peak) {
        int active;
        active = (sk_sum_sq_pairwise_f32(render_block, (size_t)render_block_len)
                  > c->active_render_threshold);
        if (c->delay_ref == delay_blocks) {
            /* Saturates at LONG_MAX, not at the hold threshold: the raw value is observable. */
            if (active && c->counter < LONG_MAX) c->counter += 1;
        } else {
            c->counter = 0;
            c->delay_ref = delay_blocks;
        }
    }

    return c->counter > c->consistent_hold_hops;
}

/* FilterAnalyzer */
void fa_init(FilterAnalyzer *m, float *h_highpass_storage, int size,
             float active_render_limit, int bounded_erl, float default_gain,
             float *abs_scratch, int abs_scratch_len,
             float *render_sq_scratch, int render_sq_scratch_len,
             int hop_size, int sample_rate) {
    m->hop_size = hop_size;
    m->sample_rate = sample_rate;
    m->bounded_erl = bounded_erl ? 1 : 0;
    m->default_gain = default_gain;
    m->consistent.active_render_threshold =
        (active_render_limit * active_render_limit) * (float)hop_size;
    /* 5 s convergence threshold and 1.5 s consistency hold, from the live grid. */
    m->convergence_threshold_hops = aec_ms_to_hops(5000.0f, hop_size, sample_rate);
    m->consistent.consistent_hold_hops = aec_ms_to_hops(1500.0f, hop_size, sample_rate);
    cd_reset(&m->consistent);
    m->region_start = 0;
    m->region_end = 0;
    m->blocks_since_reset = 0;
    m->peak_index = 0;
    m->gain = default_gain;
    m->consistent_estimate = 0;
    m->delay_blocks = 0;
    m->size = size;
    m->h_highpass = h_highpass_storage;
    m->abs_scratch = abs_scratch;
    m->abs_scratch_len = abs_scratch_len;
    m->render_sq_scratch = render_sq_scratch;
    m->render_sq_scratch_len = render_sq_scratch_len;
    /* No provider by default; fa_reset() leaves it alone since it is wiring, not analyzer state. */
    m->taps_provider = NULL;
    m->taps_provider_ctx = NULL;
    if (m->h_highpass != NULL && size > 0) {
        memset(m->h_highpass, 0, (size_t)size * sizeof(float));
    }
}

void fa_reset(FilterAnalyzer *m) {
    m->region_start = 0;
    m->region_end = 0;
    m->blocks_since_reset = 0;
    m->peak_index = 0;
    m->gain = m->default_gain;
    m->consistent_estimate = 0;
    m->delay_blocks = 0;
    cd_reset(&m->consistent);
    if (m->h_highpass != NULL && m->size > 0) {
        memset(m->h_highpass, 0, (size_t)m->size * sizeof(float));
    }
}

/* Advance the region by one block, wrapping at the end. */
static void fa_set_region(FilterAnalyzer *m, int size) {
    int block = m->hop_size;
    if (m->region_end >= size - 1) {
        m->region_start = 0;
    } else {
        m->region_start = m->region_end + 1;
    }
    m->region_end = m->region_start + block - 1;
    if (m->region_end > size - 1) m->region_end = size - 1;
}

void fa_set_taps_provider(FilterAnalyzer *m, FaTapsProvider fn, void *ctx) {
    m->taps_provider = fn;
    m->taps_provider_ctx = ctx;
}

/* Zero the region, then apply the causal 3-tap FIR. */
static void fa_preprocess(FilterAnalyzer *m, const float *filter_taps) {
    int start, end, zero_hi, i;
    start = FA_HPF_LEN - 1;
    if (m->region_start > start) start = m->region_start;
    end = m->region_end + 1;
    /* Zero only the prefix [region_start, start) the FIR loop will not reach. */
    zero_hi = (start < end) ? start : end;
    for (i = m->region_start; i < zero_hi; ++i) {
        m->h_highpass[i] = 0.0f;
    }
    if (start >= end) {
        return;
    }
    /* Causal 3-tap FIR: out = x[i]*c0 + x[i-1]*c1 + x[i-2]*c2. */
    for (i = start; i < end; ++i) {
        float v = filter_taps[i]     * FA_HPF_COEFFS[0]
                + filter_taps[i - 1] * FA_HPF_COEFFS[1]
                + filter_taps[i - 2] * FA_HPF_COEFFS[2];
        m->h_highpass[i] = v;
    }
}

void fa_update(FilterAnalyzer *m, const float *filter_taps,
               const float *render_block, int render_block_len) {
    int size = m->size;
    const float *h;
    int seg_lo, seg_hi, seg_amax;
    float cur_sq, best_sq, hp, sp;
    int sufficient;
    float peak_abs;

    /* Guard: an invalid config (size <= 0) must not reach the peak search. */
    if (size <= 0) {
        return;
    }
    /* Guard: the caller-owned scratch buffers must cover this call. */
    if (m->abs_scratch_len < size || m->render_sq_scratch_len < render_block_len) {
        return;
    }
    /* Capped at convergence_threshold_hops + 1 to avoid signed overflow; only the `>` comparison
     * below reads it. */
    if (m->blocks_since_reset <= m->convergence_threshold_hops)
        m->blocks_since_reset += 1;
    fa_set_region(m, size);
    if (m->taps_provider != NULL) {
        /* Ask the owner for exactly the taps fa_preprocess() reads: [region_start - 2, region_end],
         * clamped at 0. Everything else in filter_taps is never read by this module. */
        int span_lo = m->region_start - (FA_HPF_LEN - 1);
        if (span_lo < 0) span_lo = 0;
        m->taps_provider(m->taps_provider_ctx, span_lo, m->region_end);
    }
    fa_preprocess(m, filter_taps);
    h = m->h_highpass;

    /* Find the peak by comparing squared magnitudes. */
    if (m->peak_index >= size) {
        m->peak_index = size - 1;
    }
    hp = h[m->peak_index];
    cur_sq = hp * hp;
    seg_lo = m->region_start;
    seg_hi = m->region_end + 1;
    /* First index of the maximum of seg*seg. */
    seg_amax = sk_argmax_sq_f32(h + seg_lo, seg_hi - seg_lo);
    sp = h[seg_lo + seg_amax];
    best_sq = sp * sp;
    if (best_sq > cur_sq) {
        m->peak_index = seg_lo + seg_amax;
    }
    m->delay_blocks = m->peak_index / m->hop_size;

    /* Echo path gain = |h_highpass[peak]|. */
    sufficient = (m->blocks_since_reset > m->convergence_threshold_hops);
    peak_abs = fabsf(h[m->peak_index]);
    if (sufficient && m->consistent_estimate) {
        m->gain = peak_abs;
    } else if (m->gain != 0.0f) {
        m->gain = (m->gain > peak_abs) ? m->gain : peak_abs;
    }
    if (m->bounded_erl && m->gain != 0.0f) {
        m->gain = (m->gain > 0.01f) ? m->gain : 0.01f;
    }

    /* ConsistentFilterDetector. */
    m->consistent_estimate = cd_detect(
        &m->consistent, h, size, m->region_start, m->region_end,
        render_block, render_block_len, m->peak_index, m->delay_blocks,
        m->sample_rate,
        m->abs_scratch);
}

/* Queries */
int fa_min_filter_delay_blocks(const FilterAnalyzer *m) {
    return m->delay_blocks;
}

float fa_max_echo_path_gain(const FilterAnalyzer *m) {
    return m->gain;
}

