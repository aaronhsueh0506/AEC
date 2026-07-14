/* filter_analyzer.c — byte-equal C port of
 * python/modules/state/filter_analyzer.py. See header for the parity contract.
 * Build with -ffp-contract=off. */
#include "filter_analyzer.h"

#include <math.h>
#include <string.h>

/* Minimum-phase HPF cutoff ~600 Hz (filter_analyzer.cc:170-172 verbatim,
 * stored as float32 in Python). */
static const float FA_HPF_COEFFS[3] = {
    0.7929742f, -0.36072128f, -0.47047766f
};
#define FA_HPF_LEN 3

/* ConsistentFilterDetector thresholds (filter_analyzer.cc:264-265 verbatim). */
#define FA_FLOOR_RATIO        10.0f
#define FA_SECONDARY_RATIO     2.0f
#define FA_FLOOR_LOW_OFFSET    64
#define FA_FLOOR_HIGH_OFFSET   128

/* ---------------------------------------------------------------------------
 * numpy 1.26 pairwise summation (float32 accumulate). Mirrors
 * numpy/core/src/umath/loops_utils.h.src @pairwise_sum@:
 *   n < 8     -> sequential
 *   n <= 128  -> 8 partial accumulators, balanced-tree combine, sequential tail
 *   n > 128   -> split at n2 = (n/2) floored to a multiple of 8, recurse, add.
 * Every op is a float32 add (no f64 widening). */
float fa_f32_pairwise_sum(const float *a, size_t n) {
    if (n == 0) {
        return 0.0f;
    }
    if (n < 8) {
        float res = a[0];
        size_t i;
        for (i = 1; i < n; ++i) {
            res = res + a[i];
        }
        return res;
    }
    if (n <= 128) {
        float r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
        float r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        float res;
        size_t i;
        for (i = 8; i + 8 <= n; i += 8) {
            r0 = r0 + a[i + 0];
            r1 = r1 + a[i + 1];
            r2 = r2 + a[i + 2];
            r3 = r3 + a[i + 3];
            r4 = r4 + a[i + 4];
            r5 = r5 + a[i + 5];
            r6 = r6 + a[i + 6];
            r7 = r7 + a[i + 7];
        }
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; ++i) {
            res = res + a[i];
        }
        return res;
    } else {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return fa_f32_pairwise_sum(a, n2) + fa_f32_pairwise_sum(a + n2, n - n2);
    }
}

/* Same structure, float32-by-design (formerly a float64 accumulate variant;
 * converted for uniformity — the render active-power sum below now runs
 * entirely in float32, matching fa_f32_pairwise_sum bit-for-bit). */
float fa_f64_pairwise_sum(const float *a, size_t n) {
    if (n == 0) {
        return 0.0f;
    }
    if (n < 8) {
        float res = a[0];
        size_t i;
        for (i = 1; i < n; ++i) {
            res = res + a[i];
        }
        return res;
    }
    if (n <= 128) {
        float r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
        float r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        float res;
        size_t i;
        for (i = 8; i + 8 <= n; i += 8) {
            r0 = r0 + a[i + 0];
            r1 = r1 + a[i + 1];
            r2 = r2 + a[i + 2];
            r3 = r3 + a[i + 3];
            r4 = r4 + a[i + 4];
            r5 = r5 + a[i + 5];
            r6 = r6 + a[i + 6];
            r7 = r7 + a[i + 7];
        }
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; ++i) {
            res = res + a[i];
        }
        return res;
    } else {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return fa_f64_pairwise_sum(a, n2) + fa_f64_pairwise_sum(a + n2, n - n2);
    }
}

/* ----------------------- ConsistentFilterDetector ------------------------ */
static void cd_reset(FaConsistentDetector *c) {
    c->significant_peak = 0;
    c->floor_accum = 0.0f;
    c->secondary_peak = 0.0f;
    c->floor_low_limit = 0;
    c->floor_high_limit = 0;
    c->counter = 0;
    c->delay_ref = -10;
}

/* abs(float32 slice) pairwise-summed in f32 + f32 max over the same slice. */
static void cd_accumulate_slice(FaConsistentDetector *c, const float *taps,
                                int lo, int hi /* exclusive */, float *scratch) {
    int i, n = hi - lo;
    float seg_max;
    if (n <= 0) {
        return;
    }
    for (i = 0; i < n; ++i) {
        scratch[i] = fabsf(taps[lo + i]);
    }
    /* np.abs(seg).sum() : float32 pairwise, accumulated directly (float32-by-design) */
    c->floor_accum += fa_f32_pairwise_sum(scratch, (size_t)n);
    /* np.abs(seg).max(initial=0.0) : float32 max, stored directly (float32-by-design) */
    seg_max = scratch[0];
    for (i = 1; i < n; ++i) {
        if (scratch[i] > seg_max) seg_max = scratch[i];
    }
    if (seg_max > c->secondary_peak) {
        c->secondary_peak = seg_max;
    }
}

/* Mirrors FilterAnalyzer::ConsistentFilterDetector.detect. Returns bool. */
static int cd_detect(FaConsistentDetector *c, const float *h, int size,
                     int region_start, int region_end,
                     const float *render_block, int render_block_len,
                     int peak_index, int delay_blocks,
                     float *scratch, float *render_sq_scratch) {
    int hi_lower, lo_upper;

    if (region_start == 0) {
        c->floor_accum = 0.0f;
        c->secondary_peak = 0.0f;
        c->floor_low_limit =
            (peak_index >= FA_FLOOR_LOW_OFFSET) ? peak_index - FA_FLOOR_LOW_OFFSET : 0;
        c->floor_high_limit =
            (peak_index > size - (FA_FLOOR_HIGH_OFFSET + 1))
                ? 0
                : peak_index + FA_FLOOR_HIGH_OFFSET;
    }

    /* Lower sweep: region_start .. min(region_end+1, floor_low_limit) */
    hi_lower = region_end + 1;
    if (c->floor_low_limit < hi_lower) hi_lower = c->floor_low_limit;
    if (region_start < hi_lower) {
        cd_accumulate_slice(c, h, region_start, hi_lower, scratch);
    }

    /* Upper sweep: max(floor_high_limit, region_start) .. region_end */
    lo_upper = (c->floor_high_limit > region_start) ? c->floor_high_limit : region_start;
    if (lo_upper <= region_end) {
        cd_accumulate_slice(c, h, lo_upper, region_end + 1, scratch);
    }

    /* End-of-sweep significance check (filter_analyzer.cc:258-266). */
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
        int active, i;
        for (i = 0; i < render_block_len; ++i) {
            float v = render_block[i];  /* float32-by-design (was astype(np.float64)) */
            render_sq_scratch[i] = v * v;
        }
        active = (fa_f64_pairwise_sum(render_sq_scratch, (size_t)render_block_len)
                  > c->active_render_threshold);
        if (c->delay_ref == delay_blocks) {
            if (active) c->counter += 1;
        } else {
            c->counter = 0;
            c->delay_ref = delay_blocks;
        }
    }

    return c->counter > FA_CONSISTENT_HOLD_HOPS;
}

/* ------------------------------ FilterAnalyzer --------------------------- */
void fa_init(FilterAnalyzer *m, float *h_highpass_storage, int size,
             float active_render_limit, int bounded_erl, float default_gain) {
    m->active_render_threshold =
        (active_render_limit * active_render_limit) * (float)FA_HOP_SAMPLES;
    m->bounded_erl = bounded_erl ? 1 : 0;
    m->default_gain = default_gain;
    m->consistent.active_render_threshold = m->active_render_threshold;
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

/* filter_analyzer.cc:194-206. Cycles one block forward; wraps at end. */
static void fa_set_region(FilterAnalyzer *m, int size) {
    int block = FA_HOP_SAMPLES;
    if (m->region_end >= size - 1) {
        m->region_start = 0;
    } else {
        m->region_start = m->region_end + 1;
    }
    m->region_end = m->region_start + block - 1;
    if (m->region_end > size - 1) m->region_end = size - 1;
}

/* filter_analyzer.cc:161-187: zero region, then causal 3-tap convolution.
 * h_highpass already sized to `size` (init/reset zeroed it). */
static void fa_preprocess(FilterAnalyzer *m, const float *filter_taps) {
    int start, end, i;
    /* h_highpass[region_start:region_end+1] = 0.0 */
    for (i = m->region_start; i <= m->region_end; ++i) {
        m->h_highpass[i] = 0.0f;
    }
    start = FA_HPF_LEN - 1;                 /* _HPF_COEFFS.size - 1 = 2 */
    if (m->region_start > start) start = m->region_start;
    end = m->region_end + 1;
    if (start >= end) {
        return;
    }
    /* Causal FIR, 3 taps, all float32:
     *   out = x[i]*c0 + x[i-1]*c1 + x[i-2]*c2 */
    for (i = start; i < end; ++i) {
        float v = filter_taps[i]     * FA_HPF_COEFFS[0]
                + filter_taps[i - 1] * FA_HPF_COEFFS[1]
                + filter_taps[i - 2] * FA_HPF_COEFFS[2];
        m->h_highpass[i] = v;          /* astype(np.float32) no-op (already f32) */
    }
}

void fa_update(FilterAnalyzer *m, const float *filter_taps,
               const float *render_block, int render_block_len) {
    int size = m->size;
    const float *h;
    int seg_lo, seg_hi, i, seg_amax;
    float cur_sq, best_sq, hp, sp;
    int sufficient;
    float peak_abs;
    /* scratch buffers sized to the filter length / render length. The largest
     * abs-slice spans at most one region (HOP) plus the floor offsets; the
     * filter length (size) bounds it safely. render block bounds the sq buf. */
    float  abs_scratch[1024];           /* >= size (960); detect slice <= size  */
    float  render_sq_scratch[1024];     /* >= render_block_len (160)            */

    if (size == 0) {
        return;
    }
    m->blocks_since_reset += 1;
    fa_set_region(m, size);
    fa_preprocess(m, filter_taps);
    h = m->h_highpass;

    /* Find peak (filter_analyzer.cc:127-130). Compare squared magnitudes. */
    if (m->peak_index >= size) {
        m->peak_index = size - 1;
    }
    hp = h[m->peak_index];
    cur_sq = hp * hp;                   /* float32 */
    seg_lo = m->region_start;
    seg_hi = m->region_end + 1;         /* exclusive */
    /* np.argmax(seg*seg): first index of the maximum (strict-greater scan). */
    seg_amax = 0;
    best_sq = h[seg_lo] * h[seg_lo];
    for (i = 1; i < seg_hi - seg_lo; ++i) {
        sp = h[seg_lo + i];
        {
            float s = sp * sp;
            if (s > best_sq) {
                best_sq = s;
                seg_amax = i;
            }
        }
    }
    if (best_sq > cur_sq) {
        m->peak_index = seg_lo + seg_amax;
    }
    m->delay_blocks = m->peak_index / FA_HOP_SAMPLES;

    /* UpdateFilterGain (filter_analyzer.cc:142-159). AEC3 passes h_highpass_ to
     * UpdateFilterGain and reads filter_time_domain[peak_index] from it, so the
     * gain is |h_hpf[peak]| (matches filter_analyzer.py update()). */
    sufficient = (m->blocks_since_reset > FA_CONVERGENCE_THRESHOLD_HOPS);
    peak_abs = fabsf(h[m->peak_index]);
    if (sufficient && m->consistent_estimate) {
        m->gain = peak_abs;
    } else if (m->gain != 0.0f) {        /* `elif self._gain:` (truthy) */
        m->gain = (m->gain > peak_abs) ? m->gain : peak_abs;
    }
    if (m->bounded_erl && m->gain != 0.0f) {
        m->gain = (m->gain > 0.01f) ? m->gain : 0.01f;
    }

    /* ConsistentFilterDetector (filter_analyzer.cc:135-138). */
    (void)abs_scratch; /* sized above */
    m->consistent_estimate = cd_detect(
        &m->consistent, h, size, m->region_start, m->region_end,
        render_block, render_block_len, m->peak_index, m->delay_blocks,
        abs_scratch, render_sq_scratch);
}

/* ------------------------------ queries --------------------------------- */
int fa_min_filter_delay_blocks(const FilterAnalyzer *m) {
    return m->delay_blocks;
}

int fa_any_filter_consistent(const FilterAnalyzer *m) {
    return m->consistent_estimate;
}

float fa_max_echo_path_gain(const FilterAnalyzer *m) {
    return m->gain;
}

int fa_peak_index(const FilterAnalyzer *m) {
    return m->peak_index;
}

const float *fa_get_adjusted_filters(const FilterAnalyzer *m) {
    return m->h_highpass;
}
