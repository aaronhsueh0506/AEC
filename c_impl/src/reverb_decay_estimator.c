/* reverb_decay_estimator.c — C port of
 * python/modules/residual/reverb_decay_estimator.py.  See header for parity
 * notes.  Internal math is double throughout (matches the numpy .astype(f64)
 * casts + math.log2/pow in the Python reference); inputs arrive as float32. */
#include "reverb_decay_estimator.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── numpy 1.26 pairwise sum (8-way unrolled for n<=128) ─────────────────
 * Reproduces numpy/core/src/umath/loops_utils.h.src pairwise_sum for the small
 * arrays this module reduces (n=64 block taps, and legacy-regression slices,
 * all <=128).  A naive left-fold is NOT bit-equal. */
double rde_pairwise_sum(const double *a, int n) {
    if (n < 8) {
        double s = 0.0;
        int i;
        for (i = 0; i < n; ++i) s = s + a[i];
        return s;
    } else {
        /* n <= 128 path (this module never sums more than 128 doubles). */
        double r[8];
        double res;
        int i, j, lim;
        for (j = 0; j < 8; ++j) r[j] = a[j];
        lim = n - (n % 8);
        for (i = 8; i < lim; i += 8)
            for (j = 0; j < 8; ++j) r[j] = r[j] + a[i + j];
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res = res + a[i];
        return res;
    }
}

/* AEC3 cc:60-62 — N(N²-1)/12 in double.  N=0 → 0.0*-1.0/12.0 = -0.0. */
static double symmetric_arithmetic_sum(int N) {
    double Nd = (double)N;
    return Nd * (Nd * Nd - 1.0) / 12.0;
}

/* AEC3 cc:76-85 — mean of squared TD taps in a 64-sample block (double). */
static double block_energy_average(const float *h, int block_index) {
    int s = block_index * RDE_K_FFT_LENGTH_BY_2;
    double sq[RDE_K_FFT_LENGTH_BY_2];
    int k;
    for (k = 0; k < RDE_K_FFT_LENGTH_BY_2; ++k) {
        double v = (double)h[s + k];
        sq[k] = v * v;
    }
    return rde_pairwise_sum(sq, RDE_K_FFT_LENGTH_BY_2) / (double)RDE_K_FFT_LENGTH_BY_2;
}

/* AEC3 cc:65-73 — max squared TD tap in a 64-sample block. */
static double block_energy_peak(const float *h, int peak_block) {
    int s = peak_block * RDE_K_FFT_LENGTH_BY_2;
    double best;
    double best_val;
    int k, peak_idx = 0;
    /* np.argmax(seg*seg): first index of the max of seg², seg in float64. */
    best = (double)h[s] * (double)h[s];
    for (k = 1; k < RDE_K_FFT_LENGTH_BY_2; ++k) {
        double v = (double)h[s + k];
        double sq = v * v;
        if (sq > best) { best = sq; peak_idx = k; }
    }
    best_val = (double)h[s + peak_idx];
    return best_val * best_val;
}

/* ── LateReverbLinearRegressor ───────────────────────────────────────────── */
static void late_reset(RdeLateRegressor *L, int num_data_points) {
    int N = num_data_points;
    L->N = N;
    L->n = 0;
    L->nz = 0.0;
    L->nn = symmetric_arithmetic_sum(N);
    L->count = (N > 0) ? (-0.5 * (double)N + 0.5) : 0.0;
}

static void late_accumulate(RdeLateRegressor *L, double z) {
    L->nz += L->count * z;
    L->count += 1.0;
    L->n += 1;
}

static int late_estimate_available(const RdeLateRegressor *L) {
    return (L->n == L->N) && (L->N > 0);
}

static double late_estimate(const RdeLateRegressor *L) {
    if (L->nn == 0.0) return 0.0;
    return L->nz / L->nn;
}

/* ── EarlyReverbLengthEstimator ──────────────────────────────────────────── */
static void early_init(RdeEarlyEstimator *E, int max_blocks) {
    int nsec = max_blocks - RDE_K_BLOCKS_PER_SECTION;
    if (nsec < 0) nsec = 0;
    E->n_storage = nsec;
    E->numerators        = (double *)calloc((size_t)(nsec > 0 ? nsec : 1), sizeof(double));
    E->numerators_smooth = (double *)calloc((size_t)(nsec > 0 ? nsec : 1), sizeof(double));
    E->n_sections = 0;
    E->coefficients_counter = 0;
    E->block_counter = 0;
    E->first_point = (-0.5 * (double)RDE_K_BLOCKS_PER_SECTION
                          * (double)RDE_K_FFT_LENGTH_BY_2 + 0.5);
}

static void early_free(RdeEarlyEstimator *E) {
    free(E->numerators);        E->numerators = NULL;
    free(E->numerators_smooth); E->numerators_smooth = NULL;
    E->n_storage = 0;
}

/* AEC3 Reset(): clear numerators, counters; numerators_smooth NOT reset. */
static void early_reset(RdeEarlyEstimator *E) {
    int i;
    for (i = 0; i < E->n_storage; ++i) E->numerators[i] = 0.0;
    E->coefficients_counter = 0;
    E->block_counter = 0;
}

static void early_accumulate(RdeEarlyEstimator *E, double value, double smoothing) {
    int first_section = E->block_counter - RDE_K_BLOCKS_PER_SECTION + 1;
    int last_section  = E->block_counter;
    double x_value, value_to_inc, value_to_add;
    int section;

    if (first_section < 0) first_section = 0;
    if (last_section > E->n_storage - 1) last_section = E->n_storage - 1;

    if (last_section < 0) {
        /* Block beyond storage — still advance coeff counter. */
        E->coefficients_counter += 1;
        if (E->coefficients_counter == RDE_K_FFT_LENGTH_BY_2) {
            E->block_counter += 1;
            E->coefficients_counter = 0;
        }
        return;
    }

    x_value = (double)E->coefficients_counter + E->first_point;
    value_to_inc = (double)RDE_K_FFT_LENGTH_BY_2 * value;
    value_to_add = x_value * value
                 + (double)(E->block_counter - last_section) * value_to_inc;
    for (section = last_section; section >= first_section; --section) {
        E->numerators[section] += value_to_add;
        value_to_add += value_to_inc;
    }

    E->coefficients_counter += 1;
    if (E->coefficients_counter == RDE_K_FFT_LENGTH_BY_2) {
        if (E->block_counter >= (RDE_K_BLOCKS_PER_SECTION - 1)) {
            section = E->block_counter - (RDE_K_BLOCKS_PER_SECTION - 1);
            if (section < E->n_storage) {
                E->numerators_smooth[section] +=
                    smoothing * (E->numerators[section]
                                 - E->numerators_smooth[section]);
                E->n_sections = section + 1;
            }
        }
        E->block_counter += 1;
        E->coefficients_counter = 0;
    }
}

/* AEC3 cc:366-404 — returns size in blocks of early reverb. */
static int early_estimate(const RdeEarlyEstimator *E) {
    const int kNumSectionsToAnalyze = 9;
    int N, k, early_size_minus_1 = 0;
    double nn, numerator_11, numerator_08, min_numerator_tail;
    int have_tail = 0;

    if (E->n_sections < kNumSectionsToAnalyze) return 0;

    N = RDE_K_BLOCKS_PER_SECTION * RDE_K_FFT_LENGTH_BY_2;
    nn = symmetric_arithmetic_sum(N);
    numerator_11 = RDE_LOG2_1_1 * nn / (double)RDE_K_FFT_LENGTH_BY_2;
    numerator_08 = RDE_LOG2_0_8 * nn / (double)RDE_K_FFT_LENGTH_BY_2;

    /* tail = numerators_smooth[9 : n_sections]; min(tail). */
    for (k = kNumSectionsToAnalyze; k < E->n_sections; ++k) {
        double v = E->numerators_smooth[k];
        if (!have_tail || v < min_numerator_tail) {
            min_numerator_tail = v;
            have_tail = 1;
        }
    }
    if (!have_tail) return 0;

    for (k = 0; k < kNumSectionsToAnalyze; ++k) {
        double val = E->numerators_smooth[k];
        if (val > numerator_11
            || (val < numerator_08 && val < 0.9 * min_numerator_tail)) {
            early_size_minus_1 = k;
        }
    }
    return (early_size_minus_1 == 0) ? 0 : early_size_minus_1 + 1;
}

/* ── ReverbDecayEstimator ────────────────────────────────────────────────── */
void rde_init(ReverbDecayEstimator *r, int n_partitions, int hop_size,
              double default_decay, double mild_decay,
              int use_adaptive, int use_aec3_block_energy) {
    memset(r, 0, sizeof(*r));
    r->n_partitions = n_partitions;
    r->hop_size = hop_size;
    r->default_decay = default_decay;
    r->mild_decay = mild_decay;
    r->use_adaptive = use_adaptive ? 1 : 0;
    r->use_aec3_block_energy = use_aec3_block_energy ? 1 : 0;
    r->decay = default_decay;
    r->smoothing_constant = 0.0;

    r->filter_length_blocks = 0;
    r->has_early = 0;
    /* late_reverb_decay_estimator = _LateReverbLinearRegressor() → reset(0). */
    late_reset(&r->late, 0);
    r->late_reverb_start = 0;
    r->late_reverb_end = 0;
    r->block_to_analyze = 0;
    r->estimation_region_identified = 0;
    r->estimation_region_candidate_size = 0;
    r->previous_gains = NULL;
    r->n_previous_gains = 0;
    r->tail_gain = 0.0;
}

void rde_free(ReverbDecayEstimator *r) {
    if (r->has_early) early_free(&r->early);
    r->has_early = 0;
    free(r->previous_gains);
    r->previous_gains = NULL;
    r->n_previous_gains = 0;
}

double rde_decay_value(const ReverbDecayEstimator *r) { return r->decay; }

double rde_decay(const ReverbDecayEstimator *r, int mild) {
    if (r->use_adaptive) return r->decay;
    return mild ? r->mild_decay : r->default_decay;
}

/* AEC3 cc:151-160. */
static void reset_decay_estimation(ReverbDecayEstimator *r) {
    if (r->has_early) early_reset(&r->early);
    late_reset(&r->late, 0);
    r->block_to_analyze = 0;
    r->estimation_region_candidate_size = 0;
    r->estimation_region_identified = 0;
    r->smoothing_constant = 0.0;
    r->late_reverb_start = 0;
    r->late_reverb_end = 0;
}

void rde_reset(ReverbDecayEstimator *r) {
    r->decay = r->default_decay;
    r->smoothing_constant = 0.0;
    reset_decay_estimation(r);
}

static void lazy_init_aec3(ReverbDecayEstimator *r, int td_filter_size) {
    int new_blocks = td_filter_size / RDE_K_FFT_LENGTH_BY_2;
    if (!r->has_early || r->filter_length_blocks != new_blocks) {
        int i;
        r->filter_length_blocks = new_blocks;
        free(r->previous_gains);
        r->n_previous_gains = new_blocks;
        r->previous_gains = (double *)calloc(
            (size_t)(new_blocks > 0 ? new_blocks : 1), sizeof(double));
        for (i = 0; i < new_blocks; ++i) r->previous_gains[i] = 0.0;
        if (r->has_early) early_free(&r->early);
        early_init(&r->early, new_blocks - RDE_K_EARLY_REVERB_MIN_SIZE_BLOCKS);
        r->has_early = 1;
    }
}

/* ───────────────────────────────── legacy ──────────────────────────────── */
void rde_update_legacy(ReverbDecayEstimator *r,
                       const float *partition_energies,
                       double filter_quality, int filter_quality_is_none,
                       int filter_delay_blocks,
                       int usable_linear_filter, int stationary_signal) {
    int late_start, late_end, n, i;
    double new_smoothing;
    double e[256];          /* slice <= n_partitions */
    double log2e[256];
    double x[256];
    double tmp[256];
    double x_mean, y_mean, num, den;
    double slope_per_partition, decay_log2_per_sample, decay_new;
    int hop_eff;

    if (stationary_signal) return;
    if (!r->use_adaptive) return;
    if (!usable_linear_filter) return;
    if (filter_delay_blocks < 0) return;
    if (filter_delay_blocks >
        (r->n_partitions - RDE_K_EARLY_REVERB_MIN_SIZE_BLOCKS - 1))
        return;

    new_smoothing = (filter_quality_is_none ? 0.0 : filter_quality) * 0.2;
    if (new_smoothing > r->smoothing_constant)
        r->smoothing_constant = new_smoothing;
    if (r->smoothing_constant == 0.0) return;

    late_start = filter_delay_blocks + RDE_K_EARLY_REVERB_MIN_SIZE_BLOCKS;
    late_end = r->n_partitions;
    if ((late_end - late_start) < 2) return;

    n = late_end - late_start;
    /* e = partition_energies[late_start:late_end].astype(f64); maximum(e,1e-30) */
    for (i = 0; i < n; ++i) {
        double v = (double)partition_energies[late_start + i];
        if (v < 1e-30) v = 1e-30;
        e[i] = v;
        log2e[i] = log2(e[i]);     /* np.log2 == libm log2 here */
        x[i] = (double)i;          /* np.arange(float64) */
    }
    x_mean = rde_pairwise_sum(x, n) / (double)n;       /* x.mean() */
    y_mean = rde_pairwise_sum(log2e, n) / (double)n;   /* log2_e.mean() */
    for (i = 0; i < n; ++i)
        tmp[i] = (x[i] - x_mean) * (log2e[i] - y_mean);
    num = rde_pairwise_sum(tmp, n);
    for (i = 0; i < n; ++i) {
        double d = x[i] - x_mean;
        tmp[i] = d * d;
    }
    den = rde_pairwise_sum(tmp, n);
    if (den == 0.0) return;

    slope_per_partition = num / den;
    hop_eff = (r->hop_size > 1) ? r->hop_size : 1;   /* max(1, hop_size) */
    decay_log2_per_sample = slope_per_partition / (double)hop_eff;
    decay_new = pow(2.0, decay_log2_per_sample);
    if (0.97 * r->decay > decay_new) decay_new = 0.97 * r->decay; /* max */
    if (decay_new > RDE_K_MAX_DECAY) decay_new = RDE_K_MAX_DECAY; /* min */
    if (decay_new < RDE_K_MIN_DECAY) decay_new = RDE_K_MIN_DECAY; /* max */
    r->decay += r->smoothing_constant * (decay_new - r->decay);
    r->smoothing_constant = 0.0;
}

/* ───────────────────────────── AEC3-strict ─────────────────────────────── */

/* AEC3 cc:47-57 — returns adapting via *adapting, decaying via return. */
static int analyze_block_gain(ReverbDecayEstimator *r, const double *h2_block,
                              int previous_gain_idx, int *adapting_out) {
    double gain = rde_pairwise_sum(h2_block, RDE_K_FFT_LENGTH_BY_2)
                  / (double)RDE_K_FFT_LENGTH_BY_2;   /* np.mean */
    double prev;
    int adapting, decaying;
    if (gain < 1e-32) gain = 1e-32;     /* max(mean, 1e-32) */
    prev = r->previous_gains[previous_gain_idx];
    adapting = (prev > 1.1 * gain) || (prev < 0.9 * gain);
    decaying = gain > r->tail_gain;
    r->previous_gains[previous_gain_idx] = gain;
    *adapting_out = adapting;
    return decaying;
}

/* AEC3 cc:226-264. */
static void analyze_filter(ReverbDecayEstimator *r, const float *td_filter) {
    int idx = r->block_to_analyze;
    int s = idx * RDE_K_FFT_LENGTH_BY_2;
    double h2[RDE_K_FFT_LENGTH_BY_2];
    int adapting, above_noise_floor, k;

    for (k = 0; k < RDE_K_FFT_LENGTH_BY_2; ++k) {
        double v = (double)td_filter[s + k];
        h2[k] = v * v;
    }

    above_noise_floor = analyze_block_gain(r, h2, idx, &adapting);

    r->estimation_region_identified =
        r->estimation_region_identified || adapting || !above_noise_floor;
    if (!r->estimation_region_identified)
        r->estimation_region_candidate_size += 1;

    if (idx <= r->late_reverb_end) {
        if (idx >= r->late_reverb_start) {
            for (k = 0; k < RDE_K_FFT_LENGTH_BY_2; ++k) {
                double h2_log2 = log2(h2[k] + 1e-10);  /* math.log2 */
                late_accumulate(&r->late, h2_log2);
                if (r->has_early)
                    early_accumulate(&r->early, h2_log2, r->smoothing_constant);
            }
        } else {
            for (k = 0; k < RDE_K_FFT_LENGTH_BY_2; ++k) {
                double h2_log2 = log2(h2[k] + 1e-10);
                if (r->has_early)
                    early_accumulate(&r->early, h2_log2, r->smoothing_constant);
            }
        }
    }
}

/* AEC3 cc:162-224. */
static void estimate_decay(ReverbDecayEstimator *r, const float *td_filter,
                           int td_size, int peak_block) {
    int h_size_blocks, size_early_reverb, size_late_reverb;
    double first_reverb_gain, peak_energy;
    int sufficient_reverb_decay, valid_filter;

    r->block_to_analyze = peak_block + RDE_K_EARLY_REVERB_MIN_SIZE_BLOCKS;
    if (r->block_to_analyze > r->filter_length_blocks)
        r->block_to_analyze = r->filter_length_blocks;

    first_reverb_gain = block_energy_average(td_filter, r->block_to_analyze);
    h_size_blocks = td_size >> RDE_K_FFT_LENGTH_BY_2_LOG2;
    r->tail_gain = block_energy_average(td_filter, h_size_blocks - 1);
    peak_energy = block_energy_peak(td_filter, peak_block);
    sufficient_reverb_decay = first_reverb_gain > 4.0 * r->tail_gain;
    valid_filter = (first_reverb_gain > 2.0 * r->tail_gain)
                   && (peak_energy < 100.0);

    size_early_reverb = r->has_early ? early_estimate(&r->early) : 0;
    size_late_reverb = r->estimation_region_candidate_size - size_early_reverb;
    if (size_late_reverb < 0) size_late_reverb = 0;

    if (size_late_reverb >= 5) {
        if (valid_filter && late_estimate_available(&r->late)) {
            double slope = late_estimate(&r->late);
            double decay = pow(2.0, slope * (double)RDE_K_FFT_LENGTH_BY_2);
            if (0.97 * r->decay > decay) decay = 0.97 * r->decay;
            if (decay > RDE_K_MAX_DECAY) decay = RDE_K_MAX_DECAY;
            if (decay < RDE_K_MIN_DECAY) decay = RDE_K_MIN_DECAY;
            r->decay += r->smoothing_constant * (decay - r->decay);
        }
        late_reset(&r->late, size_late_reverb * RDE_K_FFT_LENGTH_BY_2);
        r->late_reverb_start =
            peak_block + RDE_K_EARLY_REVERB_MIN_SIZE_BLOCKS + size_early_reverb;
        r->late_reverb_end =
            r->block_to_analyze + r->estimation_region_candidate_size - 1;
    } else {
        late_reset(&r->late, 0);
        r->late_reverb_start = 0;
        r->late_reverb_end = 0;
    }

    r->estimation_region_identified = !(valid_filter && sufficient_reverb_decay);
    r->estimation_region_candidate_size = 0;
    r->smoothing_constant = 0.0;
    if (r->has_early) early_reset(&r->early);
}

void rde_update_aec3_strict(ReverbDecayEstimator *r,
                            const float *td_filter, int td_size,
                            double filter_quality, int filter_quality_is_none,
                            int filter_delay_blocks,
                            int usable_linear_filter, int stationary_signal) {
    int feasible;
    double new_smoothing;

    if (stationary_signal) return;
    lazy_init_aec3(r, td_size);

    feasible = (filter_delay_blocks
                <= (r->filter_length_blocks
                    - RDE_K_EARLY_REVERB_MIN_SIZE_BLOCKS - 1));
    feasible = feasible && (filter_delay_blocks > 0);
    feasible = feasible && usable_linear_filter;
    feasible = feasible && ((td_size % RDE_K_FFT_LENGTH_BY_2) == 0);

    if (!feasible) {
        reset_decay_estimation(r);
        return;
    }

    if (!r->use_adaptive) return;

    new_smoothing = (filter_quality_is_none ? 0.0 : filter_quality) * 0.2;
    if (new_smoothing > r->smoothing_constant)
        r->smoothing_constant = new_smoothing;
    if (r->smoothing_constant == 0.0) return;

    if (r->block_to_analyze < r->filter_length_blocks) {
        analyze_filter(r, td_filter);
        r->block_to_analyze += 1;
    } else {
        estimate_decay(r, td_filter, td_size, filter_delay_blocks);
    }
}
