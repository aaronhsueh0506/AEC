/* subband_erle.c — C port of python/modules/state/subband_erle.py.
 *
 * Per-bin ERLE estimator with onset compensation (AEC3 SubbandErleEstimator,
 * single capture channel). See subband_erle.h for the parity contract.
 *
 * All per-bin ERLE arithmetic is float32 (numpy weak promotion: Python-float
 * scalars cast to f32 before combining with the f32 arrays). The lone float64
 * quantity is _x2_band_energy_threshold; the low-render test
 *   low |= x2(f32) < thr(f64)
 * is a numpy mixed-dtype comparison -> x2 promoted to f64, compared in double.
 * Built with -ffp-contract=off so per-op float roundings are not fused.
 */
#include "subband_erle.h"

#include "aec3_scale.h"   /* aec3_per_bin_psd_threshold */

#include <string.h>

/* _set_max_erle_bands: arr[:half] = max_erle_l ; arr[half:] = max_erle_h
 * with half = (n_bins - 1) // 2  (kFftLengthBy2 // 2 analogue). */
static void set_max_erle_bands(float *arr, int n_bins,
                               float max_erle_l, float max_erle_h) {
    int half = (n_bins - 1) / 2;   /* Python floor div on non-negative */
    int k;
    for (k = 0; k < half; ++k) arr[k] = max_erle_l;
    for (k = half; k < n_bins; ++k) arr[k] = max_erle_h;
}

void subband_erle_init(SubbandErle *s, int n_bins,
                       float min_erle, float max_erle_l, float max_erle_h,
                       int use_onset_detection, int use_min_erle_during_onsets,
                       int hop_size, int e2y2_gate_enabled,
                       float e2y2_gate_threshold,
                       float *max_erle_st, float *erle_st, float *erle_oc_st,
                       float *erle_unb_st, float *erle_during_st,
                       unsigned char *coming_onset_st, int32_t *hold_st,
                       float *y2_acc_st, float *e2_acc_st,
                       unsigned char *low_render_st) {
    s->n_bins = n_bins;
    s->min_erle = min_erle;
    s->use_onset_detection = use_onset_detection ? 1 : 0;
    s->use_min_erle_during_onsets = use_min_erle_during_onsets ? 1 : 0;
    s->alpha_up = 0.05;
    s->alpha_down = 0.1;
    s->onset_release_decay = 0.97;
    s->e2y2_gate_enabled = e2y2_gate_enabled ? 1 : 0;
    s->e2y2_gate_threshold = e2y2_gate_threshold;
    /* _x2_band_energy_threshold = per_bin_psd_threshold(44015068.0, hop_size)
     * (ref_hop=160), a float64 scalar. */
    s->x2_band_energy_threshold =
        aec3_per_bin_psd_threshold(SE_X2_BAND_ENERGY_THRESHOLD, hop_size, 160);

    s->max_erle = max_erle_st;
    s->erle = erle_st;
    s->erle_onset_compensated = erle_oc_st;
    s->erle_unbounded = erle_unb_st;
    s->erle_during_onsets = erle_during_st;
    s->coming_onset = coming_onset_st;
    s->hold_counters = hold_st;
    s->y2_acc = y2_acc_st;
    s->e2_acc = e2_acc_st;
    s->low_render_energy = low_render_st;

    set_max_erle_bands(s->max_erle, n_bins, max_erle_l, max_erle_h);
    subband_erle_reset(s);
}

void subband_erle_reset(SubbandErle *s) {
    int k;
    for (k = 0; k < s->n_bins; ++k) {
        s->erle[k] = s->min_erle;
        s->erle_onset_compensated[k] = s->min_erle;
        s->erle_unbounded[k] = s->min_erle;
        s->erle_during_onsets[k] = s->min_erle;
        s->coming_onset[k] = 1;
        s->hold_counters[k] = 0;
    }
    /* _reset_accumulated_spectra */
    for (k = 0; k < s->n_bins; ++k) {
        s->y2_acc[k] = 0.0f;
        s->e2_acc[k] = 0.0f;
        s->low_render_energy[k] = 0;
    }
    s->num_points = 0;
}

/* _smoothed_update(prev, new_val, low_render, min_v, max_v) -> float (f32).
 *
 * numpy scalar promotion: prev/new_val are np.float32 scalars; alpha/min_v/
 * max_v are Python floats (f64). So:
 *   diff = new_val - prev          -> float32 (two f32 scalars)
 *   v    = prev + alpha * diff     -> float64 (f32_scalar * pyfloat -> f64)
 *   v    = np.clip(v, min_v, max_v)-> float64
 *   store to f32 array             -> (float)v
 * alpha (0.05/0.1/0.0) is exactly representable so passing it as float is fine
 * inside the f64 multiply; min_v/max_v are passed as double for the clip. */
static float smoothed_update(const SubbandErle *s, float prev, float new_val,
                             int low_render, double min_v, double max_v) {
    double alpha = s->alpha_up;
    float  diff = new_val - prev;                  /* f32 subtraction */
    double v;
    if (new_val < prev) {
        alpha = low_render ? 0.0 : s->alpha_down;
    }
    v = (double)prev + alpha * (double)diff;       /* f64 */
    /* np.clip(v, min_v, max_v): clamp low then high (f64). */
    if (v < min_v) v = min_v;
    if (v > max_v) v = max_v;
    return (float)v;                               /* store to f32 array */
}

/* _update_accumulated_spectra */
static void update_accumulated_spectra(SubbandErle *s,
                                       const float *x2, const float *y2,
                                       const float *e2, int converged_filter) {
    int k;
    if (!converged_filter) return;
    if (s->num_points == SE_POINTS_TO_ACCUMULATE) {
        s->num_points = 0;
        for (k = 0; k < s->n_bins; ++k) {
            s->y2_acc[k] = 0.0f;
            s->e2_acc[k] = 0.0f;
            s->low_render_energy[k] = 0;
        }
    }
    for (k = 0; k < s->n_bins; ++k) {
        s->y2_acc[k] = s->y2_acc[k] + y2[k];     /* f32 += f32 */
        s->e2_acc[k] = s->e2_acc[k] + e2[k];     /* f32 += f32 */
        /* low_render |= x2 < thr : x2(f32) promoted to f64, compared in double */
        if ((double)x2[k] < s->x2_band_energy_threshold) {
            s->low_render_energy[k] = 1;
        }
    }
    s->num_points += 1;
}

/* _update_bands */
static void update_bands(SubbandErle *s, int converged_filter,
                         const unsigned char *coh_gate_mask) {
    int k;
    int n = s->n_bins;
    if (!converged_filter) return;
    if (s->num_points != SE_POINTS_TO_ACCUMULATE) return;

    /* Per-bin compute only for interior bins k = 1 .. n-2 (excludes endpoints).
     * We fold new_erle + is_erle_updated into a per-bin loop; for masked-out
     * bins new_erle is 0.0 (matches np.where else-branch) and is_erle_updated
     * is 0, so they never touch state. */
    for (k = 1; k <= n - 2; ++k) {
        float new_erle;
        int is_updated;
        float e2a = s->e2_acc[k];
        int low;
        /* mask = e2_acc > 0.0 ; new = where(mask, y2_acc/max(e2_acc,1e-30), 0) */
        if (e2a > 0.0f) {
            float denom = e2a > 1e-30f ? e2a : 1e-30f; /* np.maximum(e2_acc,1e-30) */
            new_erle = s->y2_acc[k] / denom;            /* f32 */
            is_updated = 1;
        } else {
            new_erle = 0.0f;
            is_updated = 0;
        }

        /* E2/Y2 gate: is_updated &= (e2_acc/max(y2_acc,1e-30) <= thr). */
        if (s->e2y2_gate_enabled && is_updated) {
            float ydenom = s->y2_acc[k] > 1e-30f ? s->y2_acc[k] : 1e-30f;
            float e2y2 = e2a / ydenom;                  /* f32 */
            if (!(e2y2 <= s->e2y2_gate_threshold)) is_updated = 0;
        }
        /* Coherence gate: is_updated &= coh_gate_mask[k]. */
        if (coh_gate_mask != NULL && is_updated) {
            if (!coh_gate_mask[k]) is_updated = 0;
        }

        low = s->low_render_energy[k] ? 1 : 0;

        /* Onset detection transition + hold. */
        if (s->use_onset_detection) {
            if (is_updated && !s->low_render_energy[k]) {
                if (s->coming_onset[k]) {
                    s->coming_onset[k] = 0;
                    if (!s->use_min_erle_during_onsets) {
                        /* v = during + alpha*(new_erle - during): the (new-during)
                         * subtraction is f32; alpha(pyfloat)*diff -> f64; clip f64;
                         * store (float). alpha = 0.3 if new<during else 0.15. */
                        float  prev_d = s->erle_during_onsets[k];
                        double alpha = (new_erle < prev_d) ? 0.3 : 0.15;
                        float  diff = new_erle - prev_d;          /* f32 */
                        double v = (double)prev_d + alpha * (double)diff; /* f64 */
                        double lo = (double)s->min_erle;
                        double hi = (double)s->max_erle[k];
                        if (v < lo) v = lo;            /* np.clip */
                        if (v > hi) v = hi;
                        s->erle_during_onsets[k] = (float)v;
                    }
                }
                s->hold_counters[k] = SE_BLOCKS_FOR_ONSET_DETECTION;
            }
        }

        /* Per-bin smoothed ERLE update (only when is_erle_updated). */
        if (is_updated) {
            s->erle[k] = smoothed_update(s, s->erle[k], new_erle, low,
                                         (double)s->min_erle,
                                         (double)s->max_erle[k]);
            if (s->use_onset_detection) {
                s->erle_onset_compensated[k] =
                    smoothed_update(s, s->erle_onset_compensated[k], new_erle,
                                    low, (double)s->min_erle,
                                    (double)s->max_erle[k]);
            }
            s->erle_unbounded[k] =
                smoothed_update(s, s->erle_unbounded[k], new_erle, low,
                                (double)s->min_erle, SE_UNBOUNDED_ERLE_MAX);
        }
    }
}

/* _decrease_erle_per_band_for_low_render_signals */
static void decrease_erle_for_low_render(SubbandErle *s) {
    int k;
    int n = s->n_bins;
    for (k = 1; k <= n - 2; ++k) {
        s->hold_counters[k] -= 1;
        if (s->hold_counters[k] <=
            (SE_BLOCKS_FOR_ONSET_DETECTION - SE_BLOCKS_TO_HOLD_ERLE)) {
            if (s->erle_onset_compensated[k] > s->erle_during_onsets[k]) {
                /* max(during, 0.97 * onset_compensated): 0.97(pyfloat) *
                 * onset_compensated(f32 scalar) -> f64; builtin max picks the
                 * f64 product or the f32 'during' (promoted in the compare);
                 * result stored to f32 array -> (float). */
                double decayed = s->onset_release_decay *
                                 (double)s->erle_onset_compensated[k];
                double during = (double)s->erle_during_onsets[k];
                s->erle_onset_compensated[k] =
                    (float)(decayed > during ? decayed : during);
            }
            if (s->hold_counters[k] <= 0) {
                s->coming_onset[k] = 1;
                s->hold_counters[k] = 0;
            }
        }
    }
}

void subband_erle_update(SubbandErle *s,
                         const float *x2, const float *y2, const float *e2,
                         int converged_filter,
                         const unsigned char *coh_gate_mask) {
    int n = s->n_bins;
    update_accumulated_spectra(s, x2, y2, e2, converged_filter);
    update_bands(s, converged_filter, coh_gate_mask);
    if (s->use_onset_detection) {
        decrease_erle_for_low_render(s);
    }
    /* Mirror first / last bin (AEC3 cc:100-109). */
    s->erle[0] = s->erle[1];
    s->erle[n - 1] = s->erle[n - 2];
    s->erle_onset_compensated[0] = s->erle_onset_compensated[1];
    s->erle_onset_compensated[n - 1] = s->erle_onset_compensated[n - 2];
    s->erle_unbounded[0] = s->erle_unbounded[1];
    s->erle_unbounded[n - 1] = s->erle_unbounded[n - 2];
}

const float *subband_erle_erle(const SubbandErle *s, int onset_compensated) {
    if (onset_compensated && s->use_onset_detection) {
        return s->erle_onset_compensated;
    }
    return s->erle;
}

const float *subband_erle_erle_unbounded(const SubbandErle *s) {
    return s->erle_unbounded;
}

const float *subband_erle_erle_during_onsets(const SubbandErle *s) {
    return s->erle_during_onsets;
}
