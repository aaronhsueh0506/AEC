/* subband_erle.c -- see subband_erle.h. Float32 throughout. */
#include "subband_erle.h"
#include "utility/simd_aec.h"

#include "aec_scale.h"

#include <string.h>

/* Low half of the bins gets max_erle_l, the rest max_erle_h. */
static void set_max_erle_bands(float *arr, int n_bins,
                               float max_erle_l, float max_erle_h) {
    int half = (n_bins - 1) / 2;
    int k;
    for (k = 0; k < half; ++k) arr[k] = max_erle_l;
    for (k = half; k < n_bins; ++k) arr[k] = max_erle_h;
}

void subband_erle_init(SubbandErle *s, int n_bins,
                       float min_erle, float max_erle_l, float max_erle_h,
                       int use_onset_detection, int use_min_erle_during_onsets,
                       int hop_size, int sample_rate,
                       float *max_erle_st, float *erle_st, float *erle_oc_st,
                       float *erle_unb_st, float *erle_during_st,
                       unsigned char *coming_onset_st, int32_t *hold_st,
                       float *y2_acc_st, float *e2_acc_st,
                       unsigned char *low_render_st) {
    s->n_bins = n_bins;
    s->min_erle = min_erle;
    s->use_onset_detection = use_onset_detection ? 1 : 0;
    s->use_min_erle_during_onsets = use_min_erle_during_onsets ? 1 : 0;
    /* The EMA constants are per 4 ms block. alpha_up/alpha_down fire once per 6-hop window and
     * onset_release_decay every hop; the factor of 6 cancels against the reference period, so plain
     * hop_size is used. */
    s->alpha_up = aec_per_block_ema_alpha_to_per_hop(
        0.05f, hop_size, sample_rate);
    s->alpha_down = aec_per_block_ema_alpha_to_per_hop(
        0.1f, hop_size, sample_rate);
    s->onset_release_decay = 1.0f - aec_per_block_ema_alpha_to_per_hop(
        1.0f - 0.97f, hop_size, sample_rate);
    /* ERLE hold 400 ms, onset detection 1 s, scaled to the live grid. */
    s->blocks_to_hold_erle = aec_ms_to_hops(400.0f, hop_size, sample_rate);
    s->blocks_for_onset_detection = aec_ms_to_hops(1000.0f, hop_size, sample_rate);
    /* Threshold scaled to the live hop (reference hop 160). */
    s->x2_band_energy_threshold =
        aec_per_bin_psd_threshold(SE_X2_BAND_ENERGY_THRESHOLD, hop_size, 160);

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
    /* Reset accumulated spectra. */
    for (k = 0; k < s->n_bins; ++k) {
        s->y2_acc[k] = 0.0f;
        s->e2_acc[k] = 0.0f;
        s->low_render_energy[k] = 0;
    }
    s->num_points = 0;
}

/* Smoothed update of prev toward new_val with alpha_up (rising) or alpha_down (falling, 0 when
 * low render), clamped to [min_v, max_v]. */
static float smoothed_update(const SubbandErle *s, float prev, float new_val,
                             int low_render, float min_v, float max_v) {
    float alpha = s->alpha_up;
    float diff = new_val - prev;
    float v;
    if (new_val < prev) {
        alpha = low_render ? 0.0f : s->alpha_down;
    }
    v = prev + alpha * diff;
    if (v < min_v) v = min_v;
    if (v > max_v) v = max_v;
    return v;
}

/* Accumulate y2/e2 and the low-render flag. */
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
    sk_erle_accumulate_f32(s->y2_acc, s->e2_acc, s->low_render_energy,
                            x2, y2, e2, s->x2_band_energy_threshold, s->n_bins);
    s->num_points += 1;
}

/* Update the per-bin estimates at the end of each accumulation window. */
static void update_bands(SubbandErle *s, int converged_filter,
                         const unsigned char *coh_gate_mask) {
    int k;
    int n = s->n_bins;
    if (!converged_filter) return;
    if (s->num_points != SE_POINTS_TO_ACCUMULATE) return;

    /* Interior bins k = 1 .. n-2 only; masked-out bins never touch state. */
    for (k = 1; k <= n - 2; ++k) {
        float new_erle;
        int is_updated;
        float e2a = s->e2_acc[k];
        int low;
        if (e2a > 0.0f) {
            float denom = e2a > 1e-30f ? e2a : 1e-30f;
            new_erle = s->y2_acc[k] / denom;
            is_updated = 1;
        } else {
            new_erle = 0.0f;
            is_updated = 0;
        }

        /* Coherence gate. */
        if (coh_gate_mask != NULL && is_updated) {
            if (!coh_gate_mask[k]) is_updated = 0;
        }

        low = s->low_render_energy[k] ? 1 : 0;

        /* Onset detection transition and hold. */
        if (s->use_onset_detection) {
            if (is_updated && !s->low_render_energy[k]) {
                if (s->coming_onset[k]) {
                    s->coming_onset[k] = 0;
                    if (!s->use_min_erle_during_onsets) {
                        /* Smooth erle_during_onsets toward new_erle (0.3 falling, 0.15 rising). */
                        float prev_d = s->erle_during_onsets[k];
                        float alpha = (new_erle < prev_d) ? 0.3f : 0.15f;
                        float diff = new_erle - prev_d;
                        float v = prev_d + alpha * diff;
                        float lo = s->min_erle;
                        float hi = s->max_erle[k];
                        if (v < lo) v = lo;
                        if (v > hi) v = hi;
                        s->erle_during_onsets[k] = v;
                    }
                }
                s->hold_counters[k] = s->blocks_for_onset_detection;
            }
        }

        /* Smoothed ERLE update, only for updated bins. */
        if (is_updated) {
            s->erle[k] = smoothed_update(s, s->erle[k], new_erle, low,
                                         s->min_erle, s->max_erle[k]);
            if (s->use_onset_detection) {
                s->erle_onset_compensated[k] =
                    smoothed_update(s, s->erle_onset_compensated[k], new_erle,
                                    low, s->min_erle, s->max_erle[k]);
            }
            s->erle_unbounded[k] =
                smoothed_update(s, s->erle_unbounded[k], new_erle, low,
                                s->min_erle, SE_UNBOUNDED_ERLE_MAX);
        }
    }
}

/* Release the onset-compensated ERLE for low-render signals. */
static void decrease_erle_for_low_render(SubbandErle *s) {
    sk_erle_release_f32(s->erle_onset_compensated + 1,
                         s->erle_during_onsets + 1, s->hold_counters + 1,
                         s->coming_onset + 1,
                         s->blocks_for_onset_detection - s->blocks_to_hold_erle,
                         s->onset_release_decay, s->n_bins - 2);
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
    /* Mirror first and last bin. */
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

