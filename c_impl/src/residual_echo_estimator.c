/* residual_echo_estimator.c -- see residual_echo_estimator.h. Float32 in the R^2 paths; the
 * reverb decay is exponentiated in double once at init. */
#include "residual_echo_estimator.h"
#include "aec_scale.h"
#include "utility/simd_aec.h"

#include <math.h>
#include <string.h>

/* Deque helpers (appendleft semantics, index 0 = newest) */
/* Rings of `count` frames with `head` as logical index 0; appendleft moves head back one slot. */
static float *ree_buf_slot(float *buf, int n_bins, int slot) {
    return buf + (size_t)slot * (size_t)n_bins;
}

static void ree_appendleft(float *buf, int n_bins, int *count, int *head,
                           const float *frame) {
    int new_head = (*head - 1 + REE_DELAY_BUF_SIZE) % REE_DELAY_BUF_SIZE;
    *head = new_head;
    if (*count < REE_DELAY_BUF_SIZE) {
        (*count)++;
    }
    memcpy(ree_buf_slot(buf, n_bins, new_head), frame,
           (size_t)n_bins * sizeof(float));
}

/* Frame at logical index i (0 = newest); requires 0 <= i < count. */
static const float *ree_buf_at(const float *buf, int n_bins, int head, int i) {
    int slot = (head + i) % REE_DELAY_BUF_SIZE;
    return buf + (size_t)slot * (size_t)n_bins;
}

void ree_init(ResidualEchoEstimator *r,
              int n_bins, int hop_size, int sample_rate,
              const ReeEchoModelConfig *echo_model,
              float default_gain, float tm_gain,
              int erle_onset_comp_in_dominant,
              double reverb_decay, double reverb_mild_decay_scale,
              int reverb_enabled, float reverb_tail_strength,
              int use_aec_residual_noise_gate,
              int use_stationarity_properties,
              int use_aec_echo_gen_window,
              int nl_r2_enabled, float nl_r2_alpha, float nl_norm_power,
              float nl_r2_cap_multiple,
              float residual_noise_gate_power,
              int noise_floor_hold_hops,
              int use_freq_response, int reverb_use_conservative,
              double reverb_smoothing_base,
              float *x2_noise_floor, int *x2_noise_floor_counter,
              float *reverb_model_storage, float *reverb_tail_storage,
              float *render_history_storage,
              float *delay_render_storage, float *reverb_render_storage,
              float *last_r2_direct_store, float *last_r2_reverb_store,
              float *scratch_store) {
    int k;
    r->n_bins = n_bins;
    /* Render-noise-power per-block growth (x2_noise_floor *= 1.1) rescaled to the live grid; 1.1 is
     * a double literal so the exponentiation keeps full precision. */
    r->noise_floor_growth_per_hop =
        aec_per_block_growth_to_per_hop(1.1, hop_size, sample_rate);
    r->echo_model = *echo_model;
    r->default_gain_early = default_gain;
    r->default_gain_late = default_gain;
    r->tm_gain_early = tm_gain;
    r->tm_gain_late = tm_gain;
    r->erle_onset_comp_in_dominant = erle_onset_comp_in_dominant ? 1 : 0;
    r->reverb_enabled = reverb_enabled ? 1 : 0;
    /* Both dominant_nearend decay branches are resolved here, once. The mild branch multiplies in
     * double before the exponentiation. */
    if (r->reverb_enabled) {
        double d_mild = reverb_decay;
        d_mild *= reverb_mild_decay_scale;
        r->reverb_decay_per_hop =
            aec_per_block_growth_to_per_hop(reverb_decay, hop_size,
                                            sample_rate);
        r->reverb_decay_per_hop_mild =
            aec_per_block_growth_to_per_hop(d_mild, hop_size, sample_rate);
    } else {
        r->reverb_decay_per_hop = 0.0f;
        r->reverb_decay_per_hop_mild = 0.0f;
    }
    r->reverb_tail_strength = reverb_tail_strength;
    r->use_aec_residual_noise_gate = use_aec_residual_noise_gate ? 1 : 0;
    r->use_stationarity_properties = use_stationarity_properties ? 1 : 0;
    r->use_aec_echo_gen_window = use_aec_echo_gen_window ? 1 : 0;
    r->nl_r2_enabled = nl_r2_enabled ? 1 : 0;
    r->nl_r2_alpha = nl_r2_alpha;
    r->nl_norm_power = nl_norm_power;
    r->nl_r2_cap_multiple = nl_r2_cap_multiple;
    r->residual_noise_gate_power = residual_noise_gate_power;
    r->noise_floor_hold_hops = noise_floor_hold_hops;
    r->render_pre_window_size = use_aec_echo_gen_window ? 1 : 0;
    r->render_post_window_size = 1;

    /* NULL storage (a lane that never runs the estimator) is tolerated: only the scalar settings above are used then. */
    r->x2_noise_floor = x2_noise_floor;
    r->x2_noise_floor_counter = x2_noise_floor_counter;
    if (x2_noise_floor) {
        for (k = 0; k < n_bins; ++k) {
            r->x2_noise_floor[k] = echo_model->min_noise_floor_power;
            r->x2_noise_floor_counter[k] = noise_floor_hold_hops;
        }
    }

    reverb_model_init(&r->reverb_model, reverb_model_storage, n_bins);
    r->have_reverb_freq_resp = use_freq_response ? 1 : 0;
    reverb_freq_resp_init(&r->reverb_freq_resp, reverb_tail_storage, n_bins,
                          reverb_use_conservative, reverb_smoothing_base);

    r->render_history = render_history_storage;
    r->render_history_size = r->render_pre_window_size +
                             r->render_post_window_size + 1;
    r->render_history_idx = 0;
    r->render_history_initialised = 0;

    r->delay_render_buf = delay_render_storage;
    r->delay_buf_count = 0;
    r->delay_buf_head = 0;
    r->reverb_render_history = reverb_render_storage;
    r->reverb_buf_count = 0;
    r->reverb_buf_head = 0;

    r->last_r2_direct = last_r2_direct_store;
    r->last_r2_reverb = last_r2_reverb_store;
    r->scratch = scratch_store;
    if (r->last_r2_direct) memset(r->last_r2_direct, 0, (size_t)n_bins * sizeof(float));
    if (r->last_r2_reverb) memset(r->last_r2_reverb, 0, (size_t)n_bins * sizeof(float));
}

void ree_reset(ResidualEchoEstimator *r) {
    int k;
    if (!r->x2_noise_floor) return;   /* no storage: nothing was ever carved */
    for (k = 0; k < r->n_bins; ++k) {
        r->x2_noise_floor[k] = r->echo_model.min_noise_floor_power;
        r->x2_noise_floor_counter[k] = r->noise_floor_hold_hops;
    }
    reverb_model_reset(&r->reverb_model);
    if (r->have_reverb_freq_resp) {
        reverb_freq_resp_reset(&r->reverb_freq_resp);
    }
}

void ree_update_reverb_models(ResidualEchoEstimator *r,
                              const float *frequency_response,
                              int n_partitions,
                              int filter_delay_blocks,
                              double filter_quality,
                              int filter_quality_is_none,
                              int stationary_block) {
    /* Only the frequency-response branch is wired. */
    if (r->have_reverb_freq_resp) {
        reverb_freq_resp_update(&r->reverb_freq_resp, frequency_response,
                                n_partitions, filter_delay_blocks,
                                filter_quality, filter_quality_is_none,
                                stationary_block);
    }
}

/* Stationary render noise floor update. */
static void ree_update_render_noise_power(ResidualEchoEstimator *r,
                                          const float *render_psd) {
    /* Snap the floor down to a lower PSD at once; otherwise count hops and, after the hold, ramp it
     * up by the rescaled growth, never below the model floor. */
    sk_noise_floor_track_f32(r->x2_noise_floor, r->x2_noise_floor_counter,
                             render_psd, r->noise_floor_growth_per_hop,
                             r->echo_model.min_noise_floor_power,
                             r->noise_floor_hold_hops, r->n_bins);
}

/* Returns the squared echo path gain. */
static float ree_echo_path_gain(const ResidualEchoEstimator *r,
                                int transparent_mode,
                                int gain_for_early_reflections) {
    float g;
    if (transparent_mode) {
        g = gain_for_early_reflections ? r->tm_gain_early : r->tm_gain_late;
    } else {
        g = gain_for_early_reflections ? r->default_gain_early
                                       : r->default_gain_late;
    }
    return g * g;
}

/* Per-hop decay (float32): a branch select between the two values cached in ree_init(). */
static float ree_reverb_decay(const ResidualEchoEstimator *r,
                              int dominant_nearend) {
    return dominant_nearend ? r->reverb_decay_per_hop_mild
                            : r->reverb_decay_per_hop;
}

/* Linear-path reverb update. */
static void ree_update_reverb_linear(ResidualEchoEstimator *r,
                                     const float *render_psd,
                                     const float *s2_linear,
                                     int dominant_nearend,
                                     int filter_length_blocks) {
    float decay = ree_reverb_decay(r, dominant_nearend);
    int offset, k;
    const float *delayed_render;
    const float *scaling_src;
    float *scaling_buf = r->scratch;
    if (decay <= 0.0f) {
        return;
    }
    offset = (filter_length_blocks > 0 ? filter_length_blocks : 0) + 1;
    if (offset >= r->reverb_buf_count) {
        return; /* buffer not warm */
    }
    delayed_render = ree_buf_at(r->reverb_render_history, r->n_bins,
                                r->reverb_buf_head, offset);
    if (r->have_reverb_freq_resp) {
        /* Scaling comes from the reverb frequency response. */
        scaling_src = r->reverb_freq_resp.tail_response;
        reverb_model_update(&r->reverb_model, delayed_render, scaling_src,
                            decay);
    } else {
        /* Fallback scaling = s2_linear / max(render_psd, 1e-10), capped at gain_cap. */
        float gain_cap = (r->default_gain_late * r->default_gain_late) * 4.0f;
        for (k = 0; k < r->n_bins; ++k) {
            float den = render_psd[k];
            float lo = 1.0e-10f;
            float d = (den > lo) ? den : lo;
            float s = s2_linear[k] / d;
            if (s > gain_cap) {
                s = gain_cap;
            }
            scaling_buf[k] = s;
        }
        reverb_model_update(&r->reverb_model, delayed_render, scaling_buf,
                            decay);
    }
}

/* Adds the scaled reverb tail to both R2 arrays and records the split. */
static void ree_add_reverb(ResidualEchoEstimator *r, float *r2,
                           float *r2_unbounded, int N) {
    int k;
    for (k = 0; k < N; ++k) {
        float reverb = r->reverb_model.reverb[k] * r->reverb_tail_strength;
        r->last_r2_direct[k] = r2[k];
        r->last_r2_reverb[k] = reverb;
        r2[k] = r2[k] + reverb;
        r2_unbounded[k] = r2_unbounded[k] + reverb;
    }
}

void ree_estimate(ResidualEchoEstimator *r,
                  const float *render_psd,
                  const float *capture_psd,
                  const float *s2_linear,
                  int dominant_nearend,
                  int usable, int saturated, int transparent_mode,
                  const float *erle, const float *erle_unbounded,
                  int filter_delay_blocks,
                  int filter_length_blocks,
                  int force_nonlinear_path,
                  float *r2, float *r2_unbounded) {
    const int N = r->n_bins;
    int k;

    /* Update the stationary render noise floor. */
    ree_update_render_noise_power(r, render_psd);

    /* Push the current render to both deques every hop (also on linear hops) so the delay buffer
     * stays current for linear-to-nonlinear transitions. */
    ree_appendleft(r->reverb_render_history, N, &r->reverb_buf_count,
                   &r->reverb_buf_head, render_psd);
    ree_appendleft(r->delay_render_buf, N, &r->delay_buf_count,
                   &r->delay_buf_head, render_psd);

    if (force_nonlinear_path) {
        usable = 0;
    }

    /* Clear the diagnostics unless one of the full-array writers below will run. */
    if (!(usable ||
          (r->echo_model.model_reverb_in_nonlinear_mode && !transparent_mode))) {
        memset(r->last_r2_reverb, 0, (size_t)N * sizeof(float));
        memset(r->last_r2_direct, 0, (size_t)N * sizeof(float));
    }

    if (usable) {
        if (saturated) {
            for (k = 0; k < N; ++k) {
                r2[k] = capture_psd[k];
                r2_unbounded[k] = capture_psd[k];
            }
        } else {
            /* R^2 = s2_linear / max(erle, 1e-30). */
            for (k = 0; k < N; ++k) {
                float eb = erle[k];
                float eu = erle_unbounded[k];
                float db = (eb > 1.0e-30f) ? eb : 1.0e-30f;
                float du = (eu > 1.0e-30f) ? eu : 1.0e-30f;
                r2[k] = s2_linear[k] / db;
                r2_unbounded[k] = s2_linear[k] / du;
            }
        }
        /* Update the reverb and add it. */
        ree_update_reverb_linear(r, render_psd, s2_linear, dominant_nearend,
                                 filter_length_blocks);
        ree_add_reverb(r, r2, r2_unbounded, N);
    } else {
        if (saturated) {
            for (k = 0; k < N; ++k) {
                r2[k] = capture_psd[k];
                r2_unbounded[k] = capture_psd[k];
            }
        } else {
            float echo_path_gain = ree_echo_path_gain(r, transparent_mode, 1);
            /* EchoGeneratingPower window walk; x2 lives in the per-instance scratch. */
            float *x2 = r->scratch;
            if (r->use_aec_echo_gen_window) {
                int delay = (filter_delay_blocks > 0 ? filter_delay_blocks : 0);
                int pre = r->render_pre_window_size;
                int post = r->render_post_window_size;
                int idx_start, idx_stop, i;
                idx_start = delay - pre;
                if (idx_start < 0) idx_start = 0;
                idx_stop = delay + post;
                if (idx_stop > r->delay_buf_count - 1) {
                    idx_stop = r->delay_buf_count - 1;
                }
                /* x2 = element-wise maximum over frames [idx_start, idx_stop]. */
                {
                    const float *first = ree_buf_at(r->delay_render_buf, N,
                                                    r->delay_buf_head,
                                                    idx_start);
                    memcpy(x2, first, (size_t)N * sizeof(float));
                }
                for (i = idx_start + 1; i <= idx_stop; ++i) {
                    const float *s = ree_buf_at(r->delay_render_buf, N,
                                                r->delay_buf_head, i);
                    for (k = 0; k < N; ++k) {
                        if (s[k] > x2[k]) {
                            x2[k] = s[k];
                        }
                    }
                }
            } else {
                /* Ring render history (used only when use_aec_echo_gen_window is off). */
                if (!r->render_history_initialised) {
                    int row;
                    for (row = 0; row < r->render_history_size; ++row) {
                        memcpy(ree_buf_slot(r->render_history, N, row),
                               render_psd, (size_t)N * sizeof(float));
                    }
                    r->render_history_initialised = 1;
                } else {
                    memcpy(ree_buf_slot(r->render_history, N,
                                        r->render_history_idx),
                           render_psd, (size_t)N * sizeof(float));
                    r->render_history_idx = (r->render_history_idx + 1) %
                                            r->render_history_size;
                }
                /* x2 = maximum over the history rows. */
                memcpy(x2, ree_buf_slot(r->render_history, N, 0),
                       (size_t)N * sizeof(float));
                {
                    int row;
                    for (row = 1; row < r->render_history_size; ++row) {
                        const float *s = ree_buf_slot(r->render_history, N, row);
                        for (k = 0; k < N; ++k) {
                            if (s[k] > x2[k]) {
                                x2[k] = s[k];
                            }
                        }
                    }
                }
            }
            /* The noise gate is skipped when use_stationarity_properties is set (the shipped
             * setting). */
            if (!r->use_stationarity_properties) {
                /* Noise gate. */
                float ng = r->use_aec_residual_noise_gate
                           ? r->residual_noise_gate_power
                           : r->echo_model.noise_gate_power;
                float slope = r->echo_model.noise_gate_slope;
                for (k = 0; k < N; ++k) {
                    if (ng > x2[k]) {
                        /* x2 = max(0, x2 - slope * (ng - x2)) where x2 < ng. */
                        float v = x2[k] - slope * (ng - x2[k]);
                        x2[k] = (v > 0.0f) ? v : 0.0f;
                    }
                }
            }
            /* Subtract the stationary noise floor, floor at 0. */
            {
                float sgs = r->echo_model.stationary_gate_slope;
                for (k = 0; k < N; ++k) {
                    float v = x2[k] - sgs * r->x2_noise_floor[k];
                    x2[k] = (v > 0.0f) ? v : 0.0f;
                }
            }
            for (k = 0; k < N; ++k) {
                /* r2 = x2 * echo_path_gain. */
                float v = x2[k] * echo_path_gain;
                r2[k] = v;
                r2_unbounded[k] = v;
            }
            if (r->nl_r2_enabled && r->nl_r2_alpha > 0.0f) {
                /* Non-linear term alpha * x2^2 / norm. */
                float alpha = r->nl_r2_alpha;
                float norm = r->nl_norm_power;
                float cap_mult = r->nl_r2_cap_multiple;
                for (k = 0; k < N; ++k) {
                    float x2sq = x2[k] * x2[k];
                    float r2_nl = alpha * x2sq / norm;
                    if (cap_mult > 0.0f) {
                        /* Cap the non-linear term at a multiple of the direct term. */
                        float cap = r2[k] * cap_mult;
                        r2_nl = (r2_nl < cap) ? r2_nl : cap;
                    }
                    r2[k] = r2[k] + r2_nl;
                    r2_unbounded[k] = r2_unbounded[k] + r2_nl;
                }
            }
        }
        /* Update the reverb (non-linear mode) and add it. */
        if (r->echo_model.model_reverb_in_nonlinear_mode && !transparent_mode) {
            float ep_late = ree_echo_path_gain(r, transparent_mode, 0);
            float decay = ree_reverb_decay(r, dominant_nearend);
            int nl_offset = (filter_delay_blocks > 0 ? filter_delay_blocks : 0)
                            + 1;
            if (nl_offset < r->reverb_buf_count) {
                const float *delayed_render = ree_buf_at(
                    r->reverb_render_history, N, r->reverb_buf_head, nl_offset);
                reverb_model_update_no_freq_shaping(&r->reverb_model,
                                                    delayed_render,
                                                    ep_late,
                                                    decay);
            }
            /* The reverb is added even when no delayed render is available. */
            ree_add_reverb(r, r2, r2_unbounded, N);
        }
    }
}
