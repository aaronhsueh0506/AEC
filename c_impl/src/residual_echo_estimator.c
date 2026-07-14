/* residual_echo_estimator.c — C port of residual_echo_estimator.py.
 * All-float32 arithmetic in the R² paths (every input is f32 and every scalar
 * multiplier is value-cast to f32 inside the numpy array op). powf() for the
 * per-hop reverb decay runs in float32 (converted from the former double
 * pow(); drift accepted). Built with -ffp-contract=off. */
#include "residual_echo_estimator.h"

#include <math.h>
#include <string.h>

/* ── deque helpers (appendleft semantics, index 0 = most recent) ──────────── */
/* The two render deques are rings of `count` frames with `head` = logical 0.
 * appendleft: head moves back one slot (mod 16), count grows up to 16. */
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

/* logical index i (0=most recent). Caller must guarantee 0 <= i < count. */
static const float *ree_buf_at(const float *buf, int n_bins, int head, int i) {
    int slot = (head + i) % REE_DELAY_BUF_SIZE;
    return buf + (size_t)slot * (size_t)n_bins;
}

void ree_init(ResidualEchoEstimator *r,
              int n_bins, int hop_size,
              const ReeEchoModelConfig *echo_model,
              float default_gain, float tm_gain,
              int erle_onset_comp_in_dominant,
              float reverb_decay, float reverb_mild_decay_scale,
              int reverb_enabled, float reverb_tail_strength,
              int use_aec3_residual_noise_gate,
              int use_stationarity_properties,
              int use_aec3_echo_gen_window,
              int nl_r2_enabled, float nl_r2_alpha, float nl_norm_power,
              float residual_noise_gate_power,
              int noise_floor_hold_hops,
              int use_freq_response, int reverb_use_conservative,
              float reverb_smoothing_base,
              float *x2_noise_floor, int *x2_noise_floor_counter,
              float *reverb_model_storage, float *reverb_tail_storage,
              float *render_history_storage,
              float *delay_render_storage, float *reverb_render_storage,
              float *last_r2_direct_store, float *last_r2_reverb_store,
              float *scratch_store) {
    int k;
    r->n_bins = n_bins;
    r->hop_size = hop_size;
    r->echo_model = *echo_model;
    r->default_gain_early = default_gain;
    r->default_gain_late = default_gain;
    r->tm_gain_early = tm_gain;
    r->tm_gain_late = tm_gain;
    r->erle_onset_comp_in_dominant = erle_onset_comp_in_dominant ? 1 : 0;
    r->reverb_decay = reverb_decay;
    r->reverb_mild_decay_scale = reverb_mild_decay_scale;
    r->reverb_enabled = reverb_enabled ? 1 : 0;
    r->reverb_tail_strength = reverb_tail_strength;
    r->use_aec3_residual_noise_gate = use_aec3_residual_noise_gate ? 1 : 0;
    r->use_stationarity_properties = use_stationarity_properties ? 1 : 0;
    r->use_aec3_echo_gen_window = use_aec3_echo_gen_window ? 1 : 0;
    r->nl_r2_enabled = nl_r2_enabled ? 1 : 0;
    r->nl_r2_alpha = nl_r2_alpha;
    r->nl_norm_power = nl_norm_power;
    r->residual_noise_gate_power = residual_noise_gate_power;
    r->noise_floor_hold_hops = noise_floor_hold_hops;
    r->render_pre_window_size = use_aec3_echo_gen_window ? 1 : 0;
    r->render_post_window_size = 1;

    r->x2_noise_floor = x2_noise_floor;
    r->x2_noise_floor_counter = x2_noise_floor_counter;
    for (k = 0; k < n_bins; ++k) {
        r->x2_noise_floor[k] = echo_model->min_noise_floor_power;
        r->x2_noise_floor_counter[k] = noise_floor_hold_hops;
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
    memset(r->last_r2_direct, 0, (size_t)n_bins * sizeof(float));
    memset(r->last_r2_reverb, 0, (size_t)n_bins * sizeof(float));
}

void ree_reset(ResidualEchoEstimator *r) {
    int k;
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
                              float filter_quality,
                              int filter_quality_is_none,
                              int stationary_block) {
    /* Only the freq-resp branch is wired (adaptive decay estimator not bound). */
    if (r->have_reverb_freq_resp) {
        reverb_freq_resp_update(&r->reverb_freq_resp, frequency_response,
                                n_partitions, filter_delay_blocks,
                                filter_quality, filter_quality_is_none,
                                stationary_block);
    }
}

/* UpdateRenderNoisePower (cc:325-359). */
static void ree_update_render_noise_power(ResidualEchoEstimator *r,
                                          const float *render_psd) {
    const int N = r->n_bins;
    const int hold = r->noise_floor_hold_hops;
    const float min_floor = r->echo_model.min_noise_floor_power;
    int k;
    for (k = 0; k < N; ++k) {
        if (render_psd[k] < r->x2_noise_floor[k]) {
            /* decrease rapidly: snap floor down (astype f32 no-op). */
            r->x2_noise_floor[k] = render_psd[k];
            r->x2_noise_floor_counter[k] = 0;
        } else {
            /* not_down */
            if (r->x2_noise_floor_counter[k] >= hold) {
                /* ramp up 10% (f32 array * pyfloat 1.1 = f32), max vs min. */
                float ramp = r->x2_noise_floor[k] * 1.1f;
                r->x2_noise_floor[k] = (ramp > min_floor) ? ramp : min_floor;
            } else {
                r->x2_noise_floor_counter[k] += 1;
            }
        }
    }
}

/* GetEchoPathGain → returns g*g (python float). */
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

/* _reverb_decay(dominant_nearend): static config path (decay estimator not
 * bound). Returns the per-hop decay (float32). */
static float ree_reverb_decay(const ResidualEchoEstimator *r,
                              int dominant_nearend) {
    float d;
    if (!r->reverb_enabled) {
        return 0.0f;
    }
    d = r->reverb_decay;
    if (dominant_nearend) {
        d *= r->reverb_mild_decay_scale;
    }
    if (r->hop_size != 64) {
        d = powf(d, (float)r->hop_size / 64.0f);
    }
    return d;
}

float ree_reverb_decay_value(const ResidualEchoEstimator *r,
                             int dominant_nearend) {
    return ree_reverb_decay(r, dominant_nearend);
}

/* _update_reverb_linear (cc:390-392). */
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
        /* scaling = tail_response.astype(f32) — tail_response already f32. */
        scaling_src = r->reverb_freq_resp.tail_response;
        reverb_model_update(&r->reverb_model, delayed_render, scaling_src,
                            decay);
    } else {
        /* fallback: scaling = s2_linear / max(render_psd, 1e-10), clip gain_cap */
        float gain_cap = (r->default_gain_late * r->default_gain_late) * 4.0f;
        for (k = 0; k < r->n_bins; ++k) {
            float den = render_psd[k];
            float lo = 1.0e-10f;
            float d = (den > lo) ? den : lo; /* np.maximum array vs pyfloat (f32) */
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

    /* Step 1: update stationary render noise floor. */
    ree_update_render_noise_power(r, render_psd);

    /* push current render to both history deques unconditionally (appendleft).
     * E10 fix: delay_render_buf was pushed only inside the nonlinear path, so
     * linear hops left it stale → wrong EchoGeneratingPower on linear→nonlinear
     * transitions.  Push at the top so every hop keeps both deques current. */
    ree_appendleft(r->reverb_render_history, N, &r->reverb_buf_count,
                   &r->reverb_buf_head, render_psd);
    ree_appendleft(r->delay_render_buf, N, &r->delay_buf_count,
                   &r->delay_buf_head, render_psd);

    if (force_nonlinear_path) {
        usable = 0;
    }

    /* diag reset */
    memset(r->last_r2_reverb, 0, (size_t)N * sizeof(float));
    memset(r->last_r2_direct, 0, (size_t)N * sizeof(float));

    if (usable) {
        if (saturated) {
            for (k = 0; k < N; ++k) {
                r2[k] = capture_psd[k];
                r2_unbounded[k] = capture_psd[k];
            }
        } else {
            /* R² = s2_linear / max(erle, 1e-30) (all f32). */
            for (k = 0; k < N; ++k) {
                float eb = erle[k];
                float eu = erle_unbounded[k];
                float db = (eb > 1.0e-30f) ? eb : 1.0e-30f;
                float du = (eu > 1.0e-30f) ? eu : 1.0e-30f;
                r2[k] = s2_linear[k] / db;
                r2_unbounded[k] = s2_linear[k] / du;
            }
        }
        /* UpdateReverb(kLinear) + AddReverb. */
        ree_update_reverb_linear(r, render_psd, s2_linear, dominant_nearend,
                                 filter_length_blocks);
        for (k = 0; k < N; ++k) {
            /* reverb = reverb_model.reverb * reverb_tail_strength
             * (f32 array * pyfloat → f32). */
            float reverb = r->reverb_model.reverb[k] *
                           r->reverb_tail_strength;
            r->last_r2_direct[k] = r2[k];
            r->last_r2_reverb[k] = reverb;
            r2[k] = r2[k] + reverb;
            r2_unbounded[k] = r2_unbounded[k] + reverb;
        }
    } else {
        float echo_path_gain = ree_echo_path_gain(r, transparent_mode, 1);
        if (saturated) {
            for (k = 0; k < N; ++k) {
                r2[k] = capture_psd[k];
                r2_unbounded[k] = capture_psd[k];
            }
        } else {
            /* EchoGeneratingPower window walk. x2 lives in per-instance scratch. */
            float *x2 = r->scratch;
            if (r->use_aec3_echo_gen_window) {
                int delay = (filter_delay_blocks > 0 ? filter_delay_blocks : 0);
                int pre = r->render_pre_window_size;   /* 1 */
                int post = r->render_post_window_size; /* 1 */
                int idx_start, idx_stop, i;
                /* delay_render_buf already pushed at the top of ree_estimate (E10). */
                idx_start = delay - pre;
                if (idx_start < 0) idx_start = 0;
                idx_stop = delay + post;
                if (idx_stop > r->delay_buf_count - 1) {
                    idx_stop = r->delay_buf_count - 1;
                }
                /* x2 = np.maximum.reduce over slices [idx_start, idx_stop].
                 * deque[i] is logical index i (0 = most recent). */
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
                /* legacy ring buffer (default-OFF). */
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
                /* x2 = max over axis 0. */
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
            /* AEC3 cc:121-129 noise gate — SKIPPED when
             * use_stationarity_properties=True (matches Python
             * `if not self._use_stationarity_properties:`). Production sets
             * this True, so the gate does not run; the earlier `transparent_mode`
             * guard here was a port bug (transparent_mode is always False on
             * this path, so the gate fired and collapsed x2 to 0). */
            if (!r->use_stationarity_properties) {
                /* noise gate (cc:121-129). */
                float ng = r->use_aec3_residual_noise_gate
                           ? r->residual_noise_gate_power
                           : r->echo_model.noise_gate_power;
                float slope = r->echo_model.noise_gate_slope; /* 0.3 */
                for (k = 0; k < N; ++k) {
                    if (ng > x2[k]) {
                        /* x2[mask] = max(0, x2 - slope*(ng - x2)) (all f32) */
                        float v = x2[k] - slope * (ng - x2[k]);
                        x2[k] = (v > 0.0f) ? v : 0.0f;
                    }
                }
            }
            /* subtract stationary noise (cc:284-288), all f32; then max 0. */
            {
                float sgs = r->echo_model.stationary_gate_slope; /* 10 */
                for (k = 0; k < N; ++k) {
                    float v = x2[k] - sgs * r->x2_noise_floor[k];
                    x2[k] = (v > 0.0f) ? v : 0.0f;
                }
            }
            for (k = 0; k < N; ++k) {
                /* r2 = x2 * echo_path_gain (f32 array * pyfloat → f32). */
                float v = x2[k] * echo_path_gain;
                r2[k] = v;
                r2_unbounded[k] = v;
            }
            if (r->nl_r2_enabled && r->nl_r2_alpha > 0.0f) {
                /* r2_nl = (alpha * x2**2 / norm).astype(f32), all f32. */
                float alpha = r->nl_r2_alpha;
                float norm = r->nl_norm_power;
                for (k = 0; k < N; ++k) {
                    float x2sq = x2[k] * x2[k];
                    float r2_nl = alpha * x2sq / norm;
                    r2[k] = r2[k] + r2_nl;
                    r2_unbounded[k] = r2_unbounded[k] + r2_nl;
                }
            }
        }
        /* UpdateReverb(kNonLinear) + AddReverb. */
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
            /* AddReverb fires even when delayed_render is None (matches
             * Python: reverb read + add happens unconditionally inside the
             * model_reverb_in_nonlinear_mode block). */
            for (k = 0; k < N; ++k) {
                float reverb = r->reverb_model.reverb[k] *
                               r->reverb_tail_strength;
                r->last_r2_direct[k] = r2[k];
                r->last_r2_reverb[k] = reverb;
                r2[k] = r2[k] + reverb;
                r2_unbounded[k] = r2_unbounded[k] + reverb;
            }
        }
    }
}
