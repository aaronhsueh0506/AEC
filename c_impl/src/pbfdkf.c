/* pbfdkf.c -- see pbfdkf.h. PBFDAF base (NLMS) plus PBFDKF (per-bin H_error gain). Float32
 * arithmetic; H_error, error_psd and R are derived from FFT outputs. */
#include "pbfdkf.h"
#include "aec_scale.h"
#include "utility/simd_aec.h"
#include "audio_window.h"
#include "utility/mem_align.h"
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static int next_pow2(int x) { int n = 1; while (n < x) n <<= 1; return n; }

static void pbfdaf_compute_sizes(int hop_size, int* out_hop, int* out_blk,
                                  int* out_fft, int* out_K) {
    int hop = (hop_size > 0) ? hop_size : 0;
    int blk = 2 * hop;
    int fft = next_pow2(blk);
    if (out_hop) *out_hop = hop;
    if (out_blk) *out_blk = blk;
    if (out_fft) *out_fft = fft;
    if (out_K)   *out_K   = fft / 2 + 1;
}

/* td_window is 1 on [0, hop) and 0 on [hop, fft_size); sqrt_hann is the periodic sqrt-Hann. */
static void pbfdaf_fill_windows(PBFDAF* p) {
    memset(p->td_window, 0, (size_t)p->fft_size * sizeof(float));
    /* Constrain: keep the causal half of each partition and clear the rest, so no wrap-around
     * energy leaks into the next valid hop. */
    for (int i = 0; i < p->hop_size; ++i) p->td_window[i] = 1.0f;
    ac_window_build(p->sqrt_hann, p->block_size);
}

static void pbfdaf_init_scalars(PBFDAF* p, int n_partitions, float mu,
                                float delta, int sample_rate) {
    p->n_partitions = n_partitions;
    p->sample_rate = sample_rate;
    p->mu = mu;
    p->delta = delta;
    p->enable_td_constraint = 1;
    p->partition_idx = 0;
    p->call_counter = 0;
    p->poor_excitation_counter = 0;
    p->saturated_capture = 0;
    p->block_stationary = 0;
    p->initial_state_active = 1;
    p->initial_state_active_render_hops = 0;
    /* Initial-state duration threshold: 2.5 s at any grid. */
    p->initial_state_threshold_hops =
        aec_ms_to_hops(2500.0f, p->hop_size, sample_rate);
    p->initial_state_far_energy_floor = 1e-4f;
    p->last_initial_state_active = 1;
    p->last_s_max_abs = 0.0f;
    p->lightweight = 0;
    p->precomputed_far_spec = NULL;
    p->far_fft_real_compute_count = 0;
    p->constraint_round_robin = 0;
    p->partition_to_constrain = 0;
    p->w_dirty = 1;
}

static void pbfdaf_zero_state(PBFDAF* p) {
    int Wsz = p->n_partitions * p->n_freqs;
    memset(p->W,     0, (size_t)Wsz * sizeof(Complex));
    p->w_dirty = 1;
    memset(p->X_buf, 0, (size_t)Wsz * sizeof(Complex));
    memset(p->x2_cache, 0, (size_t)Wsz * sizeof(float));
    memset(p->near_buffer, 0, (size_t)p->block_size * sizeof(float));
    memset(p->far_buffer,  0, (size_t)p->block_size * sizeof(float));
    memset(p->near_spec,  0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->echo_spec,  0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->error_spec, 0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->far_spec,   0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->far_cmag2_hold, 0, (size_t)p->n_freqs * sizeof(float));
    memset(p->error_spec_windowed, 0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->time_scratch, 0, (size_t)p->fft_size * sizeof(float));
    memset(p->spec_scratch, 0, (size_t)p->n_freqs * sizeof(Complex));
}

size_t pbfdaf_get_mem_size(int block_size, int n_partitions, int hop_size,
                           int with_process_scratch) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    int blk, fft, K;
    pbfdaf_compute_sizes(hop, NULL, &blk, &fft, &K);
    if (n_partitions <= 0 || K <= 0) return 0;
    size_t Wsz = ck_mul_size((size_t)n_partitions, (size_t)K);

    /* Checked size arithmetic (saturating ck_* helpers); the carve order in pbfdaf_init_static must
     * match this one. */
    size_t total = 0;
    total = ck_field_size(total, (size_t)fft, sizeof(float));
    total = ck_field_size(total, (size_t)blk, sizeof(float));
    total = ck_field_size(total, Wsz, sizeof(Complex));
    total = ck_field_size(total, Wsz, sizeof(Complex));
    total = ck_field_size(total, (size_t)blk, sizeof(float));
    total = ck_field_size(total, (size_t)blk, sizeof(float));
    total = ck_field_size(total, (size_t)K,   sizeof(Complex));
    total = ck_field_size(total, (size_t)K,   sizeof(Complex));
    total = ck_field_size(total, (size_t)K,   sizeof(Complex));
    total = ck_field_size(total, (size_t)K,   sizeof(Complex));
    total = ck_field_size(total, (size_t)K,   sizeof(Complex));
    total = ck_field_size(total, (size_t)fft, sizeof(float));
    total = ck_field_size(total, (size_t)K,   sizeof(Complex));
    total = ck_field_size(total, (size_t)hop, sizeof(float));
    if (with_process_scratch) {
        total = ck_field_size(total, (size_t)K, sizeof(float));
        total = ck_field_size(total, (size_t)K, sizeof(float));
        total = ck_field_size(total, (size_t)K, sizeof(float));
    }
    total = ck_field_size(total, ck_mul_size((size_t)n_partitions, (size_t)hop),
                         sizeof(float));
    total = ck_field_size(total, Wsz,         sizeof(float));
    total = ck_field_size(total, (size_t)K,   sizeof(float));

    if (MEM_SIZE_INVALID(total)) return 0;
    return total;
}

int pbfdaf_init_static(PBFDAF* p, void* mem, size_t mem_size,
                         int block_size, int n_partitions,
                         float mu, float delta, int hop_size,
                         int with_process_scratch, int sample_rate,
                         FftHandle* shared_fft) {
    int hop;
    if (!p || !mem || !shared_fft || sample_rate <= 0) return -1;
    hop = (hop_size > 0) ? hop_size : block_size / 2;
    if (hop <= 0) return -1;
    if (!MEM_IS_ALIGNED16(mem)) return -1;
    {
        size_t need = pbfdaf_get_mem_size(block_size, n_partitions, hop_size,
                                          with_process_scratch);
        if (need == 0 || mem_size < need) return -1;
    }

    memset(p, 0, sizeof(*p));
    pbfdaf_compute_sizes(hop, &p->hop_size, &p->block_size,
                         &p->fft_size, &p->n_freqs);

    uint8_t* ptr = (uint8_t*)mem;
    int blk = p->block_size, fft = p->fft_size, K = p->n_freqs;
    int Wsz = n_partitions * K;

    p->td_window  = (float*)ptr;   ptr += ALIGN16((size_t)fft * sizeof(float));
    p->sqrt_hann  = (float*)ptr;   ptr += ALIGN16((size_t)blk * sizeof(float));
    p->W          = (Complex*)ptr; ptr += ALIGN16((size_t)Wsz * sizeof(Complex));
    p->X_buf      = (Complex*)ptr; ptr += ALIGN16((size_t)Wsz * sizeof(Complex));
    p->near_buffer = (float*)ptr;  ptr += ALIGN16((size_t)blk * sizeof(float));
    p->far_buffer  = (float*)ptr;  ptr += ALIGN16((size_t)blk * sizeof(float));
    p->near_spec  = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    p->echo_spec  = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    p->error_spec = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    p->far_spec   = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    p->error_spec_windowed = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    /* Borrow the caller's shared_fft: fft_init()'d at this fft_size, used strictly sequentially. */
    p->fft          = shared_fft;
    p->time_scratch = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
    p->spec_scratch = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));

    p->scr_fsq      = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    if (with_process_scratch) {
        p->scr_mu_local = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
        p->scr_x2psum   = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
        p->scr_mu_eff   = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    } else {
        p->scr_mu_local = NULL; p->scr_x2psum = NULL; p->scr_mu_eff = NULL;
    }
    p->scr_ir       = (float*)ptr; ptr += ALIGN16((size_t)n_partitions * (size_t)hop * sizeof(float));
    p->x2_cache     = (float*)ptr; ptr += ALIGN16((size_t)Wsz * sizeof(float));
    p->far_cmag2_hold = (float*)ptr;
    ptr += ALIGN16((size_t)K * sizeof(float));
    (void)ptr;

    pbfdaf_init_scalars(p, n_partitions, mu, delta, sample_rate);
    pbfdaf_zero_state(p);
    pbfdaf_fill_windows(p);
    return 0;
}

void pbfdaf_free(PBFDAF* p) {
    if (!p) return;
    /* Nothing is owned here; the shared FFT handle is destroyed once by its creator, so only drop
     * the reference. */
    p->fft = NULL;
}

void pbfdaf_reset_taps(PBFDAF* p) {
    int Wsz = p->n_partitions * p->n_freqs;
    memset(p->W, 0, (size_t)Wsz * sizeof(Complex));
    p->w_dirty = 1;
    pbfdaf_clear_far_history(p);
}

void pbfdaf_clear_far_history(PBFDAF* p) {
    int Wsz = p->n_partitions * p->n_freqs;
    memset(p->X_buf, 0, (size_t)Wsz * sizeof(Complex));
    memset(p->x2_cache, 0, (size_t)Wsz * sizeof(float));
}

void pbfdaf_reset(PBFDAF* p) {
    pbfdaf_reset_taps(p);
    memset(p->near_buffer, 0, (size_t)p->block_size * sizeof(float));
    memset(p->far_buffer,  0, (size_t)p->block_size * sizeof(float));
    p->partition_idx = 0;
    memset(p->error_spec_windowed, 0, (size_t)p->n_freqs * sizeof(Complex));
    /* precomputed_far_spec is one-shot and already NULL after a hop; clear it anyway so a reset
     * between an external set and the process call cannot leave a stale pointer. */
    p->precomputed_far_spec = NULL;
    p->far_fft_real_compute_count = 0;
}

static void rfft_padded(PBFDAF* p, const float* time_in, int in_len,
                        Complex* spec_out) {
    if (in_len == p->fft_size) {
        fft_forward(p->fft, time_in, spec_out);
    } else {
        memcpy(p->time_scratch, time_in, (size_t)in_len * sizeof(float));
        memset(p->time_scratch + in_len, 0,
               (size_t)(p->fft_size - in_len) * sizeof(float));
        fft_forward_scratch(p->fft, p->time_scratch, spec_out);
    }
}

/* Shared front end: buffer shift, FFTs, far PSD, echo, error and windowed error. Returns the far
 * hop energy (the W-update gate); curr_p is stored in *out_curr_p. */
static float pbfdaf_frontend(PBFDAF* p,
                              const float* near_end, const float* far_end,
                              float* output, int* out_curr_p,
                              const float* precomputed_far_cmag2) {
    int hop = p->hop_size, blk = p->block_size, K = p->n_freqs;

    memmove(p->near_buffer, p->near_buffer + hop, (size_t)(blk - hop) * sizeof(float));
    memcpy(p->near_buffer + (blk - hop), near_end, (size_t)hop * sizeof(float));
    memmove(p->far_buffer, p->far_buffer + hop, (size_t)(blk - hop) * sizeof(float));
    memcpy(p->far_buffer + (blk - hop), far_end, (size_t)hop * sizeof(float));

    if (!p->lightweight)
        rfft_padded(p, p->near_buffer, blk, p->near_spec);
    /* far_spec is shared by main and shadow; reuse the caller's copy when given (one-shot). */
    int borrowed_far = p->precomputed_far_spec != NULL;
    if (borrowed_far) {
        memcpy(p->far_spec, p->precomputed_far_spec, (size_t)K * sizeof(Complex));
        p->precomputed_far_spec = NULL;
    } else {
        rfft_padded(p, p->far_buffer, blk, p->far_spec);
        if (p->far_fft_real_compute_count < LONG_MAX)
            p->far_fft_real_compute_count++;
    }

    int curr_p = p->partition_idx;
    *out_curr_p = curr_p;
    memcpy(p->X_buf + (size_t)curr_p * K, p->far_spec, (size_t)K * sizeof(Complex));

    /* |far_spec[k]|^2 for this hop, kept in far_cmag2_hold so the orchestrator can read the same
     * magnitudes before and after the frontend. No far-power EMA is kept: nothing reads one. */
    float *far_cmag2 = p->far_cmag2_hold;
    if (borrowed_far && precomputed_far_cmag2)
        memcpy(far_cmag2, precomputed_far_cmag2, (size_t)K * sizeof(float));
    else
        sk_cmag2_np_f32(p->far_spec, far_cmag2, K);
    /* Mirror |X_buf[curr_p]|^2 into x2_cache once, instead of recomputing every partition's
     * magnitudes. */
    memcpy(p->x2_cache + (size_t)curr_p * K, far_cmag2, (size_t)K * sizeof(float));

    memset(p->echo_spec, 0, (size_t)K * sizeof(Complex));
    /* Complex multiply-accumulate uses the fused form (re = fma(wr, xr, -(wi * xi)), im = fma(wr,
     * xi, wi * xr)), one partition at a time in partition order. */
    sk_cmac_fma_ring_f32(p->echo_spec, p->W, p->X_buf, curr_p,
                        p->n_partitions, K);

    fft_inverse(p->fft, p->echo_spec, p->time_scratch);
    for (int i = 0; i < hop; ++i)
        output[i] = p->near_buffer[blk - hop + i] - p->time_scratch[hop + i];

    p->last_s_max_abs = sk_max_abs_f32(p->time_scratch + hop, hop);

    memset(p->time_scratch, 0, (size_t)hop * sizeof(float));
    memcpy(p->time_scratch + hop, output, (size_t)hop * sizeof(float));
    if (p->fft_size > blk)
        memset(p->time_scratch + blk, 0, (size_t)(p->fft_size - blk) * sizeof(float));
    fft_forward_scratch(p->fft, p->time_scratch, p->error_spec);

    /* Windowed near (sqrt-Hann * near_buffer) minus echo_spec, for the RES analysis; skipped in
     * lightweight (shadow) mode. */
    if (!p->lightweight) {
        for (int i = 0; i < blk; ++i)
            p->time_scratch[i] = p->near_buffer[i] * p->sqrt_hann[i];
        rfft_padded(p, p->time_scratch, blk, p->spec_scratch);
        sk_csub_f32(p->error_spec_windowed, p->spec_scratch, p->echo_spec, K);
    }

    {
        float *fsq = p->scr_fsq;
        for (int i = 0; i < hop; ++i) fsq[i] = far_end[i] * far_end[i];
        float s = sk_pairwise_sum_tailfold_f32(fsq, (size_t)hop);
        return s / (float)hop;
    }
}

void pbfdaf_process_with_far_cmag2(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr,
                       float        mu_scale_scalar,
                       float*       output,
                       const float* precomputed_far_cmag2) {
    int K = p->n_freqs, N = p->n_partitions;
    int curr_p;
    float far_hop_energy = pbfdaf_frontend(p, near_end, far_end, output, &curr_p,
                                               precomputed_far_cmag2);

    /* InitialState tracking (count active render hops). */
    if (p->initial_state_active &&
        far_hop_energy > p->initial_state_far_energy_floor) {
        p->initial_state_active_render_hops += 1;
        if (p->initial_state_active_render_hops >= p->initial_state_threshold_hops)
            p->initial_state_active = 0;
    }
    p->last_initial_state_active = p->initial_state_active;

    if (far_hop_energy > 1e-4f) {
        /* The counter increments first, unconditionally, and saturates at LONG_MAX: its raw value
         * is observable, so it is not capped at the startup threshold. */
        if (p->call_counter < LONG_MAX) p->call_counter += 1;
        const float* mu_arr = mu_scale_arr;
        int any_pos;
        if (mu_arr == NULL) {
            /* With a scalar mu_scale only the scalar needs to be positive; skip the fill and scan
             * otherwise. */
            any_pos = (mu_scale_scalar > 0.0f);
            if (!any_pos) { p->partition_idx = (p->partition_idx + 1) % N; return; }
            {
                float* mu_local = p->scr_mu_local;
                for (int k = 0; k < K; ++k) mu_local[k] = mu_scale_scalar;
                mu_arr = mu_local;
            }
        } else {
            any_pos = 0;
            for (int k = 0; k < K; ++k) if (mu_arr[k] > 0.0f) { any_pos = 1; break; }
            if (!any_pos) { p->partition_idx = (p->partition_idx + 1) % N; return; }
        }
        if (p->saturated_capture) { p->partition_idx = (p->partition_idx + 1) % N; return; }
        if (p->call_counter <= N || p->poor_excitation_counter < N) {
            p->partition_idx = (p->partition_idx + 1) % N; return;
        }
        if (p->block_stationary) { p->partition_idx = (p->partition_idx + 1) % N; return; }

        /* x2_partition_sum = sum over partitions of |X_buf|^2, taken from x2_cache, summed in
         * partition order. */
        float *x2psum = p->scr_x2psum;
        memcpy(x2psum, p->x2_cache, (size_t)K * sizeof(float));
        for (int part = 1; part < N; ++part) {
            const float* x2p = p->x2_cache + (size_t)part * K;
            for (int k = 0; k < K; ++k) x2psum[k] += x2p[k];
        }
        float mu_initial_boost = p->initial_state_active ? (0.95f / 0.7f) : 1.0f;
        float ng_thr = (float)AEC_FILTER_NOISE_GATE_POWER_FLOAT;
        float *mu_eff = p->scr_mu_eff;
        for (int k = 0; k < K; ++k) {
            float denom = x2psum[k] + p->delta;
            float m = (p->mu * mu_initial_boost * mu_arr[k]) / denom;
            mu_eff[k] = (x2psum[k] >= ng_thr) ? m : 0.0f;
        }
        for (int part = 0; part < N; ++part) {
            int p_idx = curr_p - part; while (p_idx < 0) p_idx += N;
            Complex* Wp = p->W + (size_t)part * K;
            const Complex* Xp = p->X_buf + (size_t)p_idx * K;
            /* grad = error_spec * conj(X), complex multiply in the fused form; mu_eff * grad is
             * real x complex. */
            sk_wupdate_nlms_f32(Wp, Xp, p->error_spec, mu_eff, K);
            if (p->enable_td_constraint &&
                (!p->constraint_round_robin || part == p->partition_to_constrain)) {
                fft_inverse_scratch(p->fft, Wp, p->time_scratch);
                for (int i = p->hop_size; i < p->fft_size; ++i) p->time_scratch[i] *= p->td_window[i];
                fft_forward_scratch(p->fft, p->time_scratch, Wp);
            }
        }
        if (p->enable_td_constraint && p->constraint_round_robin)
            p->partition_to_constrain = (p->partition_to_constrain + 1) % N;
        p->w_dirty = 1;
    }

    p->partition_idx = (p->partition_idx + 1) % N;
}

void pbfdaf_process(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr, float mu_scale_scalar,
                       float* output) {
    pbfdaf_process_with_far_cmag2(p, near_end, far_end, mu_scale_arr,
                                mu_scale_scalar, output, NULL);
}

void pbfdaf_copy_weights_from(PBFDAF* dst, const PBFDAF* src) {
    int Wsz = dst->n_partitions * dst->n_freqs;
    memcpy(dst->W, src->W, (size_t)Wsz * sizeof(Complex));
    dst->w_dirty = 1;
}

/* Warm tap transfer: shifts the learned IR by shift_samples (positive = left, toward tap 0;
 * negative = right), zero-filling the freed end, so the cold-start cancellation survives a delay
 * realign. Raw irfft per partition -> concatenate -> shift -> rfft each hop block back into W (no
 * td_window). Uses the scr_ir scratch. A negative shift clears the far history. Returns 0 (applied)
 * or -1 (refused by the span guard, reachable from a legal config); the caller must then fall back
 * to a filter reset. */
int pbfdaf_warm_shift_ir(PBFDAF* p, int shift_samples) {
    int s = shift_samples;
    if (s == 0) return 0;
    int hop = p->hop_size, nF = p->fft_size, nP = p->n_partitions, K = p->n_freqs;
    int total = nP * hop;
    if (total <= 0 || total > 4096) return -1;
    if (s > total) s = total;
    if (s < -total) s = -total;
    float *ir = p->scr_ir;
    pbfdaf_get_time_domain_filter(p, ir);
    if (s > 0) {
        for (int i = 0; i < total - s; ++i) ir[i] = ir[i + s];
        for (int i = total - s; i < total; ++i) ir[i] = 0.0f;
    } else {
        /* Negative shift: the learned response moves toward later taps; walk backwards. */
        int r = -s;
        for (int i = total - 1; i >= r; --i) ir[i] = ir[i - r];
        for (int i = 0; i < r; ++i) ir[i] = 0.0f;
    }
    for (int part = 0; part < nP; ++part) {
        memset(p->time_scratch, 0, (size_t)nF * sizeof(float));
        for (int i = 0; i < hop; ++i) p->time_scratch[i] = ir[part * hop + i];
        fft_forward_scratch(p->fft, p->time_scratch, p->W + (size_t)part * K);
    }
    p->w_dirty = 1;
    if (s < 0) pbfdaf_clear_far_history(p);
    return 0;
}

static void pbfdkf_init_scalars(PBFDKF* p) {
    int K = p->base.n_freqs;
    for (int k = 0; k < K; ++k) {
        p->Q_high[k] = 1e-4f;
        p->Q_low[k]  = 1e-5f;
        p->Q[k]      = 1e-4f;
        p->R[k]      = 1e-2f;
        p->error_psd[k] = 1e-2f;
        p->H_error_per_bin[k] = (float)AEC_H_ERROR_INIT_FLOAT;
        p->erl_per_bin[k] = 0.1f;
        p->e2_coarse_per_bin[k] = 0.0f;
    }
    /* Measurement-noise PSD EMA retention per hop, retimed from the 10 ms reference (0.95). */
    p->alpha_r = aec_growth_rehop(0.95f, 160, 16000,
                                  p->base.hop_size, p->base.sample_rate);
    p->h_error_floor = (float)AEC_H_ERROR_FLOOR_FLOAT;
    p->h_error_ceil  = (float)AEC_H_ERROR_CEIL_FLOAT;
    /* Leakage rates rescaled to the live grid from the per-block rates 5e-5, 5e-2, 5e-3, 5e-1. */
    p->leakage_converged = aec_per_block_rate_to_per_hop(
        5e-5f, p->base.hop_size, p->base.sample_rate);
    p->leakage_diverged = aec_per_block_rate_to_per_hop(
        5e-2f, p->base.hop_size, p->base.sample_rate);
    p->leakage_converged_transient = aec_per_block_rate_to_per_hop(
        5e-3f, p->base.hop_size, p->base.sample_rate);
    p->leakage_diverged_transient = aec_per_block_rate_to_per_hop(
        5e-1f, p->base.hop_size, p->base.sample_rate);
    p->e2_coarse_per_bin_valid = 0;
    p->e2_coarse_for_refresh = 0.0f;
    p->disallow_leakage_diverged = 0;
    p->h_error_refresh_erl_floor = 0.0f;
    p->last_leakage_div_frac = 0.0f;
    p->partition_sum_x2_startup_hops = 0;
    p->base.poor_excitation_counter = 400;
}

size_t pbfdkf_get_mem_size(int block_size, int n_partitions, int hop_size) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    int blk, fft, K;
    pbfdaf_compute_sizes(hop, NULL, &blk, &fft, &K);
    (void)blk; (void)fft;
    if (n_partitions <= 0 || K <= 0) return 0;
    size_t total = pbfdaf_get_mem_size(block_size, n_partitions, hop_size, 0);
    if (total == 0) return 0;
    /* Checked size arithmetic: 12 equally sized K-length float fields, each ALIGN16'd by
     * pbfdkf_init_static. */
    total = ck_field_size_reps(total, (size_t)K, sizeof(float), 12);

    if (MEM_SIZE_INVALID(total)) return 0;
    return total;
}

int pbfdkf_init_static(PBFDKF* p, void* mem, size_t mem_size,
                         int block_size, int n_partitions,
                         float mu, float delta, int hop_size,
                         int sample_rate, FftHandle* shared_fft) {
    int hop;
    if (!p || !mem || !shared_fft || sample_rate <= 0) return -1;
    hop = (hop_size > 0) ? hop_size : block_size / 2;
    if (hop <= 0) return -1;
    if (!MEM_IS_ALIGNED16(mem)) return -1;
    {
        size_t need = pbfdkf_get_mem_size(block_size, n_partitions, hop_size);
        if (need == 0 || mem_size < need) return -1;
    }
    size_t base_sz = pbfdaf_get_mem_size(block_size, n_partitions, hop_size, 0);
    memset(p, 0, sizeof(*p));
    if (pbfdaf_init_static(&p->base, mem, base_sz,
                       block_size, n_partitions, mu, delta, hop_size, 0,
                       sample_rate, shared_fft) != 0)
        return -1;
    uint8_t* ptr = (uint8_t*)mem + base_sz;
    int K = p->base.n_freqs;
    p->Q_high    = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->Q_low     = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->Q         = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->R         = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->error_psd = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->H_error_per_bin = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->erl_per_bin = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->e2_coarse_per_bin = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->e2_ref_scratch = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->scr_mu_local = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->scr_X2       = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->scr_mu_aec   = (float*)ptr; ptr += ALIGN16((size_t)p->base.n_freqs * sizeof(float));
    (void)ptr;
    pbfdkf_init_scalars(p);
    return 0;
}

void pbfdkf_free(PBFDKF* p) {
    if (p) pbfdaf_free(&p->base);
}

/* Kalman-side re-arm shared by pbfdkf_reset and pbfdkf_reset_taps. */
static void pbfdkf_rearm_kalman(PBFDKF* p) {
    int K = p->base.n_freqs;
    for (int k = 0; k < K; ++k) {
        p->R[k] = 1e-2f;
        p->error_psd[k] = 1e-2f;
        p->Q[k] = p->Q_high[k];
    }
}

void pbfdkf_reset(PBFDKF* p) {
    pbfdaf_reset(&p->base);
    pbfdkf_rearm_kalman(p);
}

void pbfdkf_reset_taps(PBFDKF* p) {
    pbfdaf_reset_taps(&p->base);
    pbfdkf_rearm_kalman(p);
}

void pbfdaf_reset_carried_state(PBFDAF* p) {
    p->partition_to_constrain = 0;
    p->last_s_max_abs = 0.0f;
    p->call_counter = 0;
    /* The rate-scaled value used at construction and in pbfdkf_handle_echo_path_change. */
    p->poor_excitation_counter =
        aec_blocks_to_hops(1000, p->hop_size, p->sample_rate);
    p->saturated_capture = 0;
    p->block_stationary = 0;
    p->initial_state_active = 1;
    p->initial_state_active_render_hops = 0;
    p->last_initial_state_active = 1;
}

void pbfdkf_reset_carried_state(PBFDKF* p) {
    int k, K = p->base.n_freqs;
    pbfdaf_reset_carried_state(&p->base);
    p->e2_coarse_per_bin_valid = 0;
    p->e2_coarse_for_refresh = 0.0f;
    p->disallow_leakage_diverged = 0;
    p->last_leakage_div_frac = 0.0f;
    /* H_error is not re-armed by pbfdkf_rearm_kalman; the mid-stream relock path restores it via
     * pbfdkf_handle_echo_path_change(delay_change=1). Restore it here. */
    for (k = 0; k < K; ++k)
        p->H_error_per_bin[k] = (float)AEC_H_ERROR_INIT_FLOAT;
}

void pbfdkf_handle_echo_path_change(PBFDKF* p, int delay_change, int gain_change) {
    /* Counter reset (not gain_change) to the rate-scaled value used at construction. */
    if (!gain_change) {
        p->base.poor_excitation_counter =
            aec_blocks_to_hops(1000, p->base.hop_size, p->base.sample_rate);
        p->base.call_counter = 0;
    }
    if (delay_change) {
        int K = p->base.n_freqs;
        for (int k = 0; k < K; ++k)
            p->H_error_per_bin[k] = (float)AEC_H_ERROR_INIT_FLOAT;
    }
}

/* Concatenate irfft(W[p])[:hop] for partitions [first_partition, first_partition + n_parts) into
 * out[part * hop ...], at the same offsets as the whole-filter layout; other taps are untouched.
 * The range is clamped to [0, n_partitions). */
void pbfdaf_get_time_domain_filter_range(PBFDAF* p, int first_partition,
                                         int n_parts, float* out) {
    int hop = p->hop_size;
    int first = first_partition;
    int last  = first_partition + n_parts;
    if (first < 0) first = 0;
    if (last > p->n_partitions) last = p->n_partitions;
    for (int part = first; part < last; ++part) {
        const Complex* Wp = p->W + (size_t)part * p->n_freqs;
        fft_inverse(p->fft, Wp, p->time_scratch);
        for (int i = 0; i < hop; ++i)
            out[(size_t)part * hop + i] = p->time_scratch[i];
    }
}

void pbfdaf_get_time_domain_filter(PBFDAF* p, float* out) {
    pbfdaf_get_time_domain_filter_range(p, 0, p->n_partitions, out);
}

void pbfdaf_scale_filter(PBFDAF* p, float scale) {
    int Wsz = p->n_partitions * p->n_freqs;
    for (int i = 0; i < Wsz; ++i) {
        p->W[i].r = p->W[i].r * scale;
        p->W[i].i = p->W[i].i * scale;
    }
    p->w_dirty = 1;
}

void pbfdkf_scale_filter(PBFDKF* p, float scale) {
    pbfdaf_scale_filter(&p->base, scale);
}

/* H_error leakage refresh and clamp; runs every hop. */
static void pbfdkf_h_error_refresh(PBFDKF* p, const float *e2_ref_bins) {
    PBFDAF* b = &p->base;
    int K = b->n_freqs;

    float lc_eff, ld_eff;
    if (b->initial_state_active) {
        lc_eff = p->leakage_converged_transient;
        ld_eff = p->leakage_diverged_transient;
    } else {
        lc_eff = p->leakage_converged;
        ld_eff = p->leakage_diverged;
    }

    if (p->e2_coarse_per_bin_valid) {
        /* Per-bin path; e2_ref_bins[k] = |error_spec[k]|^2 is computed once by pbfdkf_process and
         * shared. */
        int n_div = sk_h_error_refresh_f32(
            p->H_error_per_bin, e2_ref_bins, p->e2_coarse_per_bin,
            p->erl_per_bin, lc_eff, ld_eff, p->h_error_refresh_erl_floor,
            p->disallow_leakage_diverged, K);
        p->last_leakage_div_frac = (float)n_div / (float)K;
    } else {
        /* Scalar fallback: e2_ref_sum = sum(error_psd) against the scalar coarse energy (the
         * orchestrator normally supplies the per-bin array). */
        float e2_ref_sum = sk_pairwise_sum_tailfold_f32(p->error_psd, (size_t)K);
        float e2_coa_sum = p->e2_coarse_for_refresh;
        int use_conv = (e2_ref_sum <= e2_coa_sum) || p->disallow_leakage_diverged;
        p->last_leakage_div_frac = use_conv ? 0.0f : 1.0f;
        float leakage = use_conv ? lc_eff : ld_eff;
        for (int k = 0; k < K; ++k)
            p->H_error_per_bin[k] = p->H_error_per_bin[k] +
                                    leakage * p->erl_per_bin[k];
    }
    sk_clip_f32(p->H_error_per_bin, p->h_error_floor, p->h_error_ceil, K);
}

/* Per-bin Kalman gain via H_error. mu_scale_arr is already post-mask. */
static void pbfdkf_update_weights(PBFDKF* p, int curr_p,
                                  const float* mu_scale_arr,
                                  const float* e2_ref_bins) {
    PBFDAF* b = &p->base;
    int K = b->n_freqs, N = b->n_partitions;

    /* X^2 source: the partition sum once call_counter is past the startup hops, else the latest
     * partition; both read x2_cache. */
    float *X2 = p->scr_X2;
    if (b->call_counter > p->partition_sum_x2_startup_hops) {
        memcpy(X2, b->x2_cache, (size_t)K * sizeof(float));
        for (int part = 1; part < N; ++part) {
            const float* x2p = b->x2_cache + (size_t)part * K;
            for (int k = 0; k < K; ++k) X2[k] += x2p[k];
        }
    } else {
        memcpy(X2, b->x2_cache + (size_t)curr_p * K, (size_t)K * sizeof(float));
    }

    float delta32 = (float)b->delta;
    float n_part = (float)N;
    float ng_thr = (float)AEC_FILTER_NOISE_GATE_POWER_FLOAT;

    /* mu[k] = H_error / (0.5 * H_error * X^2 + n * E^2_refined + delta), gated by X^2 >= noise
     * gate. */
    float *mu_aec = p->scr_mu_aec;
    for (int k = 0; k < K; ++k) {
        float e2_ref = e2_ref_bins[k];
        float He = p->H_error_per_bin[k];
        float denom = 0.5f * He * X2[k] + n_part * e2_ref + delta32;
        float m = He / denom;
        mu_aec[k] = (X2[k] >= ng_thr) ? m : 0.0f;
    }

    for (int part = 0; part < N; ++part) {
        int p_idx = curr_p - part; while (p_idx < 0) p_idx += N;
        const Complex* Xp = b->X_buf + (size_t)p_idx * K;
        Complex* Wp = b->W + (size_t)part * K;
        /* K = mu * conj(X); K_scaled = K * mu_scale_arr; W += K_scaled * error_spec, complex
         * multiply in
         * the fused form. */
        sk_wupdate_kf_f32(Wp, Xp, b->error_spec, mu_aec, mu_scale_arr, K);
        if (b->enable_td_constraint &&
            (!b->constraint_round_robin || part == b->partition_to_constrain)) {
            fft_inverse_scratch(b->fft, Wp, b->time_scratch);
            for (int i = b->hop_size; i < b->fft_size; ++i) b->time_scratch[i] *= b->td_window[i];
            fft_forward_scratch(b->fft, b->time_scratch, Wp);
        }
    }
    if (b->enable_td_constraint && b->constraint_round_robin)
        b->partition_to_constrain = (b->partition_to_constrain + 1) % N;
    b->w_dirty = 1;

    /* Scale mu by mu_scale_arr before the H_error decay so masked bins do not decay while their W
     * is frozen. H_error decay (active branch only): H_error -= 0.5 * mu * X^2 * H_error. */
    for (int k = 0; k < K; ++k) {
        float ms = mu_aec[k] * mu_scale_arr[k];
        p->H_error_per_bin[k] -= 0.5f * ms * X2[k] * p->H_error_per_bin[k];
    }

    pbfdkf_h_error_refresh(p, e2_ref_bins);
}

int pbfdkf_process_with_far_cmag2(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr,
                       float        mu_scale_scalar,
                       float*       output,
                       const float* precomputed_far_cmag2) {
    PBFDAF* b = &p->base;
    int K = b->n_freqs, N = b->n_partitions;
    int curr_p;
    int error_cmag2_valid = 0;
    float far_hop_energy = pbfdaf_frontend(b, near_end, far_end, output, &curr_p,
                                               precomputed_far_cmag2);

    /* InitialState tracking. */
    if (b->initial_state_active &&
        far_hop_energy > b->initial_state_far_energy_floor) {
        b->initial_state_active_render_hops += 1;
        if (b->initial_state_active_render_hops >= b->initial_state_threshold_hops)
            b->initial_state_active = 0;
    }
    b->last_initial_state_active = b->initial_state_active;

    if (far_hop_energy > 1e-4f) {
        /* PBFDKF update. */
        float *mu_local = p->scr_mu_local;
        const float* mu_arr = mu_scale_arr;
        if (mu_arr == NULL) {
            for (int k = 0; k < K; ++k) mu_local[k] = mu_scale_scalar;
            mu_arr = mu_local;
        }
        /* e2_ref[k] = |error_spec[k]|^2, computed once and shared by the error_psd EMA, the mu
         * gate and
         * every H_error refresh in this function. */
        float *e2_ref = p->e2_ref_scratch;
        sk_cmag2_np_f32(b->error_spec, e2_ref, K);
        error_cmag2_valid = 1;

        /* Startup, poor-excitation and saturation gates: refresh only. call_counter saturates at
         * LONG_MAX, as in pbfdaf_process. */
        if (b->call_counter < LONG_MAX) b->call_counter += 1;
        if (b->call_counter <= N || b->poor_excitation_counter < N ||
            b->saturated_capture) {
            pbfdkf_h_error_refresh(p, e2_ref);
            b->partition_idx = (b->partition_idx + 1) % N;
            return error_cmag2_valid;
        }
        if (b->block_stationary) {
            pbfdkf_h_error_refresh(p, e2_ref);
            b->partition_idx = (b->partition_idx + 1) % N;
            return error_cmag2_valid;
        }

        /* error_psd EMA and R = max(error_psd, delta), float32. alpha_r is retimed at init. */
        const float ar = p->alpha_r;
        const float oar = 1.0f - p->alpha_r;
        for (int k = 0; k < K; ++k) {
            p->error_psd[k] = ar * p->error_psd[k] + oar * e2_ref[k];
            p->R[k] = (p->error_psd[k] > b->delta) ? p->error_psd[k] : (float)b->delta;
        }

        pbfdkf_update_weights(p, curr_p, mu_arr, e2_ref);
    }

    b->partition_idx = (b->partition_idx + 1) % N;
    return error_cmag2_valid;
}

void pbfdkf_process(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr, float mu_scale_scalar,
                       float* output) {
    (void)pbfdkf_process_with_far_cmag2(p, near_end, far_end, mu_scale_arr,
                                     mu_scale_scalar, output, NULL);
}
