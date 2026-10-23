/* pbfdkf.c — PBFDAF base (NLMS) + PBFDKF derived (per-bin H_error Kalman).
 *
 * Port of python/modules/filters.py. Numerical notes:
 *  - The H_error / mu / leakage-refresh ARITHMETIC is reproduced op-for-op in
 *    the same dtype/order as Python, so it is exact GIVEN identical inputs.
 *    But H_error's gain denom consumes |error_spec|² and |X|² (FFT outputs)
 *    and error_psd accumulates |error_spec|², so H_error / error_psd / R are
 *    rtol-bounded (<1e-4), not bit-exact. Only the integer control state
 *    (call_counter / partition_idx / initial_state) is bit-exact.
 *  - np.float32 ⇒ float; np.complex64 ⇒ Complex. Scalar promotions are done
 *    directly in float32 (single literals get the `f` suffix).
 *
 * The active update is the per-bin H_error gain + always-on leakage
 * refresh.
 */
#include "pbfdkf.h"
#include "aec_scale.h"
#include "utility/simd_aec.h"
#include "audio_window.h"
#include "utility/mem_align.h"   /* ALIGN16 -- was transitively reached via fft_wrapper.h */
#include <limits.h>   /* LONG_MAX */
#include <stdint.h>
#include <string.h>
#include <math.h>

static int next_pow2(int x) { int n = 1; while (n < x) n <<= 1; return n; }

/* numpy |complex64|**2 (np.abs(z)**2) is computed by the
 * sk_cmag2_np_f32 / sk__cmag2_np_elem kernel family
 * (utility/simd_aec.h). */

/* Compute derived sizes. */
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

/* td_window + sqrt-Hann analysis window (P0.4 canonical periodic sqrt-Hann,
 * denom = block_size). */
static void pbfdaf_fill_windows(PBFDAF* p) {
    memset(p->td_window, 0, (size_t)p->fft_size * sizeof(float));
    /* Adaptive-filter Constrain: keep the causal half of each
     * partition and clear the second half exactly.  Any non-zero coefficient
     * at or beyond hop belongs to the circular/non-causal half of this
     * overlap-save partition and leaks wrap-around energy into the next valid
     * hop.  Keep every causal tap; a fade in [hop-fade_len, hop) would shorten
     * the effective filter for no overlap-save benefit. */
    for (int i = 0; i < p->hop_size; ++i) p->td_window[i] = 1.0f;
    /* [hop, fft_size) stays 0. */
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
    /* Initial-state duration threshold: 2.5 s at any grid via
     * aec_ms_to_hops. Mirrors filters.py's PBFDAF._initial_state_threshold_hops. */
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
    /* Paired with far_spec, not with X_buf -- cmag2_np(0,0) is +0.0f, so a
     * zeroed hold is what a recompute off the zeroed spectrum would produce. */
    memset(p->far_cmag2_hold, 0, (size_t)p->n_freqs * sizeof(float));
    memset(p->error_spec_windowed, 0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->time_scratch, 0, (size_t)p->fft_size * sizeof(float));
    memset(p->spec_scratch, 0, (size_t)p->n_freqs * sizeof(Complex));
}

/* ===== PBFDAF ============================================================ */

size_t pbfdaf_get_mem_size(int block_size, int n_partitions, int hop_size,
                           int with_process_scratch) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    int blk, fft, K;
    pbfdaf_compute_sizes(hop, NULL, &blk, &fft, &K);
    if (n_partitions <= 0 || K <= 0) return 0;
    size_t Wsz = ck_mul_size((size_t)n_partitions, (size_t)K);

    /* Checked size arithmetic: saturating ck_* helpers (mem_align.h);
     * MEM_SIZE_INVALID(total) at the end reports failure (0) on overflow instead
     * of a small wrapped total. pbfdaf_init_static's carve order must stay in
     * lockstep with this one. */
    size_t total = 0;
    total = ck_field_size(total, (size_t)fft, sizeof(float));   /* td_window */
    total = ck_field_size(total, (size_t)blk, sizeof(float));   /* sqrt_hann */
    total = ck_field_size(total, Wsz, sizeof(Complex));         /* W */
    total = ck_field_size(total, Wsz, sizeof(Complex));         /* X_buf */
    total = ck_field_size(total, (size_t)blk, sizeof(float));   /* near_buffer */
    total = ck_field_size(total, (size_t)blk, sizeof(float));   /* far_buffer */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* near_spec */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* echo_spec */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* error_spec */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* far_spec */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* error_spec_windowed */
    /* No nested-FFT term here: the static path always borrows a
     * caller-carved shared_fft (see pbfdaf_init_static's doc comment) instead
     * of carving its own. */
    total = ck_field_size(total, (size_t)fft, sizeof(float));   /* time_scratch */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* spec_scratch */
    /* De-stacked per-hop scratch (see pbfdkf.h PBFDAF comment). */
    total = ck_field_size(total, (size_t)hop, sizeof(float));   /* scr_fsq */
    if (with_process_scratch) {
        total = ck_field_size(total, (size_t)K, sizeof(float)); /* scr_mu_local */
        total = ck_field_size(total, (size_t)K, sizeof(float)); /* scr_x2psum */
        total = ck_field_size(total, (size_t)K, sizeof(float)); /* scr_mu_eff */
    }
    total = ck_field_size(total, ck_mul_size((size_t)n_partitions, (size_t)hop),
                         sizeof(float));                        /* scr_ir */
    /* Appended last, mirroring the struct's own tail. */
    total = ck_field_size(total, Wsz,         sizeof(float));   /* x2_cache */
    total = ck_field_size(total, (size_t)K,   sizeof(float));   /* far_cmag2_hold */

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
    /* Reject a misaligned pool base before any pool write -- hardens a caller
     * invoking this directly (the parent aec_carve always carves ALIGN16 offsets
     * off an already-aligned base, so this never fires there). */
    if (!MEM_IS_ALIGNED16(mem)) return -1;
    {
        /* pbfdaf_get_mem_size returning 0 means "invalid"; a real config's size is
         * always > 0, so 0 is an unambiguous failure here rather than
         * `mem_size < 0`, which is never true for a size_t. */
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
    /* Borrow the caller's shared_fft instead of carving/owning a
     * private FftHandle. Precondition (documented on the header decl):
     * shared_fft is already fft_init()'d at this same fft_size, and every
     * borrower's use of it is fully sequential (single-threaded, no
     * transform left in flight across calls) -- exactly how main filter,
     * shadow filter, and aec_post already call each other today. */
    p->fft          = shared_fft;
    p->time_scratch = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
    p->spec_scratch = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));

    /* De-stacked per-hop scratch (see pbfdkf.h PBFDAF comment). */
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
    /* Nothing here is owned: the arrays live in the caller's pool and p->fft
     * is a handle borrowed from aec_carve's shared FFT, destroyed exactly once
     * by whoever created it (aec_destroy). Destroying it per borrower would
     * double-free the backend's twiddle allocation, so just drop the
     * reference. */
    p->fft = NULL;
}

/* Adaptation state only — see the pbfdkf.h prototype for the split rationale.
 * pbfdaf_reset() below is defined as this plus the analysis-side buffers, so
 * the two can never drift apart on what "the filter's own state" is. */
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
    /* precomputed_far_spec is one-shot: set and consumed within the same
     * pbfdaf_frontend() call, cleared there right after the memcpy. It is
     * therefore already NULL when any reset() can observe it (reset never runs
     * mid-hop). With cross-instance sharing (aec_process_context_shared_far())
     * setting it from outside the instance, a caller that resets an instance
     * between setting the pointer and this instance's own process call would
     * otherwise leave a stale, possibly caller-freed pointer on the struct;
     * clearing it here removes that possibility without relying on every caller
     * getting the ordering right. */
    p->precomputed_far_spec = NULL;
    p->far_fft_real_compute_count = 0;
}

/* forward rfft of an input shorter than fft_size (zero-pad). */
static void rfft_padded(PBFDAF* p, const float* time_in, int in_len,
                        Complex* spec_out) {
    if (in_len == p->fft_size) {
        fft_forward(p->fft, time_in, spec_out);
    } else {
        memcpy(p->time_scratch, time_in, (size_t)in_len * sizeof(float));
        memset(p->time_scratch + in_len, 0,
               (size_t)(p->fft_size - in_len) * sizeof(float));
        /* time_scratch is dead after this call -> clobber-permitted rfft
         * (NE10 skips its input staging copy). */
        fft_forward_scratch(p->fft, p->time_scratch, spec_out);
    }
}

/* Shared front-end: buffer shift, FFTs, far PSD, echo, error, windowed-error.
 * Identical for PBFDAF and PBFDKF (matches PBFDAF.process).
 * Returns far_hop_energy (the W-update gate). curr_p stored in *out_curr_p. */
static float pbfdaf_frontend(PBFDAF* p,
                              const float* near_end, const float* far_end,
                              float* output, int* out_curr_p,
                              const float* precomputed_far_cmag2) {
    int hop = p->hop_size, blk = p->block_size, K = p->n_freqs;

    memmove(p->near_buffer, p->near_buffer + hop, (size_t)(blk - hop) * sizeof(float));
    memcpy(p->near_buffer + (blk - hop), near_end, (size_t)hop * sizeof(float));
    memmove(p->far_buffer, p->far_buffer + hop, (size_t)(blk - hop) * sizeof(float));
    memcpy(p->far_buffer + (blk - hop), far_end, (size_t)hop * sizeof(float));

    /* near_spec: consumed by the main filter only; the shadow's copy is never
     * read, so skip it in lightweight (shadow) mode — byte-equal. */
    if (!p->lightweight)
        rfft_padded(p, p->near_buffer, blk, p->near_spec);
    /* far_spec: identical between main + shadow (same far_buffer). When the
     * caller already computed it (shadow runs pre-main), reuse it instead of a
     * redundant rfft — byte-equal. One-shot. */
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

    /* |far_spec[k]|² for this hop, computed once and kept.
     *
     * The destination is the persistent far_cmag2_hold rather than scratch:
     * the orchestrator reads the same magnitudes for its RSA update (before
     * this hop's frontend, so the previous spectrum) and its stationarity
     * refresh (after, so this one), and a field written and zeroed in lockstep
     * with far_spec gives both the value they would have recomputed.
     *
     * There is deliberately no far-power EMA here: nothing reads such a
     * `power` state -- not the gain, not the noise gate, not the orchestrator,
     * not a debug accessor -- so maintaining one would cost a cold-start
     * branch, two K-wide reductions and a K-wide combine on every hop of both
     * filters for no consumer. */
    float *far_cmag2 = p->far_cmag2_hold;   /* [n_freqs], persists to the orchestrator */
    if (borrowed_far && precomputed_far_cmag2)
        memcpy(far_cmag2, precomputed_far_cmag2, (size_t)K * sizeof(float));
    else
        sk_cmag2_np_f32(p->far_spec, far_cmag2, K);
    /* X_buf's row curr_p was just set from this same far_spec, so this array
     * IS |X_buf[curr_p]|². Mirror it once here rather than have both X² sums
     * re-derive every partition's magnitudes from the complex history. */
    memcpy(p->x2_cache + (size_t)curr_p * K, far_cmag2, (size_t)K * sizeof(float));

    /* echo_spec = Σ_p W[p] * X_buf[(curr_p-p)%N] */
    memset(p->echo_spec, 0, (size_t)K * sizeof(Complex));
    /* numpy complex64 mul uses FMA on arm64: re=fma(wr,xr,-(wi*xi)),
     * im=fma(wr,xi, wi*xr). (Verified 0/300000 vs naive's 25%.) The ring
     * kernel adds each partition's product in partition order, exactly as
     * one sk_cmac_fma_f32_scalar per partition does. */
    sk_cmac_fma_ring_f32(p->echo_spec, p->W, p->X_buf, curr_p,
                        p->n_partitions, K);

    /* IFFT echo, output = near[-hop:] - echo_time[hop:block] */
    fft_inverse(p->fft, p->echo_spec, p->time_scratch);
    for (int i = 0; i < hop; ++i)
        output[i] = p->near_buffer[blk - hop + i] - p->time_scratch[hop + i];

    /* Subtractor-output s_*_max_abs = max|echo_time[hop:block]| on the
     * valid hop. (np.max(np.abs(.)) over hop floats.) */
    p->last_s_max_abs = sk_max_abs_f32(p->time_scratch + hop, hop);

    /* error_spec: zero-pad, valid region [hop, block) = output */
    memset(p->time_scratch, 0, (size_t)hop * sizeof(float));
    memcpy(p->time_scratch + hop, output, (size_t)hop * sizeof(float));
    if (p->fft_size > blk)
        memset(p->time_scratch + blk, 0, (size_t)(p->fft_size - blk) * sizeof(float));
    fft_forward_scratch(p->fft, p->time_scratch, p->error_spec);

    /* windowed near (sqrt-Hann × near_buffer) − echo_spec. RES analysis = main
     * filter only; the shadow's error_spec_windowed is never read, so skip it in
     * lightweight (shadow) mode — byte-equal. */
    if (!p->lightweight) {
        for (int i = 0; i < blk; ++i)
            p->time_scratch[i] = p->near_buffer[i] * p->sqrt_hann[i];
        rfft_padded(p, p->time_scratch, blk, p->spec_scratch);
        sk_csub_f32(p->error_spec_windowed, p->spec_scratch, p->echo_spec, K);
    }

    /* far_hop_energy = np.sum(far_end**2) / hop. Reproduce np.sum (pairwise f32)
     * over far_end**2, then divide in float32. */
    {
        float *fsq = p->scr_fsq;   /* instance scratch, not stack: [hop_size] */
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
        /* === PBFDAF NLMS _update_weights (coarse_filter_update_gain.cc) === */
        /* Increment first, unconditionally: filters.py's _update_weights ticks
         * _call_counter before it even looks at mu_scale, so a hop with an
         * all-zero mu (and, same as below, a saturated or startup/poor-
         * excitation hop) still advances the counter. Floored at LONG_MAX to
         * avoid signed overflow. call_counter's raw value is observable
         * (test_counter_saturation.c reads it) and must track Python's
         * unbounded counter, so capping at N / the startup threshold
         * -- which would still satisfy the `<= N` gate below -- would desync the raw
         * value on the next call (same constraint as erl_estimator.c's
         * hold_counter_time_domain). Capping at LONG_MAX changes nothing for any
         * practically reachable call count. */
        if (p->call_counter < LONG_MAX) p->call_counter += 1;
        const float* mu_arr = mu_scale_arr;
        int any_pos;
        if (mu_arr == NULL) {
            /* mu_local would be filled uniformly with mu_scale_scalar, so
             * 'any element > 0' degenerates to just the scalar itself --
             * skip the K-length fill and scan entirely when it's not. */
            any_pos = (mu_scale_scalar > 0.0f);
            if (!any_pos) { p->partition_idx = (p->partition_idx + 1) % N; return; }
            {
                float* mu_local = p->scr_mu_local;   /* instance scratch, not stack: [n_freqs] */
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
        /* (RenderSignalAnalyzer mask omitted: production shadow has the same
         * RSA; the parity harness captures the post-mask mu_arr so the mask is
         * already folded in.) */

        /* x2_partition_sum = (np.abs(X_buf)**2).sum(axis=0) — cmag2_np, summed
         * sequentially over partitions (n=6<8 → numpy axis-0 sequential).
         * The magnitudes come from x2_cache, which holds exactly the floats
         * cmag2_np would return for each row; zero-init then one add per
         * element per partition keeps the summation order identical. */
        float *x2psum = p->scr_x2psum;   /* instance scratch, not stack: [n_freqs] */
        memcpy(x2psum, p->x2_cache, (size_t)K * sizeof(float));   /* part 0 */
        for (int part = 1; part < N; ++part) {
            const float* x2p = p->x2_cache + (size_t)part * K;
            for (int k = 0; k < K; ++k) x2psum[k] += x2p[k];
        }
        float mu_initial_boost = p->initial_state_active ? (0.95f / 0.7f) : 1.0f;
        float ng_thr = (float)AEC_FILTER_NOISE_GATE_POWER_FLOAT;
        float *mu_eff = p->scr_mu_eff;   /* instance scratch, not stack: [n_freqs] */
        for (int k = 0; k < K; ++k) {
            float denom = x2psum[k] + p->delta;
            float m = (p->mu * mu_initial_boost * mu_arr[k]) / denom;
            mu_eff[k] = (x2psum[k] >= ng_thr) ? m : 0.0f;
        }
        for (int part = 0; part < N; ++part) {
            int p_idx = curr_p - part; while (p_idx < 0) p_idx += N;
            Complex* Wp = p->W + (size_t)part * K;
            const Complex* Xp = p->X_buf + (size_t)p_idx * K;
            /* grad = error_spec * conj(X). conj(X)=(xr,-xi); complex64 mul
             * uses numpy FMA form re=fma(er,cxr,-(ei*cxi)),
             * im=fma(er,cxi, ei*cxr). mu_eff×grad is real×complex (no FMA).
             * Matches sk_wupdate_nlms_f32's scalar reference exactly. */
            sk_wupdate_nlms_f32(Wp, Xp, p->error_spec, mu_eff, K);
            if (p->enable_td_constraint &&
                (!p->constraint_round_robin || part == p->partition_to_constrain)) {
                /* Wp is rewritten in full by the forward transform below and
                 * nothing reads it in between, so the inverse may consume it
                 * instead of staging a private copy. */
                fft_inverse_scratch(p->fft, Wp, p->time_scratch);
                /* td_window[i]==1.0f for i<hop_size (pbfdaf_fill_windows) --
                 * x*1.0f leaves every float bit pattern unchanged, so only the
                 * [hop_size, fft_size) half (where td_window==0.0f) needs the
                 * multiply. */
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

/* Warm tap-transfer: shift the learned IR by `shift_samples` -- positive =
 * LEFT (toward tap 0, the aligned far advanced), negative = RIGHT (the
 * aligned far retarded), zero-filling the freed end -- instead of zeroing
 * the filter, so the cold-start cancellation survives a delay realign.
 * Mirrors filters.py: raw irfft per partition (first hop taps) ->
 * concatenate -> left-shift by s -> rfft each hop block (zero-padded to
 * fft_size) back into W[p]. NO td_window -- Python uses a raw irfft here
 * (unlike the TD-constraint round-trip). Uses p->scr_ir instance scratch,
 * not stack -- sized exactly n_partitions*hop_size at init (see pbfdkf.h
 * PBFDAF comment).
 *
 * Owns the far-history decision too, same as filters.py's Filter.warm_shift_ir
 * (`if s < 0: self.X_buf.fill(0)`): a RETARD leaves the held far history
 * ahead of the realigned stream, so pbfdaf_clear_far_history() runs below for
 * that direction -- one spelling, not one per caller, which matches
 * Python's ownership and covers any future call site for free.
 *
 * Returns 0 (applied) / -1 (refused). The span guard below is REACHABLE from
 * a legal config (n_partitions is admissible up to 256 and hop up to 512), and
 * a silent refusal is indistinguishable from success to the caller: the taps
 * would stay at the pre-realign alignment while the caller believes the shift
 * happened. Reporting it lets the caller fall back to a filter reset. */
int pbfdaf_warm_shift_ir(PBFDAF* p, int shift_samples) {
    int s = shift_samples;
    if (s == 0) return 0;
    int hop = p->hop_size, nF = p->fft_size, nP = p->n_partitions, K = p->n_freqs;
    int total = nP * hop;
    if (total <= 0 || total > 4096) return -1;   /* safety; real max ~2080 */
    if (s > total) s = total;
    if (s < -total) s = -total;
    float *ir = p->scr_ir;
    pbfdaf_get_time_domain_filter(p, ir);
    if (s > 0) {
        for (int i = 0; i < total - s; ++i) ir[i] = ir[i + s]; /* read-ahead safe */
        for (int i = total - s; i < total; ++i) ir[i] = 0.0f;
    } else {
        /* Negative shift: the aligned far stream RETARDED by -s samples (the
         * external delay shrank), so the learned response moves toward later
         * taps; walk backwards so the copy is read-ahead safe. */
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

/* ===== PBFDKF ============================================================ */

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
    /* Measurement-noise PSD EMA, per hop: 10 ms reference, TC 194.96 ms. Mirrors
     * filters.py. */
    p->alpha_r = aec_growth_rehop(0.95f, 160, 16000,
                                  p->base.hop_size, p->base.sample_rate);
    p->h_error_floor = (float)AEC_H_ERROR_FLOOR_FLOAT;
    p->h_error_ceil  = (float)AEC_H_ERROR_CEIL_FLOAT;
    /* The AEC_LEAKAGE_*_PER_HOP #defines pre-bake the x2.5 factor for the
     * legacy hop=160/sr=16000 grid and are never recomputed for the filter's
     * actual hop_size/sample_rate. Rescaled live here via the same
     * per_block_rate_to_per_hop helper filters.py uses (raw per-block
     * rates 5e-5/5e-2/5e-3/5e-1). Mirrors filters.py's PBFDKF
     * leakage_converged/_diverged/_converged_transient/_diverged_transient. */
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
    /* Refined-filter gates (orchestrator overrides poor_excitation per config). */
    p->base.poor_excitation_counter = 400;
}

size_t pbfdkf_get_mem_size(int block_size, int n_partitions, int hop_size) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    int blk, fft, K;
    pbfdaf_compute_sizes(hop, NULL, &blk, &fft, &K);
    (void)blk; (void)fft;
    if (n_partitions <= 0 || K <= 0) return 0;
    size_t total = pbfdaf_get_mem_size(block_size, n_partitions, hop_size, 0);
    if (total == 0) return 0;   /* base PBFDAF sizing rejected its own inputs */
    /* Checked size arithmetic: 12 identically-sized K-length float fields
     * (Q_high, Q_low, Q, R, error_psd, H_error_per_bin, erl_per_bin,
     * e2_coarse_per_bin, e2_ref_scratch, scr_mu_local, scr_X2, scr_mu_aec),
     * each individually ALIGN16'd by pbfdkf_init_static below
     * (mem_align.h ck_field_size_reps composes
     * ck_mul_size/ck_align16_size/ck_add_size with the same
     * saturate-to-SIZE_MAX-on-overflow behavior). */
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
    /* Reject a misaligned pool base before any pool write (see the matching
     * guard in pbfdaf_init_static above). */
    if (!MEM_IS_ALIGNED16(mem)) return -1;
    {
        /* pbfdkf_get_mem_size returning 0 means "invalid"; same 0-vs-`mem_size < 0`
         * reasoning as pbfdaf_init_static above. */
        size_t need = pbfdkf_get_mem_size(block_size, n_partitions, hop_size);
        if (need == 0 || mem_size < need) return -1;
    }
    size_t base_sz = pbfdaf_get_mem_size(block_size, n_partitions, hop_size, 0);
    memset(p, 0, sizeof(*p));
    if (pbfdaf_init_static(&p->base, mem, base_sz,
                       block_size, n_partitions, mu, delta, hop_size, 0,
                       sample_rate, shared_fft) != 0)
        return -1;   /* base filter's pool base/size checks rejected the inputs */
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
    /* aec_carve()'s value, not pbfdaf_init_scalars()'s 0 / pbfdkf_init_scalars()'s
     * reference-grid 400: those are both overwritten at construction. Same
     * expression as pbfdkf_handle_echo_path_change above. */
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
    /* pbfdkf_rearm_kalman (inside pbfdkf_reset) re-arms R/error_psd/Q but
     * NOT H_error: the mid-stream relock path gets it from the
     * pbfdkf_handle_echo_path_change(delay_change=1) that follows at each of
     * its call sites. Same value that path and pbfdkf_init_scalars write. */
    for (k = 0; k < K; ++k)
        p->H_error_per_bin[k] = (float)AEC_H_ERROR_INIT_FLOAT;
}

void pbfdkf_handle_echo_path_change(PBFDKF* p, int delay_change, int gain_change) {
    /* CoarseFilterUpdateGain M3 counter reset (not-gain_change): set to the
     * rate-scaled value aec.c's construction path sets via the identical
     * aec_blocks_to_hops(1000, hop, sr) call, not to the reference-grid
     * constant AEC_POOR_EXC_COUNTER_INITIAL_HOPS (400). Mirrors filters.py's
     * _poor_excitation_counter. */
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

/* Concatenate irfft(W[p])[:hop] for partitions [first_partition,
 * first_partition+n_parts) → out[part*hop .. part*hop+hop-1], indexed against
 * the SAME full [n_partitions × hop] layout the whole-filter call produces.
 * Taps belonging to partitions outside the requested range are not written
 * (they keep whatever the caller's buffer already held). The range is clamped
 * to [0, n_partitions), so an out-of-range request degrades to the overlap
 * rather than reading past W. */
void pbfdaf_get_time_domain_filter_range(PBFDAF* p, int first_partition,
                                         int n_parts, float* out) {
    int hop = p->hop_size;
    int first = first_partition;
    int last  = first_partition + n_parts;     /* exclusive */
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

/* W *= np.complex64(scale): scalar (real) value-cast to f32, multiply each
 * complex component in f32. */
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

/* H_error leakage refresh + clamp.
 * Runs on every hop (gated path + after active decay). */
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
        /* per-bin path. e2_ref_bins[k] = np.abs(error_spec[k])**2 (current
         * frame), precomputed once by the caller (pbfdkf_process) and shared
         * with the error_psd EMA loop + pbfdkf_update_weights's mu gate
         * -- all three read the same untouched b->error_spec this hop, so
         * recomputing cmag2_np() per call site here would be pure waste. */
        int n_div = sk_h_error_refresh_f32(
            p->H_error_per_bin, e2_ref_bins, p->e2_coarse_per_bin,
            p->erl_per_bin, lc_eff, ld_eff, p->h_error_refresh_erl_floor,
            p->disallow_leakage_diverged, K);
        /* Diag: fraction of bins on diverged-leakage (np.mean(~mask)). */
        p->last_leakage_div_frac = (float)n_div / (float)K;
    } else {
        /* scalar fallback: e2_ref_sum = sum(error_psd) (float32); e2_coa scalar.
         * (Not exercised in production: orchestrator sets e2_coarse_per_bin
         * every hop. Kept pairwise-correct for completeness.) */
        float e2_ref_sum = sk_pairwise_sum_tailfold_f32(p->error_psd, (size_t)K);
        float e2_coa_sum = p->e2_coarse_for_refresh;
        int use_conv = (e2_ref_sum <= e2_coa_sum) || p->disallow_leakage_diverged;
        /* scalar path is all-or-nothing → frac is 0.0 or 1.0. */
        p->last_leakage_div_frac = use_conv ? 0.0f : 1.0f;
        float leakage = use_conv ? lc_eff : ld_eff;
        for (int k = 0; k < K; ++k)
            p->H_error_per_bin[k] = p->H_error_per_bin[k] +
                                    leakage * p->erl_per_bin[k];
    }
    /* np.clip(H_error, floor, ceil): low bound checked first, then high. */
    sk_clip_f32(p->H_error_per_bin, p->h_error_floor, p->h_error_ceil, K);
}

/* Per-bin Kalman gain via H_error (the reference refined-filter update
 * gain). mu_scale_arr is already post-RSA. */
static void pbfdkf_update_weights(PBFDKF* p, int curr_p,
                                  const float* mu_scale_arr,
                                  const float* e2_ref_bins) {
    PBFDAF* b = &p->base;
    int K = b->n_freqs, N = b->n_partitions;

    /* X² source. Hybrid startup: partition-sum once call_counter past startup.
     * Python: X2 = (np.abs(X_buf)**2).sum(axis=0)  [partition-sum, n=6<8 →
     * sequential over part 0..N-1] or (np.abs(X_latest)**2). Both read
     * x2_cache, whose rows are the cmag2_np results for the matching X_buf
     * rows, produced once each as the row entered history. */
    float *X2 = p->scr_X2;   /* instance scratch, not stack: [n_freqs] */
    if (b->call_counter > p->partition_sum_x2_startup_hops) {
        memcpy(X2, b->x2_cache, (size_t)K * sizeof(float));   /* part 0 */
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

    /* mu[k] = H_error / (0.5·H_error·X² + n·E²_refined + δ), gated by X²≥gate.
     * e2_ref_bins[k] = |error_spec[k]|² precomputed once by pbfdkf_process
     * (same value cmag2_np(er,ei) would give here -- error_spec is untouched
     * since pbfdaf_frontend ran at the top of this hop). */
    float *mu_aec = p->scr_mu_aec;   /* instance scratch, not stack: [n_freqs] */
    for (int k = 0; k < K; ++k) {
        float e2_ref = e2_ref_bins[k];
        float He = p->H_error_per_bin[k];
        float denom = 0.5f * He * X2[k] + n_part * e2_ref + delta32;
        float m = He / denom;
        mu_aec[k] = (X2[k] >= ng_thr) ? m : 0.0f;
    }

    /* W[p] += (mu·conj(X[p]))·mu_scale·error_spec. */
    for (int part = 0; part < N; ++part) {
        int p_idx = curr_p - part; while (p_idx < 0) p_idx += N;
        const Complex* Xp = b->X_buf + (size_t)p_idx * K;
        Complex* Wp = b->W + (size_t)part * K;
        /* K = mu*conj(X); K_scaled = K*mu_scale_arr; W += K_scaled*error_spec
         * (complex64×complex64 → numpy FMA form re=fma(ksr,er,-(ksi*ei)),
         * im=fma(ksr,ei, ksi*er)). Matches sk_wupdate_kf_f32's scalar
         * reference exactly. */
        sk_wupdate_kf_f32(Wp, Xp, b->error_spec, mu_aec, mu_scale_arr, K);
        if (b->enable_td_constraint &&
            (!b->constraint_round_robin || part == b->partition_to_constrain)) {
            /* Wp is rewritten in full by the forward transform below (see the
             * PBFDAF constraint), so the inverse may consume it. */
            fft_inverse_scratch(b->fft, Wp, b->time_scratch);
            for (int i = b->hop_size; i < b->fft_size; ++i) b->time_scratch[i] *= b->td_window[i];
            fft_forward_scratch(b->fft, b->time_scratch, Wp);
        }
    }
    if (b->enable_td_constraint && b->constraint_round_robin)
        b->partition_to_constrain = (b->partition_to_constrain + 1) % N;
    b->w_dirty = 1;

    /* Scale mu_aec by mu_scale_arr before the H_error decay so that masked
     * bins (RSA narrowband mask + DT scale) do not decay while their W is
     * frozen. H_error decay (active branch only):
     * H_error -= 0.5·mu·X²·H_error. mu_aec is fully recomputed at the top of
     * every hop, so fusing the scale into this loop (instead of writing the
     * scaled value back into mu_aec[k]) changes nothing a later reader would
     * see. */
    for (int k = 0; k < K; ++k) {
        float ms = mu_aec[k] * mu_scale_arr[k];
        p->H_error_per_bin[k] -= 0.5f * ms * X2[k] * p->H_error_per_bin[k];
    }

    /* always-on refresh + clamp. */
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
        /* === PBFDKF._update_weights ============================== */
        float *mu_local = p->scr_mu_local;   /* instance scratch, not stack: [n_freqs] */
        const float* mu_arr = mu_scale_arr;
        if (mu_arr == NULL) {
            for (int k = 0; k < K; ++k) mu_local[k] = mu_scale_scalar;
            mu_arr = mu_local;
        }
        /* e2_ref[k] = |error_spec[k]|², computed once here and shared by the
         * error_psd EMA below, pbfdkf_update_weights's mu gate, and
         * pbfdkf_h_error_refresh's per-bin branch (all 4 call sites in this
         * function, including both refresh-only early-returns below): all of
         * them read the SAME b->error_spec, set once by pbfdaf_frontend above
         * and untouched for the rest of this hop. */
        float *e2_ref = p->e2_ref_scratch;   /* instance scratch, not stack */
        sk_cmag2_np_f32(b->error_spec, e2_ref, K);
        error_cmag2_valid = 1;

        /* startup / poor-excitation / saturation gates → refresh only.
         * call_counter is floored at LONG_MAX, as in pbfdaf_process's increment
         * above; the raw value on the PBFDKF/main-filter path is part of the
         * counter-saturation contract (test_counter_saturation.c). */
        if (b->call_counter < LONG_MAX) b->call_counter += 1;
        if (b->call_counter <= N || b->poor_excitation_counter < N ||
            b->saturated_capture) {
            pbfdkf_h_error_refresh(p, e2_ref);
            b->partition_idx = (b->partition_idx + 1) % N;
            return error_cmag2_valid;
        }
        /* stationary-far gate → refresh only. */
        if (b->block_stationary) {
            pbfdkf_h_error_refresh(p, e2_ref);
            b->partition_idx = (b->partition_idx + 1) % N;
            return error_cmag2_valid;
        }
        /* (RSA mask folded into captured mu_arr — see header note.) */

        /* error_psd EMA + R = max(error_psd, delta).
         * error_psd = |error_spec|**2;
         * _error_psd = _alpha_r*_error_psd + (1-_alpha_r)*error_psd.
         * float32 throughout.
         *
         * Grid-retimed at init; the direct-PBFDKF scalar fallback of the
         * H_error refresh consumes this value. Never a literal here. */
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
