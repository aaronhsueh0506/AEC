/* pbfdkf.c — PBFDAF base (NLMS) + PBFDKF derived (v3.22 per-bin H_error Kalman).
 *
 * 1:1 port of python/modules/filters.py. Numerical-fidelity notes:
 *  - The FFT primitive (fft_pocketfft.c, vendored numpy-1.26.4 pocketfft) is now
 *    BIT-EXACT vs np.fft (parity_fft.c: 0 mismatches). FFT-derived arrays
 *    (error_spec / echo_spec / W after the TD-constraint FFT round-trip) are
 *    still rtol<1e-4 — but that residual is pbfdkf's own fp32 accumulator drift
 *    over the case, NOT the FFT. See parity_pbfdkf.c.
 *  - The H_error / mu / leakage-refresh ARITHMETIC is reproduced op-for-op in
 *    the same dtype/order as Python (parity rules in
 *    project_c_port_parity_rules.md), so it is exact GIVEN identical inputs.
 *    But H_error's gain denom consumes |error_spec|² and |X|² (FFT outputs)
 *    and error_psd accumulates |error_spec|², so H_error / error_psd / R are
 *    rtol-bounded (<1e-4), not bit-exact. Only the integer control state
 *    (call_counter / partition_idx / initial_state) is bit-exact.
 *  - np.float32 ⇒ float; np.complex64 ⇒ Complex. float32-by-design: the old
 *    "scalar pyfloat×f32 promotes through double" parity nuance is retired —
 *    Python bit-exact parity is no longer a goal, so scalar promotions here
 *    are done directly in float32 (single literals get the `f` suffix).
 *
 * v3.10 → v3.22 change: the old P-denominator Kalman body (P/Q/R Riccati +
 * KX-blend + P_floor/p_max clamps) is GONE. The active update is the AEC3
 * per-bin H_error gain + always-on leakage refresh.
 */
#include "pbfdkf.h"
#include "aec3_scale.h"
#include "aec_simd_kernels.h"
#include <limits.h>   /* LONG_MAX */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI_AEC
#define M_PI_AEC 3.14159265358979323846f
#endif

static int next_pow2(int x) { int n = 1; while (n < x) n <<= 1; return n; }

/* Checked equivalent of `t += ALIGN16(count*elem_size) * reps` — reps
 * identically-sized fields laid out back-to-back (each individually
 * ALIGN16'd by the matching init_static's pool-carve, so their sum is
 * exactly reps * ALIGN16(count*elem_size)). Saturates to SIZE_MAX on
 * overflow via mem_align.h's ck_* helpers, same as ck_field_size.
 * (Local twin of the same helper in aec.c — this file has no shared header
 * to hang a cross-TU static on, matching this file's existing per-TU-local
 * static-helper convention, e.g. next_pow2 above.) */
static size_t ck_field_size_reps(size_t total, size_t count, size_t elem_size,
                                  size_t reps) {
    size_t one = ck_align16_size(ck_mul_size(count, elem_size));
    return ck_add_size(total, ck_mul_size(one, reps));
}

/* pairwise float32 sum matching numpy's reduction EXACTLY: an 8-accumulator
 * unrolled leaf for n<=128 (numpy NPY_PW_BLOCKSIZE), then a recursive split at
 * n/2 rounded down to a multiple of 8. Verified 0/5000 mismatch vs np.sum over
 * f32 across lengths {80,160,257,320,512}. Used for np.sum(far_end**2).
 * Body delegates to simd_kernels.h's sk_pairwise_sum_tailfold_f32, which
 * reproduces this exact tree (verbatim source for that kernel's `_scalar`
 * twin) plus a bit-identical NEON path; local symbol/signature kept so call
 * sites are unchanged. */
static float pairwise_sum_f32(const float* a, int n) {
    return sk_pairwise_sum_tailfold_f32(a, (size_t)n);
}

/* numpy |complex64|**2 (np.abs(z)**2) is now computed exclusively via the
 * sk_cmag2_np_f32/sk_ema_cmag2_f32/sk__cmag2_np_elem kernel family
 * (aec_simd_kernels.h) -- this file's own former scalar `cmag2_np(er, ei)`
 * helper (a scaled-hypot with an fmaf, bit-exact to those kernels for all
 * finite inputs: they differ only in the sign of an intermediate zero, which
 * never survives into the squared result) is retired now that its last call
 * site (pbfdaf_frontend's far_psd_sum loop) was folded into the
 * sk_cmag2_np_f32 call just below it. Referenced by name only in comments
 * below (e2_ref_scratch dedup) for historical context. */

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
    int fade_len = p->hop_size / 4;
    memset(p->td_window, 0, (size_t)p->fft_size * sizeof(float));
    /* E1 fix: all causal taps [0:hop] preserved at 1.0.  Prior code placed the
     * descending fade in [hop-fade_len:hop], suppressing the last 40 causal taps
     * → dead zone in every partition of the 832-tap IR. */
    for (int i = 0; i < p->hop_size; ++i) p->td_window[i] = 1.0f;
    /* Non-causal fade [hop:hop+fade_len]: descending from ~1.0 to 0.0.
     * float32-by-design (Python double parity retired). */
    for (int i = 0; i < fade_len; ++i) {
        int idx = p->hop_size + i;
        if (idx < p->fft_size) {
            float v = 0.5f * (1.0f - cosf(M_PI_AEC * (float)(fade_len - 1 - i) / (float)fade_len));
            p->td_window[idx] = v;
        }
    }
    /* [hop+fade_len, fft_size) stays 0. */
    for (int i = 0; i < p->block_size; ++i) {
        float h = 0.5f * (1.0f - cosf(2.0f * M_PI_AEC * (float)i / (float)p->block_size));
        p->sqrt_hann[i] = sqrtf(h);
    }
}

static void pbfdaf_init_scalars(PBFDAF* p, int n_partitions, float mu,
                                float delta, int sample_rate) {
    p->n_partitions = n_partitions;
    p->sample_rate = sample_rate;
    p->mu = mu;
    p->delta = delta;
    p->alpha_power = 0.9f;
    p->enable_td_constraint = 1;
    p->partition_idx = 0;
    p->call_counter = 0;
    p->poor_excitation_counter = 0;
    p->saturated_capture = 0;
    p->block_stationary = 0;
    p->initial_state_active = 1;
    p->initial_state_active_render_hops = 0;
    /* AEC3 kInitialStateSeconds-equivalent threshold; was a bare literal 250
     * (= 2.5s only at the legacy hop=160/sr=16000 grid). Rescaled via
     * aec3_ms_to_hops so the window stays 2.5s at any grid. Mirrors
     * filters.py's PBFDAF._initial_state_threshold_hops. */
    p->initial_state_threshold_hops =
        aec3_ms_to_hops(2500.0f, p->hop_size, sample_rate);
    p->initial_state_far_energy_floor = 1e-4f;
    p->last_initial_state_active = 1;
    p->last_s_max_abs = 0.0f;
    p->lightweight = 0;
    p->precomputed_far_spec = NULL;
    p->constraint_round_robin = 0;
    p->partition_to_constrain = 0;
}

static void pbfdaf_zero_state(PBFDAF* p) {
    int Wsz = p->n_partitions * p->n_freqs;
    memset(p->W,     0, (size_t)Wsz * sizeof(Complex));
    memset(p->X_buf, 0, (size_t)Wsz * sizeof(Complex));
    memset(p->near_buffer, 0, (size_t)p->block_size * sizeof(float));
    memset(p->far_buffer,  0, (size_t)p->block_size * sizeof(float));
    memset(p->power, 0, (size_t)p->n_freqs * sizeof(float));
    memset(p->near_spec,  0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->echo_spec,  0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->error_spec, 0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->far_spec,   0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->error_spec_windowed, 0, (size_t)p->n_freqs * sizeof(Complex));
    memset(p->time_scratch, 0, (size_t)p->fft_size * sizeof(float));
    memset(p->spec_scratch, 0, (size_t)p->n_freqs * sizeof(Complex));
}

/* ===== PBFDAF ============================================================ */

int pbfdaf_init(PBFDAF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size,
                    int with_process_scratch, int sample_rate) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    pbfdaf_compute_sizes(hop, &p->hop_size, &p->block_size,
                         &p->fft_size, &p->n_freqs);
    /* Set early (before any malloc below) so pbfdaf_free(p) is safe to call
     * from the OOM path even if the very first allocation fails -- it reads
     * p->is_static to pick its heap-vs-static branch, and that must never
     * be uninitialized garbage. */
    p->is_static = 0;

    p->td_window = (float*)calloc((size_t)p->fft_size, sizeof(float));
    p->sqrt_hann = (float*)malloc((size_t)p->block_size * sizeof(float));

    int Wsz = p->n_partitions = n_partitions;
    Wsz = p->n_partitions * p->n_freqs;
    p->W     = (Complex*)calloc((size_t)Wsz, sizeof(Complex));
    p->X_buf = (Complex*)calloc((size_t)Wsz, sizeof(Complex));

    p->near_buffer = (float*)calloc((size_t)p->block_size, sizeof(float));
    p->far_buffer  = (float*)calloc((size_t)p->block_size, sizeof(float));
    p->power       = (float*)calloc((size_t)p->n_freqs, sizeof(float));

    p->near_spec   = (Complex*)calloc((size_t)p->n_freqs, sizeof(Complex));
    p->echo_spec   = (Complex*)calloc((size_t)p->n_freqs, sizeof(Complex));
    p->error_spec  = (Complex*)calloc((size_t)p->n_freqs, sizeof(Complex));
    p->far_spec    = (Complex*)calloc((size_t)p->n_freqs, sizeof(Complex));
    p->error_spec_windowed = (Complex*)calloc((size_t)p->n_freqs, sizeof(Complex));

    p->fft         = fft_create(p->fft_size);
    p->time_scratch = (float*)calloc((size_t)p->fft_size, sizeof(float));
    p->spec_scratch = (Complex*)calloc((size_t)p->n_freqs, sizeof(Complex));

    /* De-stacked per-hop scratch (see pbfdkf.h PBFDAF comment). */
    p->scr_fsq      = (float*)malloc((size_t)p->hop_size * sizeof(float));
    p->scr_mu_local = with_process_scratch
        ? (float*)malloc((size_t)p->n_freqs * sizeof(float)) : NULL;
    p->scr_x2psum   = with_process_scratch
        ? (float*)malloc((size_t)p->n_freqs * sizeof(float)) : NULL;
    p->scr_mu_eff   = with_process_scratch
        ? (float*)malloc((size_t)p->n_freqs * sizeof(float)) : NULL;
    p->scr_ir       = (float*)malloc((size_t)p->n_partitions * (size_t)p->hop_size * sizeof(float));
    p->scr_e2       = (float*)malloc((size_t)p->n_freqs * sizeof(float));

    /* F04 (extending the existing fft_create() OOM
     * convention to every malloc'd/calloc'd scratch field above): any of
     * these can return NULL under real OOM, and pbfdaf_fill_windows() below
     * writes straight through p->td_window/p->sqrt_hann unconditionally --
     * a NULL td_window/sqrt_hann would be an immediate NULL-deref crash
     * before we ever reach the old `if (!p->fft)` check. Check every
     * pointer BEFORE calling pbfdaf_init_scalars()/pbfdaf_fill_windows(),
     * not after. scr_mu_local/scr_x2psum/scr_mu_eff are only required
     * non-NULL when with_process_scratch requested them (they're
     * deliberately NULL otherwise -- not an OOM). pbfdaf_free(p) tolerates
     * a partially-NULL struct (free(NULL) is a no-op for every field, and
     * `if (p->fft) fft_destroy(p->fft);` already guards the one non-free
     * teardown call). */
    if (!p->td_window || !p->sqrt_hann || !p->W || !p->X_buf ||
        !p->near_buffer || !p->far_buffer || !p->power ||
        !p->near_spec || !p->echo_spec || !p->error_spec || !p->far_spec ||
        !p->error_spec_windowed || !p->fft || !p->time_scratch ||
        !p->spec_scratch || !p->scr_fsq ||
        (with_process_scratch &&
         (!p->scr_mu_local || !p->scr_x2psum || !p->scr_mu_eff)) ||
        !p->scr_ir || !p->scr_e2) {
        pbfdaf_free(p);
        return -1;
    }

    pbfdaf_init_scalars(p, n_partitions, mu, delta, sample_rate);
    pbfdaf_fill_windows(p);
    return 0;
}

size_t pbfdaf_get_mem_size(int block_size, int n_partitions, int hop_size,
                           int with_process_scratch) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    int blk, fft, K;
    pbfdaf_compute_sizes(hop, NULL, &blk, &fft, &K);
    if (n_partitions <= 0 || K <= 0) return 0;
    size_t Wsz = ck_mul_size((size_t)n_partitions, (size_t)K);

    /* Checked size arithmetic (F05): saturating ck_* helpers (mem_align.h);
     * MEM_SIZE_INVALID(total) at the end reports failure (0) on overflow
     * instead of a small wrapped total. Field order/grouping unchanged from
     * the previous unchecked `total += ALIGN16(...)` walk — pbfdaf_init_
     * static's carve order must stay in lockstep with this one. */
    size_t total = 0;
    total = ck_field_size(total, (size_t)fft, sizeof(float));   /* td_window */
    total = ck_field_size(total, (size_t)blk, sizeof(float));   /* sqrt_hann */
    total = ck_field_size(total, Wsz, sizeof(Complex));         /* W */
    total = ck_field_size(total, Wsz, sizeof(Complex));         /* X_buf */
    total = ck_field_size(total, (size_t)blk, sizeof(float));   /* near_buffer */
    total = ck_field_size(total, (size_t)blk, sizeof(float));   /* far_buffer */
    total = ck_field_size(total, (size_t)K,   sizeof(float));   /* power */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* near_spec */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* echo_spec */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* error_spec */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* far_spec */
    total = ck_field_size(total, (size_t)K,   sizeof(Complex)); /* error_spec_windowed */
    total = ck_add_size(total, fft_get_mem_size(fft));          /* nested FFT */
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
    total = ck_field_size(total, (size_t)K,   sizeof(float));   /* scr_e2 (also frontend's far_cmag2 scratch -- D2) */

    if (MEM_SIZE_INVALID(total)) return 0;
    return total;
}

int pbfdaf_init_static(PBFDAF* p, void* mem, size_t mem_size,
                         int block_size, int n_partitions,
                         float mu, float delta, int hop_size,
                         int with_process_scratch, int sample_rate) {
    if (!p || !mem) return -1;
    /* F07: reject a misaligned pool base before any pool write — hardens a
     * caller invoking this directly (the parent aec_carve always carves
     * ALIGN16 offsets off an already-aligned base, so this never fires
     * there). */
    if (!MEM_IS_ALIGNED16(mem)) return -1;
    {
        /* pbfdaf_get_mem_size returning 0 means "invalid" (F05); a real
         * config's size is always > 0, so 0 is an unambiguous failure here
         * rather than `mem_size < 0`, which is never true for a size_t. */
        size_t need = pbfdaf_get_mem_size(block_size, n_partitions, hop_size,
                                          with_process_scratch);
        if (need == 0 || mem_size < need) return -1;
    }

    int hop = (hop_size > 0) ? hop_size : block_size / 2;
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
    p->power      = (float*)ptr;   ptr += ALIGN16((size_t)K * sizeof(float));
    p->near_spec  = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    p->echo_spec  = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    p->error_spec = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    p->far_spec   = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    p->error_spec_windowed = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    size_t fft_sz = fft_get_mem_size(fft);
    p->fft        = fft_init(ptr, fft_sz, fft); ptr += fft_sz;
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
    p->scr_e2       = (float*)ptr; ptr += ALIGN16((size_t)p->n_freqs * sizeof(float));
    (void)ptr;

    pbfdaf_init_scalars(p, n_partitions, mu, delta, sample_rate);
    pbfdaf_zero_state(p);
    pbfdaf_fill_windows(p);
    p->is_static = 1;

    if (!p->fft) return -1;   /* F04: nested fft_init() failed (OOM) */
    return 0;
}

void pbfdaf_free(PBFDAF* p) {
    if (!p) return;
    if (p->is_static) {
        /* F04/lifecycle: NULL the handle after destroying it so a second
         * pbfdaf_free() (e.g. via a caller's double aec_destroy()) sees
         * p->fft == NULL and does not dereference a pointer into memory the
         * pool/arena owner may since have freed. */
        if (p->fft) { fft_destroy(p->fft); p->fft = NULL; }
        return;
    }
    free(p->td_window); free(p->sqrt_hann);
    free(p->W); free(p->X_buf);
    free(p->near_buffer); free(p->far_buffer); free(p->power);
    free(p->near_spec); free(p->echo_spec); free(p->error_spec);
    free(p->far_spec); free(p->error_spec_windowed);
    free(p->time_scratch); free(p->spec_scratch);
    free(p->scr_fsq); free(p->scr_mu_local); free(p->scr_x2psum);
    free(p->scr_mu_eff); free(p->scr_ir); free(p->scr_e2);
    if (p->fft) fft_destroy(p->fft);
}

void pbfdaf_reset(PBFDAF* p) {
    int Wsz = p->n_partitions * p->n_freqs;
    memset(p->W, 0, (size_t)Wsz * sizeof(Complex));
    memset(p->X_buf, 0, (size_t)Wsz * sizeof(Complex));
    memset(p->near_buffer, 0, (size_t)p->block_size * sizeof(float));
    memset(p->far_buffer,  0, (size_t)p->block_size * sizeof(float));
    memset(p->power, 0, (size_t)p->n_freqs * sizeof(float));
    p->partition_idx = 0;
    memset(p->error_spec_windowed, 0, (size_t)p->n_freqs * sizeof(Complex));
    /* Defensive hardening (Group 6): precomputed_far_spec is documented as
     * one-shot -- set and consumed within the same pbfdaf_frontend() call,
     * cleared there immediately after the memcpy. Today that means it is
     * already NULL by the time any reset() call can observe it (reset
     * never runs mid-hop), so this has never been a live bug -- but with
     * cross-instance sharing (aec_process_context_shared_far()) now
     * setting this from outside the instance, a caller that resets an
     * instance between setting the pointer and this instance's own
     * process call would otherwise leave a stale, potentially
     * caller-freed pointer sitting on the struct. Clearing it here removes
     * that possibility entirely rather than relying on every caller to
     * get the ordering right on its own. */
    p->precomputed_far_spec = NULL;
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
         * (NE10 skips its input staging copy; KISS delegates unchanged). */
        fft_forward_scratch(p->fft, p->time_scratch, spec_out);
    }
}

/* Shared front-end: buffer shift, FFTs, power EMA, echo, error, windowed-error.
 * Identical for PBFDAF and PBFDKF (matches PBFDAF.process lines 191-248).
 * Returns far_hop_energy (the W-update gate). curr_p stored in *out_curr_p. */
static float pbfdaf_frontend(PBFDAF* p,
                              const float* near_end, const float* far_end,
                              float* output, int* out_curr_p) {
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
    if (p->precomputed_far_spec != NULL) {
        memcpy(p->far_spec, p->precomputed_far_spec, (size_t)K * sizeof(Complex));
        p->precomputed_far_spec = NULL;
    } else {
        rfft_padded(p, p->far_buffer, blk, p->far_spec);
    }

    int curr_p = p->partition_idx;
    *out_curr_p = curr_p;
    memcpy(p->X_buf + (size_t)curr_p * K, p->far_spec, (size_t)K * sizeof(Complex));

    /* power EMA (cold start: direct init when sum(power)<1e-10 and active far).
     * cmag2(far_spec[k]) computed ONCE here into instance scratch, reused for
     * both the far_psd_sum reduction and the cold-start/EMA power update
     * below (previously recomputed a 2nd time inside sk_cmag2_np_f32/
     * sk_ema_cmag2_f32, which both derive |far_spec[k]|^2 from scratch
     * internally). sk_cmag2_np_f32's ternary-based abs and cmag2_np's
     * fabsf-based abs are bit-exact equivalent for all finite inputs (they
     * differ only in the sign of an intermediate zero, which never survives
     * into the squared result) -- already relied upon by the cold-start/EMA
     * branches just below, which already call these sk_ kernels on this same
     * far_spec array.
     *
     * Borrows p->scr_e2 (normally
     * pbfdaf_get_error_energy's |error_spec|^2 scratch) instead of a
     * dedicated scr_far_cmag2 field -- cross-phase reuse, safe because this
     * whole computation (write far_cmag2, read it twice below) is fully
     * self-contained within THIS call, well before pbfdaf_get_error_energy
     * runs later in the same hop (aec.c step 13, strictly after step
     * 8.5/9's pbfdaf_process/pbfdkf_process -- i.e. after pbfdaf_frontend
     * returns) and unconditionally overwrites every element before reading
     * any of them back. Same argument aec3_post.c documents for
     * x2_at_delay's cross-phase reuse. */
    float *far_cmag2 = p->scr_e2;   /* instance scratch, not stack: [n_freqs] */
    sk_cmag2_np_f32(p->far_spec, far_cmag2, K);
    float pwr_sum = 0.0f;
    for (int k = 0; k < K; ++k) pwr_sum += p->power[k];
    float far_psd_sum = 0.0f;
    for (int k = 0; k < K; ++k)
        far_psd_sum += far_cmag2[k];
    if (pwr_sum < 1e-10f && far_psd_sum > 1e-10f) {
        /* cold start: power = np.abs(far_spec)**2 (already computed above) */
        memcpy(p->power, far_cmag2, (size_t)K * sizeof(float));
    } else {
        /* power = alpha_power*power + (1-alpha_power)*far_psd; alpha_power=0.9,
         * float32-by-design -- textually identical to sk_ema_cmag2_f32_scalar's
         * own combine (state[i]=alpha*state[i]+beta*mag2), matching the
         * e2_ref/error_psd precedent below (pbfdkf.c e2_ref_scratch dedup). */
        const float a = 0.9f;
        const float b = 1.0f - 0.9f;
        for (int k = 0; k < K; ++k)
            p->power[k] = a * p->power[k] + b * far_cmag2[k];
    }

    /* echo_spec = Σ_p W[p] * X_buf[(curr_p-p)%N] */
    memset(p->echo_spec, 0, (size_t)K * sizeof(Complex));
    for (int part = 0; part < p->n_partitions; ++part) {
        int p_idx = curr_p - part;
        while (p_idx < 0) p_idx += p->n_partitions;
        const Complex* Wp = p->W     + (size_t)part  * K;
        const Complex* Xp = p->X_buf + (size_t)p_idx * K;
        /* numpy complex64 mul uses FMA on arm64: re=fma(wr,xr,-(wi*xi)),
         * im=fma(wr,xi, wi*xr). (Verified 0/300000 vs naive's 25%.) Matches
         * sk_cmac_np_f32's scalar reference expression-for-expression. */
        sk_cmac_np_f32(p->echo_spec, Wp, Xp, K);
    }

    /* IFFT echo, output = near[-hop:] - echo_time[hop:block] */
    fft_inverse(p->fft, p->echo_spec, p->time_scratch);
    for (int i = 0; i < hop; ++i)
        output[i] = p->near_buffer[blk - hop + i] - p->time_scratch[hop + i];

    /* AEC3 SubtractorOutput.s_*_max_abs = max|echo_time[hop:block]| on the
     * valid hop. (np.max(np.abs(.)) over hop floats.) */
    {
        float smax = 0.0f;
        for (int i = 0; i < hop; ++i) {
            float a = fabsf(p->time_scratch[hop + i]);
            if (a > smax) smax = a;
        }
        p->last_s_max_abs = smax;
    }

    /* error_spec: zero-pad, valid region [hop, block) = output */
    memset(p->time_scratch, 0, (size_t)p->fft_size * sizeof(float));
    for (int i = 0; i < hop; ++i) p->time_scratch[hop + i] = output[i];
    fft_forward_scratch(p->fft, p->time_scratch, p->error_spec);

    /* windowed near (sqrt-Hann × near_buffer) − echo_spec. RES analysis = main
     * filter only; the shadow's error_spec_windowed is never read, so skip it in
     * lightweight (shadow) mode — byte-equal. */
    if (!p->lightweight) {
        for (int i = 0; i < blk; ++i)
            p->time_scratch[i] = p->near_buffer[i] * p->sqrt_hann[i];
        rfft_padded(p, p->time_scratch, blk, p->spec_scratch);
        for (int k = 0; k < K; ++k) {
            p->error_spec_windowed[k].r = p->spec_scratch[k].r - p->echo_spec[k].r;
            p->error_spec_windowed[k].i = p->spec_scratch[k].i - p->echo_spec[k].i;
        }
    }

    /* far_hop_energy = np.sum(far_end**2) / hop. Reproduce np.sum (pairwise f32)
     * over far_end**2, then divide in float32 (float32-by-design). */
    {
        float *fsq = p->scr_fsq;   /* instance scratch, not stack: [hop_size] */
        for (int i = 0; i < hop; ++i) fsq[i] = far_end[i] * far_end[i];
        float s = pairwise_sum_f32(fsq, hop);
        return s / (float)hop;
    }
}

void pbfdaf_process(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr,
                       float        mu_scale_scalar,
                       float*       output) {
    int K = p->n_freqs, N = p->n_partitions;
    int curr_p;
    float far_hop_energy = pbfdaf_frontend(p, near_end, far_end, output, &curr_p);

    /* AEC3 InitialState tracking (count active render hops). */
    if (p->initial_state_active &&
        far_hop_energy > p->initial_state_far_energy_floor) {
        p->initial_state_active_render_hops += 1;
        if (p->initial_state_active_render_hops >= p->initial_state_threshold_hops)
            p->initial_state_active = 0;
    }
    p->last_initial_state_active = p->initial_state_active;

    if (far_hop_energy > 1e-4f) {
        /* === PBFDAF NLMS _update_weights (coarse_filter_update_gain.cc) === */
        float* mu_local = p->scr_mu_local;   /* instance scratch, not stack: [n_freqs] */
        const float* mu_arr = mu_scale_arr;
        if (mu_arr == NULL) {
            for (int k = 0; k < K; ++k) mu_local[k] = mu_scale_scalar;
            mu_arr = mu_local;
        }
        int any_pos = 0;
        for (int k = 0; k < K; ++k) if (mu_arr[k] > 0.0f) { any_pos = 1; break; }
        if (!any_pos) { p->partition_idx = (p->partition_idx + 1) % N; return; }
        /* E22: increment before saturation gate (mirrors AEC3/PBFDKF).
         * UBSan-confirmed signed-overflow fix, floored at LONG_MAX: unlike
         * poor_excitation_counter (fixed above with a tight
         * n_partitions cap), call_counter's raw value is read bit-exact by
         * test/parity_pbfdkf.c (`flt.base.call_counter != exp_call`,
         * explicitly documented above as one of the 3 fields this port
         * keeps bit-exact) -- capping at N/the startup threshold would
         * still satisfy the `<= N` gate below, but would desync the raw
         * value from Python's unbounded golden the very next call (same
         * trap as erl_estimator.c's hold_counter_time_domain -- see that
         * fix's comment). Capping at LONG_MAX instead is a no-op for
         * every practically-reachable call count while eliminating the UB
         * at the true overflow boundary. */
        if (p->call_counter < LONG_MAX) p->call_counter += 1;
        if (p->saturated_capture) { p->partition_idx = (p->partition_idx + 1) % N; return; }
        if (p->call_counter <= N || p->poor_excitation_counter < N) {
            p->partition_idx = (p->partition_idx + 1) % N; return;
        }
        if (p->block_stationary) { p->partition_idx = (p->partition_idx + 1) % N; return; }
        /* (RenderSignalAnalyzer mask omitted: production shadow has the same
         * RSA; the parity harness captures the post-mask mu_arr so the mask is
         * already folded in.) */

        /* x2_partition_sum = (np.abs(X_buf)**2).sum(axis=0) — cmag2_np, summed
         * sequentially over partitions (n=6<8 → numpy axis-0 sequential). */
        float *x2psum = p->scr_x2psum;   /* instance scratch, not stack: [n_freqs] */
        for (int k = 0; k < K; ++k) x2psum[k] = 0.0f;
        for (int part = 0; part < N; ++part) {
            const Complex* Xp = p->X_buf + (size_t)part * K;
            sk_cmag2_np_acc_f32(Xp, x2psum, K);
        }
        float mu_initial_boost = p->initial_state_active ? (0.95f / 0.7f) : 1.0f;
        float ng_thr = (float)AEC3_FILTER_NOISE_GATE_POWER_FLOAT;
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
                fft_inverse(p->fft, Wp, p->time_scratch);
                for (int i = 0; i < p->fft_size; ++i) p->time_scratch[i] *= p->td_window[i];
                fft_forward_scratch(p->fft, p->time_scratch, Wp);
            }
        }
        if (p->enable_td_constraint && p->constraint_round_robin)
            p->partition_to_constrain = (p->partition_to_constrain + 1) % N;
    }

    p->partition_idx = (p->partition_idx + 1) % N;
}

void pbfdaf_copy_weights_from(PBFDAF* dst, const PBFDAF* src) {
    int Wsz = dst->n_partitions * dst->n_freqs;
    memcpy(dst->W, src->W, (size_t)Wsz * sizeof(Complex));
}

/* Warm tap-transfer (v3.24.1): shift the learned IR LEFT by `shift_samples`
 * (toward tap 0), zero-filling the freed tail, INSTEAD of zeroing the filter —
 * so the cold-start cancellation survives a delay realign (the "1.14s vertical
 * line" fix). Mirrors filters.py:139-166: raw irfft per partition (first hop
 * taps) -> concatenate -> left-shift by s -> rfft each hop block (zero-padded
 * to fft_size) back into W[p]. Keeps X_buf (Python _warm_clear_xbuf default
 * False). NO td_window — Python uses raw irfft here (unlike the TD-constraint
 * round-trip). Uses p->scr_ir instance scratch, not stack — sized exactly
 * n_partitions*hop_size at init (see pbfdkf.h PBFDAF comment). */
void pbfdaf_warm_shift_ir(PBFDAF* p, int shift_samples) {
    int s = shift_samples;
    if (s <= 0) return;
    int hop = p->hop_size, nF = p->fft_size, nP = p->n_partitions, K = p->n_freqs;
    int total = nP * hop;
    if (total <= 0 || total > 4096) return;   /* safety; real max ~2080 */
    if (s > total) s = total;
    float *ir = p->scr_ir;
    for (int part = 0; part < nP; ++part) {
        fft_inverse(p->fft, p->W + (size_t)part * K, p->time_scratch);
        for (int i = 0; i < hop; ++i) ir[part * hop + i] = p->time_scratch[i];
    }
    for (int i = 0; i < total - s; ++i) ir[i] = ir[i + s];   /* read-ahead safe */
    for (int i = total - s; i < total; ++i) ir[i] = 0.0f;
    for (int part = 0; part < nP; ++part) {
        memset(p->time_scratch, 0, (size_t)nF * sizeof(float));
        for (int i = 0; i < hop; ++i) p->time_scratch[i] = ir[part * hop + i];
        fft_forward_scratch(p->fft, p->time_scratch, p->W + (size_t)part * K);
    }
}

/* ===== PBFDKF (v3.22) ==================================================== */

static void pbfdkf_init_scalars(PBFDKF* p) {
    int K = p->base.n_freqs;
    for (int k = 0; k < K; ++k) {
        p->Q_high[k] = 1e-4f;
        p->Q_low[k]  = 1e-5f;
        p->Q[k]      = 1e-4f;
        p->R[k]      = 1e-2f;
        p->error_psd[k] = 1e-2f;
        p->H_error_per_bin[k] = (float)AEC3_H_ERROR_INIT_FLOAT;
        p->erl_per_bin[k] = 0.1f;
        p->e2_coarse_per_bin[k] = 0.0f;
    }
    p->alpha_r = 0.95f;
    p->h_error_floor = (float)AEC3_H_ERROR_FLOOR_FLOAT;
    p->h_error_ceil  = (float)AEC3_H_ERROR_CEIL_FLOAT;
    /* The AEC3_LEAKAGE_*_PER_HOP #defines pre-bake the x2.5 factor for the
     * legacy hop=160/sr=16000 grid and are never recomputed for the filter's
     * actual hop_size/sample_rate. Rescaled live here via the same
     * per_block_rate_to_per_hop helper filters.py uses (raw AEC3 per-block
     * rates 5e-5/5e-2/5e-3/5e-1). Mirrors filters.py's PBFDKF
     * leakage_converged/_diverged/_converged_transient/_diverged_transient. */
    p->leakage_converged = aec3_per_block_rate_to_per_hop(
        5e-5f, p->base.hop_size, p->base.sample_rate);
    p->leakage_diverged = aec3_per_block_rate_to_per_hop(
        5e-2f, p->base.hop_size, p->base.sample_rate);
    p->leakage_converged_transient = aec3_per_block_rate_to_per_hop(
        5e-3f, p->base.hop_size, p->base.sample_rate);
    p->leakage_diverged_transient = aec3_per_block_rate_to_per_hop(
        5e-1f, p->base.hop_size, p->base.sample_rate);
    p->e2_coarse_per_bin_valid = 0;
    p->e2_coarse_for_refresh = 0.0f;
    p->disallow_leakage_diverged = 0;
    p->h_error_refresh_erl_floor = 0.0f;
    p->last_leakage_div_frac = 0.0f;
    p->partition_sum_x2_startup_hops = 0;
    /* AEC3 refined gates (orchestrator overrides poor_excitation per config). */
    p->base.poor_excitation_counter = 400;
}

int pbfdkf_init(PBFDKF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size, int sample_rate) {
    /* Set early (before any malloc below) so pbfdkf_free(p) is safe to call
     * from the OOM path even if the very first allocation fails -- see
     * pbfdaf_init's identical reasoning. */
    p->is_static = 0;
    if (pbfdaf_init(&p->base, block_size, n_partitions, mu, delta, hop_size, 0,
                     sample_rate) != 0)
        return -1;   /* F04: base filter's nested fft_create() failed (OOM);
                      * pbfdaf_init already freed its own partial state. */
    int K = p->base.n_freqs;
    p->Q_high = (float*)malloc((size_t)K * sizeof(float));
    p->Q_low  = (float*)malloc((size_t)K * sizeof(float));
    p->Q      = (float*)malloc((size_t)K * sizeof(float));
    p->R      = (float*)malloc((size_t)K * sizeof(float));
    p->error_psd = (float*)malloc((size_t)K * sizeof(float));
    p->H_error_per_bin = (float*)malloc((size_t)K * sizeof(float));
    p->erl_per_bin = (float*)malloc((size_t)K * sizeof(float));
    p->e2_coarse_per_bin = (float*)malloc((size_t)K * sizeof(float));
    p->e2_ref_scratch = (float*)malloc((size_t)K * sizeof(float));
    p->scr_mu_local = (float*)malloc((size_t)K * sizeof(float));
    p->scr_X2       = (float*)malloc((size_t)K * sizeof(float));
    p->scr_mu_aec3  = (float*)malloc((size_t)K * sizeof(float));

    /* F04: pbfdkf_init_scalars() below writes straight
     * through p->Q_high[k]/Q_low[k]/.../e2_coarse_per_bin[k] unconditionally
     * -- any NULL among the 12 mallocs above would be an immediate NULL-deref
     * crash right there, not just on some later process() call. Check every
     * pointer BEFORE calling pbfdkf_init_scalars(). pbfdkf_free(p) already
     * tolerates a partially-NULL struct (free(NULL) is a no-op for each of
     * these fields; the base filter is fully built at this point so
     * pbfdaf_free(&p->base) inside it is unaffected). */
    if (!p->Q_high || !p->Q_low || !p->Q || !p->R || !p->error_psd ||
        !p->H_error_per_bin || !p->erl_per_bin || !p->e2_coarse_per_bin ||
        !p->e2_ref_scratch || !p->scr_mu_local || !p->scr_X2 || !p->scr_mu_aec3) {
        pbfdkf_free(p);
        return -1;
    }

    pbfdkf_init_scalars(p);
    return 0;
}

size_t pbfdkf_get_mem_size(int block_size, int n_partitions, int hop_size) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    int blk, fft, K;
    pbfdaf_compute_sizes(hop, NULL, &blk, &fft, &K);
    (void)blk; (void)fft;
    if (n_partitions <= 0 || K <= 0) return 0;
    size_t total = pbfdaf_get_mem_size(block_size, n_partitions, hop_size, 0);
    if (total == 0) return 0;   /* base PBFDAF sizing rejected its own inputs */
    /* Checked size arithmetic (F05): 12 identically-sized K-length float
     * fields (Q_high, Q_low, Q, R, error_psd, H_error_per_bin, erl_per_bin,
     * e2_coarse_per_bin, e2_ref_scratch, scr_mu_local, scr_X2, scr_mu_aec3),
     * each individually ALIGN16'd by pbfdkf_init_static below — order/
     * grouping unchanged, only the arithmetic is wrapped (mem_align.h
     * ck_field_size_reps composes ck_mul_size/ck_align16_size/ck_add_size
     * with the same saturate-to-SIZE_MAX-on-overflow behavior). */
    total = ck_field_size_reps(total, (size_t)K, sizeof(float), 12);

    if (MEM_SIZE_INVALID(total)) return 0;
    return total;
}

int pbfdkf_init_static(PBFDKF* p, void* mem, size_t mem_size,
                         int block_size, int n_partitions,
                         float mu, float delta, int hop_size,
                         int sample_rate) {
    if (!p || !mem) return -1;
    /* F07: reject a misaligned pool base before any pool write (see the
     * matching guard in pbfdaf_init_static above). */
    if (!MEM_IS_ALIGNED16(mem)) return -1;
    {
        /* pbfdkf_get_mem_size returning 0 means "invalid" (F05); guard the
         * same 0-vs-`mem_size < 0` ambiguity as pbfdaf_init_static above. */
        size_t need = pbfdkf_get_mem_size(block_size, n_partitions, hop_size);
        if (need == 0 || mem_size < need) return -1;
    }
    size_t base_sz = pbfdaf_get_mem_size(block_size, n_partitions, hop_size, 0);
    memset(p, 0, sizeof(*p));
    if (pbfdaf_init_static(&p->base, mem, base_sz,
                       block_size, n_partitions, mu, delta, hop_size, 0,
                       sample_rate) != 0)
        return -1;   /* F04: base filter's nested fft_init() failed (OOM) */
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
    p->scr_mu_aec3  = (float*)ptr; ptr += ALIGN16((size_t)p->base.n_freqs * sizeof(float));
    (void)ptr;
    pbfdkf_init_scalars(p);
    p->is_static = 1;
    return 0;
}

void pbfdkf_free(PBFDKF* p) {
    if (!p) return;
    if (p->is_static) { pbfdaf_free(&p->base); return; }
    free(p->Q_high); free(p->Q_low); free(p->Q); free(p->R);
    free(p->error_psd); free(p->H_error_per_bin); free(p->erl_per_bin);
    free(p->e2_coarse_per_bin);
    free(p->e2_ref_scratch);
    free(p->scr_mu_local); free(p->scr_X2); free(p->scr_mu_aec3);
    pbfdaf_free(&p->base);
}

void pbfdkf_reset(PBFDKF* p) {
    pbfdaf_reset(&p->base);
    int K = p->base.n_freqs;
    for (int k = 0; k < K; ++k) {
        p->R[k] = 1e-2f;
        p->error_psd[k] = 1e-2f;
        p->Q[k] = p->Q_high[k];
    }
}

void pbfdkf_handle_echo_path_change(PBFDKF* p, int delay_change, int gain_change) {
    /* CoarseFilterUpdateGain M3 counter reset (not-gain_change). Was reset to
     * the frozen reference-grid AEC3_POOR_EXC_COUNTER_INITIAL_HOPS (400),
     * silently undoing the correctly rate-scaled value aec.c's construction
     * path sets via the identical aec3_blocks_to_hops(1000, hop, sr) call --
     * every echo-path-change event after construction re-clobbered it back
     * to the reference-grid constant. Mirrors filters.py's
     * _poor_excitation_counter fix. */
    if (!gain_change) {
        p->base.poor_excitation_counter =
            aec3_blocks_to_hops(1000, p->base.hop_size, p->base.sample_rate);
        p->base.call_counter = 0;
    }
    if (delay_change) {
        int K = p->base.n_freqs;
        for (int k = 0; k < K; ++k)
            p->H_error_per_bin[k] = (float)AEC3_H_ERROR_INIT_FLOAT;
    }
}

void pbfdkf_copy_weights_from(PBFDKF* dst, const PBFDKF* src) {
    pbfdaf_copy_weights_from(&dst->base, &src->base);
}

/* sum(|error_spec|**2) — cmag2_np into a scratch then the numpy-1.26 pairwise
 * float32 sum, returned as float32 (float32-by-design, no double promotion).
 * Not const: writes through p->scr_e2 instance scratch (was a fixed-size
 * stack local, n_freqs up to 8192 entries) — see the pbfdkf.h prototype
 * comment for the const rationale. */
float pbfdaf_get_error_energy(PBFDAF* p) {
    int K = p->n_freqs;
    float *e2 = p->scr_e2;   /* instance scratch, not stack: [n_freqs] */
    sk_cmag2_np_f32(p->error_spec, e2, K);
    return pairwise_sum_f32(e2, K);
}

float pbfdkf_get_error_energy(PBFDKF* p) {
    return pbfdaf_get_error_energy(&p->base);
}

/* Concatenate per-partition irfft(W[p])[:hop] → out[n_partitions × hop]. */
void pbfdaf_get_time_domain_filter(PBFDAF* p, float* out) {
    int N = p->n_partitions, hop = p->hop_size;
    for (int part = 0; part < N; ++part) {
        const Complex* Wp = p->W + (size_t)part * p->n_freqs;
        fft_inverse(p->fft, Wp, p->time_scratch);
        for (int i = 0; i < hop; ++i)
            out[(size_t)part * hop + i] = p->time_scratch[i];
    }
}

/* W *= np.complex64(scale): scalar (real) value-cast to f32, multiply each
 * complex component in f32. */
void pbfdaf_scale_filter(PBFDAF* p, float scale) {
    int Wsz = p->n_partitions * p->n_freqs;
    for (int i = 0; i < Wsz; ++i) {
        p->W[i].r = p->W[i].r * scale;
        p->W[i].i = p->W[i].i * scale;
    }
}

void pbfdkf_scale_filter(PBFDKF* p, float scale) {
    pbfdaf_scale_filter(&p->base, scale);
}

/* AEC3 H_error leakage refresh + clamp (refined_filter_update_gain.cc:128-138).
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
         * with the error_psd EMA loop + pbfdkf_update_weights_aec3's mu gate
         * -- all three read the same untouched b->error_spec this hop, so
         * recomputing cmag2_np() per call site here was pure waste (up to
         * 3x per bin per hop before this change). */
        int n_div = 0;   /* count of bins on the diverged-leakage branch */
        for (int k = 0; k < K; ++k) {
            float e2_ref = e2_ref_bins[k];
            int use_conv = (e2_ref <= p->e2_coarse_per_bin[k]) ||
                           p->disallow_leakage_diverged;
            if (!use_conv) ++n_div;
            float leakage = use_conv ? lc_eff : ld_eff;
            float erl_eff = p->erl_per_bin[k];
            if (p->h_error_refresh_erl_floor > 0.0f) {
                /* max(erl, floor), float32-by-design. */
                if (erl_eff < p->h_error_refresh_erl_floor)
                    erl_eff = p->h_error_refresh_erl_floor;
            }
            p->H_error_per_bin[k] = p->H_error_per_bin[k] + leakage * erl_eff;
        }
        /* Diag: fraction of bins on diverged-leakage (np.mean(~mask)). */
        p->last_leakage_div_frac = (float)n_div / (float)K;
    } else {
        /* scalar fallback: e2_ref_sum = sum(error_psd) (float32); e2_coa scalar.
         * (Not exercised in production: orchestrator sets e2_coarse_per_bin
         * every hop. Kept pairwise-correct for completeness.) */
        float e2_ref_sum = pairwise_sum_f32(p->error_psd, K);
        float e2_coa_sum = p->e2_coarse_for_refresh;
        int use_conv = (e2_ref_sum <= e2_coa_sum) || p->disallow_leakage_diverged;
        /* scalar path is all-or-nothing → frac is 0.0 or 1.0. */
        p->last_leakage_div_frac = use_conv ? 0.0f : 1.0f;
        float leakage = use_conv ? lc_eff : ld_eff;
        for (int k = 0; k < K; ++k)
            p->H_error_per_bin[k] = p->H_error_per_bin[k] +
                                    leakage * p->erl_per_bin[k];
    }
    /* np.clip(H_error, floor, ceil). numpy clip: out = min(max(x,lo),hi). */
    for (int k = 0; k < K; ++k) {
        float v = p->H_error_per_bin[k];
        if (v < p->h_error_floor) v = p->h_error_floor;
        if (v > p->h_error_ceil)  v = p->h_error_ceil;
        p->H_error_per_bin[k] = v;
    }
}

/* AEC3-aligned per-bin Kalman gain via H_error
 * (refined_filter_update_gain.cc:104-138). mu_scale_arr is already post-RSA. */
static void pbfdkf_update_weights_aec3(PBFDKF* p, int curr_p,
                                       const float* mu_scale_arr,
                                       const float* e2_ref_bins) {
    PBFDAF* b = &p->base;
    int K = b->n_freqs, N = b->n_partitions;

    /* X² source. Hybrid startup: partition-sum once call_counter past startup.
     * Python: X2 = (np.abs(X_buf)**2).sum(axis=0)  [partition-sum, n=6<8 →
     * sequential over part 0..N-1] or (np.abs(X_latest)**2). cmag2_np = the
     * bit-exact np.abs(.)**2. */
    float *X2 = p->scr_X2;   /* instance scratch, not stack: [n_freqs] */
    if (b->call_counter > p->partition_sum_x2_startup_hops) {
        for (int k = 0; k < K; ++k) X2[k] = 0.0f;
        for (int part = 0; part < N; ++part) {
            const Complex* Xp = b->X_buf + (size_t)part * K;
            sk_cmag2_np_acc_f32(Xp, X2, K);
        }
    } else {
        const Complex* Xl = b->X_buf + (size_t)curr_p * K;
        sk_cmag2_np_f32(Xl, X2, K);
    }

    float delta32 = (float)b->delta;
    float n_part = (float)N;
    float ng_thr = (float)AEC3_FILTER_NOISE_GATE_POWER_FLOAT;

    /* mu[k] = H_error / (0.5·H_error·X² + n·E²_refined + δ), gated by X²≥gate.
     * e2_ref_bins[k] = |error_spec[k]|² precomputed once by pbfdkf_process
     * (same value cmag2_np(er,ei) would give here -- error_spec is untouched
     * since pbfdaf_frontend ran at the top of this hop). */
    float *mu_aec3 = p->scr_mu_aec3;   /* instance scratch, not stack: [n_freqs] */
    for (int k = 0; k < K; ++k) {
        float e2_ref = e2_ref_bins[k];
        float He = p->H_error_per_bin[k];
        float denom = 0.5f * He * X2[k] + n_part * e2_ref + delta32;
        float m = He / denom;
        mu_aec3[k] = (X2[k] >= ng_thr) ? m : 0.0f;
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
        sk_wupdate_kf_f32(Wp, Xp, b->error_spec, mu_aec3, mu_scale_arr, K);
        if (b->enable_td_constraint &&
            (!b->constraint_round_robin || part == b->partition_to_constrain)) {
            fft_inverse(b->fft, Wp, b->time_scratch);
            for (int i = 0; i < b->fft_size; ++i) b->time_scratch[i] *= b->td_window[i];
            fft_forward_scratch(b->fft, b->time_scratch, Wp);
        }
    }
    if (b->enable_td_constraint && b->constraint_round_robin)
        b->partition_to_constrain = (b->partition_to_constrain + 1) % N;

    /* E2 fix: scale mu_aec3 by mu_scale_arr before H_error decay so that masked
     * bins (RSA narrowband mask + DT scale) do not decay while their W is frozen.
     * Mirrors AEC3 refined_filter_update_gain.cc:113-119. */
    for (int k = 0; k < K; ++k) mu_aec3[k] *= mu_scale_arr[k];
    /* H_error decay (active branch only): H_error -= 0.5·mu·X²·H_error. */
    for (int k = 0; k < K; ++k)
        p->H_error_per_bin[k] -= 0.5f * mu_aec3[k] * X2[k] * p->H_error_per_bin[k];

    /* always-on refresh + clamp. */
    pbfdkf_h_error_refresh(p, e2_ref_bins);
}

void pbfdkf_process(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr,
                       float        mu_scale_scalar,
                       float*       output) {
    PBFDAF* b = &p->base;
    int K = b->n_freqs, N = b->n_partitions;
    int curr_p;
    float far_hop_energy = pbfdaf_frontend(b, near_end, far_end, output, &curr_p);

    /* AEC3 InitialState tracking. */
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
         * error_psd EMA below, pbfdkf_update_weights_aec3's mu gate, and
         * pbfdkf_h_error_refresh's per-bin branch (all 4 call sites in this
         * function, including both refresh-only early-returns below): all of
         * them read the SAME b->error_spec, set once by pbfdaf_frontend above
         * and untouched for the rest of this hop. Previously recomputed via
         * cmag2_np() up to 3x per bin on the full-update path -- same
         * deterministic function of the same floats, so reusing the value
         * changes nothing bit-wise. */
        float *e2_ref = p->e2_ref_scratch;   /* instance scratch, not stack */
        sk_cmag2_np_f32(b->error_spec, e2_ref, K);

        /* startup / poor-excitation / saturation gates → refresh only.
         * UBSan-confirmed signed-overflow fix, floored at LONG_MAX -- same
         * bit-exact-parity reasoning as pbfdaf_process's call_counter
         * increment above (test/parity_pbfdkf.c asserts this exact field
         * on the PBFDKF/main-filter path). */
        if (b->call_counter < LONG_MAX) b->call_counter += 1;
        if (b->call_counter <= N || b->poor_excitation_counter < N ||
            b->saturated_capture) {
            pbfdkf_h_error_refresh(p, e2_ref);
            b->partition_idx = (b->partition_idx + 1) % N;
            return;
        }
        /* stationary-far gate → refresh only. */
        if (b->block_stationary) {
            pbfdkf_h_error_refresh(p, e2_ref);
            b->partition_idx = (b->partition_idx + 1) % N;
            return;
        }
        /* (RSA mask folded into captured mu_arr — see header note.) */

        /* error_psd EMA + R = max(error_psd, delta).
         * error_psd = |error_spec|**2;
         * _error_psd = _alpha_r*_error_psd + (1-_alpha_r)*error_psd.
         * alpha_r=0.95, float32-by-design (the old double-promotion parity
         * nuance is retired). */
        const float ar = 0.95f;
        const float oar = 1.0f - 0.95f;
        for (int k = 0; k < K; ++k) {
            p->error_psd[k] = ar * p->error_psd[k] + oar * e2_ref[k];
            p->R[k] = (p->error_psd[k] > b->delta) ? p->error_psd[k] : (float)b->delta;
        }

        pbfdkf_update_weights_aec3(p, curr_p, mu_arr, e2_ref);
    }

    b->partition_idx = (b->partition_idx + 1) % N;
}
