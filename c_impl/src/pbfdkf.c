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
 *  - np.float32 ⇒ float; np.complex64 ⇒ Complex; scalar pyfloat·f32 promotions
 *    follow the OPPOSITE-of-array weak-promotion rule (done in double where the
 *    Python forms a 0-d scalar from a pyfloat × f32).
 *
 * v3.10 → v3.22 change: the old P-denominator Kalman body (P/Q/R Riccati +
 * KX-blend + P_floor/p_max clamps) is GONE. The active update is the AEC3
 * per-bin H_error gain + always-on leakage refresh.
 */
#include "pbfdkf.h"
#include "aec3_scale.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI_AEC
#define M_PI_AEC 3.14159265358979323846
#endif

static int next_pow2(int x) { int n = 1; while (n < x) n <<= 1; return n; }

/* pairwise float32 sum matching numpy's reduction EXACTLY: an 8-accumulator
 * unrolled leaf for n<=128 (numpy NPY_PW_BLOCKSIZE), then a recursive split at
 * n/2 rounded down to a multiple of 8. Verified 0/5000 mismatch vs np.sum over
 * f32 across lengths {80,160,257,320,512}. Used for np.sum(far_end**2). */
static float pw_leaf_f32(const float* a, int n) {
    if (n < 8) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += a[i];
        return s;
    }
    float r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    int i = 8;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
}
static float pairwise_sum_f32(const float* a, int n) {
    if (n <= 128) return pw_leaf_f32(a, n);
    int n2 = n / 2; n2 -= (n2 % 8);
    return pairwise_sum_f32(a, n2) + pairwise_sum_f32(a + n2, n - n2);
}

/* |complex64|² matching numpy's `np.abs(z) ** 2` BIT-EXACTLY. numpy computes
 * np.abs on complex64 via the SIMD `simd_cabsolute` (NEON/SSE) = a scaled-hypot
 * with a FUSED multiply-add on `ratio*ratio + 1`, NOT hypotf and NOT
 * sqrtf(r²+i²) (both verified to mismatch ~35%/3.7%). Then `**2` is `m*m`.
 * `fmaf` + `sqrtf` are IEEE-754 correctly-rounded, so this scalar form is
 * deterministic & portable and reproduces np.abs(z)**2 with 0/300000 mismatch.
 * Used wherever Python writes `np.abs(...)**2` (error_psd EMA, X² partition sum,
 * e2_refined, h_error refresh). */
static float cmag2_np(float er, float ei) {
    float a = fabsf(er), b = fabsf(ei);
    float larger  = (a > b) ? a : b;
    float smaller = (a > b) ? b : a;
    float m;
    if (larger == 0.0f) {
        m = 0.0f;
    } else {
        float ratio = smaller / larger;
        m = larger * sqrtf(fmaf(ratio, ratio, 1.0f));
    }
    return m * m;
}

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
    /* Non-causal fade [hop:hop+fade_len]: descending from ~1.0 to 0.0. */
    for (int i = 0; i < fade_len; ++i) {
        int idx = p->hop_size + i;
        if (idx < p->fft_size) {
            double v = 0.5 * (1.0 - cos(M_PI_AEC * (double)(fade_len - 1 - i) / (double)fade_len));
            p->td_window[idx] = (float)v;
        }
    }
    /* [hop+fade_len, fft_size) stays 0. */
    for (int i = 0; i < p->block_size; ++i) {
        double h = 0.5 * (1.0 - cos(2.0 * M_PI_AEC * (double)i / (double)p->block_size));
        p->sqrt_hann[i] = (float)sqrt(h);
    }
}

static void pbfdaf_init_scalars(PBFDAF* p, int n_partitions, float mu, float delta) {
    p->n_partitions = n_partitions;
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
    p->initial_state_threshold_hops = 250;
    p->initial_state_far_energy_floor = 1e-4f;
    p->last_initial_state_active = 1;
    p->last_s_max_abs = 0.0f;
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

void pbfdaf_init(PBFDAF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    pbfdaf_compute_sizes(hop, &p->hop_size, &p->block_size,
                         &p->fft_size, &p->n_freqs);

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

    pbfdaf_init_scalars(p, n_partitions, mu, delta);
    pbfdaf_fill_windows(p);
    p->is_static = 0;
}

size_t pbfdaf_get_mem_size(int block_size, int n_partitions, int hop_size) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    int blk, fft, K;
    pbfdaf_compute_sizes(hop, NULL, &blk, &fft, &K);
    if (n_partitions <= 0 || K <= 0) return 0;
    int Wsz = n_partitions * K;
    size_t total = 0;
    total += ALIGN16((size_t)fft * sizeof(float));    /* td_window */
    total += ALIGN16((size_t)blk * sizeof(float));    /* sqrt_hann */
    total += ALIGN16((size_t)Wsz * sizeof(Complex));  /* W */
    total += ALIGN16((size_t)Wsz * sizeof(Complex));  /* X_buf */
    total += ALIGN16((size_t)blk * sizeof(float));    /* near_buffer */
    total += ALIGN16((size_t)blk * sizeof(float));    /* far_buffer */
    total += ALIGN16((size_t)K   * sizeof(float));    /* power */
    total += ALIGN16((size_t)K   * sizeof(Complex));  /* near_spec */
    total += ALIGN16((size_t)K   * sizeof(Complex));  /* echo_spec */
    total += ALIGN16((size_t)K   * sizeof(Complex));  /* error_spec */
    total += ALIGN16((size_t)K   * sizeof(Complex));  /* far_spec */
    total += ALIGN16((size_t)K   * sizeof(Complex));  /* error_spec_windowed */
    total += fft_get_mem_size(fft);                   /* nested FFT */
    total += ALIGN16((size_t)fft * sizeof(float));    /* time_scratch */
    total += ALIGN16((size_t)K   * sizeof(Complex));  /* spec_scratch */
    return total;
}

void pbfdaf_init_static(PBFDAF* p, void* mem, size_t mem_size,
                         int block_size, int n_partitions,
                         float mu, float delta, int hop_size) {
    if (!p || !mem) return;
    if (mem_size < pbfdaf_get_mem_size(block_size, n_partitions, hop_size)) return;

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
    p->spec_scratch = (Complex*)ptr; /* last */

    pbfdaf_init_scalars(p, n_partitions, mu, delta);
    pbfdaf_zero_state(p);
    pbfdaf_fill_windows(p);
    p->is_static = 1;
}

void pbfdaf_free(PBFDAF* p) {
    if (!p) return;
    if (p->is_static) {
        if (p->fft) fft_destroy(p->fft);  /* no-op when nested static */
        return;
    }
    free(p->td_window); free(p->sqrt_hann);
    free(p->W); free(p->X_buf);
    free(p->near_buffer); free(p->far_buffer); free(p->power);
    free(p->near_spec); free(p->echo_spec); free(p->error_spec);
    free(p->far_spec); free(p->error_spec_windowed);
    free(p->time_scratch); free(p->spec_scratch);
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
        fft_forward(p->fft, p->time_scratch, spec_out);
    }
}

/* Shared front-end: buffer shift, FFTs, power EMA, echo, error, windowed-error.
 * Identical for PBFDAF and PBFDKF (matches PBFDAF.process lines 191-248).
 * Returns far_hop_energy (the W-update gate). curr_p stored in *out_curr_p. */
static double pbfdaf_frontend(PBFDAF* p,
                              const float* near_end, const float* far_end,
                              float* output, int* out_curr_p) {
    int hop = p->hop_size, blk = p->block_size, K = p->n_freqs;

    memmove(p->near_buffer, p->near_buffer + hop, (size_t)(blk - hop) * sizeof(float));
    memcpy(p->near_buffer + (blk - hop), near_end, (size_t)hop * sizeof(float));
    memmove(p->far_buffer, p->far_buffer + hop, (size_t)(blk - hop) * sizeof(float));
    memcpy(p->far_buffer + (blk - hop), far_end, (size_t)hop * sizeof(float));

    rfft_padded(p, p->near_buffer, blk, p->near_spec);
    rfft_padded(p, p->far_buffer,  blk, p->far_spec);

    int curr_p = p->partition_idx;
    *out_curr_p = curr_p;
    memcpy(p->X_buf + (size_t)curr_p * K, p->far_spec, (size_t)K * sizeof(Complex));

    /* power EMA (cold start: direct init when sum(power)<1e-10 and active far) */
    double pwr_sum = 0.0;
    for (int k = 0; k < K; ++k) pwr_sum += (double)p->power[k];
    double far_psd_sum = 0.0;
    for (int k = 0; k < K; ++k)
        far_psd_sum += (double)cmag2_np(p->far_spec[k].r, p->far_spec[k].i);
    if (pwr_sum < 1e-10 && far_psd_sum > 1e-10) {
        /* cold start: power = np.abs(far_spec)**2 */
        for (int k = 0; k < K; ++k)
            p->power[k] = cmag2_np(p->far_spec[k].r, p->far_spec[k].i);
    } else {
        /* power = alpha_power*power + (1-alpha_power)*far_psd; alpha_power=0.9
         * (Python float) → (1-0.9) is a double cast to f32. */
        const float a = (float)0.9;
        const float b = (float)(1.0 - 0.9);
        for (int k = 0; k < K; ++k)
            p->power[k] = a * p->power[k] + b * cmag2_np(p->far_spec[k].r, p->far_spec[k].i);
    }

    /* echo_spec = Σ_p W[p] * X_buf[(curr_p-p)%N] */
    memset(p->echo_spec, 0, (size_t)K * sizeof(Complex));
    for (int part = 0; part < p->n_partitions; ++part) {
        int p_idx = curr_p - part;
        while (p_idx < 0) p_idx += p->n_partitions;
        const Complex* Wp = p->W     + (size_t)part  * K;
        const Complex* Xp = p->X_buf + (size_t)p_idx * K;
        for (int k = 0; k < K; ++k) {
            float wr = Wp[k].r, wi = Wp[k].i;
            float xr = Xp[k].r, xi = Xp[k].i;
            /* numpy complex64 mul uses FMA on arm64: re=fma(wr,xr,-(wi*xi)),
             * im=fma(wr,xi, wi*xr). (Verified 0/300000 vs naive's 25%.) */
            p->echo_spec[k].r += fmaf(wr, xr, -(wi * xi));
            p->echo_spec[k].i += fmaf(wr, xi,  (wi * xr));
        }
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
    fft_forward(p->fft, p->time_scratch, p->error_spec);

    /* windowed near (sqrt-Hann × near_buffer) − echo_spec */
    for (int i = 0; i < blk; ++i)
        p->time_scratch[i] = p->near_buffer[i] * p->sqrt_hann[i];
    rfft_padded(p, p->time_scratch, blk, p->spec_scratch);
    for (int k = 0; k < K; ++k) {
        p->error_spec_windowed[k].r = p->spec_scratch[k].r - p->echo_spec[k].r;
        p->error_spec_windowed[k].i = p->spec_scratch[k].i - p->echo_spec[k].i;
    }

    /* far_hop_energy = np.sum(far_end**2) / hop. Reproduce np.sum (pairwise f32)
     * over far_end**2, then divide as f64 (np.float32 / int → float64). */
    {
        float fsq[8192];
        for (int i = 0; i < hop; ++i) fsq[i] = far_end[i] * far_end[i];
        float s = pairwise_sum_f32(fsq, hop);
        return (double)s / (double)hop;
    }
}

void pbfdaf_process(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr,
                       float        mu_scale_scalar,
                       float*       output) {
    int K = p->n_freqs, N = p->n_partitions;
    int curr_p;
    double far_hop_energy = pbfdaf_frontend(p, near_end, far_end, output, &curr_p);

    /* AEC3 InitialState tracking (count active render hops). */
    if (p->initial_state_active &&
        far_hop_energy > (double)p->initial_state_far_energy_floor) {
        p->initial_state_active_render_hops += 1;
        if (p->initial_state_active_render_hops >= p->initial_state_threshold_hops)
            p->initial_state_active = 0;
    }
    p->last_initial_state_active = p->initial_state_active;

    if (far_hop_energy > 1e-4) {
        /* === PBFDAF NLMS _update_weights (coarse_filter_update_gain.cc) === */
        float mu_local[8192];
        const float* mu_arr = mu_scale_arr;
        if (mu_arr == NULL) {
            for (int k = 0; k < K; ++k) mu_local[k] = mu_scale_scalar;
            mu_arr = mu_local;
        }
        int any_pos = 0;
        for (int k = 0; k < K; ++k) if (mu_arr[k] > 0.0f) { any_pos = 1; break; }
        if (!any_pos) { p->partition_idx = (p->partition_idx + 1) % N; return; }
        p->call_counter += 1;  /* E22: increment before saturation gate (mirrors AEC3/PBFDKF) */
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
        float x2psum[8192];
        for (int k = 0; k < K; ++k) x2psum[k] = 0.0f;
        for (int part = 0; part < N; ++part) {
            const Complex* Xp = p->X_buf + (size_t)part * K;
            for (int k = 0; k < K; ++k)
                x2psum[k] += cmag2_np(Xp[k].r, Xp[k].i);
        }
        float mu_initial_boost = p->initial_state_active ? (float)(0.95 / 0.7) : 1.0f;
        float ng_thr = (float)AEC3_FILTER_NOISE_GATE_POWER_FLOAT;
        float mu_eff[8192];
        for (int k = 0; k < K; ++k) {
            float denom = x2psum[k] + p->delta;
            float m = (p->mu * mu_initial_boost * mu_arr[k]) / denom;
            mu_eff[k] = (x2psum[k] >= ng_thr) ? m : 0.0f;
        }
        for (int part = 0; part < N; ++part) {
            int p_idx = curr_p - part; while (p_idx < 0) p_idx += N;
            Complex* Wp = p->W + (size_t)part * K;
            const Complex* Xp = p->X_buf + (size_t)p_idx * K;
            for (int k = 0; k < K; ++k) {
                float er = p->error_spec[k].r, ei = p->error_spec[k].i;
                float xr = Xp[k].r, xi = Xp[k].i;
                /* grad = error_spec * conj(X). conj(X)=(xr,-xi); complex64 mul
                 * uses numpy FMA form re=fma(er,cxr,-(ei*cxi)),
                 * im=fma(er,cxi, ei*cxr). mu_eff×grad is real×complex (no FMA). */
                float cxr = xr, cxi = -xi;
                float gr = fmaf(er, cxr, -(ei * cxi));
                float gi = fmaf(er, cxi,  (ei * cxr));
                Wp[k].r += mu_eff[k] * gr;
                Wp[k].i += mu_eff[k] * gi;
            }
            if (p->enable_td_constraint) {
                fft_inverse(p->fft, Wp, p->time_scratch);
                for (int i = 0; i < p->fft_size; ++i) p->time_scratch[i] *= p->td_window[i];
                fft_forward(p->fft, p->time_scratch, Wp);
            }
        }
    }

    p->partition_idx = (p->partition_idx + 1) % N;
}

void pbfdaf_copy_weights_from(PBFDAF* dst, const PBFDAF* src) {
    int Wsz = dst->n_partitions * dst->n_freqs;
    memcpy(dst->W, src->W, (size_t)Wsz * sizeof(Complex));
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
    p->leakage_converged = (float)AEC3_LEAKAGE_CONVERGED_PER_HOP;
    p->leakage_diverged  = (float)AEC3_LEAKAGE_DIVERGED_PER_HOP;
    p->leakage_converged_transient = (float)AEC3_LEAKAGE_CONVERGED_TRANSIENT_PER_HOP;
    p->leakage_diverged_transient  = (float)AEC3_LEAKAGE_DIVERGED_TRANSIENT_PER_HOP;
    p->e2_coarse_per_bin_valid = 0;
    p->e2_coarse_for_refresh = 0.0f;
    p->disallow_leakage_diverged = 0;
    p->h_error_refresh_erl_floor = 0.0f;
    p->partition_sum_x2_startup_hops = 0;
    /* AEC3 refined gates (orchestrator overrides poor_excitation per config). */
    p->base.poor_excitation_counter = 400;
}

void pbfdkf_init(PBFDKF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size) {
    pbfdaf_init(&p->base, block_size, n_partitions, mu, delta, hop_size);
    int K = p->base.n_freqs;
    p->Q_high = (float*)malloc((size_t)K * sizeof(float));
    p->Q_low  = (float*)malloc((size_t)K * sizeof(float));
    p->Q      = (float*)malloc((size_t)K * sizeof(float));
    p->R      = (float*)malloc((size_t)K * sizeof(float));
    p->error_psd = (float*)malloc((size_t)K * sizeof(float));
    p->H_error_per_bin = (float*)malloc((size_t)K * sizeof(float));
    p->erl_per_bin = (float*)malloc((size_t)K * sizeof(float));
    p->e2_coarse_per_bin = (float*)malloc((size_t)K * sizeof(float));
    pbfdkf_init_scalars(p);
    p->is_static = 0;
}

size_t pbfdkf_get_mem_size(int block_size, int n_partitions, int hop_size) {
    int hop = (hop_size > 0) ? hop_size : block_size / 2;
    int blk, fft, K;
    pbfdaf_compute_sizes(hop, NULL, &blk, &fft, &K);
    (void)blk; (void)fft;
    if (n_partitions <= 0 || K <= 0) return 0;
    size_t total = pbfdaf_get_mem_size(block_size, n_partitions, hop_size);
    total += ALIGN16((size_t)K * sizeof(float));   /* Q_high */
    total += ALIGN16((size_t)K * sizeof(float));   /* Q_low */
    total += ALIGN16((size_t)K * sizeof(float));   /* Q */
    total += ALIGN16((size_t)K * sizeof(float));   /* R */
    total += ALIGN16((size_t)K * sizeof(float));   /* error_psd */
    total += ALIGN16((size_t)K * sizeof(float));   /* H_error_per_bin */
    total += ALIGN16((size_t)K * sizeof(float));   /* erl_per_bin */
    total += ALIGN16((size_t)K * sizeof(float));   /* e2_coarse_per_bin */
    return total;
}

void pbfdkf_init_static(PBFDKF* p, void* mem, size_t mem_size,
                         int block_size, int n_partitions,
                         float mu, float delta, int hop_size) {
    if (!p || !mem) return;
    if (mem_size < pbfdkf_get_mem_size(block_size, n_partitions, hop_size)) return;
    size_t base_sz = pbfdaf_get_mem_size(block_size, n_partitions, hop_size);
    memset(p, 0, sizeof(*p));
    pbfdaf_init_static(&p->base, mem, base_sz,
                       block_size, n_partitions, mu, delta, hop_size);
    uint8_t* ptr = (uint8_t*)mem + base_sz;
    int K = p->base.n_freqs;
    p->Q_high    = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->Q_low     = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->Q         = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->R         = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->error_psd = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->H_error_per_bin = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->erl_per_bin = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    p->e2_coarse_per_bin = (float*)ptr; /* last */
    pbfdkf_init_scalars(p);
    p->is_static = 1;
}

void pbfdkf_free(PBFDKF* p) {
    if (!p) return;
    if (p->is_static) { pbfdaf_free(&p->base); return; }
    free(p->Q_high); free(p->Q_low); free(p->Q); free(p->R);
    free(p->error_psd); free(p->H_error_per_bin); free(p->erl_per_bin);
    free(p->e2_coarse_per_bin);
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
    /* CoarseFilterUpdateGain M3 counter reset (not-gain_change). */
    if (!gain_change) {
        p->base.poor_excitation_counter =
            AEC3_POOR_EXC_COUNTER_INITIAL_HOPS;
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

/* float(np.sum(np.abs(error_spec)**2)) — cmag2_np into a scratch then the
 * numpy-1.26 pairwise float32 sum, returned as a C double (Python float()). */
double pbfdaf_get_error_energy(const PBFDAF* p) {
    int K = p->n_freqs;
    float e2[8192];
    for (int k = 0; k < K; ++k)
        e2[k] = cmag2_np(p->error_spec[k].r, p->error_spec[k].i);
    return (double)pairwise_sum_f32(e2, K);
}

double pbfdkf_get_error_energy(const PBFDKF* p) {
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
static void pbfdkf_h_error_refresh(PBFDKF* p) {
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
        /* per-bin path. e2_refined = np.abs(error_spec)**2 (current frame). */
        for (int k = 0; k < K; ++k) {
            float er = b->error_spec[k].r, ei = b->error_spec[k].i;
            float e2_ref = cmag2_np(er, ei);
            int use_conv = (e2_ref <= p->e2_coarse_per_bin[k]) ||
                           p->disallow_leakage_diverged;
            float leakage = use_conv ? lc_eff : ld_eff;
            float erl_eff = p->erl_per_bin[k];
            if (p->h_error_refresh_erl_floor > 0.0f) {
                /* max(erl, floor) compared in double per parity rule */
                if ((double)erl_eff < (double)p->h_error_refresh_erl_floor)
                    erl_eff = p->h_error_refresh_erl_floor;
            }
            p->H_error_per_bin[k] = p->H_error_per_bin[k] + leakage * erl_eff;
        }
    } else {
        /* scalar fallback: e2_ref_sum = float(np.sum(error_psd)) ; e2_coa scalar.
         * (Not exercised in production: orchestrator sets e2_coarse_per_bin
         * every hop. Kept pairwise-correct for completeness.) */
        double e2_ref_sum = (double)pairwise_sum_f32(p->error_psd, K);
        double e2_coa_sum = (double)p->e2_coarse_for_refresh;
        int use_conv = (e2_ref_sum <= e2_coa_sum) || p->disallow_leakage_diverged;
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
                                       const float* mu_scale_arr) {
    PBFDAF* b = &p->base;
    int K = b->n_freqs, N = b->n_partitions;

    /* X² source. Hybrid startup: partition-sum once call_counter past startup.
     * Python: X2 = (np.abs(X_buf)**2).sum(axis=0)  [partition-sum, n=6<8 →
     * sequential over part 0..N-1] or (np.abs(X_latest)**2). cmag2_np = the
     * bit-exact np.abs(.)**2. */
    float X2[8192];
    if (b->call_counter > p->partition_sum_x2_startup_hops) {
        for (int k = 0; k < K; ++k) X2[k] = 0.0f;
        for (int part = 0; part < N; ++part) {
            const Complex* Xp = b->X_buf + (size_t)part * K;
            for (int k = 0; k < K; ++k)
                X2[k] += cmag2_np(Xp[k].r, Xp[k].i);
        }
    } else {
        const Complex* Xl = b->X_buf + (size_t)curr_p * K;
        for (int k = 0; k < K; ++k)
            X2[k] = cmag2_np(Xl[k].r, Xl[k].i);
    }

    float delta32 = (float)b->delta;
    float n_part = (float)N;
    float ng_thr = (float)AEC3_FILTER_NOISE_GATE_POWER_FLOAT;

    /* mu[k] = H_error / (0.5·H_error·X² + n·E²_refined + δ), gated by X²≥gate. */
    float mu_aec3[8192];
    for (int k = 0; k < K; ++k) {
        float er = b->error_spec[k].r, ei = b->error_spec[k].i;
        float e2_ref = cmag2_np(er, ei);   /* = (np.abs(error_spec)**2) */
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
        for (int k = 0; k < K; ++k) {
            float xr = Xp[k].r, xi = Xp[k].i;
            /* K = mu * conj(X) = (mu*xr, -mu*xi) */
            float kr = mu_aec3[k] * xr;
            float ki = -(mu_aec3[k] * xi);
            /* K_scaled = K * mu_scale_arr[k] (complex×real: no FMA needed) */
            float ksr = kr * mu_scale_arr[k];
            float ksi = ki * mu_scale_arr[k];
            /* W += K_scaled * error_spec. complex64×complex64 → numpy FMA form
             * re=fma(ksr,er,-(ksi*ei)), im=fma(ksr,ei, ksi*er). */
            float er = b->error_spec[k].r, ei = b->error_spec[k].i;
            Wp[k].r += fmaf(ksr, er, -(ksi * ei));
            Wp[k].i += fmaf(ksr, ei,  (ksi * er));
        }
        if (b->enable_td_constraint) {
            fft_inverse(b->fft, Wp, b->time_scratch);
            for (int i = 0; i < b->fft_size; ++i) b->time_scratch[i] *= b->td_window[i];
            fft_forward(b->fft, b->time_scratch, Wp);
        }
    }

    /* E2 fix: scale mu_aec3 by mu_scale_arr before H_error decay so that masked
     * bins (RSA narrowband mask + DT scale) do not decay while their W is frozen.
     * Mirrors AEC3 refined_filter_update_gain.cc:113-119. */
    for (int k = 0; k < K; ++k) mu_aec3[k] *= mu_scale_arr[k];
    /* H_error decay (active branch only): H_error -= 0.5·mu·X²·H_error. */
    for (int k = 0; k < K; ++k)
        p->H_error_per_bin[k] -= 0.5f * mu_aec3[k] * X2[k] * p->H_error_per_bin[k];

    /* always-on refresh + clamp. */
    pbfdkf_h_error_refresh(p);
}

void pbfdkf_process(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr,
                       float        mu_scale_scalar,
                       float*       output) {
    PBFDAF* b = &p->base;
    int K = b->n_freqs, N = b->n_partitions;
    int curr_p;
    double far_hop_energy = pbfdaf_frontend(b, near_end, far_end, output, &curr_p);

    /* AEC3 InitialState tracking. */
    if (b->initial_state_active &&
        far_hop_energy > (double)b->initial_state_far_energy_floor) {
        b->initial_state_active_render_hops += 1;
        if (b->initial_state_active_render_hops >= b->initial_state_threshold_hops)
            b->initial_state_active = 0;
    }
    b->last_initial_state_active = b->initial_state_active;

    if (far_hop_energy > 1e-4) {
        /* === PBFDKF._update_weights ============================== */
        float mu_local[8192];
        const float* mu_arr = mu_scale_arr;
        if (mu_arr == NULL) {
            for (int k = 0; k < K; ++k) mu_local[k] = mu_scale_scalar;
            mu_arr = mu_local;
        }
        /* startup / poor-excitation / saturation gates → refresh only. */
        b->call_counter += 1;
        if (b->call_counter <= N || b->poor_excitation_counter < N ||
            b->saturated_capture) {
            pbfdkf_h_error_refresh(p);
            b->partition_idx = (b->partition_idx + 1) % N;
            return;
        }
        /* stationary-far gate → refresh only. */
        if (b->block_stationary) {
            pbfdkf_h_error_refresh(p);
            b->partition_idx = (b->partition_idx + 1) % N;
            return;
        }
        /* (RSA mask folded into captured mu_arr — see header note.) */

        /* error_psd EMA + R = max(error_psd, delta).
         * Python: error_psd = np.abs(error_spec)**2;
         *         _error_psd = _alpha_r*_error_psd + (1-_alpha_r)*error_psd.
         * _alpha_r is a Python float (double 0.95), so (1-_alpha_r) is computed
         * in DOUBLE then cast to f32 by the array op — (float)(1.0-0.95), NOT
         * 1.0f-0.95f (the all-f32 form is off by 1 ULP). */
        const float ar = (float)0.95;
        const float oar = (float)(1.0 - 0.95);
        for (int k = 0; k < K; ++k) {
            float er = b->error_spec[k].r, ei = b->error_spec[k].i;
            float e2 = cmag2_np(er, ei);
            p->error_psd[k] = ar * p->error_psd[k] + oar * e2;
            p->R[k] = (p->error_psd[k] > b->delta) ? p->error_psd[k] : (float)b->delta;
        }

        pbfdkf_update_weights_aec3(p, curr_p, mu_arr);
    }

    b->partition_idx = (b->partition_idx + 1) % N;
}
