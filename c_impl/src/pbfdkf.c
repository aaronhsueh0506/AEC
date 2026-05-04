/* pbfdkf.c — PBFDAF base + PBFDKF derived.
 *
 * Tight 1:1 port. Notes on numerical fidelity:
 *  - Python casts Complex multiplications back to complex64 at FFT
 *    boundaries via .astype(np.complex64); numpy's pocketfft uses
 *    fp64 internally then casts. kiss_fft is fp32 throughout, so we
 *    cannot achieve bit-exact parity on FFT outputs. The downstream
 *    parity gate is rtol < 1e-4 on filter weights / error spectra.
 *  - Per-bin operations carefully mirror Python op order to keep ULP
 *    drift bounded. EMA RHS uses pre-update values.
 *  - PBFDKF G1 KX blend (aec.py:856-868) is reproduced exactly: cast
 *    mu_mean to fp32 before forming KX = mu*KX_optimal + (1-mu)*KX_scaled.
 */
#include "pbfdkf.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI_AEC
#define M_PI_AEC 3.14159265358979323846
#endif

static int next_pow2(int x) { int n = 1; while (n < x) n <<= 1; return n; }

/* ===== PBFDAF ============================================================ */

void pbfdaf_init(PBFDAF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size) {
    p->hop_size  = (hop_size > 0) ? hop_size : block_size / 2;
    p->block_size = 2 * p->hop_size;
    p->fft_size  = next_pow2(p->block_size);
    p->n_partitions = n_partitions;
    p->n_freqs   = p->fft_size / 2 + 1;
    p->mu        = mu;
    p->delta     = delta;
    p->alpha_power = 0.9f;
    p->enable_td_constraint = 1;

    /* td_window: ones for [0, hop-fade_len), raised-cosine fade across
     * [hop-fade_len, hop), zeros for [hop, fft_size). Match aec.py:609-613.
     *
     * Python: fade = 0.5*(1-cos(pi*arange(fade_len)/fade_len)) — increasing.
     * Then `_td_window[hop-fade_len:hop] = fade[::-1]` — reversed (decreasing).
     * Equivalently the reversed fade at slot i (0..fade_len-1) equals
     * 0.5*(1-cos(pi*(fade_len-1-i)/fade_len)). */
    p->td_window = (float*)calloc((size_t)p->fft_size, sizeof(float));
    int fade_len = p->hop_size / 4;
    for (int i = 0; i < p->hop_size - fade_len; ++i) p->td_window[i] = 1.0f;
    for (int i = 0; i < fade_len; ++i) {
        double v = 0.5 * (1.0 - cos(M_PI_AEC * (double)(fade_len - 1 - i) / (double)fade_len));
        p->td_window[p->hop_size - fade_len + i] = (float)v;
    }
    /* [hop, fft_size) already zero from calloc. */
    /* sqrt-Hann analysis window over block_size: np.hanning uses (N-1) denom */
    p->sqrt_hann = (float*)malloc((size_t)p->block_size * sizeof(float));
    for (int i = 0; i < p->block_size; ++i) {
        double h = 0.5 * (1.0 - cos(2.0 * M_PI_AEC * (double)i / (double)(p->block_size - 1)));
        p->sqrt_hann[i] = (float)sqrt(h);
    }

    int Wsz = p->n_partitions * p->n_freqs;
    p->W     = (Complex*)calloc((size_t)Wsz, sizeof(Complex));
    p->X_buf = (Complex*)calloc((size_t)Wsz, sizeof(Complex));
    p->partition_idx = 0;

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
}

void pbfdaf_free(PBFDAF* p) {
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

/* Helper: forward rfft with zero-pad of an input shorter than fft_size. */
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

void pbfdaf_process(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr,
                       float        mu_scale_scalar,
                       float*       output) {
    int hop = p->hop_size;
    int blk = p->block_size;
    int K   = p->n_freqs;

    /* Shift near/far buffers (overlap-save) */
    memmove(p->near_buffer, p->near_buffer + hop,
            (size_t)(blk - hop) * sizeof(float));
    memcpy(p->near_buffer + (blk - hop), near_end, (size_t)hop * sizeof(float));
    memmove(p->far_buffer, p->far_buffer + hop,
            (size_t)(blk - hop) * sizeof(float));
    memcpy(p->far_buffer + (blk - hop), far_end, (size_t)hop * sizeof(float));

    /* FFT (pad to fft_size if needed) */
    rfft_padded(p, p->near_buffer, blk, p->near_spec);
    rfft_padded(p, p->far_buffer,  blk, p->far_spec);

    int curr_p = p->partition_idx;
    /* X_buf[curr_p] = far_spec */
    memcpy(p->X_buf + (size_t)curr_p * K, p->far_spec,
           (size_t)K * sizeof(Complex));

    /* Power EMA — cold start init. */
    double pwr_sum = 0.0;
    for (int k = 0; k < K; ++k) pwr_sum += (double)p->power[k];
    double far_psd_sum = 0.0;
    for (int k = 0; k < K; ++k) {
        float r = p->far_spec[k].r, i = p->far_spec[k].i;
        far_psd_sum += (double)(r * r + i * i);
    }
    if (pwr_sum < 1e-10 && far_psd_sum > 1e-10) {
        for (int k = 0; k < K; ++k) {
            float r = p->far_spec[k].r, i = p->far_spec[k].i;
            p->power[k] = r * r + i * i;
        }
    } else {
        const float a = p->alpha_power, b = 1.0f - p->alpha_power;
        for (int k = 0; k < K; ++k) {
            float r = p->far_spec[k].r, i = p->far_spec[k].i;
            float far_psd = r * r + i * i;
            p->power[k] = a * p->power[k] + b * far_psd;
        }
    }

    /* echo_spec = sum_p W[p] * X_buf[(curr_p - p) % n_partitions] */
    memset(p->echo_spec, 0, (size_t)K * sizeof(Complex));
    for (int part = 0; part < p->n_partitions; ++part) {
        int p_idx = (curr_p - part);
        while (p_idx < 0) p_idx += p->n_partitions;
        const Complex* Wp = p->W     + (size_t)part  * K;
        const Complex* Xp = p->X_buf + (size_t)p_idx * K;
        for (int k = 0; k < K; ++k) {
            float wr = Wp[k].r, wi = Wp[k].i;
            float xr = Xp[k].r, xi = Xp[k].i;
            p->echo_spec[k].r += wr * xr - wi * xi;
            p->echo_spec[k].i += wr * xi + wi * xr;
        }
    }

    /* IFFT echo */
    fft_inverse(p->fft, p->echo_spec, p->time_scratch);

    /* Output = near_buffer[-hop:] - echo_time[hop:block_size] */
    for (int i = 0; i < hop; ++i) {
        output[i] = p->near_buffer[blk - hop + i] - p->time_scratch[hop + i];
    }

    /* error_time[fft_size] zero-padded except [hop, block_size) = output */
    memset(p->time_scratch, 0, (size_t)p->fft_size * sizeof(float));
    for (int i = 0; i < hop; ++i) p->time_scratch[hop + i] = output[i];
    fft_forward(p->fft, p->time_scratch, p->error_spec);

    /* Windowed near for RES (sqrt-Hann × near_buffer), then minus echo_spec */
    for (int i = 0; i < blk; ++i) {
        p->time_scratch[i] = p->near_buffer[i] * p->sqrt_hann[i];
    }
    rfft_padded(p, p->time_scratch, blk, p->spec_scratch);
    for (int k = 0; k < K; ++k) {
        p->error_spec_windowed[k].r = p->spec_scratch[k].r - p->echo_spec[k].r;
        p->error_spec_windowed[k].i = p->spec_scratch[k].i - p->echo_spec[k].i;
    }

    /* Far-hop energy gate */
    double sum_far_sq = 0.0;
    for (int i = 0; i < hop; ++i) {
        double v = (double)far_end[i];
        sum_far_sq += v * v;
    }
    double far_hop_energy = sum_far_sq / (double)hop;
    if (far_hop_energy > 1e-4) {
        /* PBFDAF NLMS update — overridable by subclass via separate
         * function, so this base does its own NLMS. */
        /* Decide effective per-bin mu_scale */
        const float* mu_arr = mu_scale_arr;
        float mu_local[8192];   /* assumes n_freqs <= 8192 */
        if (mu_arr == NULL) {
            for (int k = 0; k < K; ++k) mu_local[k] = mu_scale_scalar;
            mu_arr = mu_local;
        }
        int any_pos = 0;
        for (int k = 0; k < K; ++k) if (mu_arr[k] > 0.0f) { any_pos = 1; break; }
        if (any_pos) {
            /* power floor: max(power, max(local_floor, global_floor)) */
            double pwr_mean = 0.0;
            for (int k = 0; k < K; ++k) pwr_mean += (double)p->power[k];
            pwr_mean /= (double)K;
            float global_floor = (float)(pwr_mean * 0.001 + p->delta);
            for (int k = 0; k < K; ++k) {
                float local_floor = p->power[k] * 0.01f + p->delta;
                float floor_v = local_floor > global_floor ? local_floor : global_floor;
                float pf = p->power[k] > floor_v ? p->power[k] : floor_v;
                float mu_eff = (p->mu * mu_arr[k]) /
                               (pf * (float)p->n_partitions + p->delta);
                /* W[p] += mu_eff * conj(X_buf[p_idx]) * error_spec */
                /* (NLMS path — only PBFDAF uses this. PBFDKF replaces.) */
                /* Loop partitions */
                for (int part = 0; part < p->n_partitions; ++part) {
                    int p_idx = curr_p - part;
                    while (p_idx < 0) p_idx += p->n_partitions;
                    Complex* Wp = p->W + (size_t)part * K;
                    const Complex* Xp = p->X_buf + (size_t)p_idx * K;
                    float er = p->error_spec[k].r, ei = p->error_spec[k].i;
                    float xr = Xp[k].r, xi = Xp[k].i;
                    /* grad = error_spec * conj(X) = (er + j ei)(xr - j xi)
                     *     = (er*xr + ei*xi) + j(ei*xr - er*xi) */
                    float gr = er * xr + ei * xi;
                    float gi = ei * xr - er * xi;
                    Wp[k].r += mu_eff * gr;
                    Wp[k].i += mu_eff * gi;
                }
            }
            if (p->enable_td_constraint) {
                /* Apply TD constraint per partition */
                for (int part = 0; part < p->n_partitions; ++part) {
                    Complex* Wp = p->W + (size_t)part * K;
                    fft_inverse(p->fft, Wp, p->time_scratch);
                    for (int i = 0; i < p->fft_size; ++i) {
                        p->time_scratch[i] *= p->td_window[i];
                    }
                    fft_forward(p->fft, p->time_scratch, Wp);
                }
            }
        }
    }

    p->partition_idx = (p->partition_idx + 1) % p->n_partitions;
}

void pbfdaf_copy_weights_from(PBFDAF* dst, const PBFDAF* src) {
    int Wsz = dst->n_partitions * dst->n_freqs;
    memcpy(dst->W, src->W, (size_t)Wsz * sizeof(Complex));
}

/* ===== PBFDKF =========================================================== */

void pbfdkf_init(PBFDKF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size) {
    pbfdaf_init(&p->base, block_size, n_partitions, mu, delta, hop_size);
    int K = p->base.n_freqs;
    int sz = n_partitions * K;
    p->P      = (float*)malloc((size_t)sz * sizeof(float));
    p->Q_high = (float*)malloc((size_t)K  * sizeof(float));
    p->Q_low  = (float*)malloc((size_t)K  * sizeof(float));
    p->Q      = (float*)malloc((size_t)K  * sizeof(float));
    p->R      = (float*)malloc((size_t)K  * sizeof(float));
    p->error_psd = (float*)malloc((size_t)K * sizeof(float));
    for (int i = 0; i < sz; ++i) p->P[i] = 0.01f;
    for (int k = 0; k < K; ++k) {
        p->Q_high[k] = 1e-4f;
        p->Q_low[k]  = 1e-5f;
        p->Q[k]      = 1e-4f;
        p->R[k]      = 1e-2f;
        p->error_psd[k] = 1e-2f;
    }
    p->alpha_r = 0.95f;
    p->p_max_override        = 0.5f;
    p->p_max_override_frames = -1;
    p->p_floor_beta          = 0.1f;
    p->p_floor_beta_frames   = -1;
}

void pbfdkf_free(PBFDKF* p) {
    free(p->P); free(p->Q_high); free(p->Q_low); free(p->Q); free(p->R);
    free(p->error_psd);
    pbfdaf_free(&p->base);
}

void pbfdkf_reset(PBFDKF* p) {
    pbfdaf_reset(&p->base);
    int K = p->base.n_freqs;
    int sz = p->base.n_partitions * K;
    for (int i = 0; i < sz; ++i) p->P[i] = 0.01f;
    for (int k = 0; k < K; ++k) {
        p->R[k] = 1e-2f;
        p->error_psd[k] = 1e-2f;
        p->Q[k] = p->Q_high[k];
    }
    p->p_max_override        = 0.5f;
    p->p_max_override_frames = -1;
    p->p_floor_beta          = 0.1f;
    p->p_floor_beta_frames   = -1;
}

void pbfdkf_copy_weights_from(PBFDKF* dst, const PBFDKF* src) {
    pbfdaf_copy_weights_from(&dst->base, &src->base);
}

/* PBFDKF process replicates PBFDAF.process up to weight update, but
 * substitutes the Kalman update for the NLMS update.
 *
 * Implementation: run base process with NULL mu_scale to skip its NLMS
 * update, then perform Kalman update if far-hop energy > 1e-4.
 *
 * Cleaner: replicate base process body + invoke Kalman update inline.
 * Since base PBFDAF process already has NLMS hard-coded, PBFDKF needs
 * a separate process function. We duplicate the buffer/FFT/echo logic
 * and then run Kalman update (avoiding NLMS).
 */
void pbfdkf_process(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale_arr,
                       float        mu_scale_scalar,
                       float*       output) {
    PBFDAF* b = &p->base;
    int hop = b->hop_size;
    int blk = b->block_size;
    int K   = b->n_freqs;
    int N   = b->n_partitions;

    /* Shift buffers */
    memmove(b->near_buffer, b->near_buffer + hop, (size_t)(blk - hop) * sizeof(float));
    memcpy(b->near_buffer + (blk - hop), near_end, (size_t)hop * sizeof(float));
    memmove(b->far_buffer, b->far_buffer + hop, (size_t)(blk - hop) * sizeof(float));
    memcpy(b->far_buffer + (blk - hop), far_end, (size_t)hop * sizeof(float));

    rfft_padded(b, b->near_buffer, blk, b->near_spec);
    rfft_padded(b, b->far_buffer,  blk, b->far_spec);

    int curr_p = b->partition_idx;
    memcpy(b->X_buf + (size_t)curr_p * K, b->far_spec, (size_t)K * sizeof(Complex));

    /* power EMA / cold start (same as PBFDAF) */
    double pwr_sum = 0.0;
    for (int k = 0; k < K; ++k) pwr_sum += (double)b->power[k];
    double far_psd_sum = 0.0;
    for (int k = 0; k < K; ++k) {
        float r = b->far_spec[k].r, i = b->far_spec[k].i;
        far_psd_sum += (double)(r * r + i * i);
    }
    if (pwr_sum < 1e-10 && far_psd_sum > 1e-10) {
        for (int k = 0; k < K; ++k) {
            float r = b->far_spec[k].r, i = b->far_spec[k].i;
            b->power[k] = r * r + i * i;
        }
    } else {
        const float a = b->alpha_power, B = 1.0f - b->alpha_power;
        for (int k = 0; k < K; ++k) {
            float r = b->far_spec[k].r, i = b->far_spec[k].i;
            b->power[k] = a * b->power[k] + B * (r * r + i * i);
        }
    }

    /* echo_spec */
    memset(b->echo_spec, 0, (size_t)K * sizeof(Complex));
    for (int part = 0; part < N; ++part) {
        int p_idx = curr_p - part;
        while (p_idx < 0) p_idx += N;
        const Complex* Wp = b->W     + (size_t)part  * K;
        const Complex* Xp = b->X_buf + (size_t)p_idx * K;
        for (int k = 0; k < K; ++k) {
            float wr = Wp[k].r, wi = Wp[k].i;
            float xr = Xp[k].r, xi = Xp[k].i;
            b->echo_spec[k].r += wr * xr - wi * xi;
            b->echo_spec[k].i += wr * xi + wi * xr;
        }
    }

    /* IFFT echo */
    fft_inverse(b->fft, b->echo_spec, b->time_scratch);
    for (int i = 0; i < hop; ++i) {
        output[i] = b->near_buffer[blk - hop + i] - b->time_scratch[hop + i];
    }

    /* error_spec */
    memset(b->time_scratch, 0, (size_t)b->fft_size * sizeof(float));
    for (int i = 0; i < hop; ++i) b->time_scratch[hop + i] = output[i];
    fft_forward(b->fft, b->time_scratch, b->error_spec);

    /* Windowed near */
    for (int i = 0; i < blk; ++i) b->time_scratch[i] = b->near_buffer[i] * b->sqrt_hann[i];
    rfft_padded(b, b->time_scratch, blk, b->spec_scratch);
    for (int k = 0; k < K; ++k) {
        b->error_spec_windowed[k].r = b->spec_scratch[k].r - b->echo_spec[k].r;
        b->error_spec_windowed[k].i = b->spec_scratch[k].i - b->echo_spec[k].i;
    }

    /* Far-hop energy gate */
    double sum_far_sq = 0.0;
    for (int i = 0; i < hop; ++i) {
        double v = (double)far_end[i]; sum_far_sq += v * v;
    }
    double far_hop_energy = sum_far_sq / (double)hop;
    if (far_hop_energy > 1e-4) {
        /* === PBFDKF._update_weights ============================ */
        float mu_local[8192];
        const float* mu_arr;
        if (mu_scale_arr) mu_arr = mu_scale_arr;
        else { for (int k = 0; k < K; ++k) mu_local[k] = mu_scale_scalar; mu_arr = mu_local; }

        /* error_psd EMA + R = max(error_psd, delta), then R *= R_scale */
        const float ar = p->alpha_r, oar = 1.0f - p->alpha_r;
        for (int k = 0; k < K; ++k) {
            float er = b->error_spec[k].r, ei = b->error_spec[k].i;
            float epsd = er * er + ei * ei;
            p->error_psd[k] = ar * p->error_psd[k] + oar * epsd;
            float Rk = p->error_psd[k] > b->delta ? p->error_psd[k] : b->delta;
            p->R[k] = Rk;
        }

        /* mu_mean = float(np.mean(mu_arr)). To match Python's fp32 mean
         * via fp64 promote, then float() back: use fp32 pairwise (same
         * 8-block scheme used in detectors). */
        double mu_sum = 0.0;
        for (int k = 0; k < K; ++k) mu_sum += (double)mu_arr[k];
        float mu_mean = (float)(mu_sum / (double)K);

        /* R *= 0.1 + 0.9*(1-mu_mean); Q_modulated = Q * (0.1 + 0.9*mu_mean) */
        float R_scale = 0.1f + 0.9f * (1.0f - mu_mean);
        for (int k = 0; k < K; ++k) p->R[k] *= R_scale;
        float q_scale = 0.1f + 0.9f * mu_mean;

        /* far_activity_mask: power > mean(power)*0.01 + 1e-6 */
        double power_mean = 0.0;
        for (int k = 0; k < K; ++k) power_mean += (double)b->power[k];
        power_mean /= (double)K;
        float pwr_thr = (float)(power_mean * 0.01 + 1e-6);

        /* P_floor_beta countdown */
        float beta = p->p_floor_beta;
        if (p->p_floor_beta_frames > 0) {
            p->p_floor_beta_frames--;
            if (p->p_floor_beta_frames <= 0) p->p_floor_beta = 0.1f;
        }
        /* p_max countdown */
        float p_max_v = p->p_max_override;
        if (p->p_max_override_frames > 0) {
            p->p_max_override_frames--;
            if (p->p_max_override_frames <= 0) p->p_max_override = 0.5f;
        }

        /* Compute Q_gated[k] and P_floor[k] */
        float Q_gated[8192];
        float P_floor[8192];
        for (int k = 0; k < K; ++k) {
            float Qm = p->Q[k] * q_scale;
            float Qf = Qm * 0.05f;
            Q_gated[k] = (b->power[k] > pwr_thr) ? Qm : Qf;
            P_floor[k] = p->Q_high[k] * beta;
        }

        /* Global denominator: total_echo_var + R + delta
         *   total_echo_var[k] = sum_p (P[p][k] * |X_buf[(curr_p-p)%N][k]|^2) */
        float denom[8192];
        for (int k = 0; k < K; ++k) denom[k] = p->R[k] + b->delta;
        for (int part = 0; part < N; ++part) {
            int p_idx = curr_p - part;
            while (p_idx < 0) p_idx += N;
            const Complex* Xp = b->X_buf + (size_t)p_idx * K;
            const float*   Pp = p->P    + (size_t)part * K;
            for (int k = 0; k < K; ++k) {
                float xr = Xp[k].r, xi = Xp[k].i;
                float xm2 = xr * xr + xi * xi;
                denom[k] += Pp[k] * xm2;
            }
        }

        /* Per-partition Kalman update */
        for (int part = 0; part < N; ++part) {
            int p_idx = curr_p - part;
            while (p_idx < 0) p_idx += N;
            const Complex* Xp = b->X_buf + (size_t)p_idx * K;
            Complex*       Wp = b->W    + (size_t)part  * K;
            float*         Pp = p->P    + (size_t)part  * K;
            for (int k = 0; k < K; ++k) {
                float xr = Xp[k].r, xi = Xp[k].i;
                float xm2 = xr * xr + xi * xi;
                /* K_optimal = (P[p] * conj(X)) / denom -> complex */
                float ko_r = (Pp[k] * xr) / denom[k];
                float ko_i = -(Pp[k] * xi) / denom[k];
                /* K_scaled = K_optimal * mu_scale_arr[k] */
                float ks_r = ko_r * mu_arr[k];
                float ks_i = ko_i * mu_arr[k];
                /* W[p] += K_scaled * error_spec */
                float er = b->error_spec[k].r, ei = b->error_spec[k].i;
                Wp[k].r += ks_r * er - ks_i * ei;
                Wp[k].i += ks_r * ei + ks_i * er;

                /* KX_optimal = real(K_optimal * X)
                 *   K_optimal = (Pp*xr/denom, -Pp*xi/denom)
                 *   X = (xr, xi)
                 *   real(K*X) = Pp*xr*xr/denom - (-Pp*xi)*xi/denom
                 *             = Pp*(xr^2 + xi^2)/denom = Pp*xm2/denom
                 * Same for KX_scaled with multiplier mu_arr[k]. Both
                 * collapse to Pp*xm2/denom * (1 or mu) — matching
                 * np.real(K*X) since real of complex k * complex x. */
                float KX_optimal = Pp[k] * xm2 / denom[k];
                float KX_scaled  = KX_optimal * mu_arr[k];
                /* Cast mu_mean to fp32 (Python np.float32(mu_mean)) */
                float mu_mean_f32 = mu_mean;
                float KX = mu_mean_f32 * KX_optimal +
                           (1.0f - mu_mean_f32) * KX_scaled;
                float Pnew = (1.0f - KX) * Pp[k] + Q_gated[k];
                if (Pnew < P_floor[k]) Pnew = P_floor[k];
                if (Pnew > p_max_v)    Pnew = p_max_v;
                Pp[k] = Pnew;
            }
            /* TD constraint */
            if (b->enable_td_constraint) {
                fft_inverse(b->fft, Wp, b->time_scratch);
                for (int i = 0; i < b->fft_size; ++i) b->time_scratch[i] *= b->td_window[i];
                fft_forward(b->fft, b->time_scratch, Wp);
            }
        }
    }

    b->partition_idx = (b->partition_idx + 1) % N;
}
