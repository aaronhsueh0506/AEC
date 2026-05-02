/* pbfdkf_v2.h — 1:1 port of python/aec.py PBFDAF (lines 584-739) and
 * PBFDKF (lines 741-910).
 *
 * Partitioned Block Frequency-Domain Adaptive Filter with overlap-save.
 * PBFDAF = NLMS update; PBFDKF = Kalman gain (P/Q/R) per-bin update with
 * v3.7 G1 KX-blended P-update (aec.py:856-868).
 *
 * Design rules:
 *   - All array dtypes mirror Python (np.complex64 ⇒ Complex; np.float32
 *     ⇒ float). EMA scalars (mu_mean, etc.) computed in fp64 then cast to
 *     fp32 to match Python's `np.float32(mu_mean)` step.
 *   - One struct per Python class. PBFDKF embeds a PBFDAF as `base` —
 *     this matches Python inheritance functionally without C++.
 *   - All buffers heap-allocated at create; size derived from hop_size /
 *     n_partitions (User Fix 6: no [160] magic numbers).
 */
#ifndef AEC_V2_PBFDKF_H
#define AEC_V2_PBFDKF_H

#include <stddef.h>
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PBFDAFV2 {
    int     hop_size;
    int     block_size;        /* 2 * hop_size */
    int     fft_size;          /* next pow2 >= block_size */
    int     n_partitions;
    int     n_freqs;           /* fft_size/2 + 1 */
    float   mu;
    float   delta;
    float   alpha_power;       /* 0.9 */
    int     enable_td_constraint;

    float*  td_window;         /* [fft_size] */
    float*  sqrt_hann;         /* [block_size] */

    /* W [n_partitions][n_freqs] complex64 */
    Complex* W;
    /* X_buf [n_partitions][n_freqs] complex64 (history of far_spec) */
    Complex* X_buf;
    int      partition_idx;

    float*  near_buffer;       /* [block_size] */
    float*  far_buffer;        /* [block_size] */
    float*  power;             /* [n_freqs] far PSD EMA */

    Complex* near_spec;        /* [n_freqs] */
    Complex* echo_spec;        /* [n_freqs] */
    Complex* error_spec;       /* [n_freqs] */
    Complex* far_spec;         /* [n_freqs] */
    Complex* error_spec_windowed; /* [n_freqs] */

    /* FFT scratch (private) */
    FftHandle* fft;
    float*     time_scratch;   /* [fft_size] */
    Complex*   spec_scratch;   /* [n_freqs] */
} PBFDAFV2;

void pbfdaf_v2_init(PBFDAFV2* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size);
void pbfdaf_v2_free(PBFDAFV2* p);
void pbfdaf_v2_reset(PBFDAFV2* p);

/* mu_scale: NULL ⇒ scalar 1.0; else array of n_freqs fp32. Returns nothing
 * (output written to `output`, length hop_size). */
void pbfdaf_v2_process(PBFDAFV2* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,   /* NULL or [n_freqs] */
                       float        mu_scale_scalar,
                       float*       output);

void pbfdaf_v2_copy_weights_from(PBFDAFV2* dst, const PBFDAFV2* src);

/* ── PBFDKF (Kalman) ─────────────────────────────────────────── */

typedef struct PBFDKFV2 {
    PBFDAFV2 base;

    /* Per-partition per-bin error covariance [n_partitions][n_freqs] */
    float* P;
    /* Process noise [n_freqs] */
    float* Q_high;             /* 1e-4 init */
    float* Q_low;              /* 1e-5 init */
    float* Q;
    /* Measurement noise [n_freqs] */
    float* R;
    float* error_psd;
    float  alpha_r;            /* 0.95 */

    /* Transient overrides (PR-G2 / EPC); 0 / -1 sentinels */
    float p_max_override;        /* default 0.5 */
    int   p_max_override_frames; /* < 0 means unset */
    float p_floor_beta;          /* default 0.1 */
    int   p_floor_beta_frames;   /* < 0 means unset */
} PBFDKFV2;

void pbfdkf_v2_init(PBFDKFV2* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size);
void pbfdkf_v2_free(PBFDKFV2* p);
void pbfdkf_v2_reset(PBFDKFV2* p);
/* PBFDKF process = PBFDAF process with overridden _update_weights. */
void pbfdkf_v2_process(PBFDKFV2* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,
                       float        mu_scale_scalar,
                       float*       output);
/* W copy only (P/Q/R untouched). */
void pbfdkf_v2_copy_weights_from(PBFDKFV2* dst, const PBFDKFV2* src);

#ifdef __cplusplus
}
#endif

#endif /* AEC_V2_PBFDKF_H */
