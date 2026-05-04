/* pbfdkf.h — 1:1 port of python/aec.py PBFDAF (lines 584-739) and
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
#ifndef AEC_PBFDKF_H
#define AEC_PBFDKF_H

#include <stddef.h>
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PBFDAF {
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
} PBFDAF;

void pbfdaf_init(PBFDAF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size);
void pbfdaf_free(PBFDAF* p);
void pbfdaf_reset(PBFDAF* p);

/* mu_scale: NULL ⇒ scalar 1.0; else array of n_freqs fp32. Returns nothing
 * (output written to `output`, length hop_size). */
void pbfdaf_process(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,   /* NULL or [n_freqs] */
                       float        mu_scale_scalar,
                       float*       output);

void pbfdaf_copy_weights_from(PBFDAF* dst, const PBFDAF* src);

/* ── PBFDKF (Kalman) ─────────────────────────────────────────── */

typedef struct PBFDKF {
    PBFDAF base;

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
} PBFDKF;

void pbfdkf_init(PBFDKF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size);
void pbfdkf_free(PBFDKF* p);
void pbfdkf_reset(PBFDKF* p);
/* PBFDKF process = PBFDAF process with overridden _update_weights. */
void pbfdkf_process(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,
                       float        mu_scale_scalar,
                       float*       output);
/* W copy only (P/Q/R untouched). */
void pbfdkf_copy_weights_from(PBFDKF* dst, const PBFDKF* src);

#ifdef __cplusplus
}
#endif

#endif /* AEC_PBFDKF_H */
