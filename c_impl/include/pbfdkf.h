/* pbfdkf.h — C port of python/modules/filters.py PBFDAF + PBFDKF (v3.22).
 *
 * Partitioned Block Frequency-Domain Adaptive Filter with overlap-save.
 *   PBFDAF — NLMS update (shadow / coarse filter).
 *   PBFDKF — v3.21/v3.22 redesign: per-bin AEC3 H_error Kalman gain + leakage
 *            refresh (refined_filter_update_gain.cc:104-138). REPLACES the old
 *            v3.10 P-denominator Kalman body. The legacy P/Q/R fields are gone.
 *
 * v3.10 → v3.22 PBFDKF change (the crux of this port):
 *   OLD (v3.10): K = P·conj(X) / (Σ_p P·|X|² + R + δ), with R-EMA, Q-modulation,
 *                P-Riccati update (1-KX)·P + Q, KX-blend, P_floor/p_max clamps.
 *   NEW (v3.22): per-bin H_error state. mu[k] = H_error / (0.5·H_error·X² +
 *                n·E²_refined + δ); W[p] += mu·conj(X[p])·mu_scale·error_spec;
 *                H_error -= 0.5·mu·X²·H_error (decay, active branch only);
 *                always-on refresh H_error += leakage·erl, clamp[floor,ceil].
 *                leakage = converged/diverged chosen per-bin by E²_ref≤E²_coa,
 *                transient (initial_state) profile for first 250 active hops.
 *
 * Design rules (parity):
 *   - All array dtypes mirror Python: np.complex64 ⇒ Complex; np.float32 ⇒
 *     float. The FFT primitive (fft_pocketfft.c, vendored numpy pocketfft) is
 *     now BIT-EXACT vs np.fft. FFT-derived arrays here are still rtol<1e-4 (NOT
 *     bit-exact), but the residual is pbfdkf's own fp32 accumulator drift, not
 *     the FFT. The H_error / mu / leakage-refresh ARITHMETIC is exact
 *     given identical inputs, but its inputs (|error_spec|², |X|²) come THROUGH
 *     the FFT, so H_error / error_psd / R inherit the ≤1-ULP FFT drift too —
 *     they are rtol-bounded, NOT bit-exact. Only the integer control state
 *     (call_counter / partition_idx / initial_state) is bit-exact.
 *   - One struct per Python class. PBFDKF embeds a PBFDAF as `base`.
 *   - Built with -ffp-contract=off (Makefile) — no FMA fusion.
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
    float*  sqrt_hann;         /* [block_size] sqrt-Hann analysis window */

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

    /* AEC3 SubtractorOutput.s_*_max_abs — time-domain peak of the echo
     * predictor on the valid hop (max|echo_time[hop:block]|). Recomputed every
     * frontend pass; consumed by AecState SaturationDetector. */
    float   last_s_max_abs;

    /* AEC3 CoarseFilterUpdateGain startup / poor-excitation / saturation gates */
    long    call_counter;
    long    poor_excitation_counter;
    int     saturated_capture;
    int     block_stationary;

    /* AEC3 InitialState tracking (shared field; only PBFDKF consumes it). */
    int     initial_state_active;
    long    initial_state_active_render_hops;
    long    initial_state_threshold_hops;
    float   initial_state_far_energy_floor;
    int     last_initial_state_active;

    /* FFT scratch (private) */
    FftHandle* fft;
    float*     time_scratch;   /* [fft_size] */
    Complex*   spec_scratch;   /* [n_freqs] */

    int is_static;             /* 1 = state placed in caller buffer */
} PBFDAF;

void   pbfdaf_init(PBFDAF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size);
size_t pbfdaf_get_mem_size(int block_size, int n_partitions, int hop_size);
void   pbfdaf_init_static(PBFDAF* p, void* mem, size_t mem_size,
                           int block_size, int n_partitions,
                           float mu, float delta, int hop_size);
void pbfdaf_free(PBFDAF* p);
void pbfdaf_reset(PBFDAF* p);

/* mu_scale: NULL ⇒ scalar; else per-bin array of n_freqs fp32. Output written
 * to `output` (length hop_size). */
void pbfdaf_process(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,   /* NULL or [n_freqs] */
                       float        mu_scale_scalar,
                       float*       output);

void pbfdaf_copy_weights_from(PBFDAF* dst, const PBFDAF* src);

/* ── PBFDKF (per-bin H_error Kalman) ────────────────────────────── */

typedef struct PBFDKF {
    PBFDAF base;

    /* Process noise [n_freqs] — kept as benign config state (orchestrator sets
     * Q_high/Q_low/Q at construction); NOT consumed by the H_error update. */
    float* Q_high;             /* config kalman_q_high */
    float* Q_low;              /* config kalman_q_low */
    float* Q;
    /* Measurement-noise EMA [n_freqs] (error_psd EMA + R = max(error_psd,δ)).
     * R is computed in the active path but no longer feeds the H_error gain;
     * retained because the Python class still maintains it per hop. */
    float* R;
    float* error_psd;
    float  alpha_r;            /* 0.95 */

    /* === v3.22 AEC3 H_error per-bin state =========================== */
    float* H_error_per_bin;    /* [n_freqs], init 10000 */
    float  h_error_floor;      /* 1e-3 */
    float  h_error_ceil;       /* 2.0 */
    float  leakage_converged;  /* 1.25e-4 steady */
    float  leakage_diverged;   /* 0.125  steady */
    float  leakage_converged_transient;  /* 0.0125 (initial_state) */
    float  leakage_diverged_transient;   /* 1.25   (initial_state) */

    /* Per-hop orchestrator-set inputs to the refresh formula. */
    float* e2_coarse_per_bin;  /* [n_freqs] or NULL (then scalar fallback) */
    int    e2_coarse_per_bin_valid;
    float  e2_coarse_for_refresh;   /* scalar fallback */
    float* erl_per_bin;        /* [n_freqs] = Σ_p|W_p|² (orchestrator-set) */
    int    disallow_leakage_diverged;
    float  h_error_refresh_erl_floor;   /* 0.0 = OFF */
    /* Diag: fraction of bins on the diverged-leakage branch in the most
     * recent H_error refresh (np.mean(~use_converged_mask)). Read by the
     * orchestrator Track F sustained-leakage gate. */
    float  last_leakage_div_frac;

    long   partition_sum_x2_startup_hops;  /* 0 = partition-sum from hop 1 */

    int is_static;               /* 1 = state placed in caller buffer */
} PBFDKF;

void   pbfdkf_init(PBFDKF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size);
size_t pbfdkf_get_mem_size(int block_size, int n_partitions, int hop_size);
void   pbfdkf_init_static(PBFDKF* p, void* mem, size_t mem_size,
                           int block_size, int n_partitions,
                           float mu, float delta, int hop_size);
void pbfdkf_free(PBFDKF* p);
void pbfdkf_reset(PBFDKF* p);

/* mu_scale: NULL ⇒ scalar broadcast; else per-bin [n_freqs] fp32 (already
 * post-RSA-mask, matching the array the Python _update_weights_aec3 receives). */
void pbfdkf_process(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,
                       float        mu_scale_scalar,
                       float*       output);

/* AEC3 RefinedFilterUpdateGain::HandleEchoPathChange — H_error reset to init
 * on delay_change + CoarseFilterUpdateGain counter reset. */
void pbfdkf_handle_echo_path_change(PBFDKF* p, int delay_change, int gain_change);

/* W copy only (H_error / R untouched). */
void pbfdkf_copy_weights_from(PBFDKF* dst, const PBFDKF* src);

/* float(np.sum(np.abs(error_spec)**2)) — bit-exact (cmag2_np + pairwise). */
double pbfdaf_get_error_energy(const PBFDAF* p);
double pbfdkf_get_error_energy(const PBFDKF* p);

/* get_time_domain_filter: concat per-partition irfft(W[p])[:hop] →
 * out[n_partitions × hop_size] (caller-owned). Mirrors filters.py:397. */
void pbfdaf_get_time_domain_filter(PBFDAF* p, float* out);

/* scale_filter: W *= (float32)scale (in place, all partitions). */
void pbfdaf_scale_filter(PBFDAF* p, float scale);
void pbfdkf_scale_filter(PBFDKF* p, float scale);

#ifdef __cplusplus
}
#endif

#endif /* AEC_PBFDKF_H */
