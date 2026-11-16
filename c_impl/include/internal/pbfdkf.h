/* Partitioned-block frequency-domain adaptive filters with overlap-save: PBFDAF (NLMS update,
 * shadow / coarse filter) and PBFDKF (per-bin H_error gain with leakage refresh, refined filter;
 * embeds a PBFDAF as `base`).
 * PBFDKF update: mu[k] = H_error / (0.5 * H_error * X^2 + n * E^2_refined + delta);
 * W[p] += mu * conj(X[p]) * mu_scale * error_spec; in the active branch
 * H_error -= 0.5 * mu * X^2 * H_error; always H_error += leakage * erl, clamped to [floor, ceil].
 * Leakage is the converged or diverged rate per bin (E^2_ref <= E^2_coa), with a transient profile
 * while the initial state lasts (2.5 s of active render). Python reference: modules/filters.py
 * (PBFDAF, PBFDKF). */
#ifndef AEC_PBFDKF_H
#define AEC_PBFDKF_H

#include <stddef.h>
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PBFDAF {
    int     hop_size;
    int     sample_rate;       /* drives the wall-clock-rescaled fields */
    int     block_size;
    int     fft_size;
    int     n_partitions;
    int     n_freqs;
    float   mu;
    float   delta;
    int     enable_td_constraint;

    float*  td_window;
    float*  sqrt_hann;

    /* W [n_partitions][n_freqs] */
    Complex* W;
    /* X_buf [n_partitions][n_freqs] (far_spec history) */
    Complex* X_buf;
    int      partition_idx;

    float*  near_buffer;
    float*  far_buffer;

    Complex* near_spec;
    Complex* echo_spec;
    Complex* error_spec;
    Complex* far_spec;
    Complex* error_spec_windowed;

    /* FFT sharing between main and shadow filters: when the caller already computed far_spec this
     * hop it points it here and the frontend reuses it (one-shot, borrowed). `lightweight` (shadow)
     * skips near_spec and error_spec_windowed. */
    const Complex* precomputed_far_spec;
    int      lightweight;
    /* Counts hops in which this filter itself ran a far-end rfft; read via
     * aec_far_fft_real_compute_count(). */
    long     far_fft_real_compute_count;

    /* Round-robin time-domain constraint, one partition per hop. aec_create sets
     * constraint_round_robin = 1 on both filters (config, never reset); the cursor survives
     * pbfdaf_reset() and is re-armed only by pbfdaf_reset_carried_state(). */
    int      constraint_round_robin;
    int      partition_to_constrain;

    /* Time-domain peak max |echo_time[hop:block]| of the last frontend pass, used by the
     * SaturationDetector. */
    float   last_s_max_abs;

    /* Coarse-filter update-gain gates */
    long    call_counter;
    long    poor_excitation_counter;
    int     saturated_capture;
    int     block_stationary;

    /* InitialState tracking (PBFDKF only) */
    int     initial_state_active;
    long    initial_state_active_render_hops;
    long    initial_state_threshold_hops;
    float   initial_state_far_energy_floor;

    /* FFT scratch (private) */
    FftHandle* fft;
    float*     time_scratch;
    Complex*   spec_scratch;

    /* Per-hop scratch in instance storage, not on the stack. */
    float*  scr_fsq;
    float*  scr_mu_local;
    float*  scr_x2psum;
    float*  scr_mu_eff;
    float*  scr_ir;

    /* x2_cache [n_partitions][n_freqs] holds |X_buf[p][k]|^2 row for row; every X_buf writer
     * updates it in the same step. far_cmag2_hold [n_freqs] holds |far_spec[k]|^2 and is zeroed
     * where far_spec is; before a hop's frontend it holds the previous hop's spectrum, after it the
     * current one. */
    float*  x2_cache;
    float*  far_cmag2_hold;

    /* Set by every writer of W and by init. Its one consumer (the per-bin ERL publish) clears it
     * after taking the sum; a writer that skips it serves a stale ERL to the H_error refresh. */
    int     w_dirty;
} PBFDAF;

/* Buffers a filter writes before it reads inside one call, lent to the *_borrowed constructors when the filters run one after another.
 * pbfdaf_scratch_layout() returns the bytes (s == NULL) or also points *s into `mem`, for a main filter (follower 0) or a follower
 * (follower 1); 0 for an unusable geometry. */
typedef struct PbfdafScratch {
    float*   td_window;      /* not scratch: the constant tables (fft_size and block_size floats), filled by every init that borrows */
    float*   sqrt_hann;      /* main only; the layout leaves both NULL for the caller to point */
    float*   time_scratch;   /* fft_size floats */
    Complex* spec_scratch;   /* n_freqs bins; main only */
    float*   scr_fsq;        /* hop_size floats */
    float*   scr_ir;         /* n_partitions * hop_size floats */
    float*   scr_mu_local;   /* n_freqs floats each; follower only */
    float*   scr_x2psum;
    float*   scr_mu_eff;
    float*   kf_mu_local;    /* n_freqs floats each; main only */
    float*   kf_x2;
    float*   kf_mu_aec;
} PbfdafScratch;
size_t pbfdaf_scratch_layout(PbfdafScratch* s, void* mem, int block_size, int n_partitions, int hop_size, int follower);

/* Sizing never includes a nested FFT: pbfdaf_init_static() borrows a caller-supplied shared_fft.
 * with_process_scratch = 1 carves the pbfdaf_process()-only scratch; pass 0 for an instance driven
 * only via the frontend (the PBFDKF base). */
size_t pbfdaf_get_mem_size(int block_size, int n_partitions, int hop_size,
                           int with_process_scratch);
/* shared_fft: required, already fft_init()'d at this fft_size; borrowed, never destroyed here
 * (its creator destroys it once). Use is single-threaded and sequential. Returns 0, or -1 if the
 * pool checks reject the inputs, shared_fft is NULL or sample_rate/hop is non-positive (nothing is
 * written). */
int    pbfdaf_init_static(PBFDAF* p, void* mem, size_t mem_size,
                           int block_size, int n_partitions,
                           float mu, float delta, int hop_size,
                    int with_process_scratch, int sample_rate,
                    FftHandle* shared_fft);
/* Follower: a lightweight shadow of far_owner (an initialised filter at the same n_partitions and hop_size). It lacks sqrt_hann
 * (NULL), near_spec, error_spec_windowed and spec_scratch, and shares far_owner's X_buf and x2_cache, so both must see the same far input and the
 * same far-history clears and partition steps every hop. Process scratch is included. */
size_t pbfdaf_get_mem_size_follower(int block_size, int n_partitions, int hop_size);
int    pbfdaf_init_static_follower(PBFDAF* p, void* mem, size_t mem_size,
                                   int block_size, int n_partitions,
                                   float mu, float delta, int hop_size,
                                   int sample_rate, FftHandle* shared_fft,
                                   const PBFDAF* far_owner);
/* The same follower without its own scratch rows: they come from `scratch` (a pbfdaf_scratch_layout() with follower 1). */
size_t pbfdaf_get_mem_size_follower_borrowed(int block_size, int n_partitions, int hop_size);
int    pbfdaf_init_static_follower_borrowed(PBFDAF* p, void* mem, size_t mem_size,
                                            int block_size, int n_partitions,
                                            float mu, float delta, int hop_size,
                                            int sample_rate, FftHandle* shared_fft,
                                            const PBFDAF* far_owner, const PbfdafScratch* scratch);
/* Drops the borrowed shared_fft reference only. */
void pbfdaf_free(PBFDAF* p);
void pbfdaf_reset(PBFDAF* p);

/* Adaptation-only wipe: taps (W) and far history (X_buf); the time-domain analysis buffers are
 * kept. pbfdaf_reset() adds those buffers. */
void pbfdaf_reset_taps(PBFDAF* p);

/* Restores every scalar pbfdaf_reset() carries forward to its construction value (gates,
 * InitialState pair, round-robin cursor, last subtractor peak). Used by aec_reset(). */
void pbfdaf_reset_carried_state(PBFDAF* p);

/* Wipes X_buf and its |.|^2 mirror x2_cache together; taps and analysis buffers are kept. */
void pbfdaf_clear_far_history(PBFDAF* p);

/* mu_scale: NULL for a scalar, else per-bin float[n_freqs]. Writes hop_size samples to `output`. */
void pbfdaf_process(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,
                       float        mu_scale_scalar,
                       float*       output);

/* As pbfdaf_process with an optional exact |far_spec|^2 from this library's cmag2 kernel; NULL
 * makes the frontend compute it. */
void pbfdaf_process_with_far_cmag2(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale, float mu_scale_scalar,
                       float* output, const float* precomputed_far_cmag2);

void pbfdaf_copy_weights_from(PBFDAF* dst, const PBFDAF* src);
/* Warm tap transfer: shifts the learned IR by shift_samples (positive: toward tap 0). Returns 0,
 * or -1 when refused (n_partitions * hop_size exceeds the staging scratch; the filter is untouched
 * and the caller must not report warm success). A negative shift also clears the far history. */
int pbfdaf_warm_shift_ir(PBFDAF* p, int shift_samples);

/* PBFDKF */

typedef struct PBFDKF {
    PBFDAF base;

    float* error_psd;
    float  alpha_r;            /* error-PSD EMA retention, retimed from the 160 @ 16 kHz authoring grid */

    /* H_error per-bin state */
    float* H_error_per_bin;    /* init 10000 */
    float  h_error_floor;
    float  h_error_ceil;
    float  leakage_converged;
    float  leakage_diverged;
    float  leakage_converged_transient;
    float  leakage_diverged_transient;

    /* Per-hop orchestrator-set inputs to the refresh. */
    float* e2_coarse_per_bin;  /* [n_freqs] or NULL (scalar fallback) */
    int    e2_coarse_per_bin_valid;
    float  e2_coarse_for_refresh;
    float* erl_per_bin;        /* [n_freqs] = sum_p |W_p|^2 */
    int    disallow_leakage_diverged;
    float  h_error_refresh_erl_floor;   /* 0 = off */
    /* Fraction of bins on the diverged-leakage branch in the last refresh. */
    float  last_leakage_div_frac;

    long   partition_sum_x2_startup_hops;

    /* Per-hop |error_spec|^2 [n_freqs], computed once in pbfdkf_process_with_far_cmag2 and shared. */
    float* e2_ref_scratch;

    /* More per-hop scratch, n_freqs each. */
    float* scr_mu_local;
    float* scr_X2;
    float* scr_mu_aec;

} PBFDKF;

size_t pbfdkf_get_mem_size(int block_size, int n_partitions, int hop_size);
/* shared_fft is forwarded to pbfdaf_init_static(). Returns 0, or -1 as that function does. */
int    pbfdkf_init_static(PBFDKF* p, void* mem, size_t mem_size,
                           int block_size, int n_partitions,
                           float mu, float delta, int hop_size,
                           int sample_rate, FftHandle* shared_fft);
/* The same main filter without its scratch rows: they come from `scratch` (a pbfdaf_scratch_layout() with follower 0). */
size_t pbfdkf_get_mem_size_borrowed(int block_size, int n_partitions, int hop_size);
int    pbfdkf_init_static_borrowed(PBFDKF* p, void* mem, size_t mem_size,
                                   int block_size, int n_partitions,
                                   float mu, float delta, int hop_size,
                                   int sample_rate, FftHandle* shared_fft, const PbfdafScratch* scratch);
void pbfdkf_free(PBFDKF* p);
void pbfdkf_reset(PBFDKF* p);
/* pbfdaf_reset_taps plus the error_psd re-arm. */
void pbfdkf_reset_taps(PBFDKF* p);
/* pbfdaf_reset_carried_state plus the orchestrator-fed refresh inputs and the per-bin H_error. */
void pbfdkf_reset_carried_state(PBFDKF* p);

/* As pbfdaf_process_with_far_cmag2. mu_scale: NULL for a scalar broadcast, else per-bin float[n_freqs]
 * (after the narrow-band mask). Returns 1 when e2_ref_scratch holds this hop's |error_spec|^2, 0 on
 * the far-silent path. */
int pbfdkf_process_with_far_cmag2(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale, float mu_scale_scalar,
                       float* output, const float* precomputed_far_cmag2);

/* Resets H_error on a delay change and re-arms the coarse-filter gain counter. */
void pbfdkf_handle_echo_path_change(PBFDKF* p, int delay_change, int gain_change);

/* The filter's reach in taps, used by the warm-transfer eligibility checks. */
static inline int pbfdaf_tap_span(const PBFDAF* p) {
    return p->n_partitions * p->hop_size;
}

/* Concatenates irfft(W[p])[:hop] per partition into out[n_partitions * hop_size]; writes only
 * partitions [first_partition, first_partition + n_parts) and leaves other taps untouched. */
void pbfdaf_get_time_domain_filter_range(PBFDAF* p, int first_partition,
                                         int n_parts, float* out);

/* W *= scale, all partitions. */
void pbfdkf_scale_filter(PBFDKF* p, float scale);

#ifdef __cplusplus
}
#endif

#endif
