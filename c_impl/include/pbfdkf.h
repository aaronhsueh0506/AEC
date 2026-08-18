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
    int     sample_rate;       /* drives every wall-clock-rescaled field below
                                 * (initial_state_threshold_hops here;
                                 * PBFDKF's leakage rates and its
                                 * poor_excitation_counter reset, via
                                 * base.sample_rate) */
    int     block_size;        /* 2 * hop_size */
    int     fft_size;          /* next pow2 >= block_size */
    int     n_partitions;
    int     n_freqs;           /* fft_size/2 + 1 */
    float   mu;
    float   delta;
    float   alpha_power;       /* far-end power EMA retention. AUTHORING value
                                * 0.9 at hop=160/16000 (10 ms); the RUNTIME
                                * value is retimed to this instance's grid by
                                * pbfdaf_init_scalars() and equals 0.9 only on
                                * a 10 ms hop. Read the effective value, never
                                * this comment -- test_rate_structural (d2). */
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

    /* FFT dedup (main+shadow share an identical far_buffer: lockstep shift +
     * paired reset). When the caller already computed far_spec this hop (the
     * shadow runs pre-main), it points it here so the frontend reuses it instead
     * of a redundant rfft; one-shot (cleared after use). `lightweight` (set on
     * the shadow) skips near_spec + error_spec_windowed, which no consumer reads.
     * Both are byte-equal. Borrowed pointer — not pool-owned. */
    const Complex* precomputed_far_spec;
    int      lightweight;
    /* Far-end-FFT-sharing instrumentation: incremented exactly once per hop this
     * filter ACTUALLY runs a far-end rfft (the else-branch in
     * pbfdaf_frontend() below) -- i.e. NOT when precomputed_far_spec was
     * set (internal shadow->main borrow OR an external caller's
     * aec_process_context_shared_far()). Never read internally; exposed
     * read-only via aec_far_fft_real_compute_count() for tests/instrumentation
     * that need to prove a multi-lane caller's far-FFT count actually
     * dropped after switching lanes over to the shared-spectrum entry
     * point, without conflating it with the near/error/W-tap FFTs that
     * also flow through the same underlying fft_forward_scratch(). */
    long     far_fft_real_compute_count;

    /* AEC3 round-robin TD constraint (adaptive_fir_filter.cc:686-689): constrain
     * ONE partition per hop (cycling `partition_to_constrain`) instead of all
     * partitions every hop — saves ~58% of the per-hop constraint FFTs and
     * deepens convergence (the old all-every-hop over-constrained). aec_create
     * sets `constraint_round_robin`=1 on BOTH filters (production default). Each
     * filter keeps its own counter; persists across reset() (mirrors Python —
     * reset() does not touch it). */
    int      constraint_round_robin;
    int      partition_to_constrain;

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

    /* Per-hop scratch, instance storage (not stack) — keeps pbfdaf_process /
     * pbfdaf_frontend / pbfdaf_get_error_energy / pbfdaf_warm_shift_ir stack
     * usage independent of n_freqs / n_partitions (embedded task stacks are
     * small; these were float[8192]/float[4096] locals, a stack-overflow
     * hazard). Mirrors the e2_ref_scratch pattern below on PBFDKF. One field
     * per distinct concurrent live range (see pbfdkf.c comments at each use). */
    float*  scr_fsq;           /* [hop_size]     pbfdaf_frontend far_end**2 */
    float*  scr_mu_local;      /* [n_freqs]      pbfdaf_process mu broadcast */
    float*  scr_x2psum;        /* [n_freqs]      pbfdaf_process X_buf**2 partition sum */
    float*  scr_mu_eff;        /* [n_freqs]      pbfdaf_process per-bin mu */
    float*  scr_ir;            /* [n_partitions*hop_size] pbfdaf_warm_shift_ir IR concat */
    float*  scr_e2;            /* [n_freqs] pbfdaf_get_error_energy |error_spec|**2 --
                                 * ALSO pbfdaf_frontend's far_spec cmag2 scratch (far_psd_sum
                                 * + cold-start/EMA power update), reused across the two
                                 * phases -- frontend fully writes-then-
                                 * consumes it within its own call, and get_error_energy fully
                                 * overwrites-then-reads it within its own call (its only
                                 * caller, aec.c step 13, runs strictly after that hop's
                                 * frontend, itself invoked from pbfdaf_process/pbfdkf_process
                                 * step 9/8.5), so nothing ever reads either phase's data
                                 * through the other's stale contents -- see pbfdaf_frontend's
                                 * own comment for the full argument. Saves one
                                 * ALIGN16(n_freqs*4)-byte field per instance (main + shadow). */

    int is_static;             /* 1 = state placed in caller buffer */
} PBFDAF;

/* with_process_scratch: 1 allocates the pbfdaf_process()-only scratch
 * (scr_mu_local/scr_x2psum/scr_mu_eff). Pass 0 when the instance is only
 * driven via pbfdaf_frontend() (the PBFDKF base) — pbfdaf_process() must
 * NOT be called on such an instance. */
/* F04: returns 0 on success, -1 if the nested fft_create() allocation failed
 * (OOM) — the filter's arrays are all freed before returning (mirrors
 * pbfdaf_free's heap branch) rather than leaving a half-built filter around
 * whose FFT calls would silently no-op on the NULL handle.
 * Also -1, with nothing written to `p`, if p is NULL or sample_rate/hop are
 * non-positive: sample_rate retimes alpha_power and
 * initial_state_threshold_hops, so an unusable rate must be refused here
 * rather than producing a filter whose EMAs never adapt. */
int    pbfdaf_init(PBFDAF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size,
                    int with_process_scratch, int sample_rate);
/* Static-path sizing never includes a nested FFT: pbfdaf_init_static() always
 * runs against a caller-supplied `shared_fft` (one FftHandle shared
 * by main filter / shadow filter / aec3_post within a single Aec instance,
 * carved once by aec_get_mem_size()/aec_carve() rather than once per
 * sub-module). Only the heap path (pbfdaf_init(), pbfdaf_get_mem_size() has
 * no heap counterpart because the heap path never queries a size up front)
 * owns its own FFT. */
size_t pbfdaf_get_mem_size(int block_size, int n_partitions, int hop_size,
                           int with_process_scratch);
/* shared_fft: REQUIRED, non-NULL — an FftHandle already fft_init()'d by the
 * caller at this instance's fft_size (pbfdaf_compute_sizes(hop_size, ...)).
 * `p` borrows it for its lifetime and never destroys it; the caller (whoever
 * constructed shared_fft) remains solely responsible for fft_destroy()-ing it
 * exactly once, after every borrower is done with it. Precondition:
 * single-threaded, fully sequential use — at most one transform in flight
 * across ALL borrowers of shared_fft at any time, and no borrower may retain
 * a pointer into the handle's internal scratch across calls. F04: returns 0
 * on success, -1 if the pool base/size checks reject the inputs (F05/F07,
 * nothing written to `mem`), shared_fft is NULL, or sample_rate/resolved hop
 * is non-positive. Invalid arguments are rejected before writing `p` or mem. */
int    pbfdaf_init_static(PBFDAF* p, void* mem, size_t mem_size,
                           int block_size, int n_partitions,
                           float mu, float delta, int hop_size,
                    int with_process_scratch, int sample_rate,
                    FftHandle* shared_fft);
void pbfdaf_free(PBFDAF* p);
void pbfdaf_reset(PBFDAF* p);

/* Adaptation-only wipe: taps (W), the per-partition far SPECTRUM history
 * (X_buf) and the far-power normalizer, leaving the time-domain analysis
 * buffers (near_buffer/far_buffer, the block_size overlap-save history that
 * feeds near_spec/far_spec/error_spec_windowed and supplies the near term of
 * the linear output) untouched. pbfdaf_reset() is this plus those buffers.
 *
 * For a caller that must restart the filter mid-stream WITHOUT splicing a
 * half-empty analysis frame into the spectra a post-filter consumes: the
 * taps are what a realign invalidates, the analysis history is not. */
void pbfdaf_reset_taps(PBFDAF* p);

/* mu_scale: NULL ⇒ scalar; else per-bin array of n_freqs fp32. Output written
 * to `output` (length hop_size). */
void pbfdaf_process(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,   /* NULL or [n_freqs] */
                       float        mu_scale_scalar,
                       float*       output);

void pbfdaf_copy_weights_from(PBFDAF* dst, const PBFDAF* src);
/* Warm tap-transfer: shift the learned IR by shift_samples (positive = left,
 * toward tap 0; negative = right). Returns 0 when the shift was applied and
 * -1 when it was REFUSED by the scr_ir span guard (a filter whose
 * n_partitions*hop_size falls outside the scratch this routine can stage the
 * concatenated IR in). A refusal leaves the filter untouched, so a caller
 * that changed its alignment must not report warm success on -1: the taps
 * still sit at the old alignment. */
int pbfdaf_warm_shift_ir(PBFDAF* p, int shift_samples);

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
    float  alpha_r;            /* error-PSD EMA retention. AUTHORING value 0.95
                                * at hop=160/16000 (10 ms, same reference as
                                * alpha_power); retimed at pbfdkf_init(), so
                                * it equals 0.95 only on a 10 ms hop. Read the
                                * effective value, never this comment. */

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

    /* Per-hop scratch [n_freqs]: |error_spec|² computed once per hop in
     * pbfdkf_process and shared by the error_psd EMA + the AEC3 mu gate +
     * the H_error refresh (they all read the same untouched error_spec).
     * Instance storage, not stack — keeps aec_process stack usage
     * independent of n_freqs (embedded task stacks are small). */
    float* e2_ref_scratch;

    /* More de-stacked per-hop scratch (same rationale as e2_ref_scratch
     * above): each was a float[8192] local in pbfdkf.c. scr_mu_local and
     * scr_X2/scr_mu_aec3 have overlapping live ranges within their call
     * (mu_scale_arr / X2 / mu_aec3 are all read together at the tail of
     * pbfdkf_update_weights_aec3), so each gets its own field — see the
     * use sites in pbfdkf.c for the exact live-range reasoning. */
    float* scr_mu_local;       /* [n_freqs] pbfdkf_process mu broadcast */
    float* scr_X2;             /* [n_freqs] pbfdkf_update_weights_aec3 X_buf**2 */
    float* scr_mu_aec3;        /* [n_freqs] pbfdkf_update_weights_aec3 per-bin mu */

    int is_static;               /* 1 = state placed in caller buffer */
} PBFDKF;

/* F04: returns 0 on success, -1 if the nested pbfdaf_init() (base filter's
 * fft_create()) failed (OOM) or rejected p/sample_rate/hop — alpha_r is
 * retimed off the same grid, so a bad rate is refused here too. */
int    pbfdkf_init(PBFDKF* p, int block_size, int n_partitions,
                    float mu, float delta, int hop_size, int sample_rate);
size_t pbfdkf_get_mem_size(int block_size, int n_partitions, int hop_size);
/* shared_fft: forwarded verbatim to the base filter's pbfdaf_init_static() —
 * see that function's doc comment (REQUIRED, non-NULL, borrowed not
 * owned). F04: returns 0 on success, -1 if the pool base/size checks reject
 * the inputs (F05/F07, nothing written to `mem`), shared_fft is NULL, or
 * sample_rate/resolved hop is non-positive. Invalid arguments are rejected
 * before writing `p` or mem. */
int    pbfdkf_init_static(PBFDKF* p, void* mem, size_t mem_size,
                           int block_size, int n_partitions,
                           float mu, float delta, int hop_size,
                           int sample_rate, FftHandle* shared_fft);
void pbfdkf_free(PBFDKF* p);
void pbfdkf_reset(PBFDKF* p);
/* pbfdaf_reset_taps for the Kalman filter: same adaptation-only scope, plus
 * the per-bin R/error_psd/Q re-arm pbfdkf_reset performs. */
void pbfdkf_reset_taps(PBFDKF* p);

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

/* sum(|error_spec|**2) via cmag2_np + pairwise float32 sum (float32-by-design,
 * no double promotion). Not const: needs p->scr_e2 instance scratch (was a
 * float[8192] stack local) — all call sites already pass non-const Aec-owned
 * filters, so dropping const costs nothing and avoids a separate mutable-
 * pointer workaround. */
float pbfdaf_get_error_energy(PBFDAF* p);
float pbfdkf_get_error_energy(PBFDKF* p);

/* get_time_domain_filter: concat per-partition irfft(W[p])[:hop] →
 * out[n_partitions × hop_size] (caller-owned). Mirrors filters.py:397.
 *
 * _range materializes only partitions [first_partition, first_partition+
 * n_parts), writing them at their natural offsets in that same full-length
 * layout and leaving every other tap in `out` untouched; the whole-filter
 * call is exactly the all-partitions case of it. One irfft per partition is
 * the dominant cost here, so a consumer that reads back only part of the
 * impulse response should ask for that part: the taps it does not read must
 * not be paid for. The range is clamped to the partition count. */
void pbfdaf_get_time_domain_filter_range(PBFDAF* p, int first_partition,
                                         int n_parts, float* out);
void pbfdaf_get_time_domain_filter(PBFDAF* p, float* out);

/* scale_filter: W *= (float32)scale (in place, all partitions). */
void pbfdaf_scale_filter(PBFDAF* p, float scale);
void pbfdkf_scale_filter(PBFDKF* p, float scale);

#ifdef __cplusplus
}
#endif

#endif /* AEC_PBFDKF_H */
