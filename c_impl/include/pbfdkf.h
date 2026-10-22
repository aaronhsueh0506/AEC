/* pbfdkf.h — C port of python/modules/filters.py PBFDAF + PBFDKF.
 *
 * Partitioned Block Frequency-Domain Adaptive Filter with overlap-save.
 *   PBFDAF — NLMS update (shadow / coarse filter).
 *   PBFDKF — per-bin H_error Kalman gain + leakage refresh
 *            (the reference refined-filter update gain).
 *
 * PBFDKF update:
 *   per-bin H_error state. mu[k] = H_error / (0.5·H_error·X² +
 *   n·E²_refined + δ); W[p] += mu·conj(X[p])·mu_scale·error_spec;
 *   H_error -= 0.5·mu·X²·H_error (decay, active branch only);
 *   always-on refresh H_error += leakage·erl, clamp[floor,ceil].
 *   leakage = converged/diverged chosen per-bin by E²_ref≤E²_coa,
 *   transient (initial_state) profile for the first 250 active hops.
 *
 * Design rules (parity):
 *   - All array dtypes mirror Python: np.complex64 ⇒ Complex; np.float32 ⇒
 *     float. The H_error / mu / leakage-refresh ARITHMETIC is exact given
 *     identical inputs, but its inputs (|error_spec|², |X|²) come THROUGH the
 *     FFT, so H_error / error_psd / R are rtol-bounded (<1e-4) against
 *     Python, NOT bit-exact. Only the integer control state (call_counter /
 *     partition_idx / initial_state) is bit-exact.
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
     * that need to prove a multi-lane caller's far-FFT count drops when
     * its lanes use the shared-spectrum entry
     * point, without conflating it with the near/error/W-tap FFTs that
     * also flow through the same underlying fft_forward_scratch(). */
    long     far_fft_real_compute_count; /* saturates at LONG_MAX */

    /* Round-robin TD constraint: constrain
     * ONE partition per hop (cycling `partition_to_constrain`) instead of all
     * partitions every hop — saves ~58% of the per-hop constraint FFTs and
     * deepens convergence (constraining all of them every hop over-constrains). aec_create
     * sets `constraint_round_robin`=1 on BOTH filters (production default). Each
     * filter keeps its own counter. `constraint_round_robin` is pure config and
     * never reset; the cursor beside it survives pbfdaf_reset() (the mid-stream
     * relock path keeps cycling where it left off, mirroring Python) and is
     * re-armed only by pbfdaf_reset_carried_state(), i.e. by aec_reset(). */
    int      constraint_round_robin;
    int      partition_to_constrain;

    /* Subtractor-output s_*_max_abs — time-domain peak of the echo
     * predictor on the valid hop (max|echo_time[hop:block]|). Recomputed every
     * frontend pass; consumed by AecState SaturationDetector. */
    float   last_s_max_abs;

    /* Coarse-filter update-gain startup / poor-excitation / saturation gates */
    long    call_counter;
    long    poor_excitation_counter;
    int     saturated_capture;
    int     block_stationary;

    /* InitialState tracking (shared field; only PBFDKF consumes it). */
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
     * pbfdaf_frontend / pbfdaf_warm_shift_ir stack
     * usage independent of n_freqs / n_partitions (embedded task stacks are
     * small; these were float[8192]/float[4096] locals, a stack-overflow
     * hazard). Mirrors the e2_ref_scratch pattern below on PBFDKF. One field
     * per distinct concurrent live range (see pbfdkf.c comments at each use). */
    float*  scr_fsq;           /* [hop_size]     pbfdaf_frontend far_end**2 */
    float*  scr_mu_local;      /* [n_freqs]      pbfdaf_process mu broadcast */
    float*  scr_x2psum;        /* [n_freqs]      pbfdaf_process X_buf**2 partition sum */
    float*  scr_mu_eff;        /* [n_freqs]      pbfdaf_process per-bin mu */
    float*  scr_ir;            /* [n_partitions*hop_size] pbfdaf_warm_shift_ir IR concat */

    /* Appended fields (kept last so every offset above is unchanged).
     *
     * x2_cache [n_partitions][n_freqs] float — |X_buf[p][k]|² mirrored row for
     * row. The two X² partition sums (pbfdaf_process's NLMS gate and
     * pbfdkf_update_weights's Kalman gain) re-derived this from X_buf with
     * the scaled-hypot cmag2 kernel on every update call, over a history where
     * exactly one row changes per hop and that row's |.|² is already computed
     * by the frontend. The cached float IS the value cmag2 would return -- it
     * is produced by the same kernel, once, at the hop the partition enters
     * history -- so the sums stay byte-identical while the divide+sqrt per bin
     * per partition disappears. Every X_buf writer must write this in the same
     * breath: pbfdaf_zero_state, pbfdaf_reset_taps, the frontend's row store
     * and pbfdaf_clear_far_history (the only spelling a caller outside this
     * file has for wiping the far history).
     *
     * far_cmag2_hold [n_freqs] float — |far_spec[k]|², written by the frontend
     * and zeroed exactly where far_spec is zeroed, so it tracks far_spec's
     * lifecycle rather than X_buf's. That distinction is load-bearing: a
     * realign clears X_buf (and x2_cache) but NOT far_spec, so a consumer
     * wanting |far_spec|² must read this, not an x2_cache row. Read before this
     * hop's frontend it holds the previous hop's spectrum; read after, the
     * current one -- which is exactly how the orchestrator's two consumers
     * (previous-hop RSA, current-hop stationarity) want it. */
    float*  x2_cache;
    float*  far_cmag2_hold;

    /* Set by EVERY writer of W in this file -- the two update paths (including
     * their round-robin TD constraint), copy_weights_from, warm_shift_ir,
     * scale_filter, reset_taps, zero_state -- and by init, so the first hop
     * always recomputes. It exists for one consumer: the orchestrator's
     * per-bin ERL publish, Σ_p|W_p[k]|², which ran every hop over a W that
     * changes on roughly a quarter of them. That consumer CLEARS the flag when
     * it has taken the sum; a second consumer must not be added without giving
     * it its own flag, because clearing is what makes the skip work. Anything
     * that writes W without setting this serves a stale ERL to the H_error
     * leakage refresh, which is audible, not cosmetic. */
    int     w_dirty;
} PBFDAF;

/* Sizing never includes a nested FFT: pbfdaf_init_static() always runs
 * against a caller-supplied `shared_fft` (one FftHandle shared by main filter /
 * shadow filter / aec_post within a single Aec instance, carved once by
 * aec_get_mem_size()/aec_carve() rather than once per sub-module).
 *
 * with_process_scratch: 1 carves the pbfdaf_process()-only scratch
 * (scr_mu_local/scr_x2psum/scr_mu_eff). Pass 0 when the instance is only
 * driven via pbfdaf_frontend() (the PBFDKF base) -- pbfdaf_process() must NOT
 * be called on such an instance. */
size_t pbfdaf_get_mem_size(int block_size, int n_partitions, int hop_size,
                           int with_process_scratch);
/* shared_fft: REQUIRED, non-NULL — an FftHandle already fft_init()'d by the
 * caller at this instance's fft_size (pbfdaf_compute_sizes(hop_size, ...)).
 * `p` borrows it for its lifetime and never destroys it; the caller (whoever
 * constructed shared_fft) remains solely responsible for fft_destroy()-ing it
 * exactly once, after every borrower is done with it. Precondition:
 * single-threaded, fully sequential use — at most one transform in flight
 * across ALL borrowers of shared_fft at any time, and no borrower may retain
 * a pointer into the handle's internal scratch across calls. Returns 0 on
 * success, -1 if the pool base/size checks reject the inputs (nothing written
 * to `mem`), shared_fft is NULL, or sample_rate/resolved hop is non-positive.
 * Invalid arguments are rejected before writing `p` or mem. */
int    pbfdaf_init_static(PBFDAF* p, void* mem, size_t mem_size,
                           int block_size, int n_partitions,
                           float mu, float delta, int hop_size,
                    int with_process_scratch, int sample_rate,
                    FftHandle* shared_fft);
/* Teardown only drops the borrowed shared_fft reference: the arrays live in the
 * caller's pool and the FFT handle belongs to whoever created it. */
void pbfdaf_free(PBFDAF* p);
void pbfdaf_reset(PBFDAF* p);

/* Adaptation-only wipe: taps (W), the per-partition far SPECTRUM history
 * (X_buf), leaving the time-domain analysis
 * buffers (near_buffer/far_buffer, the block_size overlap-save history that
 * feeds near_spec/far_spec/error_spec_windowed and supplies the near term of
 * the linear output) untouched. pbfdaf_reset() is this plus those buffers.
 *
 * For a caller that must restart the filter mid-stream WITHOUT splicing a
 * half-empty analysis frame into the spectra a post-filter consumes: the
 * taps are what a realign invalidates, the analysis history is not. */
void pbfdaf_reset_taps(PBFDAF* p);

/* Restore the construction-time value of every scalar pbfdaf_reset() carries
 * forward: the startup/excitation/saturation gates (call_counter,
 * poor_excitation_counter, saturated_capture, block_stationary), the
 * InitialState tracking pair, the round-robin constraint cursor, and the
 * last-hop SubtractorOutput peak.
 *
 * Separate from pbfdaf_reset() on purpose. pbfdaf_reset()'s other caller is
 * the mid-stream delay-relock path, which re-arms the excitation gates itself
 * at each of its call sites (pbfdkf_handle_echo_path_change for the main
 * filter, the explicit shadow counter writes beside it) and deliberately
 * carries the rest forward; folding these writes into pbfdaf_reset() would
 * change what that path does. aec_reset() has no such call site -- its
 * contract is a fresh instance -- so it calls this straight after.
 *
 * poor_excitation_counter is restored to the same rate-scaled expression
 * aec_carve() and pbfdkf_handle_echo_path_change() use, NOT to the
 * reference-grid literal the init path writes before aec_carve overwrites
 * it. */
void pbfdaf_reset_carried_state(PBFDAF* p);

/* Wipe the per-partition far SPECTRUM history (X_buf) and its |.|² mirror,
 * leaving taps and analysis buffers alone. The mirror is why this exists as a
 * function: a caller memset-ing X_buf on its own would leave x2_cache holding
 * the magnitudes of a history that is now zero, and the next update would
 * subtract echo against them. One spelling, both arrays. */
void pbfdaf_clear_far_history(PBFDAF* p);

/* mu_scale: NULL ⇒ scalar; else per-bin array of n_freqs fp32. Output written
 * to `output` (length hop_size). */
void pbfdaf_process(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,   /* NULL or [n_freqs] */
                       float        mu_scale_scalar,
                       float*       output);

/* Same process with optional exact |precomputed_far_spec|². Used only when
 * precomputed_far_spec is non-NULL; otherwise the frontend computes it.
 * The power must come from this library's cmag2 kernel, for the same spectrum
 * and hop. Copies into the normal persistent fields; caller storage is borrowed
 * only for this call. NULL preserves pbfdaf_process behavior. */
void pbfdaf_process_with_far_cmag2(PBFDAF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale, float mu_scale_scalar,
                       float* output, const float* precomputed_far_cmag2);

void pbfdaf_copy_weights_from(PBFDAF* dst, const PBFDAF* src);
/* Warm tap-transfer: shift the learned IR by shift_samples (positive = left,
 * toward tap 0; negative = right). Returns 0 when the shift was applied and
 * -1 when it was REFUSED by the scr_ir span guard (a filter whose
 * n_partitions*hop_size falls outside the scratch this routine can stage the
 * concatenated IR in). A refusal leaves the filter untouched, so a caller
 * that changed its alignment must not report warm success on -1: the taps
 * still sit at the old alignment.
 *
 * Also owns the far-history decision for a RETARD (shift_samples < 0): on a
 * successful shift it calls pbfdaf_clear_far_history() itself, mirroring
 * filters.py's Filter.warm_shift_ir (`if s < 0: self.X_buf.fill(0)`), which
 * makes the same call internally for the same reason -- the held history is
 * ahead of the realigned stream and would re-expose echo for dozens of hops
 * if kept. Callers never need (and must not add) their own clear; the
 * decision lives here once, not once per call site. */
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
     * R is computed in the active path but does not feed the H_error gain;
     * retained because the Python class maintains it per hop. */
    float* R;
    float* error_psd;
    float  alpha_r;            /* error-PSD EMA retention. AUTHORING value 0.95
                                * at hop=160/16000 (10 ms); retimed at
                                * construction, so it equals 0.95 only on a
                                * 10 ms hop. Read the effective value, never
                                * this comment. */

    /* === H_error per-bin state ======================================= */
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
     * pbfdkf_process and shared by the error_psd EMA + the mu gate +
     * the H_error refresh (they all read the same untouched error_spec).
     * Instance storage, not stack — keeps aec_process stack usage
     * independent of n_freqs (embedded task stacks are small). */
    float* e2_ref_scratch;

    /* More per-hop scratch (same rationale as e2_ref_scratch above), kept off
     * the stack: each is n_freqs long (up to 8192). scr_mu_local and
     * scr_X2/scr_mu_aec have overlapping live ranges within their call
     * (mu_scale_arr / X2 / mu_aec are all read together at the tail of
     * pbfdkf_update_weights), so each gets its own field — see the use
     * sites in pbfdkf.c for the exact live-range reasoning. */
    float* scr_mu_local;       /* [n_freqs] pbfdkf_process mu broadcast */
    float* scr_X2;             /* [n_freqs] pbfdkf_update_weights X_buf**2 */
    float* scr_mu_aec;        /* [n_freqs] pbfdkf_update_weights per-bin mu */

} PBFDKF;

size_t pbfdkf_get_mem_size(int block_size, int n_partitions, int hop_size);
/* shared_fft: forwarded verbatim to the base filter's pbfdaf_init_static() —
 * see that function's doc comment (REQUIRED, non-NULL, borrowed not
 * owned). Returns 0 on success, -1 if the pool base/size checks reject the
 * inputs (nothing written to `mem`), shared_fft is NULL, or
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
/* pbfdaf_reset_carried_state for the Kalman filter: the same scope on the
 * shared base, plus the four orchestrator-fed refresh inputs (the coarse
 * energy pair, the leakage-branch veto and its diagnostic fraction) and the
 * per-bin H_error the Kalman re-arm leaves alone. Same split rationale --
 * see the PBFDAF prototype. */
void pbfdkf_reset_carried_state(PBFDKF* p);

/* mu_scale: NULL ⇒ scalar broadcast; else per-bin [n_freqs] fp32 (already
 * post-RSA-mask, matching the array the Python weight update receives). */
void pbfdkf_process(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale,
                       float        mu_scale_scalar,
                       float*       output);

/* Optional paired far magnitudes, with the same contract as the PBFDAF
 * variant. Returns 1 when e2_ref_scratch contains this hop's |error_spec|²;
 * returns 0 on the far-silent path, which leaves that scratch untouched. */
int pbfdkf_process_with_far_cmag2(PBFDKF* p,
                       const float* near_end, const float* far_end,
                       const float* mu_scale, float mu_scale_scalar,
                       float* output, const float* precomputed_far_cmag2);

/* Refined-filter update-gain HandleEchoPathChange — H_error reset to init
 * on delay_change + coarse-filter update-gain counter reset. */
void pbfdkf_handle_echo_path_change(PBFDKF* p, int delay_change, int gain_change);

/* The filter's reach in taps (n_partitions * hop): the single spelling of
 * the span every warm-transfer eligibility check compares a delay against. */
static inline int pbfdaf_tap_span(const PBFDAF* p) {
    return p->n_partitions * p->hop_size;
}

/* get_time_domain_filter: concat per-partition irfft(W[p])[:hop] →
 * out[n_partitions × hop_size] (caller-owned). Mirrors the Python filter's get_time_domain_filter.
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
