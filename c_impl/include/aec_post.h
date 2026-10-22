/* aec_post.h — C port of the post-filter DRIVER (the post-filter step of
 * AEC in python/modules/orchestrator.py).
 *
 * This is the STATEFUL INTEGRATION layer that ties the sub-modules together
 * and adds CNG + OLA. The per-bin float math of AecState /
 * ResidualEchoEstimator / SuppressionGain / the reverb models / the ERLE +
 * stationarity estimators is owned (and parity-tested) by those modules; THIS
 * driver owns the assembly logic the orchestrator method implements directly:
 *
 *   1. PSD derivation from the fp32 spectra  (|c64|² · _PSD_SCALE, all-f32):
 *        near_psd / far_psd / echo_psd / error_psd
 *   2. windowed-capture Y² for ERLE          (erle_windowed_capture_psd)
 *   3. avg-render-reverb x2_reverb_for_erle: an owned ReverbModel fed
 *        X²[delay+1] with the steady decay, output X²[delay] + reverb
 *        (+ erle_render_x2_psd_scale).
 *   4. coherence Γ²(Ŷ,Y) EMA + ERLE coh-gate mask   (erle_coh_gate, default ON)
 *   5. CNG N2 tracking → comfort_noise: y2_smoothed EMA, n2 track-down/
 *        slow-up, n2_initial transient, noise-floor clamp. Input PSD is
 *        near_psd by default, or the windowed-capture PSD from stage 2
 *        when cng_n2_windowed_psd is set (default OFF).
 *   6. output-base select (output_capture_when_linear_unusable, |E|>|Y|
 *        guard), gain apply, CNG injection (LCG sin-table indices + sqrt(2)·
 *        sin LUT, DC/Nyquist zeroed), irfft, sqrt-Hann synth window, OLA.
 *
 * The sub-module results (usable_linear / saturated_echo / r2 / r2_unb / the
 * final per-bin gain) are INPUTS here: the orchestrator computes them by
 * calling the already-ported modules in between the driver stages. The driver
 * port re-derives every quantity it owns from the raw fp32 spectra exactly as
 * the .py does and accepts the sub-module outputs as parameters.
 *
 * PARITY (numpy 1.26 → C, -ffp-contract=off):
 *   - _PSD_SCALE = 32768² (int16 max²). PSD = (|c64|² in f32) · _PSD_SCALE in
 *     f32 (array · pyfloat → f32, numpy<2 rule), then .astype(f32) no-op.
 *   - coherence Γ²: sye is complex64 (component-wise f32 EMA, scalars value-
 *     cast to f32); syy/see are float32 (the EMA is computed in f32
 *     throughout; drift versus the numpy reference accepted). gamma2 =
 *     (|sye|²/max(syy·see,1e-30)).astype(f32).
 *   - CNG N2 tracking: y2_smoothed / n2 / n2_initial are float32 arrays; the
 *     per-hop α/freshness/retention/slow-up are python-float scalars value-cast
 *     to f32 in each array op (all-f32 computation), then np.where → .astype
 *     (f32). noise_floor clamp = np.maximum compares in float (f32 array vs f32
 *     scalar).
 *   - CNG LCG: seed (int) recurrence (seed*69069+1) & 0x7FFFFFFF, ix = seed>>26
 *     (0..31), im = (ix+8)&31. uint32 arithmetic reproduces it exactly.
 *   - irfft via the library FFT backend (1/N normalization).
 *   - synth window = block_size sqrt-Hann (MATLAB-canonical denom N), provided
 *     by the caller (orchestrator builds it once).
 *
 * Caller owns all backing storage (malloc-free audio path).
 */
#ifndef AEC_POST_H
#define AEC_POST_H

#include <stddef.h>
#include <stdint.h>

#include "fft_wrapper.h"     /* Complex, FftHandle */
#include "reverb_model.h"    /* avg-render-reverb */

/* The full-orchestration driver (aec_post_run) needs the sub-module structs. */
#include "aec_state.h"
#include "residual_echo_estimator.h"
#include "suppression_gain.h"
#include "stationarity_estimator.h"
#include "linear_filter_output.h"
#include "pbfdkf.h"          /* PBFDAF -- the live observer */

#ifdef __cplusplus
extern "C" {
#endif

/* Driver config + wall-clock-rescaled CNG constants (orchestrator computes
 * these at construction; pass them verbatim). */
typedef struct {
    int    n_bins;          /* fft_size/2 + 1 (257) */
    int    fft_size;        /* 512 */
    int    block_size;      /* 320 (analysis window length) */
    int    hop_size;        /* 160 */

    /* feature flags (balanced defaults shown) */
    int    erle_coh_gate_enabled;            /* 1 */
    int    erle_windowed_capture_psd;        /* 1 */
    int    erle_render_x2_psd_scale;         /* 1 */
    int    output_capture_when_linear_unusable; /* 1 */
    int    enable_cng;                       /* 1 */
    /* CNG N2 tracker coordinate fix (mirrors python/modules/config.py's
     * cng_n2_windowed_psd). Default OFF (0, NOT corpus-validated). When set,
     * aec_post_compute_comfort_noise() reads capture_psd_win (the windowed
     * capture PSD already shared with erle_windowed_capture_psd) instead of
     * near_psd for the y2_smoothed init + per-hop EMA; requires
     * aec_post_compute_psds() to have populated capture_psd_win this hop
     * (gated on erle_windowed_capture_psd || cng_n2_windowed_psd, below). */
    int    cng_n2_windowed_psd;              /* 0 */

    /* coherence Γ²(Ŷ,Y) EMA */
    float  erle_coh_gate_alpha;               /* 0.05 */
    float  erle_coh_gate_threshold;           /* 0.5  */

    /* CNG wall-clock-rescaled constants */
    float  cng_y2_alpha;                      /* y2_smoothed EMA α */
    float  cng_n2_track_freshness;
    float  cng_n2_track_retention;
    float  cng_n2_slow_up;
    float  cng_n2_initial_alpha;
    int    cng_n2_update_onset_hops;
    int    cng_n2_initial_duration_hops;
    float  noise_floor_int16sq;               /* GetNoiseFloorFactor(dbfs) */

    /* Length of the `synth_window` array passed to aec_post_init. The
     * selected signal-grid row provides one value per block sample;
     * aec_post_apply_output asserts this equals block_size before OLA. */
    int    synth_window_len;

    /* ── live observer ──────────────────────────────────────
     * A second, independent PBFDAF reading the far-end reference ring
     * earlier-shifted than the main filter's own read (caller-driven ring
     * logic; see AecPostRunIn.observer_far_end). Its evidence is trusted
     * into R² (`r2 = max(r2, ev)`, bounded R² only, never r2_unb) only
     * while its own LF residual energy is no worse than the MAIN filter's
     * own residual on the SAME hop (observer_err_ema_alpha-smoothed EMA
     * pair, margin observer_corrob_margin) -- comparing against the
     * observer's OWN past state (an earlier design, since abandoned)
     * cannot tell correct extra cancellation from accidental near-end
     * contamination, since both lower the observer's own residual;
     * comparing against a different, independently-adapted filter seeing
     * the same input can. Evidence is additionally capped at
     * min(|observer.echo_spec - sel_echo|^2, |sel_esw|^2): a wrong
     * observer can never claim more residual echo than the main filter's
     * own residual actually contains. observer_max_bin bounds which bins
     * the evidence (and the observer's own EMA sum) ever touches (LF-only,
     * ceil(live_observer_max_hz / bin_hz)). Default OFF, byte-equal when
     * off (factor-free: the whole block is skipped). */
    int    observer_enabled;          /* 0 */
    int    observer_max_bin;
    /* double, not float -- orchestrator.py's `_a`/`_observer_corrob_margin`
     * are Python `float` (C double) and the EMA recursion they drive is
     * double arithmetic throughout; see observer_obs_err_ema's comment in
     * the AecPost struct below for why this needs to match. */
    double observer_err_ema_alpha;    /* exp(-hop_ms/live_observer_corrob_ms) */
    double observer_corrob_margin;    /* 10^(live_observer_corrob_margin_db/10) */
} AecPostConfig;

/* Audio-passive per-hop trace capture. The three scalars below are decided
 * inside aec_post_run as locals (not otherwise persisted); the driver stashes
 * them here so the per-frame "logr" trace can read them after the call. All
 * other trace columns are re-readable via the AecState / SuppressionGain
 * accessors. Zeroed in aec_post_init; writing it has no effect on output. */
typedef struct {
    int    aec_converged;   /* per-frame convergence (refined||coarse) */
    int    far_active;       /* active_render (far_pwr > threshold)         */
    float  gain_mean;        /* mean of the SuppressionGain output array     */
} AecPostTrace;

/* Mutable driver state. All per-bin arrays are caller-owned. */
typedef struct {
    AecPostConfig cfg;

    /* audio-passive trace stash (see AecPostTrace) */
    AecPostTrace trace;

    /* Live-computed convergence-flags y2 threshold (step 4, aec_post_run).
     * See aec_post_init()'s comment. */
    float y2_thr;

    /* synthesis window (block_size) + sqrt(2)·sin LUT (32), caller-owned,
     * read-only after init. */
    const float *synth_window;     /* [block_size] */
    const float *sqrt2_sin_lut;    /* [32] */

    /* CNG state */
    uint32_t cng_seed;             /* LCG state (init 42) */
    int      noise_initialized;    /* bool */
    int      n2_counter;
    float   *y2_smoothed;          /* [n_bins] */
    float   *n2;                   /* [n_bins] */
    float   *n2_initial;           /* [n_bins] */

    /* coherence Γ²(Ŷ,Y) EMA (sye complex64 split; syy/see float32) */
    float   *coh_sye_re;           /* [n_bins] */
    float   *coh_sye_im;           /* [n_bins] */
    float   *coh_syy;              /* [n_bins] */
    float   *coh_see;              /* [n_bins] */

    /* avg-render-reverb (owned ReverbModel) */
    ReverbModel avg_reverb;        /* uses avg_reverb_storage */

    /* OLA buffer (block_size) */
    float   *ola_buf;              /* [block_size] */

    /* per-hop scratch (caller-owned) */
    float   *near_psd;             /* [n_bins] */
    float   *far_psd;              /* [n_bins] */
    float   *echo_psd;             /* [n_bins] */
    float   *error_psd;            /* [n_bins] */
    float   *capture_psd_win;     /* [n_bins] */
    float   *x2_reverb_for_erle;   /* [n_bins] */
    unsigned char *coh_gate_mask;  /* [n_bins] */
    float   *comfort_noise;        /* [n_bins] (== n2 or n2_initial alias copy) */
    float   *nf;                   /* [n_bins] sqrt(cn/PSD) */
    Complex *e_out_spec;           /* [n_bins] */
    float   *e_out_full;           /* [fft_size] irfft output */

    FftHandle *fft;                /* caller-owned, sized fft_size */

    /* Live observer corroboration EMAs. Zeroed in
     * aec_post_init and by aec_post_reset (mirrors orchestrator.py's
     * _observer_obs_err_ema/_observer_main_err_ema, which are also
     * zeroed outside the post-chain reset path by the realign
     * handler -- see aec_reset_filter_for_realign's own explicit zeroing,
     * since that path does not call aec_post_reset). */
    /* double, not float: orchestrator.py accumulates this EMA and compares
     * it against a hard `<=` threshold in Python `float` (C double)
     * throughout -- a float32 EMA here would drift from Python's and could
     * flip the comparison's outcome near the boundary (same class of bug
     * `aec_scale.c`'s `ms_to_hops_f32` exists to prevent for a different
     * quantity: a value exactly at a threshold in one precision need not
     * be on the same side in another). The observer's own corroboration
     * check (whether a hop's evidence gets injected into R^2 at all) is a
     * discrete decision, not an ordinary numeric drift the project's usual
     * float32-FFT tolerance already covers. */
    double observer_obs_err_ema;
    double observer_main_err_ema;
} AecPost;

/* Fill cfg with the balanced defaults the orchestrator uses (the caller still
 * overrides the wall-clock constants + flags with the captured values). */
void aec_post_config_defaults(AecPostConfig *cfg);

/* Initialise. Caller supplies cfg, the read-only window/LUT, the FFT handle,
 * and all per-bin backing storage. avg_reverb_storage is the ReverbModel's
 * n_bins float buffer. */
void aec_post_init(AecPost *p, const AecPostConfig *cfg,
                   FftHandle *fft,
                   const float *synth_window, const float *sqrt2_sin_lut,
                   float *avg_reverb_storage,
                   float *y2_smoothed, float *n2, float *n2_initial,
                   float *coh_sye_re, float *coh_sye_im,
                   float *coh_syy, float *coh_see,
                   float *ola_buf,
                   float *near_psd, float *far_psd, float *echo_psd,
                   float *error_psd, float *capture_psd_win,
                   float *x2_reverb_for_erle, unsigned char *coh_gate_mask,
                   float *comfort_noise, float *nf,
                   Complex *e_out_spec, float *e_out_full);

/* Reset the driver-owned state (mirrors the orchestrator's post-chain reset): OLA / CNG / coherence
 * EMAs / avg-reverb. Matches the orchestrator's filter-derived reset path
 * (preserve_render_side=True — render-side trackers live in other modules). */
void aec_post_reset(AecPost *p);

/* Per-hop captured magnitudes. numpy's complex64 abs uses a SIMD path that is
 * not portably reproducible in C; the driver takes np.abs(...) of several
 * spectra, so the magnitudes are passed in (captured from the reference) and
 * the driver does the parity-relevant (a*a)·_PSD_SCALE arithmetic in f32. */
typedef struct {
    const float *abs_near;     /* |near_spec|       (near_psd, coh see)  */
    const float *abs_far;      /* |far_spec|        (far_psd)            */
    const float *abs_sel_echo; /* |echo_spec_sel|   (echo_psd)           */
    const float *abs_error;    /* |error_spec|=|sel_esw| (error_psd) */
    const float *abs_echo_coh; /* |filter.echo_spec| (coh syy)          */
    const float *abs_nsw_e1;   /* |esw_orig+echo_orig| (capture_psd for ERLE) */
    /* Read only by aec_post_apply_output()'s output-capture guard, so aec_post_run()
     * leaves it NULL whenever that call cannot happen (context_only). Any
     * new consumer outside apply_output() must compute it for itself. */
    const float *abs_ybase;    /* |sel_esw+sel_echo|   (output-capture guard) */
} AecPostAbs;

/* ─────────────────────────────────────────────────────────────────────────
 * Stage functions — the pieces of the per-hop post-filter step. The
 * full-orchestration driver (aec_post_run) calls these interleaved with the
 * sub-module calls in the exact order of the orchestrator's post-filter step.
 * ──────────────────────────────────────────────────────────────────────── */

/* Stage 1-2: near/far/echo/error PSDs + capture_psd_win. */
void aec_post_compute_psds(AecPost *p, const AecPostAbs *mag);

/* Stage 3: avg-render-reverb → p->x2_reverb_for_erle (only when x2_present). */
void aec_post_compute_x2_reverb(AecPost *p, int x2_present,
                                const float *x2_at_delay, const float *x2_past,
                                float decay_steady);

/* Ring slots the average-render reverb reads. delay_idx is the partition
 * `delay` blocks older than the newest one (partition_idx - 1), past_idx the
 * one just older than that. The ring holds n_part blocks, so delay is capped
 * at n_part - 2 (and floored at 0): beyond that past_idx would wrap onto the
 * newest partition. Requires n_part >= 1. */
void aec_post_avg_reverb_slots(int n_part, int partition_idx, int delay,
                               int *delay_idx, int *past_idx);

/* Stage 4: coherence Γ²(Ŷ,Y) EMA → p->coh_gate_mask (when erle_coh_gate). */
void aec_post_compute_coherence(AecPost *p,
                                const Complex *echo_spec_for_coh,
                                const Complex *near_spec,
                                const AecPostAbs *mag);

/* Stage 5: CNG N2 tracking → p->comfort_noise. */
void aec_post_compute_comfort_noise(AecPost *p, int saturated_capture);

/* Stage 6: output-base select + gain apply + CNG inject + irfft + OLA. */
void aec_post_apply_output(AecPost *p,
                           const Complex *error_spec,
                           const Complex *echo_spec_sel,
                           const AecPostAbs *mag,
                           int usable_linear,
                           const float *gain,
                           float *out);

/* ─────────────────────────────────────────────────────────────────────────
 * aec_post_run — the FULL orchestration of the orchestrator's post-filter step.
 * It drives AecState + ResidualEchoEstimator + SuppressionGain +
 * StationarityEstimator + LinearFilterSelect itself, reproducing every line of
 * the Python method end-to-end.
 *
 * The caller owns + constructs all the sub-module objects (exactly as the
 * orchestrator's __init__ does, captured by the golden); aec_post_run only
 * wires the per-hop call sequence. Stationarity is updated UPSTREAM (in the
 * process loop, not here): the caller must have advanced it for this hop before
 * the call, and supplies the active-hops / converge-hops counters.
 * ──────────────────────────────────────────────────────────────────────── */

/* DelayAdjustment enum (delay/delay_types.py): NONE/BUFFER_FLUSH/NEW_DETECTED. */
#define AEC_POST_DELAY_ADJ_NONE          0
#define AEC_POST_DELAY_ADJ_BUFFER_FLUSH  1
#define AEC_POST_DELAY_ADJ_NEW_DETECTED  2

/* DelayQuality.REFINED (delay/delay_types.py). */
#define AEC_POST_DELAY_QUALITY_REFINED   2

/* All per-hop inputs aec_post_run reads, plus the persistent objects it
 * drives. A struct keeps the signature manageable. Pointers are caller-owned.*/
typedef struct {
    /* ── linear-filter spectra (PBFDKF main), length n_freqs ────────────── */
    const Complex *near_spec;            /* filter.near_spec               */
    const Complex *far_spec;             /* filter.far_spec                */
    const Complex *echo_spec;            /* filter.echo_spec               */
    const Complex *error_spec_windowed;  /* filter.error_spec_windowed     */
    const Complex *W0;                   /* filter.W[0] (bridge irfft)     */
    const Complex *W_all;                /* filter.W flat (n_part×n_freqs) */
    const Complex *X_buf;                /* filter.X_buf flat (n_part×nf)  */
    const float   *x2_cache;             /* filter.x2_cache flat (n_part×nf):
                                          * |X_buf|² row for row. NULL → the
                                          * two avg-render-reverb rows are
                                          * derived from X_buf instead, which
                                          * is what a caller driving this
                                          * module without a PBFDAF behind it
                                          * (the parity harness) does.       */
    const float   *sqrt_hann;            /* filter._sqrt_hann_analysis     */
    const float   *kalman_P;             /* filter Kalman P (bridge div);
                                          * NULL/0 → divergence_indicator 0 */
    size_t         kalman_P_len;
    int            partition_idx;        /* filter.partition_idx           */
    int            n_partitions;         /* filter.n_partitions            */
    const float   *filter_taps_full;     /* get_time_domain_filter(); NULL */
    int            filter_taps_full_len;

    /* ── shadow filter ──────────────────────────────────────────────────── */
    int            shadow_present;       /* shadow_filter is not None      */
    const Complex *shadow_error_spec;    /* shadow.error_spec (coarse e2)  */
    const float   *last_shadow_output_time; /* _last_shadow_output_time;
                                             * NULL → not available        */
    float          s_ref_max;            /* filter._last_s_max_abs         */
    float          s_coa_max;            /* shadow._last_s_max_abs (or 0)  */
    float          main_error_energy;    /* bridge e_ratio numerator/denom */
    float          shadow_error_energy;

    /* ── args ───────────────────────────────────────────────────────────── */
    const float   *raw_output;           /* [hop]  e_refined_time          */
    const float   *near_end;             /* [hop]                          */
    const float   *far_end;              /* [hop]                          */

    /* ── delay tracker / saturation / pending EPV flags ─────────────────── */
    int            current_delay;        /* int(self._current_delay)       */
    int            delay_active;         /* self._delay_active             */
    float          saturation_level;     /* self._saturation_level         */
    int            nearend_recent;        /* self._ne_recent_frames > 0: the energy-
                                             * based near-recent latch (bool) */
    int            pending_gain_change;  /* pending echo-path gain change (in) */
    int            pending_delay_change; /* pending echo-path delay change:
                                          *  -1 == None, else DelayAdjustment */

    /* ── stationarity read state (updated upstream) ─────────────────────── */
    int            stationarity_active_hops;
    int            stationarity_converge_hops;
    int            stationary_block;       /* raw is_block_stationary result;
                                             * not convergence-gated          */

    /* ── config flags ───────────────────────────────────────────────────── */
    int            erle_coh_gate_enabled;
    int            use_stationarity_properties; /* sg echo_audibility flag */
    int            dt_floor_enabled;      /* AecConfig.dt_aware_res_floor_enabled */
    int            context_only;          /* expose gain/R²/CNG seam but skip
                                             * unused private time synthesis */
    int            linear_context_only;   /* nothing reads R², comfort noise
                                             * or G_res: stop after the AecState
                                             * update (none of the skipped
                                             * state feeds the linear filter) */
    int            spatial_linear_context; /* this lane's own G_res is never
                                             * consumed (a downstream fused/
                                             * beamformed stage recomputes an
                                             * equivalent gain) -- skip
                                             * computing it, but keep the
                                             * DominantNearend hold-state
                                             * advancing every hop, since the
                                             * NEXT hop's ERLE onset decision
                                             * (Step 13) and therefore r2
                                             * (consumed downstream) depend on
                                             * it. comfort_noise (Step 19) is
                                             * computed independently of DNE --
                                             * it is one of DNE's own INPUTS,
                                             * not something DNE affects -- and
                                             * stays correct in this mode
                                             * regardless. Only meaningful, and
                                             * only validated, when context_only is
                                             * also set -- aec_validate_config()
                                             * enforces this precondition. */
    float          active_render_threshold;     /* _ar_thr = 5.96e-4       */

    /* ── live observer ────────────────────────────────────── */
    /* self._render_activity.is_active -- the persistent RenderActivityDetector
     * updated once per hop by the caller BEFORE this call, distinct from
     * this struct's active_render_threshold (a different, instantaneous
     * quantity computed INSIDE aec_post_run for AecState.update()). */
    int            render_activity_is_active;
    /* observer_far_end non-NULL iff live_observer_enabled AND this hop's
     * shifted ring read was valid (mirrors Python's
     * `self._observer_far_end is not None`). No separate bool field for
     * this condition -- a non-NULL pointer already is one. */
    const float   *observer_far_end;      /* [hop], or NULL              */
} AecPostRunIn;

/* Sub-module object bundle the driver wires. All caller-owned + constructed. */
typedef struct {
    AecState              *state;
    ResidualEchoEstimator *ree;
    SuppressionGain       *sg;
    StationarityEstimator *stationarity;
    LinearFilterSelect    *lfs;
    FftHandle             *fft;          /* sized fft_size (bridge irfft)  */
    /* Live observer: NULL when live_observer_enabled==0.
     * Driven (pbfdaf_process) from INSIDE aec_post_run, not by the
     * caller, since it needs this hop's dominant_ne/usable_linear
     * verdicts -- see aec_post_run's own injection block. */
    PBFDAF                *observer_filter;
} AecPostRunObj;

/* Scratch buffers the driver needs (caller-owned, sized to the geometry). */
typedef struct {
    Complex *sel_esw;            /* [n_freqs] selected error_spec_windowed */
    Complex *sel_echo;           /* [n_freqs] selected echo_spec           */
    Complex *nsw_e1;             /* [n_freqs] esw_orig + echo_orig       */
    Complex *ybase;              /* [n_freqs] sel_esw + sel_echo (guard)  */
    float   *abs_near;           /* [n_freqs] magnitude buffers …          */
    float   *abs_far;
    float   *abs_sel_echo;
    float   *abs_error;
    float   *abs_echo_coh;
    float   *abs_nsw_e1;
    float   *abs_ybase;
    float   *x2_at_delay;        /* [n_freqs] */
    float   *x2_past;            /* [n_freqs] */
    float   *w_mag2;             /* [n_part × n_freqs] frequency_response  */
    float   *render_block_scaled;/* [hop] far_end × 32768                  */
    float   *r2;                 /* [n_freqs] */
    float   *r2_unb;             /* [n_freqs] */
    float   *nearend_pwr;        /* [n_freqs] */
    unsigned char *stat_mask;    /* [n_freqs] band_stationary_mask         */

    /* Live observer only (cfg.observer_enabled): per-hop corroboration
     * scratch. observer_out discards pbfdaf_process's time-domain output
     * (only its spectra are read); observer_diff_spec holds
     * observer.echo_spec - sel_echo (plain complex subtraction, exact --
     * NOT sk_cadd_f32 with a negated operand); observer_diff_mag2 /
     * observer_cap_mag2 hold |observer_diff_spec|^2 / |sel_esw|^2 via
     * sk_cmag2_np_f32 (the scaled-hypot kernel, to match numpy's
     * sqrt-then-square np.abs(c64)**2 bit-for-bit -- re^2+im^2 directly
     * would NOT). */
    float   *observer_out;       /* [hop] */
    Complex *observer_diff_spec; /* [n_freqs] */
    float   *observer_diff_mag2; /* [n_freqs] */
    float   *observer_cap_mag2;  /* [n_freqs] */
} AecPostRunScratch;

/* Process one hop end-to-end. `out[hop]` is always written: the synthesized
 * internal-RES signal normally, or in->raw_output when context_only is set.
 * The pending EPV flags in `in` are consumed (the caller's mirror is cleared
 * via the returned values): *out_pending_gain_change / *out_pending_delay_change
 * receive the post-call state (cleared to 0 / -1 when an event fired). */
int aec_post_run(AecPost *p,
                 const AecPostRunIn *in,
                 const AecPostRunObj *obj,
                 AecPostRunScratch *sc,
                 float *out,
                 int *out_pending_gain_change,
                 int *out_pending_delay_change);

#ifdef __cplusplus
}
#endif

#endif /* AEC_POST_H */
