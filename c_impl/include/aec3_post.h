/* aec3_post.h — C port of the AEC3 post-filter DRIVER (AEC._aec3_post in
 * python/modules/orchestrator.py). WS5 Phase 5.5.
 *
 * This is the STATEFUL INTEGRATION layer that ties the already-bit-exact
 * sub-modules together and adds CNG + OLA. The per-bin float math of
 * AecState / ResidualEchoEstimator / SuppressionGain / the reverb models /
 * the ERLE + stationarity estimators is owned (and parity-tested) by those
 * modules; THIS driver owns the assembly logic the orchestrator method
 * implements directly:
 *
 *   1. PSD derivation from the fp32 spectra  (|c64|² · _PSD_SCALE, all-f32):
 *        near_psd / far_psd / echo_psd / error_psd
 *   2. E1 windowed-capture Y² for ERLE          (erle_windowed_capture_psd)
 *   3. avg-render-reverb x2_reverb_for_erle: an owned ReverbModel fed
 *        X²[delay+1] with the steady decay, output X²[delay] + reverb
 *        (+ erle_render_x2_psd_scale).
 *   4. coherence Γ²(Ŷ,Y) EMA + ERLE coh-gate mask   (erle_coh_gate, default ON)
 *   5. CNG N2 tracking → comfort_noise: y2_smoothed EMA, n2 track-down/
 *        slow-up, n2_initial transient, noise-floor clamp.
 *   6. E2 output-base select (output_capture_when_linear_unusable, |E|>|Y|
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
 *     cast to f32); syy/see are float32 (Stage 3a: the f64 widen previously
 *     used to match the numpy reference is retired — same EMA computed in f32
 *     throughout, structure/order unchanged, drift accepted). gamma2 =
 *     (|sye|²/max(syy·see,1e-30)).astype(f32).
 *   - CNG N2 tracking: y2_smoothed / n2 / n2_initial are float32 arrays; the
 *     per-hop α/freshness/retention/slow-up are python-float scalars value-cast
 *     to f32 in each array op (all-f32 computation), then np.where → .astype
 *     (f32). noise_floor clamp = np.maximum compares in float (f32 array vs f32
 *     scalar).
 *   - CNG LCG: seed (int) recurrence (seed*69069+1) & 0x7FFFFFFF, ix = seed>>26
 *     (0..31), im = (ix+8)&31. uint32 arithmetic reproduces it exactly.
 *   - irfft via fft_pocketfft.c (vendored numpy pocketfft; BIT-EXACT vs
 *     np.fft.irfft, 1/N normalization, fp64 internal).
 *   - synth window = block_size sqrt-Hann (MATLAB-canonical denom N), provided
 *     by the caller (orchestrator builds it once).
 *
 * Caller owns all backing storage (malloc-free audio path).
 */
#ifndef AEC3_POST_H
#define AEC3_POST_H

#include <stddef.h>
#include <stdint.h>

#include "fft_wrapper.h"     /* Complex, FftHandle */
#include "reverb_model.h"    /* avg-render-reverb */

/* The full-orchestration driver (aec3_post_run) needs the sub-module structs. */
#include "aec_state.h"
#include "residual_echo_estimator.h"
#include "suppression_gain.h"
#include "stationarity_estimator.h"
#include "linear_filter_output.h"
#include "filter_state_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

/* numpy complex64 |z| (scaled-hypot with a FUSED multiply-add). Equals the `m`
 * inside cmag2_np; `np.abs(c64)`. Used wherever the .py writes np.abs(c64). */
float cabs_np(float re, float im);

/* numpy-1.26-bit-exact pairwise float32 sum (exposed for the run driver). */
float aec3_post_pairwise_sum_f32(const float *a, size_t n);

/* Driver config + wall-clock-rescaled CNG constants (orchestrator computes
 * these at construction; pass them verbatim). */
typedef struct {
    int    n_bins;          /* fft_size/2 + 1 (257) */
    int    fft_size;        /* 512 */
    int    block_size;      /* 320 (analysis window length) */
    int    hop_size;        /* 160 */

    /* feature flags (balanced defaults shown) */
    int    erle_coh_gate_enabled;            /* 1 */
    int    erle_windowed_capture_psd;        /* 1 (E1) */
    int    erle_render_x2_psd_scale;         /* 1 */
    int    output_capture_when_linear_unusable; /* 1 (E2) */
    int    enable_cng;                       /* 1 */

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

    /* Length of the `synth_window` array passed to aec3_post_init. The
     * selected signal-grid row provides one value per block sample;
     * aec3_post_apply_output asserts this equals block_size before OLA. */
    int    synth_window_len;
} Aec3PostConfig;

/* Audio-passive per-hop trace capture. The three scalars below are decided
 * inside aec3_post_run as locals (not otherwise persisted); the driver stashes
 * them here so the per-frame "logr" trace can read them after the call. All
 * other trace columns are re-readable via the AecState / SuppressionGain
 * accessors. Zeroed in aec3_post_init; writing it has no effect on output. */
typedef struct {
    int    aec3_converged;   /* AEC3 per-frame convergence (refined||coarse) */
    int    far_active;       /* active_render (far_pwr > threshold)         */
    float  gain_mean;        /* mean of the SuppressionGain output array     */
} Aec3PostTrace;

/* Mutable driver state. All per-bin arrays are caller-owned. */
typedef struct {
    Aec3PostConfig cfg;

    /* audio-passive trace stash (see Aec3PostTrace) */
    Aec3PostTrace trace;

    /* Live-computed convergence-flags y2 threshold (step 4, aec3_post_run).
     * See aec3_post_init()'s comment; was a frozen literal 3.73e-4f. */
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
    float   *capture_psd_erle;     /* [n_bins] (E1) */
    float   *x2_reverb_for_erle;   /* [n_bins] */
    unsigned char *coh_gate_mask;  /* [n_bins] */
    float   *comfort_noise;        /* [n_bins] (== n2 or n2_initial alias copy) */
    float   *nf;                   /* [n_bins] sqrt(cn/PSD) */
    Complex *e_out_spec;           /* [n_bins] */
    float   *e_out_full;           /* [fft_size] irfft output */

    FftHandle *fft;                /* caller-owned, sized fft_size */
} Aec3Post;

/* Fill cfg with the balanced defaults the orchestrator uses (the caller still
 * overrides the wall-clock constants + flags with the captured values). */
void aec3_post_config_defaults(Aec3PostConfig *cfg);

/* Initialise. Caller supplies cfg, the read-only window/LUT, the FFT handle,
 * and all per-bin backing storage. avg_reverb_storage is the ReverbModel's
 * n_bins float buffer. */
void aec3_post_init(Aec3Post *p, const Aec3PostConfig *cfg,
                    FftHandle *fft,
                    const float *synth_window, const float *sqrt2_sin_lut,
                    float *avg_reverb_storage,
                    float *y2_smoothed, float *n2, float *n2_initial,
                    float *coh_sye_re, float *coh_sye_im,
                    float *coh_syy, float *coh_see,
                    float *ola_buf,
                    float *near_psd, float *far_psd, float *echo_psd,
                    float *error_psd, float *capture_psd_erle,
                    float *x2_reverb_for_erle, unsigned char *coh_gate_mask,
                    float *comfort_noise, float *nf,
                    Complex *e_out_spec, float *e_out_full);

/* Reset the driver-owned state (mirrors _reset_aec3_post): OLA / CNG / coherence
 * EMAs / avg-reverb. Matches the orchestrator's filter-derived reset path
 * (preserve_render_side=True — render-side trackers live in other modules). */
void aec3_post_reset(Aec3Post *p);

/* Per-hop captured magnitudes. numpy's complex64 abs uses a SIMD path that is
 * not portably reproducible in C; the driver takes np.abs(...) of several
 * spectra, so the magnitudes are passed in (captured from the reference) and
 * the driver does the parity-relevant (a*a)·_PSD_SCALE arithmetic in f32. */
typedef struct {
    const float *abs_near;     /* |near_spec|       (near_psd, coh see)  */
    const float *abs_far;      /* |far_spec|        (far_psd)            */
    const float *abs_sel_echo; /* |echo_spec_sel|   (echo_psd)           */
    const float *abs_error;    /* |error_spec|=|sel_esw| (error_psd, E2) */
    const float *abs_echo_coh; /* |filter.echo_spec| (coh syy)          */
    const float *abs_nsw_e1;   /* |esw_orig+echo_orig| (E1 capture_psd)  */
    /* Read only by aec3_post_apply_output()'s E2 guard, so aec3_post_run()
     * leaves it NULL whenever that call cannot happen (context_only). Any
     * new consumer outside apply_output() must compute it for itself. */
    const float *abs_ybase;    /* |sel_esw+sel_echo|   (E2 guard)        */
} Aec3PostAbs;

/* Process one hop. Inputs are the raw fp32 spectra the driver reads (already
 * post shadow-output-selection: error_spec == _sel_esw, echo_spec_sel ==
 * _sel_echo_spec; echo_spec_for_coh == filter.echo_spec for the Γ² EMA), the
 * captured magnitudes (Aec3PostAbs), the avg-reverb X² slices + steady decay,
 * the time-domain far_end block, and the sub-module outputs
 * (usable_linear / saturated_capture / gain[n_bins]).
 *
 *   near_spec / error_spec / echo_spec_sel / echo_spec_for_coh
 *                          : Complex[n_bins]
 *   mag                    : captured f32 magnitudes (see Aec3PostAbs)
 *   x2_present             : 1 if the avg-reverb X² slices are valid
 *   x2_at_delay / x2_past  : float[n_bins] (only read if x2_present)
 *   decay_steady           : the steady ReverbDecay the .py passes to the
 *                            avg-reverb update (computed inside REE; captured)
 *   far_end                : float[hop_size]  (unused by the output path; kept
 *                            for signature completeness / render_block_scaled)
 *   saturated_capture      : (_saturation_level > 0.5)
 *   usable_linear          : aec_state.usable_linear_estimate()
 *   gain                   : float[n_bins] (SuppressionGain output)
 *   out                    : float[hop_size] result (caller-owned).
 *
 * After the call, the driver-derived intermediates live in p->near_psd /
 * far_psd / echo_psd / error_psd / capture_psd_erle / x2_reverb_for_erle /
 * coh_gate_mask / comfort_noise (for the per-stage parity assertion). */
void aec3_post_process(Aec3Post *p,
                       const Complex *near_spec,
                       const Complex *error_spec,
                       const Complex *echo_spec_sel,
                       const Complex *echo_spec_for_coh,
                       const Aec3PostAbs *mag,
                       int x2_present,
                       const float *x2_at_delay, const float *x2_past,
                       float decay_steady,
                       const float *far_end,
                       int saturated_capture,
                       int usable_linear,
                       const float *gain,
                       float *out);

/* ─────────────────────────────────────────────────────────────────────────
 * Stage functions — the six pieces aec3_post_process is decomposed into. The
 * full-orchestration driver (aec3_post_run) calls these interleaved with the
 * sub-module calls in the exact order of AEC._aec3_post. aec3_post_process
 * calls them in sequence so the driver-only parity test stays byte-equal.
 * ──────────────────────────────────────────────────────────────────────── */

/* Stage 1-2: near/far/echo/error PSDs + (E1) capture_psd_erle. */
void aec3_post_compute_psds(Aec3Post *p, const Aec3PostAbs *mag);

/* Stage 3: avg-render-reverb → p->x2_reverb_for_erle (only when x2_present). */
void aec3_post_compute_x2_reverb(Aec3Post *p, int x2_present,
                                 const float *x2_at_delay, const float *x2_past,
                                 float decay_steady);

/* Stage 4: coherence Γ²(Ŷ,Y) EMA → p->coh_gate_mask (when erle_coh_gate). */
void aec3_post_compute_coherence(Aec3Post *p,
                                 const Complex *echo_spec_for_coh,
                                 const Complex *near_spec,
                                 const Aec3PostAbs *mag);

/* Stage 5: CNG N2 tracking → p->comfort_noise. */
void aec3_post_compute_comfort_noise(Aec3Post *p, int saturated_capture);

/* Stage 6: E2 output-base select + gain apply + CNG inject + irfft + OLA. */
void aec3_post_apply_output(Aec3Post *p,
                            const Complex *error_spec,
                            const Complex *echo_spec_sel,
                            const Aec3PostAbs *mag,
                            int usable_linear,
                            const float *gain,
                            float *out);

/* ─────────────────────────────────────────────────────────────────────────
 * aec3_post_run — the FULL orchestration of AEC._aec3_post. Unlike
 * aec3_post_process (which takes the sub-module results gain/usable_linear as
 * INPUTS), this drives AecState + ResidualEchoEstimator + SuppressionGain +
 * StationarityEstimator + LinearFilterSelect itself, reproducing every line of
 * the Python method bit-exactly end-to-end.
 *
 * The caller owns + constructs all the sub-module objects (exactly as the
 * orchestrator's __init__ does, captured by the golden); aec3_post_run only
 * wires the per-hop call sequence. Stationarity is updated UPSTREAM (in the
 * process loop, not here): the caller must have advanced it for this hop before
 * the call, and supplies the active-hops / converge-hops counters.
 * ──────────────────────────────────────────────────────────────────────── */

/* DelayAdjustment enum (delay/delay_types.py): NONE/BUFFER_FLUSH/NEW_DETECTED. */
#define AEC3_DELAY_ADJ_NONE          0
#define AEC3_DELAY_ADJ_BUFFER_FLUSH  1
#define AEC3_DELAY_ADJ_NEW_DETECTED  2

/* DelayQuality.REFINED (delay/delay_types.py). */
#define AEC3_DELAY_QUALITY_REFINED   2

/* All per-hop inputs aec3_post_run reads, plus the persistent objects it
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
    int            pending_gain_change;  /* _aec3_pending_gain_change (in) */
    int            pending_delay_change; /* _aec3_pending_delay_change:
                                          *  -1 == None, else DelayAdjustment */

    /* ── stationarity read state (updated upstream) ─────────────────────── */
    int            stationarity_active_hops;
    int            stationarity_converge_hops;
    int            stationary_block;       /* raw is_block_stationary result;
                                             * not convergence-gated          */

    /* ── config flags ───────────────────────────────────────────────────── */
    int            erle_coh_gate_enabled;
    int            use_stationarity_properties; /* sg echo_audibility flag */
    int            context_only;          /* expose gain/R²/CNG seam but skip
                                             * unused private time synthesis */
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
} Aec3PostRunIn;

/* Sub-module object bundle the driver wires. All caller-owned + constructed. */
typedef struct {
    AecState              *state;
    ResidualEchoEstimator *ree;
    SuppressionGain       *sg;
    StationarityEstimator *stationarity;
    LinearFilterSelect    *lfs;
    FftHandle             *fft;          /* sized fft_size (bridge irfft)  */
} Aec3PostRunObj;

/* Scratch buffers the driver needs (caller-owned, sized to the geometry). */
typedef struct {
    Complex *sel_esw;            /* [n_freqs] selected error_spec_windowed */
    Complex *sel_echo;           /* [n_freqs] selected echo_spec           */
    Complex *nsw_e1;             /* [n_freqs] esw_orig + echo_orig (E1)    */
    Complex *ybase;              /* [n_freqs] sel_esw + sel_echo (E2 guard)*/
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
} Aec3PostRunScratch;

/* Process one hop end-to-end. `out[hop]` is always written: the synthesized
 * internal-RES signal normally, or in->raw_output when context_only is set.
 * The pending EPV flags in `in` are consumed (the caller's mirror is cleared
 * via the returned values): *out_pending_gain_change / *out_pending_delay_change
 * receive the post-call state (cleared to 0 / -1 when an event fired). */
int aec3_post_run(Aec3Post *p,
                  const Aec3PostRunIn *in,
                  const Aec3PostRunObj *obj,
                  Aec3PostRunScratch *sc,
                  float *out,
                  int *out_pending_gain_change,
                  int *out_pending_delay_change);

#ifdef __cplusplus
}
#endif

#endif /* AEC3_POST_H */
