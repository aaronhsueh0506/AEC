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
 *     cast to f32); syy/see are float64 (the f32 a·|·|² term widened to f64 in
 *     the add). gamma2 = (|sye|²/max(syy·see,1e-30)).astype(f32).
 *   - CNG N2 tracking: y2_smoothed / n2 / n2_initial are float32 arrays; the
 *     per-hop α/freshness/retention/slow-up are python-float scalars value-cast
 *     to f32 in each array op (all-f32 computation), then np.where → .astype
 *     (f32). noise_floor clamp = np.maximum compares in float (f32 array vs f32
 *     scalar).
 *   - CNG LCG: seed (int) recurrence (seed*69069+1) & 0x7FFFFFFF, ix = seed>>26
 *     (0..31), im = (ix+8)&31. uint32 arithmetic reproduces it exactly.
 *   - irfft via fft_fp64.c (numpy irfft 1/N normalization, fp64 internal).
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
    int    erle_windowed_capture_psd;        /* 1 (E1) */
    int    erle_render_x2_psd_scale;         /* 1 */
    int    output_capture_when_linear_unusable; /* 1 (E2) */
    int    enable_cng;                       /* 1 */

    /* coherence Γ²(Ŷ,Y) EMA */
    double erle_coh_gate_alpha;              /* 0.05 */
    double erle_coh_gate_threshold;          /* 0.5  */

    /* CNG wall-clock-rescaled constants */
    double cng_y2_alpha;                     /* y2_smoothed EMA α */
    double cng_n2_track_freshness;
    double cng_n2_track_retention;
    double cng_n2_slow_up;
    double cng_n2_initial_alpha;
    int    cng_n2_update_onset_hops;
    int    cng_n2_initial_duration_hops;
    double noise_floor_int16sq;              /* GetNoiseFloorFactor(dbfs) */
} Aec3PostConfig;

/* Mutable driver state. All per-bin arrays are caller-owned. */
typedef struct {
    Aec3PostConfig cfg;

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

    /* coherence Γ²(Ŷ,Y) EMA (sye complex64 split; syy/see float64) */
    float   *coh_sye_re;           /* [n_bins] */
    float   *coh_sye_im;           /* [n_bins] */
    double  *coh_syy;              /* [n_bins] */
    double  *coh_see;              /* [n_bins] */

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
                    double *coh_syy, double *coh_see,
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
                       double decay_steady,
                       const float *far_end,
                       int saturated_capture,
                       int usable_linear,
                       const float *gain,
                       float *out);

#ifdef __cplusplus
}
#endif

#endif /* AEC3_POST_H */
