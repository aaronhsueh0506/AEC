/* linear_filter_output.h — C port of
 * AEC._aec3_select_linear_filter_output (python/modules/orchestrator.py
 * lines 2894-2999): AEC3 UseRefinedOutput + FormLinearFilterOutput parity.
 *
 * Per hop it picks the cleaner of the refined (main) / coarse (shadow) linear
 * filter time-domain residuals via the AEC3 UseRefinedOutput predicate, runs a
 * 30-sample SignalTransition crossfade on selection change, OLA-windows the
 * formed output through the analysis sqrt-Hann + WindowedPaddedFft, and emits
 * the selected windowed error spectrum + selected echo spectrum that feed RES +
 * SuppressionGain.
 *
 * Numerical fidelity (float32-by-design; converted for uniformity as part of
 * the f32 campaign, drift accepted — see project_c_port_parity_rules.md for
 * the retired fp64 parity history):
 *  - the e2/y2/s2 block sum-of-squares are np.sum over float32 (each f32
 *    sample squared in float32) → PAIRWISE-summed in float32
 *    (pairwise_sum_f32, matching pbfdkf.c's pairwise_sum_f32 tree shape).
 *  - thresholds are float32 expressions.
 *  - the crossfade ramp stays float32 (s = (k+1)/(kT+1) is f32_array / pyfloat
 *    → f32; (1-s) is pyfloat - f32_array → f32; products/sum f32).
 *  - the rfft of [e_old | e_form]·sqrt_hann (length block_size, zero-padded to
 *    fft_size) goes through the vendored bit-exact pocketfft (fft_forward).
 *  - complex64 add / subtract are elementwise (no FMA needed).
 *  - built with -ffp-contract=off so each float op rounds separately.
 *
 * Pure additive port (not wired into aec.c). Heap-only init/reset (matches the
 * orchestrator state, which has no static-memory variant).
 */
#ifndef LINEAR_FILTER_OUTPUT_H
#define LINEAR_FILTER_OUTPUT_H

#include "fft_wrapper.h"

typedef struct {
    int   hop;
    int   block_size;
    int   fft_size;
    int   n_freqs;

    /* persistent selection state (matches Python init) */
    float *prev_output_time;   /* owned; length hop. _form_prev_output_time */
    int    prev_output_valid;  /* 0 ⇒ Python None ⇒ use zeros(hop) */
    int    form_last_selection;       /* _form_last_selection,  init True (1) */
    int    refined_last_selected;     /* _refined_filter_output_last_selected */

    /* scratch (owned) */
    float   *e_form;        /* length hop */
    float   *block_win;     /* length block_size */
    Complex *sel_esw;       /* length n_freqs (selected_esw, also returned) */

    /* Per-hop scratch, instance storage (not stack) — these were float[8192]
     * locals in linear_filter_select() (stack-overflow hazard on small
     * embedded task stacks). Each gets its own field: sq is reused
     * sequentially across several sum_sq_f32() calls (safe, since each call
     * fully consumes it before the next), but sref/scoa/tin are separate
     * because sq is read again (as scratch) while sref's/scoa's own values
     * are still needed by the caller in the surrounding scope. */
    float   *scr_sq;         /* length hop:      sum_sq_f32 scratch (e2/y2/s2) */
    float   *scr_sref;       /* length hop:      s_refined_time = near - e_refined */
    float   *scr_scoa;       /* length hop:      s_coarse_time  = near - e_coarse */
    float   *scr_tin;        /* length fft_size: zero-padded rfft input */
} LinearFilterSelect;

/* Allocate scratch + persistent buffers; sets state to Python init
 * (_form_prev_output_time=None, _form_last_selection=True,
 *  _refined_filter_output_last_selected=True). Returns 0 on success, -1 on OOM.
 */
int  linear_filter_select_init(LinearFilterSelect *s,
                               int hop, int block_size, int fft_size,
                               int n_freqs);

/* Reset persistent state to Python init (does not free buffers). */
void linear_filter_select_reset(LinearFilterSelect *s);

/* Free owned buffers. */
void linear_filter_select_free(LinearFilterSelect *s);

/* One hop of UseRefinedOutput + FormLinearFilterOutput.
 *
 *   e_refined_time      : raw_output, float32[hop]
 *   near_end            : near_end_block, float32[hop]
 *   e_coarse_time       : _last_shadow_output_time, float32[hop]
 *   error_spec_windowed : filter.error_spec_windowed, complex64[n_freqs]
 *   echo_spec           : filter.echo_spec, complex64[n_freqs]
 *   sqrt_hann           : filter._sqrt_hann_analysis, float32[block_size]
 *   fft                 : FFT handle for fft_size
 *   out_sel_esw         : selected_esw,        complex64[n_freqs]
 *   out_sel_echo        : selected_echo_spec,  complex64[n_freqs]
 *
 * Updates persistent state (prev_output_time, form_last_selection,
 * refined_last_selected) exactly as the Python tail does.
 */
void linear_filter_select(LinearFilterSelect *s,
                          const float *e_refined_time,
                          const float *near_end,
                          const float *e_coarse_time,
                          const Complex *error_spec_windowed,
                          const Complex *echo_spec,
                          const float *sqrt_hann,
                          FftHandle *fft,
                          Complex *out_sel_esw,
                          Complex *out_sel_echo);

#endif /* LINEAR_FILTER_OUTPUT_H */
