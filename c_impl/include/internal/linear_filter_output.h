/* Linear-filter output selector. Per hop it picks the cleaner of the refined (main) and coarse
 * (shadow) time-domain residuals, crossfades over 30 samples on a selection change, applies the
 * analysis sqrt-Hann window over [previous | current] hops, and emits the selected windowed
 * error spectrum and echo spectrum for RES and SuppressionGain. Float32 throughout.
 * Without a shadow filter the caller passes the refined residual for both candidates and the
 * selector acts as the stateful WOLA formatter. The formed residual is always one of the two filter
 * residuals, never the capture: the output guard of the post chain (aec_post_run) only chooses the
 * rendering base and leaves this analysis stream untouched. Python reference: modules/orchestrator.py
 * (UseRefinedOutput, FormLinearFilterOutput). */
#ifndef LINEAR_FILTER_OUTPUT_H
#define LINEAR_FILTER_OUTPUT_H

#include "fft_wrapper.h"

/* Candidates, in crossfade index order. */
#define LFS_SEL_COARSE  0
#define LFS_SEL_REFINED 1

typedef struct {
    int   hop;
    int   block_size;
    int   fft_size;
    int   n_freqs;

    float *prev_output_time;
    int    prev_output_valid;
    int    form_last_selection;       /* LFS_SEL_* chosen by the previous hop (init LFS_SEL_REFINED) */

    float   *e_form;
    Complex *sel_esw;

    float   *scr_sq;
    float   *scr_sref;
    float   *scr_scoa;
    float   *scr_tin;
} LinearFilterSelect;

/* Resets the persistent selection state. */
void linear_filter_select_reset(LinearFilterSelect *s);

/* One hop of selection and output formation.
 * e_refined_time, near_end, e_coarse_time: float32[hop].
 * error_spec_windowed, echo_spec: Complex[n_freqs] from the filter.
 * sqrt_hann: float32[block_size]. fft: handle for fft_size.
 * out_sel_esw, out_sel_echo: Complex[n_freqs] selected error and echo spectra. */
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

#endif
