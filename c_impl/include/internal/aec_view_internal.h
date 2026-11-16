/* aec_view_internal.h -- the complete views (library and tests only, not installed): aec.h carries the fields a caller outside
 * the library reads, these add the rest. The getters behave as aec_get_linear() / aec_get_residual() / aec_array_get_timing(). */
#ifndef AEC_VIEW_INTERNAL_H
#define AEC_VIEW_INTERNAL_H

#include "aec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AecLinearViewFull {
    size_t       struct_size;
    int          hop_size;
    int          n_freqs;
    const float* linear_hop;
    const Complex* echo_spec;        /* main filter echo / near / far spectra (pre-FORM; the far one is rectangular-block) */
    const Complex* near_spec;
    const Complex* far_spec;
    const float* aligned_far_hop;
    int          delay_samples;
    float        delay_confidence;
    AecLinearDelayState delay_state;
    unsigned int generation;         /* the alignment generation (saturating) */
    float        far_power;
    float        erle_factor;
    float        dt_indicator;
    float        divergence;
    float        saturation_level;
    float        erl_estimate;
    float        shadow_dt;
    int          is_stationary_dt;
    int          filter_converged;
    int          filter_once_converged;
    int          epc_active;         /* echo-path-change regime */
} AecLinearViewFull;

typedef struct AecResidualViewFull {
    size_t         struct_size;
    int            n_freqs;
    int            hop_size;
    const float*   formed_hop;
    const Complex* error_spec;
    const Complex* echo_spec;
    const Complex* near_spec;
    const Complex* far_spec;
    const Complex* base_spec;
    int            guard_active;
    const float*   res_gain;
    const float*   r2;
    const float*   comfort_noise;
    int            res_floor_protect;  /* this hop's near-recent DT-floor decision */
    int            usable_linear;
    uint32_t       post_reset_serial;
    int            all_converged;
    float          max_dt;
    float          max_saturation;
    int            clock_drift;
    unsigned int   guard_lanes;
    const float*   far_hop;
} AecResidualViewFull;

/* As aec_get_linear() / aec_get_residual(): struct_size is an input (at least sizeof the Full struct), the record is zeroed as the
 * public getter zeroes its own. */
int aec_get_linear_full(const Aec* a, AecLinearViewFull* out);
int aec_get_residual_full(const Aec* a, AecResidualViewFull* out);

#ifdef __cplusplus
}
#endif

#endif /* AEC_VIEW_INTERNAL_H */
