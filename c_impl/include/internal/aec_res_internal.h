/* aec_res_internal.h -- library-internal residual-echo suppression gain stage (the array instance's fused
 * gain; aec_array.c). Not installed. Configuration comes from the same builder as the library's own lanes.
 * Memory: caller-owned 16-byte-aligned block (aec_res_get_mem_size / aec_res_init); no allocation
 * or global state; an instance is not thread-safe, distinct instances are independent.
 * Scales: error_spec and near_spec are audio scale ([-1, 1]); r2 and comfort_noise are int16^2
 * power; render_hop is audio-scale samples; the returned gain is an amplitude in [0, 1]. */
#ifndef AEC_AEC_RES_INTERNAL_H
#define AEC_AEC_RES_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"
#include "aec_tuning.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AecRes AecRes;

/* max_dt above this arms the gain stage's DT protection; the fused view reports the same decision. */
#define AEC_RES_DT_PROTECT_THRESHOLD 0.2f

/* One hop of inputs. All arrays have n_bins (= fft_size/2 + 1) elements except
 * render_hop, which has hop_size (= fft_size/2) elements. error_spec, near_spec, r2 and
 * comfort_noise and render_hop are required. */
typedef struct AecResInput {
    const Complex* error_spec;   /* linear-AEC residual E(f) */
    const Complex* near_spec;    /* capture spectrum matching error_spec */
    const float*   r2;           /* residual-echo PSD R^2(f), used as bounded and unbounded estimate */
    const float*   comfort_noise;/* comfort-noise PSD N^2(f) */
    const float*   render_hop;   /* far-end samples of this hop */
    int   all_converged;         /* nonzero: every filter has converged (near power = min(|E|^2, |Y|^2); initial state left) */
    float max_dt;                /* worst double-talk indicator; above 0.2 the DT gain floor applies */
    float max_saturation;        /* worst saturation level; above 0.5 the echo is treated as saturated */
    int   clock_drift;           /* nonzero: far/near clocks drift */
} AecResInput;

/* Bytes needed for `cfg`; 0 if cfg is NULL or invalid. Only sample_rate, fft_size and level of cfg influence the suppressor. */
size_t aec_res_get_mem_size(const AecConfig* cfg);

/* Builds an instance inside `mem` (16-byte aligned, at least
 * aec_res_get_mem_size(cfg) bytes). Returns NULL on invalid input. The memory
 * must stay valid while the instance is used; there is nothing to destroy. */
AecRes* aec_res_init(void* mem, size_t bytes, const AecConfig* cfg);

/* The same for an already-resolved configuration (the array instance passes its own tuning). */
size_t  aec_res_get_mem_size_resolved(const AecResolved* resolved);
AecRes* aec_res_init_resolved(void* mem, size_t bytes, const AecResolved* resolved);

/* Clears all suppressor state. The configured floor (including one set by
 * aec_res_set_level) is kept. */
void aec_res_reset(AecRes* r);

/* Number of frequency bins and hop length the instance was built for; 0 for
 * NULL. */
int aec_res_n_bins(const AecRes* r);
int aec_res_hop_size(const AecRes* r);

/* One hop. Returns the gain, n_bins floats owned by `r` and valid until the
 * next call on `r`; NULL if r or in is NULL or a required input array is NULL. */
const float* aec_res_process(AecRes* r, const AecResInput* in);

/* Time over which a level change walks the far-end-active floor (linear in dB); aec_set_level()
 * and aec_res_set_level() use it, there is no per-call value. */
#define AEC_LEVEL_RAMP_MS 100.0f

/* Retargets the far-end-active gain floor to the level's floor, reached after AEC_LEVEL_RAMP_MS,
 * and the DT-lifted floor, which steps immediately. Same semantics as aec_set_level(). Returns 0,
 * or -1 for NULL or an unknown level (nothing changes). */
int aec_res_set_level(AecRes* r, AecLevel level);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_RES_INTERNAL_H */
