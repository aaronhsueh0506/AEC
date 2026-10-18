/* aec_res.h -- residual-echo suppression gain as a standalone stage.
 *
 * The AEC library computes a per-bin gain G_res(f) that removes the residual
 * echo the linear filter leaves behind. Inside Aec it runs as part of the
 * post chain. This header exposes the same suppressor for callers that fuse
 * several linear AEC outputs first and need one RES gain on the fused
 * spectra (a multi-lane pipeline), together with the comfort-noise fill that
 * pairs with it.
 *
 * Configuration comes from the same builder the library uses for its own
 * lanes, so an AecRes created from an AecConfig suppresses exactly as a lane
 * built from that config would, for equal inputs.
 *
 * Memory: caller-owned block (aec_res_get_mem_size / aec_res_init), 16-byte
 * aligned. No allocation, no stdio, no global state; one instance is not
 * thread-safe, distinct instances are independent.
 *
 * Scales:
 *   error_spec, near_spec  audio scale ([-1, 1] samples), as AecResContext
 *   r2, comfort_noise      int16^2 power scale, as AecResContext.r2/.comfort_noise
 *   render_hop             audio scale time-domain samples, hop_size long
 *   returned gain          amplitude, [0, 1]
 */
#ifndef AEC_AEC_RES_H
#define AEC_AEC_RES_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AecRes AecRes;

/* One hop of inputs. All arrays have n_bins (= fft_size/2 + 1) elements except
 * render_hop, which has hop_size (= fft_size/2) elements. All are required. */
typedef struct AecResInput {
    const Complex* error_spec;   /* linear-AEC residual E(f) */
    const Complex* near_spec;    /* capture spectrum matching error_spec */
    const float*   r2;           /* residual-echo PSD R^2(f); used as both the
                                  * bounded and the unbounded estimate */
    const float*   comfort_noise;/* comfort-noise PSD N^2(f) */
    const float*   render_hop;   /* far-end samples of this hop */
    int   all_converged;         /* nonzero: every contributing filter has
                                  * converged; the near-end power reference is
                                  * min(|E|^2, |Y|^2) instead of |Y|^2 and the
                                  * suppressor leaves its initial state */
    float max_dt;                /* worst double-talk indicator; above 0.2 the
                                  * double-talk gain floor applies */
    float max_saturation;        /* worst saturation level; above 0.5 the echo
                                  * is treated as saturated */
    int   clock_drift;           /* nonzero: far/near clocks drift */
} AecResInput;

/* Bytes needed for `cfg`; 0 if cfg is NULL or invalid. Only sample_rate,
 * fft_size and the two minimum-gain floors of cfg influence the suppressor. */
size_t aec_res_get_mem_size(const AecConfig* cfg);

/* Builds an instance inside `mem` (16-byte aligned, at least
 * aec_res_get_mem_size(cfg) bytes). Returns NULL on invalid input. The memory
 * must stay valid while the instance is used; there is nothing to destroy. */
AecRes* aec_res_init(void* mem, size_t bytes, const AecConfig* cfg);

/* Clears all suppressor state. The configured floor (including one set by
 * aec_res_set_preset) is kept. */
void aec_res_reset(AecRes* r);

/* Number of frequency bins and hop length the instance was built for; 0 for
 * NULL. */
int aec_res_n_bins(const AecRes* r);
int aec_res_hop_size(const AecRes* r);

/* One hop. Returns the gain, n_bins floats owned by `r` and valid until the
 * next call on `r`; NULL if r or in is NULL or an input array is NULL. */
const float* aec_res_process(AecRes* r, const AecResInput* in);

/* Retargets the far-end-active gain floor to the preset's floor, reaching it
 * after ramp_ms (0: next hop); also retargets the DT-lifted floor (only
 * aggressive's differs from the shared -20 dB default -- see aec.h's
 * aec_preset_dt_floor_db()), which has no ramp and steps immediately
 * regardless of ramp_ms -- same semantics as aec_set_preset(). Returns 0, or
 * -1 for NULL, an unknown preset or an out-of-range ramp, in which case
 * nothing changes. */
int aec_res_set_preset(AecRes* r, AecPreset preset, float ramp_ms);

/* Reads the far-end-active floor as a power ratio: `live` is the value in force
 * now, `target` the one it is heading to (kept across aec_res_reset). Either
 * pointer may be NULL. Returns 0, or -1 if r is NULL. */
int aec_res_split_floor(const AecRes* r, float* live, float* target);

/* Comfort-noise fill for the bins the suppressor cut. Adds to spec[k], for
 * k = 1 .. n - 2:
 *
 *   amplitude = sqrt(cn[k] / 32768^2) * sqrt(1 - g_res[k]^2) * max(g_nr[k], nr_floor)
 *
 * (a negative cn or 1 - g_res^2 counts as 0) times a random phase taken from a
 * 32-entry sqrt(2)*sin table: one linear-congruential step per bin advances
 * *rng, its top five bits pick the real part's table entry and the entry a
 * quarter turn ahead the imaginary part's. The same recipe, step order and
 * table the library's own fill uses, but the arithmetic is the plain-libm form
 * of the multi-lane pipelines, not the library's batched one.
 *
 * g_nr may be NULL (no second gain: treated as 1). cn, g_res, spec and rng
 * are required; nothing is touched if any is NULL or n < 3. n is the number of
 * bins of the spectrum. */
void aec_cng_fill(uint32_t* rng, Complex* spec, const float* cn,
                  const float* g_res, const float* g_nr, float nr_floor, int n);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_RES_H */
