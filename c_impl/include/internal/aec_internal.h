/* aec_internal.h -- declarations that are not part of the AEC API (library-internal; tests and the C/Python parity driver may
 * include it): the level lookup table, the comfort-noise fill with a caller-chosen floor, and the shared-far linear entry. */
#ifndef AEC_AEC_INTERNAL_H
#define AEC_AEC_INTERNAL_H

#include "aec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* What a level sets: the far-active min-gain floor (dB; mild -20, balanced -28, aggressive -38), the DT-lifted floor
 * (min_gain_floor_dt_db; only aggressive differs from the shared -20) and min_gain_split_floor_enabled. no_floor repeats
 * balanced's two floors as inert placeholders and switches the split floor off. */
typedef struct AecLevelFloors {
    float far_active_db;
    float dt_db;
    int   split_floor_enabled;
} AecLevelFloors;

/* Returns 0 and fills *out for an in-enum level; -1 and writes nothing for an out-of-enum level or NULL out
 * (aec_config_from_level instead falls back to balanced). */
int aec_level_floors(AecLevel level, AecLevelFloors* out);

/* Adds comfort noise to the bins 1 .. n-2 the suppressor cut (amplitude from cn, g_res and, if non-NULL, max(g_nr, nr_floor); one LCG
 * step per bin on *rng, seeded AEC_CNG_SEED at a fresh instance and every hard post-chain reset). With g_nr == NULL it is
 * bit-identical to the library's own comfort noise (aec_process(), residual FULL, enable_cng). cn, g_res, spec and rng are
 * required; nothing is touched if any is NULL or n < 3 (n = bins of the spectrum). NaN inputs are not supported. */
void aec_cng_fill(uint32_t* rng, Complex* spec, const float* cn,
                  const float* g_res, const float* g_nr, float nr_floor, int n);

/* aec_process_linear() with the far spectrum and its squared magnitudes shared from far_owner, which must have completed
 * this same hop. Both instances must consume identical ref samples with synchronized far-buffer histories. No owner
 * pointer is retained; a NULL, self or incompatible-grid owner falls back to a local computation. A refused hop is
 * AEC_ERR_DSP as for aec_process_linear() (aec.h, "Refused hop"); far_owner is not reset. */
int aec_process_linear_shared_far_from(
        Aec* a, const float* mic, const float* ref, const Aec* far_owner);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_INTERNAL_H */
