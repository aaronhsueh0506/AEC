/* aec_internal.h -- declarations that are not part of the AEC API (library-internal; tests and the C/Python parity driver may
 * include it): the level lookup tables, the comfort-noise fill with a caller-chosen floor, and the shared-far linear entry. */
#ifndef AEC_AEC_INTERNAL_H
#define AEC_AEC_INTERNAL_H

#include "aec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Far-active min-gain floor (dB) for a level: mild -20, balanced -28, aggressive -38 (no_floor: -28, an inert placeholder).
 * Returns 0 and writes *db_out for an in-enum level, -1 otherwise; aec_config_from_level instead falls back to balanced. */
int aec_level_floor_db(AecLevel level, float* db_out);

/* DT-lifted floor (min_gain_floor_dt_db) for a level; only aggressive differs from the shared -20 dB (no_floor: inert -20).
 * Returns 0 and writes *db_out for an in-enum level, -1 (nothing written) otherwise. */
int aec_level_dt_floor_db(AecLevel level, float* db_out);

/* min_gain_split_floor_enabled for a level. Returns 0 and writes *enabled_out for an in-enum level, -1 (nothing written)
 * otherwise. */
int aec_level_split_floor_enabled(AecLevel level, int* enabled_out);

/* Comfort-noise fill for the bins the suppressor cut. For k = 1 .. n-2 adds to spec[k]
 *   amplitude * (sqrt(2) * sin table entry)
 * with amplitude = sqrt(cn[k] / 32768^2) * sqrt(1 - g_res[k]^2) * max(g_nr[k], nr_floor), evaluated as
 * (sqrt(1 - g_res^2) * max(g_nr, nr_floor)) * (sqrt(cn / 32768^2) * table entry) (negative
 * radicands count as 0; the g_nr factor is omitted entirely when g_nr is NULL). The phase is one LCG step
 * per bin on *rng (seed = (seed * 69069 + 1) & 0x7FFFFFFF, table index = seed >> 26 for the real part and
 * (index + 8) & 31 for the imaginary part), drawn in bin order for bins 1 .. n-2 only; bins 0 and n-1 get
 * no noise and draw nothing.
 * The library's own output stage (aec_process() with residual FULL and enable_cng) calls this function on
 * the gain-weighted base with g_res = its RES gain and g_nr = NULL, so with g_nr == NULL the result is
 * bit-identical to the internal comfort noise on the same inputs (same square roots, same product order
 * sqrt(1 - g^2) * (sqrt(cn / 32768^2) * table entry), same draw schedule). The seed of a fresh instance and after every
 * hard post-chain reset is AEC_CNG_SEED; see aec_post_reset_serial().
 * cn, g_res, spec and rng are required; nothing is touched if any is NULL or n < 3. n is the number of
 * bins of the spectrum. NaN inputs are not supported. */
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
