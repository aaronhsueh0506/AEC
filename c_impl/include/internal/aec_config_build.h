/* aec_config_build.h -- configuration builders shared by aec.c and aec_res.c (library-internal). */
#ifndef AEC_CONFIG_BUILD_H
#define AEC_CONFIG_BUILD_H

#include "aec.h"
#include "aec_tuning.h"
#include "suppression_gain.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Suppression-gain config and tuning for a configuration produced by aec_resolve() / aec_resolve_tuned(). The EMR tables of
 * the tuning point at static per-rate data; the four ENR/SU tables are NULL until aec_sg_tables_decode(). Returns 0, or -1
 * when no signal grid or per-rate row exists. */
int aec_build_sg_config(const AecResolved* rs, SuppressionGainConfig* out_cfg,
                        SuppressionGainTuning* out_tun);

/* Retargets a suppressor to `level`: the far-end-active floor (reached after AEC_LEVEL_RAMP_MS), the DT floor and the split-floor
 * gate. Returns 0, or -1 for an unknown level or a refused value (the lookups come first and the suppressor writes nothing on a
 * refusal, so the first refusal leaves it unchanged). */
int aec_sg_apply_level(SuppressionGain* sg, AecLevel level);

/* Bytes of the four decoded ENR/SU tables of an n_bins grid (each table padded to 16 bytes). */
size_t aec_sg_tables_bytes(int n_bins);

/* Expands the run-length ENR/SU tables of rs's rate row into `mem` (aec_sg_tables_bytes(n_bins) bytes, 16-byte aligned, owned
 * by the caller for as long as *tun is used) and points the four ENR/SU tables of *tun at it. Returns 0, or -1 for a NULL
 * argument or a grid without a rate row. */
int aec_sg_tables_decode(const AecResolved* rs, void* mem, SuppressionGainTuning* tun);

/* Writes the synthesis window of length n into w[0, n): rebuilt bit for bit from the stored half table, n a power of two that
 * divides AECB_SYNTH_HALF_N (every whitelisted block length does). */
void aec_synth_window_build(float* w, int n);

#ifdef __cplusplus
}
#endif

#endif /* AEC_CONFIG_BUILD_H */
