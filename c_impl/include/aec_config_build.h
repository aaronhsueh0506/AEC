/* aec_config_build.h -- configuration builders shared by aec.c and aec_res.c.
 *
 * Library-internal: not part of the public API. Both entry points exist so a
 * lane inside Aec and a standalone residual-echo suppressor (aec_res.h) derive
 * their suppression-gain configuration from the same code. */
#ifndef AEC_CONFIG_BUILD_H
#define AEC_CONFIG_BUILD_H

#include "aec.h"
#include "suppression_gain.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Validates `in` and writes the resolved config (deprecated fields folded) to
 * `out`. Returns 1 on success, 0 if `in` is NULL or invalid; `out` is
 * untouched on failure. */
int aec_resolve_valid_config(const AecConfig* in, AecConfig* out);

/* Suppression-gain config and tuning for a config produced by
 * aec_resolve_valid_config. The tuning tables point at static per-rate data.
 * Returns 0, or -1 when no signal grid or per-rate row exists. */
int aec_build_sg_config(const AecConfig* cfg, SuppressionGainConfig* out_cfg,
                        SuppressionGainTuning* out_tun);

#ifdef __cplusplus
}
#endif

#endif /* AEC_CONFIG_BUILD_H */
