/* aec_tuning.c -- the internal tuning (defaults, level, range checks); see aec_tuning.h. */
#include "aec_tuning.h"
#include "aec_internal.h"

#include <math.h>
#include <string.h>

void aec_tuning_defaults(AecTuning* tun, const AecConfig* cfg) {
    float far_db, dt_db;
    int split;
    memset(tun, 0, sizeof(*tun));
    /* 64 ms at sr >= 44100, 52 ms otherwise; 0 for an unsupported rate (rejected by the validator), so the millisecond product
     * is never formed from an arbitrary int. */
    tun->filter_length = !aec_is_valid_sample_rate(cfg->sample_rate) ? 0
                       : (cfg->sample_rate >= 44100) ? (cfg->sample_rate * 64 / 1000)
                                                      : (cfg->sample_rate * 52 / 1000);
    tun->n_partitions = 0;
    tun->mu = 0.3f;
    tun->delta = 1e-8f;
    tun->enable_highpass = 1;
    tun->enable_saturation = 1;
    tun->enable_shadow = 1;
    tun->live_observer_enabled = 0;
    tun->max_delay_ms = 1024.0f;
    tun->highpass_cutoff_hz = 80.0f;
    tun->saturation_threshold = 0.95f;
    tun->saturation_softclip_ref = 1;
    tun->cng_n2_windowed_psd = 0;
    /* Frame-count fields are 16 kHz / 10 ms reference-grid values, retimed when the instance is built. */
    tun->shadow_err_alpha = 0.80f;
    tun->shadow_mu_min = 0.5f;
    tun->shadow_mu_nlms = 0.5f;
    tun->warmup_frames = 100;
    tun->epc_hangover = 20;
    tun->epc_total_rise = 1.5f;
    tun->epc_delta_threshold = 0.3f;
    tun->delay_est_init_s = 0.3f;
    tun->delay_est_period_s = 0.0f;
    tun->delay_acquire_protect_converged = 1;
    tun->delay_acquire_warm_transfer = 1;
    tun->delay_acquire_inst_erle_db = 4.0f;
    tun->dt_aware_recovery_soft = 1;
    tun->dt_aware_res_floor_enabled = 1;
    tun->ne_recent_threshold = 0.3f;
    tun->ne_recent_hold = 150;
    tun->ne_recent_sustain = 3;
    tun->filter_misadjustment_stable_frames = 30;
    tun->filter_misadjustment_hangover_frames = 100;
    tun->filter_misadjustment_scale_min = 0.5f;
    tun->filter_misadjustment_scale_max = 2.0f;
    /* An out-of-enum level reads as balanced. */
    tun->min_gain_floor_far_active_db = (aec_level_floor_db(cfg->level, &far_db) == 0) ? far_db : -28.0f;
    tun->min_gain_floor_dt_db = (aec_level_dt_floor_db(cfg->level, &dt_db) == 0) ? dt_db : -20.0f;
    tun->min_gain_split_floor_enabled = (aec_level_split_floor_enabled(cfg->level, &split) == 0) ? split : 1;
    tun->live_observer_shift_ms = 24.0f;
    tun->live_observer_max_hz = 1000.0f;
    tun->live_observer_mu = 0.3f;
    tun->live_observer_corrob_ms = 100.0f;
    tun->live_observer_corrob_margin_db = 0.0f;
}

int aec_apply_level(AecConfig* cfg, AecTuning* tun, AecLevel level) {
    float far_db, dt_db;
    int split;
    if (aec_level_floor_db(level, &far_db) != 0 || aec_level_dt_floor_db(level, &dt_db) != 0 ||
        aec_level_split_floor_enabled(level, &split) != 0) return -1;
    cfg->level = level;
    tun->min_gain_floor_far_active_db = far_db;
    tun->min_gain_floor_dt_db = dt_db;
    tun->min_gain_split_floor_enabled = split;
    return 0;
}

int aec_tuning_valid(const AecTuning* tun, const AecConfig* cfg) {
    if (tun->filter_length <= 0 || tun->filter_length > 4096) return 0;
    if (tun->n_partitions != 0 && (tun->n_partitions < 1 || tun->n_partitions > 256)) return 0;

#define AEC_CK_BOOL(field) \
    do { if (tun->field != 0 && tun->field != 1) return 0; } while (0)
#define AEC_CK_RANGE_F(field, lo, hi) \
    do { if (!isfinite(tun->field) || tun->field < (lo) || tun->field > (hi)) return 0; } while (0)
#define AEC_CK_RANGE_I(field, lo, hi) \
    do { if (tun->field < (lo) || tun->field > (hi)) return 0; } while (0)

    AEC_CK_BOOL(cng_n2_windowed_psd);
    AEC_CK_BOOL(enable_highpass);
    AEC_CK_BOOL(enable_saturation);
    AEC_CK_BOOL(enable_shadow);
    AEC_CK_BOOL(saturation_softclip_ref);
    AEC_CK_BOOL(delay_acquire_protect_converged);
    AEC_CK_BOOL(delay_acquire_warm_transfer);
    AEC_CK_BOOL(dt_aware_recovery_soft);
    AEC_CK_BOOL(dt_aware_res_floor_enabled);

    AEC_CK_RANGE_F(mu,                          0.0f,    10.0f);
    AEC_CK_RANGE_F(delta,                       0.0f,     1.0f);
    AEC_CK_RANGE_F(shadow_err_alpha,            0.0f,     1.0f);
    AEC_CK_RANGE_F(shadow_mu_min,               0.0f,    10.0f);
    AEC_CK_RANGE_F(shadow_mu_nlms,              0.0f,    10.0f);
    AEC_CK_RANGE_F(epc_total_rise,              0.0f,  1000.0f);
    AEC_CK_RANGE_F(epc_delta_threshold,         0.0f,  1000.0f);
    AEC_CK_RANGE_F(max_delay_ms,                0.0f, 60000.0f);
    AEC_CK_RANGE_F(delay_est_init_s,            0.0f,  3600.0f);
    AEC_CK_RANGE_F(delay_est_period_s,          0.0f,  3600.0f);
    /* With enable_highpass the cutoff must satisfy hpf_params_valid(): 0 < cutoff < 0.45 * sample_rate. */
    if (tun->enable_highpass) {
        if (!isfinite(tun->highpass_cutoff_hz) || tun->highpass_cutoff_hz <= 0.0f) return 0;
        if ((double)tun->highpass_cutoff_hz >= 0.45 * (double)cfg->sample_rate) return 0;
    } else {
        AEC_CK_RANGE_F(highpass_cutoff_hz,      0.0f, 20000.0f);
    }
    AEC_CK_RANGE_F(saturation_threshold,        0.0f,    10.0f);
    AEC_CK_RANGE_F(delay_acquire_inst_erle_db,-100.0f,   100.0f);
    AEC_CK_RANGE_F(min_gain_floor_dt_db,     -300.0f,    50.0f);
    AEC_CK_RANGE_F(ne_recent_threshold,          0.0f,  1000.0f);
    AEC_CK_RANGE_F(min_gain_floor_far_active_db,-300.0f,  50.0f);
    AEC_CK_RANGE_F(filter_misadjustment_scale_min, 0.0f, 1000.0f);
    AEC_CK_RANGE_F(filter_misadjustment_scale_max, 0.0f, 1000.0f);

    AEC_CK_RANGE_I(warmup_frames,                       0, 1000000);
    AEC_CK_RANGE_I(epc_hangover,                         0, 1000000);
    AEC_CK_RANGE_I(ne_recent_hold,                       0, 1000000);
    AEC_CK_RANGE_I(ne_recent_sustain,                    0, 1000000);
    AEC_CK_RANGE_I(filter_misadjustment_stable_frames,   0, 1000000);
    AEC_CK_RANGE_I(filter_misadjustment_hangover_frames, 0, 1000000);

    AEC_CK_BOOL(live_observer_enabled);
    AEC_CK_BOOL(min_gain_split_floor_enabled);
    AEC_CK_RANGE_F(live_observer_shift_ms,       0.0f, 1.0e9f);
    AEC_CK_RANGE_F(live_observer_max_hz,         0.0f, 1.0e9f);
    AEC_CK_RANGE_F(live_observer_corrob_ms,      0.0f, 1.0e9f);
    if (!(tun->live_observer_mu > 0.0f) || tun->live_observer_mu > 2.0f) return 0;
    if (!isfinite(tun->live_observer_corrob_margin_db)) return 0;

#undef AEC_CK_BOOL
#undef AEC_CK_RANGE_F
#undef AEC_CK_RANGE_I
    return 1;
}
