/* aec_tuning.c -- the internal tuning (defaults, level, range checks), the level and signal-grid tables, aec_config_from_level() and
 * aec_resolve(); see aec_tuning.h. */
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

/* Level strength table in AecLevel order; NO_FLOOR repeats balanced's
 * (unused) values. */
typedef struct { float far_active_db, dt_db; int split_floor_enabled; } AecLevelFloors;
static const AecLevelFloors AEC_LEVEL_FLOORS[] = {
    [AEC_LEVEL_MILD]       = { -20.0f, -20.0f, 1 },
    [AEC_LEVEL_BALANCED]   = { -28.0f, -20.0f, 1 },
    [AEC_LEVEL_AGGRESSIVE] = { -38.0f, -30.0f, 1 },
    [AEC_LEVEL_NO_FLOOR]   = { -28.0f, -20.0f, 0 },
};

/* Row for an in-enum level, NULL otherwise (every lookup below then fails). */
static const AecLevelFloors* aec_level_row(AecLevel p) {
    return ((unsigned)p < sizeof(AEC_LEVEL_FLOORS) / sizeof(AEC_LEVEL_FLOORS[0]))
               ? &AEC_LEVEL_FLOORS[p] : NULL;
}

const char* aec_level_name(AecLevel p) {
    static const char* const names[] = {
        [AEC_LEVEL_MILD]       = "mild",
        [AEC_LEVEL_BALANCED]   = "balanced",
        [AEC_LEVEL_AGGRESSIVE] = "aggressive",
        [AEC_LEVEL_NO_FLOOR]   = "no_floor",
    };
    return aec_level_row(p) ? names[p] : "balanced";
}

int aec_level_floor_db(AecLevel p, float* db_out) {
    const AecLevelFloors* r = aec_level_row(p);
    if (db_out == NULL || r == NULL) return -1;
    *db_out = r->far_active_db;
    return 0;
}

int aec_level_dt_floor_db(AecLevel p, float* db_out) {
    const AecLevelFloors* r = aec_level_row(p);
    if (db_out == NULL || r == NULL) return -1;
    *db_out = r->dt_db;
    return 0;
}

int aec_level_split_floor_enabled(AecLevel p, int* enabled_out) {
    const AecLevelFloors* r = aec_level_row(p);
    if (enabled_out == NULL || r == NULL) return -1;
    *enabled_out = r->split_floor_enabled;
    return 0;
}

void aec_config_from_level(AecConfig* cfg, AecLevel p, int sr) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->sample_rate = sr;
    /* The only place a default fft_size is applied (0 for an unsupported rate, which the validator rejects). */
    cfg->fft_size = aec_default_fft_size(sr);
    cfg->n_mics = 1;
    cfg->level = aec_level_row(p) ? p : AEC_LEVEL_BALANCED;
    cfg->residual = AEC_RESIDUAL_FULL;
    /* MATCHED with 5 filters is the default; fixed_delay_samples must be -1 (0 is a legal FIXED delay). */
    cfg->delay_mode = AEC_DELAY_MATCHED;
    cfg->fixed_delay_samples = -1;
    cfg->delay_num_filters = AEC_DELAY_NUM_FILTERS_MAX;
    cfg->enable_cng = 1;
    cfg->capture_proxy_channel = 0;
    cfg->delay_backward_quarantine_enabled = 0;
    cfg->delay_backward_quarantine_s = 1.0f;
}

/* Signal-grid table: admissible (sample_rate, fft_size) pairs and their
 * geometry. */
static const AecSignalGrid AEC_GRID_TABLE[] = {
    { 16000,  256,   256,  128,     129, 0 },
    { 16000,  512,   512,  256,     257, 0 },
    { 48000, 1024,  1024,  512,     513, 0 },
    {  8000,  256,   256,  128,     129, 1 },
};

/* First table row for sample_rate whose fft_size matches (fft_size <= 0: any), or NULL. */
static const AecSignalGrid* aec_find_grid(int sample_rate, int fft_size) {
    for (size_t i = 0; i < sizeof(AEC_GRID_TABLE) / sizeof(AEC_GRID_TABLE[0]); ++i) {
        if (AEC_GRID_TABLE[i].sample_rate == sample_rate &&
            (fft_size <= 0 || AEC_GRID_TABLE[i].fft_size == fft_size))
            return &AEC_GRID_TABLE[i];
    }
    return NULL;
}

int aec_resolve_signal_grid(int sample_rate, int fft_size, AecSignalGrid* out) {
    const AecSignalGrid* g;
    if (fft_size <= 0) return AEC_ERR_ARG;
    g = aec_find_grid(sample_rate, fft_size);
    if (!g) return AEC_ERR_ARG;
    if (out) *out = *g;
    return AEC_OK;
}

int aec_is_valid_sample_rate(int sample_rate) {
    return aec_find_grid(sample_rate, 0) != NULL;
}

int aec_default_fft_size(int sample_rate) {
    const AecSignalGrid* g = aec_find_grid(sample_rate, 0);
    return g ? g->fft_size : 0;
}

int aec_resolve_tuned(const AecConfig* in, const AecTuning* tun, AecResolved* out) {
    AecResolved r;
    if (!in || !out) return 0;
    r.cfg = *in;
    if (tun) r.tun = *tun; else aec_tuning_defaults(&r.tun, in);
    if (!aec_resolved_valid(&r)) return 0;
    *out = r;
    return 1;
}

int aec_resolve(const AecConfig* in, AecResolved* out) {
    return aec_resolve_tuned(in, NULL, out);
}
