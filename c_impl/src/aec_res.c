/* aec_res.c -- residual-echo suppression gain stage; see aec_res_internal.h. */
#include "aec_res_internal.h"
#include "aec_config_build.h"
#include "aec_internal.h"
#include "aec_balanced_config.h"
#include "utility/simd_aec.h"
#include "utility/fast_math.h"
#include "utility/mem_align.h"
#include "suppression_gain.h"

#include <math.h>
#include <string.h>

/* int16^2 power scale of r2 / comfort_noise relative to audio-scale spectra */
#define AEC_RES_PSD_SCALE (32768.0f * 32768.0f)
#define AEC_RES_SATURATION_THRESHOLD 0.5f
/* Same bound as the suppressor's render statistics */
#define AEC_RES_STATS_MAX_HOP 1024

struct AecRes {
    int n_bins;
    int hop_size;
    /* Configuration as built; reset re-applies it and a level change updates it. */
    SuppressionGainConfig sg_cfg;
    SuppressionGainTuning sg_tun;
    SuppressionGain sg;
    float* sg_storage;      /* (10 + nearend_smoother_n) * n_bins floats */
    float* near_power;      /* n_bins */
    float* render_scaled;   /* hop_size */
};

/* Float count of the suppressor state and scratch: three per-bin history arrays, the smoother
 * ring and seven per-bin work arrays. */
static size_t sg_float_count(int n_bins, int smoother_n) {
    return ck_mul_size((size_t)(10 + smoother_n), (size_t)n_bins);
}

static int resolve(const AecConfig* cfg, AecConfig* resolved,
                   SuppressionGainConfig* sg_cfg, SuppressionGainTuning* sg_tun) {
    if (!cfg || !aec_resolve_valid_config(cfg, resolved)) return 0;
    return aec_build_sg_config(resolved, sg_cfg, sg_tun) == 0;
}

size_t aec_res_get_mem_size(const AecConfig* cfg) {
    AecConfig resolved;
    SuppressionGainConfig sg_cfg;
    SuppressionGainTuning sg_tun;
    AecSignalGrid grid;
    size_t total;
    if (!resolve(cfg, &resolved, &sg_cfg, &sg_tun)) return 0;
    if (aec_resolve_signal_grid(resolved.sample_rate, resolved.fft_size, &grid) != AEC_OK) return 0;
    if (sg_cfg.nearend_smoother_n < 1) return 0;
    total = ck_align16_size(sizeof(struct AecRes));
    total = ck_field_size(total, sg_float_count(grid.n_freqs, sg_cfg.nearend_smoother_n),
                          sizeof(float));
    total = ck_field_size(total, (size_t)grid.n_freqs, sizeof(float));
    total = ck_field_size(total, (size_t)grid.hop_size, sizeof(float));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

/* Re-creates the suppressor from the stored configuration over zeroed state. */
static void rebuild(AecRes* r) {
    const int n = r->n_bins;
    const int ma_n = r->sg_cfg.nearend_smoother_n;
    float* c = r->sg_storage;
    float *last_gain = c, *last_near, *last_echo, *ma_buf, *near_s, *weighted;
    float *min_gain, *max_gain, *raw_gain, *gain, *sum;
    c += n; last_near = c;
    c += n; last_echo = c;
    c += n; ma_buf = c;
    c += (size_t)ma_n * n; near_s = c;
    c += n; weighted = c;
    c += n; min_gain = c;
    c += n; max_gain = c;
    c += n; raw_gain = c;
    c += n; gain = c;
    c += n; sum = c;
    memset(r->sg_storage, 0, sg_float_count(n, ma_n) * sizeof(float));
    suppression_gain_init(&r->sg, &r->sg_cfg, &r->sg_tun, last_gain, last_near,
                          last_echo, ma_buf, near_s, weighted, min_gain, max_gain,
                          raw_gain, gain, sum);
}

AecRes* aec_res_init(void* mem, size_t bytes, const AecConfig* cfg) {
    AecConfig resolved;
    SuppressionGainConfig sg_cfg;
    SuppressionGainTuning sg_tun;
    AecSignalGrid grid;
    size_t need;
    uint8_t* p;
    AecRes* r;
    if (!mem || !MEM_IS_ALIGNED16(mem)) return NULL;
    need = aec_res_get_mem_size(cfg);
    if (need == 0 || bytes < need) return NULL;
    if (!resolve(cfg, &resolved, &sg_cfg, &sg_tun)) return NULL;
    if (aec_resolve_signal_grid(resolved.sample_rate, resolved.fft_size, &grid) != AEC_OK) return NULL;

    r = (AecRes*)mem;
    memset(r, 0, sizeof *r);
    r->n_bins = grid.n_freqs;
    r->hop_size = grid.hop_size;
    r->sg_cfg = sg_cfg;
    r->sg_tun = sg_tun;
    p = (uint8_t*)mem + ck_align16_size(sizeof *r);
    r->sg_storage = (float*)p;
    p += ck_align16_size(sg_float_count(r->n_bins, sg_cfg.nearend_smoother_n) * sizeof(float));
    r->near_power = (float*)p;
    p += ck_align16_size((size_t)r->n_bins * sizeof(float));
    r->render_scaled = (float*)p;
    rebuild(r);
    return r;
}

void aec_res_reset(AecRes* r) {
    if (r) rebuild(r);
}

int aec_res_n_bins(const AecRes* r) { return r ? r->n_bins : 0; }
int aec_res_hop_size(const AecRes* r) { return r ? r->hop_size : 0; }

const float* aec_res_process(AecRes* r, const AecResInput* in) {
    int k;
    float sum_sq, peak_sq;
    if (!r || !in || !in->error_spec || !in->near_spec || !in->r2 || !in->comfort_noise)
        return NULL;

    /* Near-end power on the int16^2 scale; once every filter has converged the residual power caps
     * it. */
    for (k = 0; k < r->n_bins; ++k) {
        float n2 = in->near_spec[k].r * in->near_spec[k].r +
                   in->near_spec[k].i * in->near_spec[k].i;
        if (in->all_converged) {
            float e2 = in->error_spec[k].r * in->error_spec[k].r +
                       in->error_spec[k].i * in->error_spec[k].i;
            n2 = fminf(e2, n2);
        }
        r->near_power[k] = n2 * AEC_RES_PSD_SCALE;
    }
    if (in->render_hop) {
        sk_scale_f32(r->render_scaled, in->render_hop, 32768.0f, r->hop_size);
        suppression_gain_render_stats(r->render_scaled, r->hop_size, &sum_sq, &peak_sq);
    } else {
        sum_sq = in->render_sum_sq;
        peak_sq = in->render_peak_sq;
    }

    if (r->sg.initial_state && in->all_converged)
        suppression_gain_set_initial_state(&r->sg, 0);
    r->sg.dt_protect_active = in->max_dt > AEC_RES_DT_PROTECT_THRESHOLD;
    return suppression_gain_get_gain_stats(&r->sg, r->near_power, in->r2, in->r2,
                                           in->comfort_noise, sum_sq, peak_sq,
                                           in->clock_drift,
                                           in->max_saturation > AEC_RES_SATURATION_THRESHOLD);
}

void aec_render_stats(const float* hop, int n, AecRenderStats* out) {
    float scaled[AEC_RES_STATS_MAX_HOP];
    if (!out) return;
    out->sum_sq = 0.0f;
    out->peak_sq = 0.0f;
    if (!hop || n <= 0 || n > AEC_RES_STATS_MAX_HOP) return;
    sk_scale_f32(scaled, hop, 32768.0f, n);
    suppression_gain_render_stats(scaled, n, &out->sum_sq, &out->peak_sq);
}

int aec_res_set_level(AecRes* r, AecLevel level) {
    float db, dt_db;
    int en;
    if (!r) return -1;
    if (aec_level_floor_db(level, &db) != 0) return -1;
    if (aec_level_dt_floor_db(level, &dt_db) != 0) return -1;
    if (aec_level_split_floor_enabled(level, &en) != 0) return -1;
    /* The suppressor validates and applies first and writes nothing on refusal. The ramp governs the
     * far-active floor only; the DT floor and the split-floor gate step immediately. */
    if (suppression_gain_set_split_floor_far_active_db(&r->sg, db, AEC_LEVEL_RAMP_MS) != 0)
        return -1;
    if (suppression_gain_set_split_floor_dt_db(&r->sg, dt_db) != 0)
        return -1;
    if (suppression_gain_set_split_floor_enabled(&r->sg, en) != 0)
        return -1;
    r->sg_cfg.split_floor_far_active = r->sg.cfg.split_floor_far_active;
    r->sg_cfg.split_floor_dt = r->sg.cfg.split_floor_dt;
    r->sg_cfg.split_floor_enabled = r->sg.cfg.split_floor_enabled;
    return 0;
}

int aec_res_split_floor(const AecRes* r, float* live, float* target) {
    if (!r) return -1;
    if (live) *live = r->sg.split_floor_far_active_live;
    if (target) *target = r->sg_cfg.split_floor_far_active;
    return 0;
}

/* ---- test-only hooks (aec_test_hooks.h; compiled only under -DAEC_TESTING) ------------------------------------------------ */
#ifdef AEC_TESTING
typedef struct AecTestingRes AecTestingRes;   /* opaque in aec_test_hooks.h; an AecRes here */

size_t aec_testing_res_mem_size(const AecConfig* cfg) { return aec_res_get_mem_size(cfg); }

AecTestingRes* aec_testing_res_init(void* mem, size_t bytes, const AecConfig* cfg) {
    return (AecTestingRes*)aec_res_init(mem, bytes, cfg);
}

const float* aec_testing_res_process(AecTestingRes* r, const AecResidualView* view) {
    AecResInput in;
    if (!view) return NULL;
    memset(&in, 0, sizeof in);
    in.error_spec = view->error_spec;
    in.near_spec = view->near_spec;
    in.r2 = view->r2;
    in.comfort_noise = view->comfort_noise;
    in.render_hop = view->far_hop;
    in.all_converged = view->all_converged;
    in.max_dt = view->max_dt;
    in.max_saturation = view->max_saturation;
    in.clock_drift = view->clock_drift;
    return aec_res_process((AecRes*)r, &in);
}
#endif
