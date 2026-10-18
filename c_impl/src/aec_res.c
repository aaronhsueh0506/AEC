/* aec_res.c -- standalone residual-echo suppression gain; see aec_res.h. */
#include "aec_res.h"
#include "aec_config_build.h"
#include "aec3_balanced_config.h" /* AEC3B_SQRT2_SIN_LUT only */
#include "aec_simd_kernels.h"
#include "mem_align.h"
#include "suppression_gain.h"

#include <math.h>
#include <string.h>

/* int16^2 power scale of r2 / comfort_noise relative to audio-scale spectra */
#define AEC_RES_PSD_SCALE (32768.0f * 32768.0f)
#define AEC_RES_DT_PROTECT_THRESHOLD 0.2f
#define AEC_RES_SATURATION_THRESHOLD 0.5f

struct AecRes {
    int n_bins;
    int hop_size;
    /* Configuration as built; reset re-applies it, and a preset change
     * updates it so a reset keeps the retargeted floor. */
    SuppressionGainConfig sg_cfg;
    SuppressionGainTuning sg_tun;
    SuppressionGain sg;
    float* sg_storage;      /* (10 + nearend_smoother_n) * n_bins floats */
    float* near_power;      /* n_bins */
    float* render_scaled;   /* hop_size */
};

/* Float count of the suppressor's state and scratch: last_gain, last_nearend,
 * last_echo, the smoother ring, and seven per-bin work arrays. */
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
    if (!aec_resolve_signal_grid(resolved.sample_rate, resolved.fft_size, &grid)) return 0;
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
    if (!aec_resolve_signal_grid(resolved.sample_rate, resolved.fft_size, &grid)) return NULL;

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
    if (!r || !in || !in->error_spec || !in->near_spec || !in->r2 ||
        !in->comfort_noise || !in->render_hop)
        return NULL;

    /* Near-end power on the int16^2 scale; once every filter has converged the
     * residual's own power caps it. Same op order as a plain |z|^2: re*re +
     * im*im, no fused multiply-add. */
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
    sk_scale_f32(r->render_scaled, in->render_hop, 32768.0f, r->hop_size);

    if (r->sg.initial_state && in->all_converged)
        suppression_gain_set_initial_state(&r->sg, 0);
    r->sg.dt_protect_active = in->max_dt > AEC_RES_DT_PROTECT_THRESHOLD;
    return suppression_gain_get_gain(&r->sg, r->near_power, in->r2, in->r2,
                                     in->comfort_noise, r->render_scaled,
                                     in->clock_drift,
                                     in->max_saturation > AEC_RES_SATURATION_THRESHOLD);
}

int aec_res_set_preset(AecRes* r, AecPreset preset, float ramp_ms) {
    float db;
    if (!r) return -1;
    if (aec_preset_floor_db(preset, &db) != 0) return -1;
    /* The suppressor validates and applies first and writes nothing on refusal,
     * so a rejected ramp leaves the stored configuration untouched. */
    if (suppression_gain_set_split_floor_far_active_db(&r->sg, db, ramp_ms) != 0)
        return -1;
    r->sg_cfg.split_floor_far_active = r->sg.cfg.split_floor_far_active;
    return 0;
}

int aec_res_split_floor(const AecRes* r, float* live, float* target) {
    if (!r) return -1;
    if (live) *live = r->sg.split_floor_far_active_live;
    if (target) *target = r->sg_cfg.split_floor_far_active;
    return 0;
}

void aec_cng_fill(uint32_t* rng, Complex* spec, const float* cn,
                  const float* g_res, const float* g_nr, float nr_floor, int n) {
    int k;
    if (!rng || !spec || !cn || !g_res || n < 3) return;
    for (k = 1; k < n - 1; ++k) {
        float n_amp = cn[k] / AEC_RES_PSD_SCALE;
        float gain2 = 1.0f - g_res[k] * g_res[k];
        float nr = g_nr ? g_nr[k] : 1.0f;
        float amplitude;
        uint32_t ix;
        n_amp = n_amp > 0.0f ? sqrtf(n_amp) : 0.0f;
        gain2 = gain2 > 0.0f ? sqrtf(gain2) : 0.0f;
        amplitude = n_amp * gain2 * fmaxf(nr, nr_floor);
        *rng = (*rng * 69069u + 1u) & 0x7FFFFFFFu;
        ix = *rng >> 26;
        spec[k].r += amplitude * AEC3B_SQRT2_SIN_LUT[ix];
        spec[k].i += amplitude * AEC3B_SQRT2_SIN_LUT[(ix + 8u) & 31u];
    }
}
