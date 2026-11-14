/* Exact paired far-spectrum/magnitude reuse, including silent and reset hops. */
#include "aec.h"
#include "aec_white_box.h"
#include "aec_test_tuning.h"
#include "aec_internal.h"
#include "aec_pool_test_util.h"
#include "utility/simd_aec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static unsigned long checks;

static void same(const void *a, const void *b, size_t bytes, const char *field) {
    ++checks;
    if (memcmp(a, b, bytes)) {
        if (failures < 8) fprintf(stderr, "FAIL %s (%zu bytes)\n", field, bytes);
        ++failures;
    }
}

static void same_filter(const PBFDAF *a, const PBFDAF *b) {
    size_t k = (size_t)a->n_freqs, nk = k * a->n_partitions;
#define ARRAY(field, count) same(a->field, b->field, (count) * sizeof(*a->field), #field)
#define VALUE(field) same(&a->field, &b->field, sizeof(a->field), #field)
#define ARRAY_IF(field, count) do { if (a->field) ARRAY(field, count); } while (0)   /* NULL in the shadow filter */
    ARRAY(W, nk); ARRAY(X_buf, nk); ARRAY(x2_cache, nk);
    ARRAY(far_cmag2_hold, k); ARRAY(far_spec, k); ARRAY_IF(near_spec, k);
    ARRAY(error_spec, k); ARRAY_IF(error_spec_windowed, k); ARRAY(echo_spec, k);
    ARRAY(near_buffer, (size_t)a->block_size); ARRAY(far_buffer, (size_t)a->block_size);
    VALUE(partition_idx); VALUE(call_counter); VALUE(poor_excitation_counter);
    VALUE(initial_state_active); VALUE(initial_state_active_render_hops);
    VALUE(partition_to_constrain); VALUE(last_s_max_abs); VALUE(w_dirty);
#undef ARRAY
#undef ARRAY_IF
#undef VALUE
}

static void same_aec(const Aec *a, const Aec *b) {
    const PBFDKF *x = &a->main_filter, *y = &b->main_filter;
    size_t k = (size_t)a->n_freqs;
    same_filter(&x->base, &y->base);
    if (a->has_shadow) same_filter(&a->shadow_filter, &b->shadow_filter);
#define STATE(field) same(x->field, y->field, k * sizeof(*x->field), #field)
    STATE(H_error_per_bin); STATE(error_psd);
    STATE(e2_ref_scratch); STATE(e2_coarse_per_bin); STATE(erl_per_bin);
#undef STATE
    /* The no-shadow path does not initialize this entire scratch array. */
    if (a->has_shadow)
        same(a->scr_e2ref_arr, b->scr_e2ref_arr, k * sizeof(float), "scr_e2ref_arr");
    AecResidualView ca, cb;
    AecLinearView la, lb;
    memset(&ca, 0, sizeof ca); ca.struct_size = sizeof ca; cb = ca;
    memset(&la, 0, sizeof la); la.struct_size = sizeof la; lb = la;
    if (aec_get_residual(a, &ca) != AEC_OK || aec_get_residual(b, &cb) != AEC_OK ||
        aec_get_linear(a, &la) != AEC_OK || aec_get_linear(b, &lb) != AEC_OK) exit(2);
    same(ca.formed_hop, cb.formed_hop, (size_t)a->hop_size * sizeof(float), "formed_hop");
    same(ca.error_spec, cb.error_spec, k * sizeof(Complex), "context_error");
    if (ca.r2 && cb.r2) same(ca.r2, cb.r2, k * sizeof(float), "context_r2");
    if (ca.comfort_noise && cb.comfort_noise)
        same(ca.comfort_noise, cb.comfort_noise, k * sizeof(float), "context_comfort");
#define VIEW(v, field) same(&v##a.field, &v##b.field, sizeof(v##a.field), #field)
    VIEW(l, far_power); VIEW(l, dt_indicator); VIEW(l, filter_converged);
    VIEW(c, usable_linear); VIEW(l, saturation_level);
#undef VIEW
}

/* One hop of an ESTIMATE-mode instance: the linear stage (far spectrum local or borrowed from `owner`) and the residual
 * stage. */
static void hop_shared(Aec *a, const float *mic, const float *far, const Aec *owner) {
    if (aec_process_linear_shared_far_from(a, mic, far, owner) != AEC_OK ||
        aec_process_residual(a, NULL) != AEC_OK) exit(2);
}

static void hop_local(Aec *a, const float *mic, const float *far) {
    if (aec_test_split_hop(a, mic, far) != AEC_OK) exit(2);
}

static Aec *new_aec(const AecTestCfg *cfg) {
    Aec *a = aec_tcfg_create(cfg);
    if (!a) exit(2);
    ++checks;
    if (((uintptr_t)a->heap_arena & 15u) != 0u) {
        fprintf(stderr, "FAIL heap arena alignment\n");
        ++failures;
    }
    return a;
}

static void free_aec(Aec *a) {
    aec_destroy(a);
}

static void run_grid(int sr, int fft, int shadow) {
    AecTestCfg cfg;
    aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, sr);
    cfg.c.fft_size = fft; cfg.t.enable_shadow = shadow;
    cfg.c.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    cfg.c.fixed_delay_samples = -1; cfg.c.residual = AEC_RESIDUAL_ESTIMATE;
    Aec *owner = new_aec(&cfg), *ref = new_aec(&cfg), *borrower = new_aec(&cfg);
    Aec *fallback = new_aec(&cfg);
    if (!owner || !ref || !borrower || !fallback) exit(2);
    int hop = aec_hop_size(owner);
    float *far = calloc((size_t)hop, sizeof(float));
    float *mic = calloc((size_t)hop, sizeof(float));
    float *owner_mic = calloc((size_t)hop, sizeof(float));
    float *power = malloc((size_t)owner->n_freqs * sizeof(float));
    if (!far || !mic || !owner_mic || !power) exit(2);
    for (int frame = 0; frame < 180; ++frame) {
        for (int i = 0; i < hop; ++i) {
            double t = (double)(frame * hop + i);
            float f = (frame >= 105 && frame < 135) ? 0.0f :
                (float)(0.22 * sin(t * 0.117) + 0.13 * sin(t * 0.071));
            far[i] = f; owner_mic[i] = 0.31f * f;
            mic[i] = 0.65f * f + (float)(0.04 * sin(t * 0.031));
        }
        if (frame == 60) {
            (void)aec_apply_external_realign(ref, -hop / 2);
            (void)aec_apply_external_realign(borrower, -hop / 2);
            (void)aec_apply_external_realign(fallback, -hop / 2);
        }
        if (frame == 90) {
            aec_reset(owner); aec_reset(ref); aec_reset(borrower); aec_reset(fallback);
        }
        hop_local(owner, owner_mic, far);
        hop_local(ref, mic, far);
        /* Mixed histories need one local hop before the half-overlap refills. */
        if (frame == 60) hop_local(borrower, mic, far);
        else hop_shared(borrower, mic, far, owner);
        /* NULL and self owners must both select the local path. */
        hop_shared(fallback, mic, far, frame & 1 ? fallback : NULL);
        same_aec(ref, borrower); same_aec(ref, fallback);
        sk_cmag2_np_f32(borrower->main_filter.base.far_spec, power, borrower->n_freqs);
        same(power, borrower->main_filter.base.far_cmag2_hold,
             (size_t)borrower->n_freqs * sizeof(float), "fresh_far_power");
        if (shadow) {
            sk_cmag2_np_f32(borrower->main_filter.base.error_spec, power, borrower->n_freqs);
            same(power, borrower->scr_e2ref_arr,
                 (size_t)borrower->n_freqs * sizeof(float), "fresh_main_error_power");
        }
        if (frame % 17 == 0) {
            const PBFDAF *b = &borrower->main_filter.base;
            for (int part = 0; part < b->n_partitions; ++part) {
                sk_cmag2_np_f32(b->X_buf + (size_t)part * b->n_freqs, power, b->n_freqs);
                same(power, b->x2_cache + (size_t)part * b->n_freqs,
                     (size_t)b->n_freqs * sizeof(float), "fresh_x2_history");
            }
        }
    }
    /* A wrong grid owner must not supply its differently sized spectrum. */
    AecTestCfg wrong_cfg = cfg; wrong_cfg.c.fft_size = fft == 256 ? 512 : 256;
    if (sr == 48000) wrong_cfg.c.sample_rate = 16000;
    Aec *wrong = new_aec(&wrong_cfg);
    if (!wrong) exit(2);
    hop_local(ref, mic, far);
    hop_shared(fallback, mic, far, wrong);
    same_aec(ref, fallback);
    free_aec(wrong);
    free_aec(owner); free_aec(ref); free_aec(borrower); free_aec(fallback);
    free(far); free(mic); free(owner_mic); free(power);
}

static void test_error_cache_validity(void) {
    PBFDKF p;
    FftHandle *fft = fft_create(256);
    size_t pool_bytes = pbfdkf_get_mem_size(256, 4, 128);
    void *pool = aec_test_pool_alloc(pool_bytes);
    if (!fft || !pool ||
        pbfdkf_init_static(&p, pool, pool_bytes, 256, 4, 0.5f, 1e-5f, 128,
                           16000, fft) != 0) exit(2);
    float near[128], far[128], out[128];
    for (int i = 0; i < 128; ++i) { near[i] = 0.05f; far[i] = 0.25f; }
    int valid = pbfdkf_process_with_far_cmag2(&p, near, far, NULL, 0.0f, out, NULL);
    ++checks; if (!valid) ++failures;
    float *saved = malloc((size_t)p.base.n_freqs * sizeof(float));
    float *fresh = malloc((size_t)p.base.n_freqs * sizeof(float));
    if (!saved || !fresh) exit(2);
    sk_cmag2_np_f32(p.base.error_spec, fresh, p.base.n_freqs);
    same(fresh, p.e2_ref_scratch, (size_t)p.base.n_freqs * sizeof(float), "active_error_cache");
    p.base.call_counter = p.base.n_partitions + 1;
    p.base.poor_excitation_counter = p.base.n_partitions + 1;
    p.base.block_stationary = 1;
    valid = pbfdkf_process_with_far_cmag2(&p, near, far, NULL, 0.0f, out, NULL);
    ++checks; if (!valid) ++failures;
    sk_cmag2_np_f32(p.base.error_spec, fresh, p.base.n_freqs);
    same(fresh, p.e2_ref_scratch, (size_t)p.base.n_freqs * sizeof(float), "stationary_error_cache");
    p.base.block_stationary = 0;
    valid = pbfdkf_process_with_far_cmag2(&p, near, far, NULL, 0.5f, out, NULL);
    ++checks; if (!valid) ++failures;
    sk_cmag2_np_f32(p.base.error_spec, fresh, p.base.n_freqs);
    same(fresh, p.e2_ref_scratch, (size_t)p.base.n_freqs * sizeof(float), "updated_error_cache");
    p.base.saturated_capture = 1;
    valid = pbfdkf_process_with_far_cmag2(&p, near, far, NULL, 0.5f, out, NULL);
    ++checks; if (!valid) ++failures;
    sk_cmag2_np_f32(p.base.error_spec, fresh, p.base.n_freqs);
    same(fresh, p.e2_ref_scratch, (size_t)p.base.n_freqs * sizeof(float), "saturated_error_cache");
    p.base.saturated_capture = 0;
    memcpy(saved, p.e2_ref_scratch, (size_t)p.base.n_freqs * sizeof(float));
    memset(far, 0, sizeof(far));
    valid = pbfdkf_process_with_far_cmag2(&p, near, far, NULL, 0.0f, out, NULL);
    ++checks; if (valid) ++failures;
    same(saved, p.e2_ref_scratch, (size_t)p.base.n_freqs * sizeof(float), "silent_cache_unchanged");
    free(saved); free(fresh); pbfdkf_free(&p);
    fft_destroy(fft); free(pool);
}

int main(void) {
    for (int shadow = 0; shadow <= 1; ++shadow) {
        run_grid(16000, 256, shadow); run_grid(16000, 512, shadow);
        run_grid(48000, 1024, shadow);
    }
    test_error_cache_validity();
    printf("spectral reuse: %lu checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
