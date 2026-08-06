/* Historical parity_reverb_decay_estimator.c — replay the binary golden from
 * python/diag/gen_reverb_decay_estimator_golden.py through the C
 * ReverbDecayEstimator and assert exact (bit-for-bit) match of the recorded
 * per-update state.  WS5 Phase 5.1 gate.
 *
 * Build (standalone, from c_impl/):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Itest/support \
 *       test/support/reverb_decay_estimator.c test/historical/parity_reverb_decay_estimator.c \
 *       -lm -o /tmp/p_rde
 *   python3 ../python/diag/gen_reverb_decay_estimator_golden.py /tmp/rde_golden.bin
 *   /tmp/p_rde /tmp/rde_golden.bin
 */
#include "reverb_decay_estimator.h"

#include <stdio.h>
#include <stdlib.h>

#define MAGIC 0x52444543

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

/* Read recorded post-update state and compare against the live estimator.
 * Returns number of mismatched fields. */
static int check_state(FILE *f, const ReverbDecayEstimator *r, int idx,
                       const char *tag) {
    double ef[5];
    int    ei[7];
    int    m = 0;
    if (!rd(f, ef, sizeof ef) || !rd(f, ei, sizeof ei)) {
        fprintf(stderr, "short read state %s %d\n", tag, idx);
        exit(2);
    }
#define CKF(name, got, exp) do { \
    if ((got) != (exp)) { \
        m++; \
        printf("  MISMATCH %s[%d] %-18s got=%.17g exp=%.17g\n", \
               tag, idx, name, (double)(got), (double)(exp)); \
    } } while (0)
#define CKI(name, got, exp) do { \
    if ((got) != (exp)) { \
        m++; \
        printf("  MISMATCH %s[%d] %-18s got=%d exp=%d\n", \
               tag, idx, name, (int)(got), (int)(exp)); \
    } } while (0)
    CKF("decay",      r->decay,                ef[0]);
    CKF("smoothing",  r->smoothing_constant,   ef[1]);
    CKF("late.nz",    r->late.nz,              ef[2]);
    CKF("late.nn",    r->late.nn,              ef[3]);
    CKF("tail_gain",  r->tail_gain,            ef[4]);
    CKI("block2anal", r->block_to_analyze,            ei[0]);
    CKI("late_start", r->late_reverb_start,           ei[1]);
    CKI("late_end",   r->late_reverb_end,             ei[2]);
    CKI("cand_size",  r->estimation_region_candidate_size, ei[3]);
    CKI("region_id",  r->estimation_region_identified, ei[4]);
    CKI("late.n",     r->late.n,                       ei[5]);
    CKI("late.N",     r->late.N,                       ei[6]);
#undef CKF
#undef CKI
    return m;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/rde_golden.bin";
    FILE *f = fopen(path, "rb");
    int magic, mism = 0, updates = 0, i;
    ReverbDecayEstimator r;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &magic, 4) || magic != MAGIC) {
        fprintf(stderr, "bad magic\n"); return 2;
    }

    /* ───────────────────────────── legacy ─────────────────────────────── */
    {
        int n_part, hop, n_legacy;
        double dd, md;
        float *pe;
        if (!rd(f, &n_part, 4) || !rd(f, &hop, 4) || !rd(f, &n_legacy, 4) ||
            !rd(f, &dd, 8) || !rd(f, &md, 8)) { fprintf(stderr, "hdr\n"); return 2; }
        pe = malloc((size_t)n_part * sizeof(float));
        rde_init(&r, n_part, hop, dd, md, /*use_adaptive=*/1,
                 /*use_aec3_block_energy=*/0);
        for (i = 0; i < n_legacy; ++i) {
            double fq; int fq_none, delay, usable, stationary;
            if (!rd(f, pe, (size_t)n_part * sizeof(float)) ||
                !rd(f, &fq, 8) || !rd(f, &fq_none, 4) ||
                !rd(f, &delay, 4) || !rd(f, &usable, 4) || !rd(f, &stationary, 4)) {
                fprintf(stderr, "legacy in %d\n", i); return 2;
            }
            rde_update_legacy(&r, pe, fq, fq_none, delay, usable, stationary);
            mism += check_state(f, &r, i, "legacy");
            updates++;
        }
        free(pe);
        rde_free(&r);
    }

    /* ─────────────────────────── aec3-strict ──────────────────────────── */
    {
        int td_size, n_aec3, n_part2, hop2;
        double dd2, md2;
        float *td;
        if (!rd(f, &td_size, 4) || !rd(f, &n_aec3, 4) ||
            !rd(f, &dd2, 8) || !rd(f, &md2, 8) ||
            !rd(f, &n_part2, 4) || !rd(f, &hop2, 4)) { fprintf(stderr, "hdr2\n"); return 2; }
        td = malloc((size_t)td_size * sizeof(float));
        if (!rd(f, td, (size_t)td_size * sizeof(float))) { fprintf(stderr, "td\n"); return 2; }
        rde_init(&r, n_part2, hop2, dd2, md2, /*use_adaptive=*/1,
                 /*use_aec3_block_energy=*/1);
        for (i = 0; i < n_aec3; ++i) {
            double fq; int fq_none, delay, usable, stationary;
            if (!rd(f, &fq, 8) || !rd(f, &fq_none, 4) ||
                !rd(f, &delay, 4) || !rd(f, &usable, 4) || !rd(f, &stationary, 4)) {
                fprintf(stderr, "aec3 in %d\n", i); return 2;
            }
            rde_update_aec3_strict(&r, td, td_size, fq, fq_none, delay,
                                   usable, stationary);
            mism += check_state(f, &r, i, "aec3");
            updates++;
        }
        free(td);
        rde_free(&r);
    }

    fclose(f);
    printf("reverb_decay_estimator parity: %d updates, mismatches=%d\n",
           updates, mism);
    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
