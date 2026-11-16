/* test_runtime_preset.c -- regression test for the runtime strength retarget:
 * suppression_gain_set_split_floor_far_active_db() and its aec_set_level() wrapper (a fixed 100 ms ramp).
 *
 * mild/balanced differ in exactly one tuning value, min_gain_floor_far_active_db; aggressive also differs in
 * min_gain_floor_dt_db (both derived from AecConfig.level). Both reach the suppressor as a single scalar clamp read once per hop, so a
 * level change needs no rebuild (which would discard the filter, the delay lock and every smoothing history). The floor is
 * a hard clamp and mild<->aggressive is an 18 dB step, so the retarget must also be able to walk instead of jump.
 *
 * Everything here drives real aec_process() hops, no test-only hook. The ramp advances inside suppression_gain_get_gain(),
 * so the trajectory is observable per hop through a->a3_sg even on hops where the clamp is not binding.
 *
 * What only this test proves (the byte-equal gate cannot):
 *   1. A ramp of 0 ms (suppression_gain_set_split_floor_far_active_db) and a completed aec_set_level() ramp both land on
 *      EXACTLY the float a fresh instance constructed with that level holds.
 *   2. The 100 ms ramp advances once per hop, is monotonic, never overshoots, and lands exactly on the hop the ms->hops
 *      conversion predicts (a geometric walk that merely converges leaves a residue and stays nominally live); the
 *      suppressor's own ramp_ms conversion is checked at half-hop inputs.
 *   3. A call during a ramp restarts from the CURRENT live value: it neither snaps back to the old target nor inherits the
 *      old ratio.
 *   4. aec_reset() abandons ramp progress but keeps the target (the ramp lives in the state block, the target in cfg, which
 *      aec_reset carries forward verbatim).
 *   5. Rejection is TOTAL: an out-of-range argument leaves the suppressor bit-identical. The dB bound is the config
 *      validator's own [-300, 50], so a runtime call cannot install a value init would have refused.
 *   6. The setter disturbs nothing else: the far-active latch, DominantNearend counters, initial_state and the three
 *      smoothing histories survive.
 *   7. A lane whose AecConfig.residual is below FULL (STATE, ESTIMATE) never reaches get_gain, so a retarget there is inert by construction
 *      (the ramp does not step); this fails if the 4-channel product's setter is simplified into per-lane calls.
 *   8. The level integers (mild 0, balanced 1, aggressive 2, no_floor 3), the three lookup getters and the balanced fallback of
 *      aec_config_from_level.
 *
 * Mutation coverage: run_landing_is_load_bearing() replays the same walk with the exact-landing branch removed and asserts
 * that the residue appears, so assertion 2 is shown to be capable of failing.
 */
#include "aec.h"
#include "aec_white_box.h"
#include "aec_internal.h"
#include "aec_tuning.h"
#include "aec_scale.h"
#include "suppression_gain.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; } \
        else         { printf("pass: %s\n", (msg)); g_pass++; } \
    } while (0)

#define LEVEL_DB_MILD       (-20.0f)
#define LEVEL_DB_BALANCED   (-28.0f)
#define LEVEL_DB_AGGRESSIVE (-38.0f)

static float floor_of(float db) { return powf(10.0f, db / 10.0f); }

/* Deterministic far/near so an instance can be driven far enough to latch
 * far-active and accumulate smoothing history. Content does not matter, only
 * that it is identical every run. */
static void step(Aec *a, int hop, int h) {
    float mic[1024], ref[1024], out[1024];
    int i;
    for (i = 0; i < hop; ++i) {
        float t = (float)(h * hop + i);
        float r = 0.6f * sinf(0.06f * t) + 0.2f * sinf(0.31f * t);
        ref[i] = r;
        mic[i] = 0.5f * r + 0.03f * sinf(0.017f * t);
    }
    aec_process(a, mic, ref, out);
}

static void run_hops(Aec *a, int hop, int first, int count) {
    int h;
    for (h = first; h < first + count; ++h) step(a, hop, h);
}

/* residual: the AecConfig.residual of the instance (FULL is the default). */
static Aec *open_aec(AecLevel level, AecResidualMode residual) {
    AecConfig cfg;
    aec_config_from_level(&cfg, level, 16000);
    cfg.residual = residual;
    return aec_create(&cfg);
}

/* ── 1: a ramp of 0 ms, and a completed level ramp, land on the constructed float ─────────────────────────── */

static void run_immediate_matches_construction(void) {
    Aec *live = NULL, *fresh = NULL, *ramped = NULL;
    int hop, i;
    CHECK((live = open_aec(AEC_LEVEL_BALANCED, AEC_RESIDUAL_FULL)) != NULL, "immediate: created");
    CHECK((ramped = open_aec(AEC_LEVEL_BALANCED, AEC_RESIDUAL_FULL)) != NULL, "immediate: level-ramp instance created");
    CHECK((fresh = open_aec(AEC_LEVEL_AGGRESSIVE, AEC_RESIDUAL_FULL)) != NULL,
          "immediate: reference instance created");
    hop = aec_hop_size(live);
    CHECK(suppression_gain_set_split_floor_far_active_db(&live->a3_sg, LEVEL_DB_AGGRESSIVE, 0.0f) == 0,
          "immediate: retarget accepted");
    CHECK(live->a3_sg.split_floor_far_active_live ==
              fresh->a3_sg.split_floor_far_active_live,
          "ramp 0 lands on the CONSTRUCTED float exactly");
    CHECK(live->a3_sg.cfg.split_floor_far_active ==
              fresh->a3_sg.cfg.split_floor_far_active,
          "ramp 0 retargets cfg to the constructed float exactly");

    CHECK(aec_set_level(ramped, AEC_LEVEL_AGGRESSIVE) == 0, "immediate: level retarget accepted");
    CHECK(ramped->tun.min_gain_floor_far_active_db == LEVEL_DB_AGGRESSIVE && ramped->cfg.level == AEC_LEVEL_AGGRESSIVE,
          "the level is stored as the instance's level and its dB");
    CHECK(ramped->a3_sg.split_floor_far_active_live != fresh->a3_sg.split_floor_far_active_live,
          "a level change does not land at once: the live floor still trails the target (the 100 ms ramp)");
    for (i = 0; i < 40; ++i) step(ramped, hop, i);
    CHECK(ramped->a3_sg.split_floor_far_active_live ==
              fresh->a3_sg.split_floor_far_active_live,
          "a completed level ramp lands on the CONSTRUCTED float exactly");
    aec_destroy(live);
    aec_destroy(ramped);
    aec_destroy(fresh);
}

/* ── 2: the ramp advances once per hop, monotonically, and lands exactly ── */

static void run_ramp_timing_and_shape(void) {
    Aec *a = NULL;
    int hop, i, landed_at = -1;
    int want_hops;
    float target, prev, cur;

    CHECK((a = open_aec(AEC_LEVEL_MILD, AEC_RESIDUAL_FULL)) != NULL, "ramp: created");
    hop = aec_hop_size(a);
    want_hops = (int)(100.0 * 1e-3 * 16000.0 / (double)hop);
    run_hops(a, hop, 0, 20);

    prev = a->a3_sg.split_floor_far_active_live;
    CHECK(prev == floor_of(LEVEL_DB_MILD),
          "ramp: 20 untouched hops leave the live floor at its construction "
          "value");
    CHECK(aec_set_level(a, AEC_LEVEL_AGGRESSIVE) == 0,
          "ramp: accepted");
    target = a->a3_sg.cfg.split_floor_far_active;
    CHECK(a->a3_sg.split_floor_far_active_live == prev,
          "ramp: the setter itself does not move the live floor");

    for (i = 1; i <= want_hops + 8; ++i) {
        step(a, hop, 20 + i);
        cur = a->a3_sg.split_floor_far_active_live;
        if (cur > prev) { CHECK(0, "ramp: descending walk never rises"); break; }
        if (cur < target) { CHECK(0, "ramp: never overshoots"); break; }
        if (landed_at < 0 && cur == target) landed_at = i;
        prev = cur;
    }
    CHECK(landed_at == want_hops,
          "ramp: lands on exactly the hop the ms->hops conversion predicts "
          "(one advance per aec_process)");
    CHECK(a->a3_sg.split_floor_far_active_live == target,
          "ramp: a landed ramp stays landed");

    /* Ascending leg: same properties with the comparison reversed. */
    CHECK(aec_set_level(a, AEC_LEVEL_MILD) == 0,
          "ramp: ascending leg accepted");
    target = a->a3_sg.cfg.split_floor_far_active;
    prev = a->a3_sg.split_floor_far_active_live;
    for (i = 1; i <= want_hops + 8; ++i) {
        step(a, hop, 200 + i);
        cur = a->a3_sg.split_floor_far_active_live;
        if (cur < prev) { CHECK(0, "ramp: ascending walk never falls"); break; }
        if (cur > target) { CHECK(0, "ramp: ascending never overshoots"); break; }
        prev = cur;
    }
    CHECK(a->a3_sg.split_floor_far_active_live == target,
          "ramp: ascending leg lands EXACTLY on the target");
    aec_destroy(a);
}

/* ── 3: a retarget mid-ramp continues from the live value ────────────────── */

static void run_retarget_mid_ramp(void) {
    Aec *a = NULL;
    int hop, i;
    float start, mid, target, prev, cur;

    CHECK((a = open_aec(AEC_LEVEL_MILD, AEC_RESIDUAL_FULL)) != NULL, "mid-ramp: created");
    hop = aec_hop_size(a);
    start = a->a3_sg.split_floor_far_active_live;
    aec_set_level(a, AEC_LEVEL_AGGRESSIVE);
    run_hops(a, hop, 0, 3);
    mid = a->a3_sg.split_floor_far_active_live;
    CHECK(mid < start && mid > a->a3_sg.cfg.split_floor_far_active,
          "mid-ramp: the walk is genuinely part-way when we retarget");

    /* Back to the original floor: this leg must REVERSE, which a restart that
     * inherited the old descending ratio could not do. */
    CHECK(aec_set_level(a, AEC_LEVEL_MILD) == 0,
          "mid-ramp: retarget accepted");
    CHECK(a->a3_sg.split_floor_far_active_live == mid,
          "mid-ramp: the retarget continues from the CURRENT live value");
    target = a->a3_sg.cfg.split_floor_far_active;
    CHECK(target > mid, "mid-ramp: the second leg genuinely reverses");
    prev = mid;
    for (i = 0; i < 40; ++i) {
        step(a, hop, 100 + i);
        cur = a->a3_sg.split_floor_far_active_live;
        if (cur < prev) { CHECK(0, "mid-ramp: reversed leg ascends"); break; }
        prev = cur;
    }
    CHECK(a->a3_sg.split_floor_far_active_live == target && target == start,
          "mid-ramp: reversed leg lands exactly back on the original floor");

    /* A 0 ms retarget during a ramp lands now. */
    aec_set_level(a, AEC_LEVEL_AGGRESSIVE);
    step(a, hop, 200);
    CHECK(suppression_gain_set_split_floor_far_active_db(&a->a3_sg, LEVEL_DB_BALANCED, 0.0f) == 0,
          "mid-ramp: immediate retarget accepted");
    CHECK(a->a3_sg.split_floor_far_active_live ==
              a->a3_sg.cfg.split_floor_far_active &&
              a->a3_sg.split_floor_far_active_live ==
                  floor_of(LEVEL_DB_BALANCED),
          "mid-ramp: a 0 ms call during a ramp lands now");
    aec_destroy(a);
}

/* ── 4 / 6: reset semantics, and nothing else moves ──────────────────────── */

static void run_reset_and_isolation(void) {
    Aec *a = NULL;
    int hop;
    int latch, dne_hold, dne_trig, init_state;
    float g0, n0, e0;

    CHECK((a = open_aec(AEC_LEVEL_BALANCED, AEC_RESIDUAL_FULL)) != NULL, "isolation: created");
    hop = aec_hop_size(a);
    run_hops(a, hop, 0, 40);

    latch = a->a3_sg.far_active_latched;
    dne_hold = a->a3_sg.dne_hold_counter;
    dne_trig = a->a3_sg.dne_trigger_counter;
    init_state = a->a3_sg.initial_state;
    g0 = a->a3_sg.last_gain[0];
    n0 = a->a3_sg.last_nearend[0];
    e0 = a->a3_sg.last_echo[0];
    CHECK(aec_set_level(a, AEC_LEVEL_MILD) == 0,
          "isolation: ramped retarget accepted mid-stream");
    CHECK(a->a3_sg.far_active_latched == latch &&
              a->a3_sg.dne_hold_counter == dne_hold &&
              a->a3_sg.dne_trigger_counter == dne_trig &&
              a->a3_sg.initial_state == init_state &&
              a->a3_sg.last_gain[0] == g0 &&
              a->a3_sg.last_nearend[0] == n0 &&
              a->a3_sg.last_echo[0] == e0,
          "isolation: far-active latch, DominantNearend counters, "
          "initial_state and the three smoothing histories all survive");

    CHECK(aec_set_level(a, AEC_LEVEL_AGGRESSIVE) == 0,
          "reset: long ramp armed");
    run_hops(a, hop, 40, 3);
    CHECK(a->a3_sg.split_floor_far_active_live !=
              a->a3_sg.cfg.split_floor_far_active,
          "reset: the ramp is genuinely mid-flight beforehand");
    aec_reset(a);
    CHECK(a->a3_sg.split_floor_far_active_live ==
              a->a3_sg.cfg.split_floor_far_active &&
              a->a3_sg.split_floor_far_active_live ==
                  floor_of(LEVEL_DB_AGGRESSIVE),
          "reset: lands on the latest TARGET and drops ramp progress");
    CHECK(a->a3_sg.cfg.split_floor_dt == floor_of(-30.0f),
          "reset: the dt floor (no ramp, already landed) survives reset "
          "at aggressive's target too, not just far-active");
    aec_destroy(a);
}

/* ── 5: rejection is total ───────────────────────────────────────────────── */

static void run_rejection_is_total(void) {
    static const struct { float db; float ramp; const char *what; } BAD[] = {
        { (float)NAN,      0.0f,        "db NaN" },
        { (float)INFINITY, 0.0f,        "db Inf" },
        { -301.0f,         0.0f,        "db below the validator's range" },
        {   51.0f,         0.0f,        "db above the validator's range" },
        {  -28.0f, (float)NAN,          "ramp_ms NaN" },
        {  -28.0f, (float)INFINITY,     "ramp_ms Inf" },
        {  -28.0f,        -1.0f,        "ramp_ms negative" },
        {  -28.0f,     60001.0f,        "ramp_ms beyond the 60 s bound" },
    };
    Aec *a = NULL;
    unsigned i;

    CHECK((a = open_aec(AEC_LEVEL_MILD, AEC_RESIDUAL_FULL)) != NULL, "rejection: created");
    run_hops(a, aec_hop_size(a), 0, 8);
    for (i = 0; i < sizeof(BAD) / sizeof(BAD[0]); ++i) {
        SuppressionGain before = a->a3_sg;
        char msg[128];
        snprintf(msg, sizeof(msg), "rejected and untouched: %s", BAD[i].what);
        CHECK(suppression_gain_set_split_floor_far_active_db(
                  &a->a3_sg, BAD[i].db, BAD[i].ramp) == -1 &&
                  memcmp(&before, &a->a3_sg, sizeof(before)) == 0, msg);
    }
    CHECK(suppression_gain_set_split_floor_far_active_db(NULL, -28.0f, 0.0f)
              == -1, "rejection: NULL suppressor");
    CHECK(aec_set_level(NULL, AEC_LEVEL_MILD) == -1,
          "rejection: NULL instance");

    /* suppression_gain_set_split_floor_dt_db() has no ramp_ms, so only the
     * db range needs this table -- same rejection discipline as the
     * far-active setter above, checked independently since it is a
     * separate function with its own validation. */
    {
        static const struct { float db; const char *what; } DT_BAD[] = {
            { (float)NAN,      "db NaN" },
            { (float)INFINITY, "db Inf" },
            { -301.0f,         "db below the validator's range" },
            {   51.0f,         "db above the validator's range" },
        };
        for (i = 0; i < sizeof(DT_BAD) / sizeof(DT_BAD[0]); ++i) {
            SuppressionGain before = a->a3_sg;
            char msg[128];
            snprintf(msg, sizeof(msg), "dt rejected and untouched: %s",
                     DT_BAD[i].what);
            CHECK(suppression_gain_set_split_floor_dt_db(
                      &a->a3_sg, DT_BAD[i].db) == -1 &&
                      memcmp(&before, &a->a3_sg, sizeof(before)) == 0, msg);
        }
        CHECK(suppression_gain_set_split_floor_dt_db(NULL, -20.0f) == -1,
              "rejection: NULL suppressor (dt setter)");
    }
    CHECK(aec_set_level(a, (AecLevel)77) == -1 &&
              a->tun.min_gain_floor_far_active_db == LEVEL_DB_MILD && a->cfg.level == AEC_LEVEL_MILD,
          "rejection: out-of-enum level REFUSED and the stored level untouched (the config "
          "factory falls back to balanced; a setter must not)");
    aec_destroy(a);
}

/* ── 7: a lane below AEC_RESIDUAL_FULL never reaches get_gain ───────────── */

static void run_residual_below_full_is_inert(void) {
    static const AecResidualMode MODES[2] = { AEC_RESIDUAL_STATE, AEC_RESIDUAL_ESTIMATE };
    static const char *const NAMES[2] = { "STATE", "ESTIMATE" };
    int m;
    for (m = 0; m < 2; ++m) {
        Aec *a = NULL;
        int hop;
        float before_live;
        char msg[200];

        snprintf(msg, sizeof(msg), "%s lane: created", NAMES[m]);
        CHECK((a = open_aec(AEC_LEVEL_BALANCED, MODES[m])) != NULL, msg);
        hop = aec_hop_size(a);
        run_hops(a, hop, 0, 40);

        snprintf(msg, sizeof(msg), "%s lane: the setter still reports success", NAMES[m]);
        CHECK(aec_set_level(a, AEC_LEVEL_AGGRESSIVE) == 0, msg);
        before_live = a->a3_sg.split_floor_far_active_live;
        run_hops(a, hop, 40, 40);
        snprintf(msg, sizeof(msg),
                 "%s lane: 40 hops advance the ramp ZERO times -- the lane never reaches get_gain, so a per-lane "
                 "retarget is inert by construction and the 4-channel product must retarget its shared post-stage "
                 "suppressor instead", NAMES[m]);
        CHECK(a->a3_sg.split_floor_far_active_live == before_live, msg);
        aec_destroy(a);
    }
}

/* ── C/Python rounding parity at a HALF hop ──────────────────────────────
 * The ms->hops conversion has three plausible spellings that agree on every
 * whole-hop input and disagree on a half: truncation, round-half-up, and the
 * round-half-to-even that lrintf (and Python's round()) actually perform. The
 * 100 ms cases above land on a whole hop and cannot tell them apart, so this
 * pins the three halves that can. ms is derived from the instance's real hop
 * so the case holds on any grid. */

static void run_half_hop_rounding_parity(void) {
    /* ramp_ms is given directly rather than derived from a "want" in hops:
     * the whole point is that the float expression does NOT land on the clean
     * half these values look like, and deriving it back would hide that. Each
     * expected count below is what the shipping conversion actually produces
     * at 16 kHz / hop 128, and the Python reference's own parity table
     * (python/tests/test_runtime_preset.py) carries the SAME pairs. */
    static const struct { float ramp_ms; int hops; const char *what; } CASES[] = {
        { 12.0f, 2, "12 ms -> 2 hops (float expr is exactly 1.5; truncation "
                    "would say 1)" },
        { 20.0f, 3, "20 ms -> 3 hops (float expr is 2.5000002, NOT the 2.5 a "
                    "double would give -- a double here would say 2)" },
        { 28.0f, 4, "28 ms -> 4 hops (float expr is exactly 3.5; truncation "
                    "would say 3)" },
        { 24.0f, 3, "24 ms -> 3 hops (whole-hop control)" },
        { 40.0f, 5, "40 ms -> 5 hops (float expr is 5.0000005)" },
        { 56.0f, 7, "56 ms -> 7 hops (whole-hop control)" },
    };
    Aec *a = NULL;
    int hop, i, c, landed;
    float ramp_ms, target;

    CHECK((a = open_aec(AEC_LEVEL_MILD, AEC_RESIDUAL_FULL)) != NULL, "half-hop: created");
    hop = aec_hop_size(a);
    CHECK(hop == 128, "half-hop: the table below is authored at hop 128");

    for (c = 0; c < (int)(sizeof(CASES) / sizeof(CASES[0])); ++c) {
        char msg[160];
        ramp_ms = CASES[c].ramp_ms;
        CHECK(suppression_gain_set_split_floor_far_active_db(&a->a3_sg, LEVEL_DB_AGGRESSIVE, ramp_ms) == 0,
              "half-hop: retarget accepted");
        target = a->a3_sg.cfg.split_floor_far_active;
        landed = -1;
        for (i = 1; i <= CASES[c].hops + 4; ++i) {
            step(a, hop, 1000 * (c + 1) + i);
            if (landed < 0 &&
                a->a3_sg.split_floor_far_active_live == target) landed = i;
        }
        snprintf(msg, sizeof(msg), "half-hop: %s", CASES[c].what);
        CHECK(landed == CASES[c].hops, msg);
        /* Back to the start floor for the next case, immediately. */
        suppression_gain_set_split_floor_far_active_db(&a->a3_sg, LEVEL_DB_MILD, 0.0f);
    }
    aec_destroy(a);
}

/* ── 8: NO_FLOOR runtime switching. The gate STEPS immediately (a boolean has no continuous path to walk), exactly like the
 * DT floor; see suppression_gain_set_split_floor_enabled() in suppression_gain.h. ───────────────────────────────────── */

static void run_no_floor_switch(void) {
    Aec *live = NULL, *fresh_nf = NULL, *fresh_bal = NULL;

    /* INTO NO_FLOOR: the gate steps off on the same call that retargets it,
     * landing field-for-field on a freshly constructed NO_FLOOR instance --
     * the same "ramp_ms==0 matches construction" property test 1 proves for
     * the dB floors, proven here for the boolean gate too. */
    CHECK((live = open_aec(AEC_LEVEL_BALANCED, AEC_RESIDUAL_FULL)) != NULL,
          "no_floor: created (balanced)");
    CHECK((fresh_nf = open_aec(AEC_LEVEL_NO_FLOOR, AEC_RESIDUAL_FULL)) != NULL,
          "no_floor: reference NO_FLOOR instance created");
    CHECK(live->a3_sg.cfg.split_floor_enabled == 1,
          "no_floor: balanced starts with the gate enabled");
    CHECK(aec_set_level(live, AEC_LEVEL_NO_FLOOR) == 0,
          "no_floor: retarget INTO NO_FLOOR accepted");
    CHECK(live->a3_sg.cfg.split_floor_enabled == 0,
          "no_floor: the gate steps off immediately (no ramp -- a boolean "
          "has no continuous path)");
    CHECK(live->a3_sg.cfg.split_floor_enabled ==
              fresh_nf->a3_sg.cfg.split_floor_enabled,
          "no_floor: lands field-for-field on a freshly constructed "
          "NO_FLOOR instance's gate");
    CHECK(live->tun.min_gain_split_floor_enabled == 0 && live->cfg.level == AEC_LEVEL_NO_FLOOR,
          "no_floor: stored as the instance's level");

    /* AWAY FROM NO_FLOOR: the gate steps back on AND the far-active/dt scalars are written through
     * the existing setters; flipping only the boolean (or only the scalar) cannot happen here
     * because aec_set_level() always writes all three together. */
    CHECK((fresh_bal = open_aec(AEC_LEVEL_BALANCED, AEC_RESIDUAL_FULL)) != NULL,
          "no_floor: reference BALANCED instance created");
    CHECK(aec_set_level(live, AEC_LEVEL_BALANCED) == 0,
          "no_floor: retarget AWAY FROM NO_FLOOR accepted");
    CHECK(live->a3_sg.cfg.split_floor_enabled == 1,
          "no_floor: the gate steps back on immediately");
    CHECK(live->a3_sg.cfg.split_floor_far_active ==
              fresh_bal->a3_sg.cfg.split_floor_far_active &&
              live->a3_sg.split_floor_far_active_live ==
              fresh_bal->a3_sg.split_floor_far_active_live,
          "no_floor: the far-active floor also lands on balanced's target "
          "exactly -- not left behind at NO_FLOOR's inert value");
    CHECK(live->a3_sg.cfg.split_floor_dt == fresh_bal->a3_sg.cfg.split_floor_dt,
          "no_floor: the dt floor also lands on balanced's target exactly");
    CHECK(live->tun.min_gain_split_floor_enabled == 1 && live->cfg.level == AEC_LEVEL_BALANCED,
          "no_floor: stored back as the instance's level");

    aec_destroy(live);
    aec_destroy(fresh_nf);
    aec_destroy(fresh_bal);
}

/* ── 9: round trip BALANCED -> NO_FLOOR -> AGGRESSIVE, each step matching a
 * freshly constructed instance of the target level field-for-field. This
 * is the proof that the setter covers the flip-only-one-field hazard,
 * rather than merely lacking a guard. ────────── */

static void run_no_floor_round_trip(void) {
    Aec *a = NULL, *fresh = NULL;
    int hop;

    CHECK((a = open_aec(AEC_LEVEL_BALANCED, AEC_RESIDUAL_FULL)) != NULL, "round-trip: created");
    hop = aec_hop_size(a);
    run_hops(a, hop, 0, 8);   /* a few real hops so the gain path has state */

    CHECK(aec_set_level(a, AEC_LEVEL_NO_FLOOR) == 0,
          "round-trip: BALANCED -> NO_FLOOR accepted");
    CHECK((fresh = open_aec(AEC_LEVEL_NO_FLOOR, AEC_RESIDUAL_FULL)) != NULL,
          "round-trip: fresh NO_FLOOR reference created");
    CHECK(a->a3_sg.cfg.split_floor_enabled ==
              fresh->a3_sg.cfg.split_floor_enabled,
          "round-trip: leg 1 gate matches a fresh NO_FLOOR instance");
    aec_destroy(fresh);
    run_hops(a, hop, 8, 8);

    CHECK(aec_set_level(a, AEC_LEVEL_AGGRESSIVE) == 0,
          "round-trip: NO_FLOOR -> AGGRESSIVE accepted");
    CHECK((fresh = open_aec(AEC_LEVEL_AGGRESSIVE, AEC_RESIDUAL_FULL)) != NULL,
          "round-trip: fresh AGGRESSIVE reference created");
    CHECK(a->a3_sg.cfg.split_floor_enabled ==
              fresh->a3_sg.cfg.split_floor_enabled &&
              a->a3_sg.cfg.split_floor_far_active ==
              fresh->a3_sg.cfg.split_floor_far_active &&
              a->a3_sg.cfg.split_floor_dt == fresh->a3_sg.cfg.split_floor_dt,
          "round-trip: leg 2 gate AND both scalars match a fresh AGGRESSIVE "
          "instance -- the gate re-arming does not leave either scalar "
          "behind at NO_FLOOR's inert value");
    aec_destroy(fresh);
    run_hops(a, hop, 16, 8);
    aec_destroy(a);
}

/* AecRes-side NO_FLOOR + aec_res_reset() coverage lives in test_aec_res.c: AecRes is opaque in aec_res_internal.h, and that
 * test includes src/aec_res.c for internal-state assertions. */

/* ── mutation: the exact-landing branch must be load-bearing ─────────────── */

static void run_landing_is_load_bearing(void) {
    Aec *a = NULL;
    float target, live;
    int i, hop, want_hops;

    CHECK((a = open_aec(AEC_LEVEL_MILD, AEC_RESIDUAL_FULL)) != NULL, "mutation: created");
    hop = aec_hop_size(a);
    want_hops = (int)(100.0 * 1e-3 * 16000.0 / (double)hop);
    aec_set_level(a, AEC_LEVEL_AGGRESSIVE);
    target = a->a3_sg.cfg.split_floor_far_active;
    /* Replay the same walk WITHOUT the snap-to-target branch. */
    live = a->a3_sg.split_floor_far_active_live;
    for (i = 0; i < want_hops; ++i) live *= a->a3_sg.split_floor_ramp_ratio;
    CHECK(live != target,
          "mutation: without the exact-landing branch the walk leaves a "
          "residue -- so the landing assertions above can genuinely fail");
    aec_destroy(a);
}

/* ── level numbering and the lookup getters ───────────────────────────────
 * The integers are the CLI and board-script contract, so they are pinned here. */

static void run_level_numbering_and_lookups(void) {
    static const float far_db[4] = { -20.0f, -28.0f, -38.0f, -28.0f };
    static const float dt_db[4]  = { -20.0f, -20.0f, -30.0f, -20.0f };
    static const int   split[4]  = { 1, 1, 1, 0 };
    AecConfig cfg;
    AecTuning tn;
    int l;

    CHECK(AEC_LEVEL_MILD == 0 && AEC_LEVEL_BALANCED == 1 && AEC_LEVEL_AGGRESSIVE == 2 && AEC_LEVEL_NO_FLOOR == 3,
          "levels: mild 0, balanced 1, aggressive 2, no_floor 3");
    CHECK(aec_level_floors(AEC_LEVEL_BALANCED, NULL) == -1, "aec_level_floors rejects a NULL out");

    for (l = 0; l < 4; ++l) {
        AecLevelFloors fl = { 123.0f, 123.0f, 77 };
        CHECK(aec_level_floors((AecLevel)l, &fl) == 0 && fl.far_active_db == far_db[l] && fl.dt_db == dt_db[l] &&
                  fl.split_floor_enabled == split[l],
              "aec_level_floors table row");
        aec_config_from_level(&cfg, (AecLevel)l, 16000);
        aec_tuning_defaults(&tn, &cfg);
        CHECK(cfg.level == (AecLevel)l && tn.min_gain_floor_far_active_db == far_db[l] && tn.min_gain_floor_dt_db == dt_db[l] &&
                  tn.min_gain_split_floor_enabled == split[l],
              "aec_config_from_level stores the level and the tuning applies the table row");
    }
    for (l = -1; l <= 9; l += 10) {
        AecLevelFloors fl = { 123.0f, 123.0f, 77 };
        CHECK(aec_level_floors((AecLevel)l, &fl) == -1 && fl.far_active_db == 123.0f && fl.dt_db == 123.0f &&
                  fl.split_floor_enabled == 77,
              "aec_level_floors rejects an out-of-enum level and writes nothing");
        aec_config_from_level(&cfg, (AecLevel)l, 16000);
        aec_tuning_defaults(&tn, &cfg);
        CHECK(cfg.level == AEC_LEVEL_BALANCED && tn.min_gain_floor_far_active_db == -28.0f && tn.min_gain_floor_dt_db == -20.0f &&
                  tn.min_gain_split_floor_enabled == 1,
              "aec_config_from_level falls back to balanced for an out-of-enum level");
        cfg.level = (AecLevel)l;
        CHECK(aec_get_mem_size(&cfg) == 0, "an out-of-enum AecConfig.level is refused by the size query");
    }
}

/* aec_level_name: the four names, and the same fallback to balanced that aec_config_from_level applies to an out-of-enum
 * level. The public float scale constants equal the library's internal double ones. */
static void run_level_names(void) {
    CHECK(strcmp(aec_level_name(AEC_LEVEL_MILD), "mild") == 0 && strcmp(aec_level_name(AEC_LEVEL_BALANCED), "balanced") == 0 &&
          strcmp(aec_level_name(AEC_LEVEL_AGGRESSIVE), "aggressive") == 0 &&
          strcmp(aec_level_name(AEC_LEVEL_NO_FLOOR), "no_floor") == 0, "aec_level_name: the four level names");
    CHECK(strcmp(aec_level_name((AecLevel)77), "balanced") == 0 && strcmp(aec_level_name((AecLevel)-1), "balanced") == 0,
          "aec_level_name: an out-of-enum level reads balanced, as aec_config_from_level treats it");
    CHECK((double)AEC_PSD_SCALE_F == AEC_PSD_SCALE && (double)AEC_PSD_SCALE_INV_F == AEC_PSD_SCALE_INV &&
          AEC_PSD_SCALE_F == 1073741824.0f, "AEC_PSD_SCALE_F / AEC_PSD_SCALE_INV_F equal the internal scale 2^30");
}

int main(void) {
    run_level_names();
    run_immediate_matches_construction();
    run_ramp_timing_and_shape();
    run_half_hop_rounding_parity();
    run_retarget_mid_ramp();
    run_reset_and_isolation();
    run_rejection_is_total();
    run_residual_below_full_is_inert();
    run_no_floor_switch();
    run_no_floor_round_trip();
    run_landing_is_load_bearing();
    run_level_numbering_and_lookups();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
