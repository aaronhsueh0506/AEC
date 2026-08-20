/* test_runtime_preset.c — permanent regression test for the runtime strength
 * retarget: suppression_gain_set_split_floor_far_active_db() and its
 * aec_set_preset() wrapper.
 *
 * Why the seam exists: the three shipped presets differ in exactly one field,
 * min_gain_floor_far_active_db (aec_config_from_preset), and that field
 * reaches the suppressor as a single scalar clamp read once per hop. So a
 * preset change never actually needed a rebuild -- but the only way to get one
 * was to construct a new Aec, discarding the filter, the delay lock and every
 * smoothing history. The floor is a hard clamp with nothing downstream to
 * smooth it and mild<->aggressive is an 18 dB step, so the retarget also has
 * to be able to walk instead of jump.
 *
 * Everything here drives real aec_process() hops rather than a test-only hook.
 * The ramp advances inside suppression_gain_get_gain(), so the trajectory is
 * observable per hop through a->a3_sg even on hops where the clamp is not
 * binding -- which is what makes the timing assertions meaningful rather than
 * scene-dependent.
 *
 * What only this test proves (the byte-equal gate cannot):
 *   1. ramp_ms == 0 lands on EXACTLY the float a fresh instance constructed
 *      with that preset holds -- the property that makes the setter a true
 *      preset change and not an approximation of one. The byte-equal gate only
 *      ever exercises the untouched instance, so it cannot see this at all.
 *   2. ramp_ms > 0 advances ONCE PER HOP, is monotonic, never overshoots, and
 *      lands EXACTLY on the hop the ms->hops conversion predicts. A geometric
 *      walk that merely converges leaves a residue and stays nominally live
 *      for the rest of the stream; nothing downstream would report that.
 *   3. A call during a ramp restarts from the CURRENT live value -- it
 *      neither snaps back to the old target nor inherits the old ratio.
 *   4. aec_reset() abandons ramp PROGRESS while keeping the TARGET. That split
 *      is the whole reason the ramp lives in the state block and the target in
 *      cfg: aec_reset carries cfg/tun forward verbatim.
 *   5. Rejection is TOTAL: an out-of-range argument leaves the suppressor
 *      bit-identical, so a caller cannot half-apply a strength change. The dB
 *      bound is the config validator's own [-300, 50], so a runtime call
 *      cannot install a value init would have refused.
 *   6. The setter disturbs NOTHING else -- far-active latch, DominantNearend
 *      counters, initial_state and the three smoothing histories all survive.
 *   7. A lane configured with spatial_linear_context never reaches get_gain,
 *      so a retarget there is inert BY CONSTRUCTION -- the ramp does not even
 *      step. Pinned here so the 4-channel product's own setter cannot be
 *      "simplified" into per-lane calls without this failing.
 *
 * Mutation coverage: run_landing_is_load_bearing() replays the same walk with
 * the exact-landing branch removed and asserts the residue really appears, so
 * assertion 2 is shown to be capable of failing.
 */
#include "aec.h"
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

#define PRESET_DB_MILD       (-20.0f)
#define PRESET_DB_BALANCED   (-28.0f)
#define PRESET_DB_AGGRESSIVE (-38.0f)

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

static int open_aec(Aec *a, AecPreset preset, int spatial_lane) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, preset, 16000);
    if (spatial_lane) {
        cfg.enable_res = 0;
        cfg.return_res_context = 1;
        cfg.spatial_linear_context = 1;
    }
    return aec_create(a, &cfg);
}

/* ── 1: ramp_ms == 0 lands on the constructed float ──────────────────────── */

static void run_immediate_matches_construction(void) {
    Aec live, fresh;
    CHECK(open_aec(&live, AEC_PRESET_BALANCED, 0) == 0, "immediate: created");
    CHECK(open_aec(&fresh, AEC_PRESET_AGGRESSIVE, 0) == 0,
          "immediate: reference instance created");
    CHECK(aec_set_preset(&live, AEC_PRESET_AGGRESSIVE, 0.0f) == 0,
          "immediate: retarget accepted");
    CHECK(live.a3_sg.split_floor_far_active_live ==
              fresh.a3_sg.split_floor_far_active_live,
          "ramp_ms=0 lands on the CONSTRUCTED float exactly");
    CHECK(live.a3_sg.cfg.split_floor_far_active ==
              fresh.a3_sg.cfg.split_floor_far_active,
          "ramp_ms=0 retargets cfg to the constructed float exactly");
    CHECK(live.cfg.min_gain_floor_far_active_db == PRESET_DB_AGGRESSIVE,
          "ramp_ms=0 mirrors the dB into AecConfig");
    aec_destroy(&live);
    aec_destroy(&fresh);
}

/* ── 2: the ramp advances once per hop, monotonically, and lands exactly ── */

static void run_ramp_timing_and_shape(void) {
    Aec a;
    int hop, i, landed_at = -1;
    int want_hops;
    float target, prev, cur;

    CHECK(open_aec(&a, AEC_PRESET_MILD, 0) == 0, "ramp: created");
    hop = aec_hop_size(&a);
    want_hops = (int)(100.0 * 1e-3 * 16000.0 / (double)hop);
    run_hops(&a, hop, 0, 20);

    prev = a.a3_sg.split_floor_far_active_live;
    CHECK(prev == floor_of(PRESET_DB_MILD),
          "ramp: 20 untouched hops leave the live floor at its construction "
          "value");
    CHECK(aec_set_preset(&a, AEC_PRESET_AGGRESSIVE, 100.0f) == 0,
          "ramp: accepted");
    target = a.a3_sg.cfg.split_floor_far_active;
    CHECK(a.a3_sg.split_floor_far_active_live == prev,
          "ramp: the setter itself does not move the live floor");

    for (i = 1; i <= want_hops + 8; ++i) {
        step(&a, hop, 20 + i);
        cur = a.a3_sg.split_floor_far_active_live;
        if (cur > prev) { CHECK(0, "ramp: descending walk never rises"); break; }
        if (cur < target) { CHECK(0, "ramp: never overshoots"); break; }
        if (landed_at < 0 && cur == target) landed_at = i;
        prev = cur;
    }
    CHECK(landed_at == want_hops,
          "ramp: lands on exactly the hop the ms->hops conversion predicts "
          "(one advance per aec_process)");
    CHECK(a.a3_sg.split_floor_far_active_live == target,
          "ramp: a landed ramp stays landed");

    /* Ascending leg: same properties with the comparison reversed. */
    CHECK(aec_set_preset(&a, AEC_PRESET_MILD, 100.0f) == 0,
          "ramp: ascending leg accepted");
    target = a.a3_sg.cfg.split_floor_far_active;
    prev = a.a3_sg.split_floor_far_active_live;
    for (i = 1; i <= want_hops + 8; ++i) {
        step(&a, hop, 200 + i);
        cur = a.a3_sg.split_floor_far_active_live;
        if (cur < prev) { CHECK(0, "ramp: ascending walk never falls"); break; }
        if (cur > target) { CHECK(0, "ramp: ascending never overshoots"); break; }
        prev = cur;
    }
    CHECK(a.a3_sg.split_floor_far_active_live == target,
          "ramp: ascending leg lands EXACTLY on the target");
    aec_destroy(&a);
}

/* ── 3: a retarget mid-ramp continues from the live value ────────────────── */

static void run_retarget_mid_ramp(void) {
    Aec a;
    int hop, i;
    float start, mid, target, prev, cur;

    CHECK(open_aec(&a, AEC_PRESET_MILD, 0) == 0, "mid-ramp: created");
    hop = aec_hop_size(&a);
    start = a.a3_sg.split_floor_far_active_live;
    aec_set_preset(&a, AEC_PRESET_AGGRESSIVE, 100.0f);
    run_hops(&a, hop, 0, 3);
    mid = a.a3_sg.split_floor_far_active_live;
    CHECK(mid < start && mid > a.a3_sg.cfg.split_floor_far_active,
          "mid-ramp: the walk is genuinely part-way when we retarget");

    /* Back to the original floor: this leg must REVERSE, which a restart that
     * inherited the old descending ratio could not do. */
    CHECK(aec_set_preset(&a, AEC_PRESET_MILD, 100.0f) == 0,
          "mid-ramp: retarget accepted");
    CHECK(a.a3_sg.split_floor_far_active_live == mid,
          "mid-ramp: the retarget continues from the CURRENT live value");
    target = a.a3_sg.cfg.split_floor_far_active;
    CHECK(target > mid, "mid-ramp: the second leg genuinely reverses");
    prev = mid;
    for (i = 0; i < 40; ++i) {
        step(&a, hop, 100 + i);
        cur = a.a3_sg.split_floor_far_active_live;
        if (cur < prev) { CHECK(0, "mid-ramp: reversed leg ascends"); break; }
        prev = cur;
    }
    CHECK(a.a3_sg.split_floor_far_active_live == target && target == start,
          "mid-ramp: reversed leg lands exactly back on the original floor");

    /* An immediate call during a ramp lands now. */
    aec_set_preset(&a, AEC_PRESET_AGGRESSIVE, 100.0f);
    step(&a, hop, 200);
    CHECK(aec_set_preset(&a, AEC_PRESET_BALANCED, 0.0f) == 0,
          "mid-ramp: immediate retarget accepted");
    CHECK(a.a3_sg.split_floor_far_active_live ==
              a.a3_sg.cfg.split_floor_far_active &&
              a.a3_sg.split_floor_far_active_live ==
                  floor_of(PRESET_DB_BALANCED),
          "mid-ramp: an immediate call during a ramp lands now");
    aec_destroy(&a);
}

/* ── 4 / 6: reset semantics, and nothing else moves ──────────────────────── */

static void run_reset_and_isolation(void) {
    Aec a;
    int hop;
    int latch, dne_hold, dne_trig, init_state;
    float g0, n0, e0;

    CHECK(open_aec(&a, AEC_PRESET_BALANCED, 0) == 0, "isolation: created");
    hop = aec_hop_size(&a);
    run_hops(&a, hop, 0, 40);

    latch = a.a3_sg.far_active_latched;
    dne_hold = a.a3_sg.dne_hold_counter;
    dne_trig = a.a3_sg.dne_trigger_counter;
    init_state = a.a3_sg.initial_state;
    g0 = a.a3_sg.last_gain[0];
    n0 = a.a3_sg.last_nearend[0];
    e0 = a.a3_sg.last_echo[0];
    CHECK(aec_set_preset(&a, AEC_PRESET_MILD, 100.0f) == 0,
          "isolation: ramped retarget accepted mid-stream");
    CHECK(a.a3_sg.far_active_latched == latch &&
              a.a3_sg.dne_hold_counter == dne_hold &&
              a.a3_sg.dne_trigger_counter == dne_trig &&
              a.a3_sg.initial_state == init_state &&
              a.a3_sg.last_gain[0] == g0 &&
              a.a3_sg.last_nearend[0] == n0 &&
              a.a3_sg.last_echo[0] == e0,
          "isolation: far-active latch, DominantNearend counters, "
          "initial_state and the three smoothing histories all survive");

    CHECK(aec_set_preset(&a, AEC_PRESET_AGGRESSIVE, 1000.0f) == 0,
          "reset: long ramp armed");
    run_hops(&a, hop, 40, 3);
    CHECK(a.a3_sg.split_floor_far_active_live !=
              a.a3_sg.cfg.split_floor_far_active,
          "reset: the ramp is genuinely mid-flight beforehand");
    aec_reset(&a);
    CHECK(a.a3_sg.split_floor_far_active_live ==
              a.a3_sg.cfg.split_floor_far_active &&
              a.a3_sg.split_floor_far_active_live ==
                  floor_of(PRESET_DB_AGGRESSIVE),
          "reset: lands on the latest TARGET and drops ramp progress");
    aec_destroy(&a);
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
    Aec a;
    unsigned i;

    CHECK(open_aec(&a, AEC_PRESET_MILD, 0) == 0, "rejection: created");
    run_hops(&a, aec_hop_size(&a), 0, 8);
    for (i = 0; i < sizeof(BAD) / sizeof(BAD[0]); ++i) {
        SuppressionGain before = a.a3_sg;
        char msg[128];
        snprintf(msg, sizeof(msg), "rejected and untouched: %s", BAD[i].what);
        CHECK(suppression_gain_set_split_floor_far_active_db(
                  &a.a3_sg, BAD[i].db, BAD[i].ramp) == -1 &&
                  memcmp(&before, &a.a3_sg, sizeof(before)) == 0, msg);
    }
    CHECK(suppression_gain_set_split_floor_far_active_db(NULL, -28.0f, 0.0f)
              == -1, "rejection: NULL suppressor");
    CHECK(aec_set_preset(NULL, AEC_PRESET_MILD, 0.0f) == -1,
          "rejection: NULL instance");
    CHECK(aec_set_preset(&a, (AecPreset)77, 0.0f) == -1 &&
              a.cfg.min_gain_floor_far_active_db == PRESET_DB_MILD,
          "rejection: out-of-enum preset REFUSED and cfg untouched (the config "
          "factory falls back to balanced; a setter must not)");
    CHECK(aec_set_preset(&a, AEC_PRESET_AGGRESSIVE, 60001.0f) == -1 &&
              a.cfg.min_gain_floor_far_active_db == PRESET_DB_MILD,
          "rejection: a refused ramp_ms leaves AecConfig untouched");
    aec_destroy(&a);
}

/* ── 7: a spatial_linear_context lane never reaches get_gain ─────────────── */

static void run_spatial_lane_is_inert(void) {
    Aec a;
    int hop;
    float before_live;

    CHECK(open_aec(&a, AEC_PRESET_BALANCED, 1) == 0, "spatial lane: created");
    hop = aec_hop_size(&a);
    run_hops(&a, hop, 0, 40);

    CHECK(aec_set_preset(&a, AEC_PRESET_AGGRESSIVE, 100.0f) == 0,
          "spatial lane: the setter still reports success");
    before_live = a.a3_sg.split_floor_far_active_live;
    run_hops(&a, hop, 40, 40);
    CHECK(a.a3_sg.split_floor_far_active_live == before_live,
          "spatial lane: 40 hops advance the ramp ZERO times -- the lane never "
          "reaches get_gain, so a per-lane retarget is inert by construction "
          "and the 4-channel product must retarget its shared post-stage "
          "suppressor instead");
    aec_destroy(&a);
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
    Aec a;
    int hop, i, c, landed;
    float ramp_ms, target;

    CHECK(open_aec(&a, AEC_PRESET_MILD, 0) == 0, "half-hop: created");
    hop = aec_hop_size(&a);
    CHECK(hop == 128, "half-hop: the table below is authored at hop 128");

    for (c = 0; c < (int)(sizeof(CASES) / sizeof(CASES[0])); ++c) {
        char msg[160];
        ramp_ms = CASES[c].ramp_ms;
        CHECK(aec_set_preset(&a, AEC_PRESET_AGGRESSIVE, ramp_ms) == 0,
              "half-hop: retarget accepted");
        target = a.a3_sg.cfg.split_floor_far_active;
        landed = -1;
        for (i = 1; i <= CASES[c].hops + 4; ++i) {
            step(&a, hop, 1000 * (c + 1) + i);
            if (landed < 0 &&
                a.a3_sg.split_floor_far_active_live == target) landed = i;
        }
        snprintf(msg, sizeof(msg), "half-hop: %s", CASES[c].what);
        CHECK(landed == CASES[c].hops, msg);
        /* Back to the start floor for the next case, immediately. */
        aec_set_preset(&a, AEC_PRESET_MILD, 0.0f);
    }
    aec_destroy(&a);
}

/* ── mutation: the exact-landing branch must be load-bearing ─────────────── */

static void run_landing_is_load_bearing(void) {
    Aec a;
    float target, live;
    int i, hop, want_hops;

    CHECK(open_aec(&a, AEC_PRESET_MILD, 0) == 0, "mutation: created");
    hop = aec_hop_size(&a);
    want_hops = (int)(100.0 * 1e-3 * 16000.0 / (double)hop);
    aec_set_preset(&a, AEC_PRESET_AGGRESSIVE, 100.0f);
    target = a.a3_sg.cfg.split_floor_far_active;
    /* Replay the same walk WITHOUT the snap-to-target branch. */
    live = a.a3_sg.split_floor_far_active_live;
    for (i = 0; i < want_hops; ++i) live *= a.a3_sg.split_floor_ramp_ratio;
    CHECK(live != target,
          "mutation: without the exact-landing branch the walk leaves a "
          "residue -- so the landing assertions above can genuinely fail");
    aec_destroy(&a);
}

int main(void) {
    run_immediate_matches_construction();
    run_ramp_timing_and_shape();
    run_half_hop_rounding_parity();
    run_retarget_mid_ramp();
    run_reset_and_isolation();
    run_rejection_is_total();
    run_spatial_lane_is_inert();
    run_landing_is_load_bearing();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
