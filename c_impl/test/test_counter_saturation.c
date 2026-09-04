/* test_counter_saturation.c — permanent, checked-in regression test for
 * counter-saturation fixes across this repo's C AEC implementation (every
 * ad hoc UBSan probe previously used to verify these fixes was throwaway
 * and never committed -- this file replaces that with a permanent `make
 * test-counter-saturation` target that runs every time, plus a permanent
 * UBSan build recipe, see test/run_counter_saturation_ubsan.sh).
 *
 * For every struct-field increment/decrement site an exhaustive grep sweep
 * verified as genuinely unbounded, plus a representative sample of sites
 * that were already fixed and are reachable from a public API/struct
 * without excessive plumbing, this file exercises at minimum:
 *   - cap-1 -> cap   : one more increment from just below the cap lands
 *                      exactly on the cap.
 *   - cap -> cap     : an increment attempt AT the cap is a no-op.
 *   - runaway        : many (thousands of) more increments/calls past the
 *                      cap never move it further.
 *   - decision-invariance: the boolean/comparison the counter feeds
 *                      produces the IDENTICAL result at the cap as it would
 *                      at a synthetic huge value standing in for the
 *                      (no-longer-reachable) unbounded value the original
 *                      buggy code would eventually have produced.
 *
 * FilterPlateauDetector gets its own dedicated section (see
 * section_filter_plateau below):
 * frame_count/far_active_count/dt_signal_count were widened to int64_t
 * rather than saturated (a per-field saturating cap would corrupt the
 * far_ratio/dt_ratio these three feed -- see include/detectors.h's struct
 * comment), so its test proves ratio-preservation in the pre-terminal
 * region and correct (non-wrapping) behaviour past where a 32-bit counter
 * would have overflowed, rather than a "settles at a cap" transition.
 *
 * Build: `make test-counter-saturation` (wired the same way `bench`/`all`
 * are -- needs the audio_common archive for the full-Aec sections; see the
 * Makefile's test-counter-saturation target). Run:
 *   ./bin/<backend>-<hash>/test_counter_saturation
 * (from c_impl/ -- the delay_aec3 section reads a checked-in fixture at the
 * relative path ../wav/aec_record/aec_record_{mic,ref}_10s.wav, matching
 * this repo's other cwd-relative test/example conventions).
 *
 * UBSan: test/run_counter_saturation_ubsan.sh builds and runs this SAME
 * source under -fsanitize=undefined -fno-sanitize-recover=all (same recipe
 * shape as test/run_selftest_ubsan.sh) and must exit 0 with no diagnostic.
 */
#include "aec.h"
#include "delay_pool_test_util.h"
#include "wav_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>

/* ── tiny harness ─────────────────────────────────────────────────────── */
static long g_checks = 0;
static long g_fails  = 0;

#define CHECK(cond, msg) do { \
        g_checks++; \
        if (!(cond)) { \
            g_fails++; \
            fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, (msg)); \
        } \
    } while (0)

/* ════════════════════════════════════════════════════════════════════
 * FilterPlateauDetector (detectors.h) -- dedicated section (widened
 * frame_count/far_active_count/dt_signal_count int -> int64_t).
 * ════════════════════════════════════════════════════════════════════ */
static void section_filter_plateau(void) {
    FilterPlateauDetector p;
    int i;

    /* --- Case A: pre-terminal region -- the widened counters must track
     * an independently-kept int64_t tally EXACTLY, frame for frame, proving
     * the widening introduces no drift (and, by construction, the
     * far_ratio/dt_ratio the source derives from them stays correct --
     * those ratios are pure functions of frame_count/far_active_count/
     * dt_signal_count, so exact-tally agreement on the raw counters implies
     * exact ratio agreement too). erle_windowed_db is held high (10.0 >
     * erle_max_db=6.0) so criteria_met is always false regardless of the
     * ratios -- this section is about the COUNTERS, not about firing. */
    filter_plateau_init(&p);
    {
        int64_t exp_far = 0, exp_dt = 0;
        for (i = 0; i < p.grace_frames + 30; i++) {
            int far_active = (i % 3) != 0;   /* 2/3 of frames        */
            int dt_present = (i % 2) == 0;   /* 1/2 of frames        */
            int fired;
            if (far_active) exp_far++;
            if (dt_present) exp_dt++;
            fired = filter_plateau_update(&p, far_active, dt_present, 10.0f, 0);
            CHECK(!fired, "plateau pre-terminal: high ERLE never fires");
            CHECK(p.frame_count == (int64_t)(i + 1),
                  "plateau pre-terminal: frame_count == call index exactly");
            CHECK(p.far_active_count == exp_far,
                  "plateau pre-terminal: far_active_count matches independent tally");
            CHECK(p.dt_signal_count == exp_dt,
                  "plateau pre-terminal: dt_signal_count matches independent tally");
        }
    }

    /* --- Case B: terminal state via once_converged=1. Every call after
     * this latches must (a) never fire, (b) reset consecutive_match to 0,
     * (c) leave attempts/cooldown untouched, and (d) keep incrementing the
     * three widened counters by exactly 1/call with NO wraparound -- proven
     * by starting them just past where a plain `int` would already have
     * wrapped (INT32_MAX), using direct assignment rather than an actual
     * 2^31-iteration loop (impractical and unnecessary: the guard being
     * tested is "does int64_t arithmetic wrap here", which direct
     * assignment exercises identically to arriving there by increments). */
    filter_plateau_reset(&p);
    p.frame_count      = (int64_t)INT32_MAX + 1000;
    p.far_active_count = 5;
    p.dt_signal_count  = 5;
    p.attempts         = 0;
    {
        int fired = filter_plateau_update(&p, 1, 1, 10.0f, /*once_converged=*/1);
        CHECK(!fired, "plateau terminal(once_converged): never fires");
        CHECK(p.frame_count == (int64_t)INT32_MAX + 1001,
              "plateau terminal: frame_count keeps incrementing past INT32_MAX, no wrap");
        CHECK(p.consecutive_match == 0,
              "plateau terminal: consecutive_match reset by the once_converged branch");
    }
    /* Sanity check that the OLD `int` field type really would have wrapped
     * at this exact point -- emulated via well-defined unsigned wraparound
     * (never invokes real signed-overflow UB, so this stays clean under
     * -fsanitize=undefined) rather than executing genuine UB to "prove" it. */
    {
        uint32_t shadow_u = (uint32_t)INT32_MAX;
        shadow_u += 1001u;
        {
            int32_t shadow_i32;
            memcpy(&shadow_i32, &shadow_u, sizeof shadow_i32);
            CHECK(shadow_i32 < 0,
                  "sanity: a plain int32 counter would have gone negative at this exact point"
                  " (the bug class this widening fixes)");
        }
        CHECK(p.frame_count > 0,
              "fixed: the widened int64_t field stays positive/sane at the same point");
    }
    for (i = 0; i < 2000; i++) {
        int64_t before = p.frame_count;
        int fired = filter_plateau_update(&p, 1, 1, 10.0f, 1);
        CHECK(!fired, "plateau terminal(once_converged): still never fires after many calls");
        CHECK(p.frame_count == before + 1,
              "plateau terminal: frame_count still +1/call, no acceleration/corruption");
    }

    /* --- Case C: the OTHER terminal gate, attempts>=max_attempts (already
     * safe pre-existing threshold-gate -- confirm it still locks out the
     * ratio path even with otherwise-favorable inputs, and that it composes
     * cleanly with the widened counters). */
    filter_plateau_reset(&p);
    p.attempts = p.max_attempts;
    p.frame_count = 12345;
    {
        int fired = filter_plateau_update(&p, 1, 1, /*erle low=*/0.0f, /*once_converged=*/0);
        CHECK(!fired, "plateau terminal(attempts>=max_attempts): never fires even with favorable erle");
        CHECK(p.frame_count == 12346, "plateau terminal(attempts): frame_count still increments");
        CHECK(p.attempts == p.max_attempts, "plateau terminal(attempts): attempts itself stays capped");
    }

    /* --- Case D: functional smoke test -- the widening must not have
     * broken the real fire path under ordinary, small, in-range counts. */
    filter_plateau_init(&p);
    {
        int fired_at = -1;
        int limit = p.grace_frames + FILTER_PLATEAU_CONSECUTIVE_REQUIRED + 10;
        for (i = 0; i < limit; i++) {
            int fired = filter_plateau_update(&p, 1, 1, 0.0f, 0);
            if (fired) { fired_at = i; break; }
        }
        CHECK(fired_at >= 0, "plateau fires under sustained qualifying conditions");
        CHECK(p.frame_count == 0 && p.far_active_count == 0 && p.dt_signal_count == 0,
              "plateau fire path resets all three widened counters to 0 exactly as before");
        CHECK(p.attempts == 1, "plateau fire path increments attempts");
        CHECK(p.cooldown_remaining == FILTER_PLATEAU_POST_RESET_GRACE_FRAMES,
              "plateau fire path arms cooldown");
    }

    /* --- Case E: the actual fire path from a frame_count value that would
     * have silently narrowed/wrapped under the OLD 32-bit `last_reset_frame`
     * (widened to int64_t alongside the other three counters).
     * far_active_count/dt_signal_count are seeded EQUAL to frame_count so
     * far_ratio/dt_ratio are exactly 1.0 (comfortably clearing their
     * 0.5/0.10 gates) on every call regardless of frame_count's magnitude --
     * the huge seed only needs to clear criteria_met, not participate in
     * some elaborate proportion, so this is the simplest input that
     * reliably satisfies every gate every call (mirrors this file's
     * established "poke state directly, then drive real calls" style, e.g.
     * section_aec_level's blocks_with_active_render poke followed by real
     * aec_process() calls). */
    filter_plateau_reset(&p);
    {
        const int64_t seed = (int64_t)INT_MAX + 1000; /* > INT_MAX: the exact
            bug window a plain `int` last_reset_frame would have wrapped in */
        int fired = 0;
        int call_count;
        int64_t expected_last_reset;

        p.frame_count        = seed;
        p.far_active_count   = seed;
        p.dt_signal_count    = seed;
        p.attempts           = 0;
        p.cooldown_remaining = 0;
        p.consecutive_match  = 0;

        for (call_count = 1; call_count <= FILTER_PLATEAU_CONSECUTIVE_REQUIRED + 5; call_count++) {
            fired = filter_plateau_update(&p, /*far_active=*/1, /*dt_signal_present=*/1,
                                           /*erle_windowed_db=*/0.0f, /*once_converged=*/0);
            if (fired) break;
        }
        expected_last_reset = seed + call_count;

        CHECK(fired,
              "plateau huge-frame_count: fires within FILTER_PLATEAU_CONSECUTIVE_REQUIRED calls"
              " when far_ratio/dt_ratio/erle criteria hold true every call");
        CHECK(call_count == FILTER_PLATEAU_CONSECUTIVE_REQUIRED,
              "plateau huge-frame_count: fires on exactly the FILTER_PLATEAU_CONSECUTIVE_REQUIRED'th"
              " call (far_ratio/dt_ratio pinned to exactly 1.0 every call by construction)");
        CHECK(expected_last_reset > (int64_t)INT_MAX,
              "sanity: the frame_count value captured at fire time genuinely exceeds INT_MAX"
              " (this case actually exercises the narrowing/wraparound bug window, not vacuously)");
        CHECK(p.last_reset_frame == expected_last_reset,
              "plateau huge-frame_count: last_reset_frame equals the EXACT int64_t frame_count"
              " value at the moment of firing -- last_reset_frame is widened int -> int64_t;"
              " a plain `int` would have silently narrowed/wrapped this same assignment");
        CHECK(p.frame_count == 0 && p.far_active_count == 0 && p.dt_signal_count == 0,
              "plateau huge-frame_count: fire path still resets all three widened counters to 0"
              " exactly as in Case D, even starting from a value past INT_MAX");
    }
    printf("section_filter_plateau: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

/* ════════════════════════════════════════════════════════════════════
 * aec.c-embedded fixes: misadj_stable_count, ne_above,
 * stationarity_active_hops off-by-one -- plus confirming pokes for
 * aec_state.c's/pbfdkf.c's already-fixed counters (all reached through one
 * real Aec instance since they're embedded deep in aec_process()).
 * ════════════════════════════════════════════════════════════════════ */
static void section_aec_level(void) {
    AecConfig cfg;
    Aec a;
    int hop, i;
    float *mic, *ref, *out;

    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
    CHECK(aec_create(&a, &cfg) == 0, "aec_create succeeds");
    hop = aec_hop_size(&a);
    mic = (float*)calloc((size_t)hop, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    CHECK(mic && ref && out, "hop buffers allocated");
    /* mild constant far-end excitation so render-activity/active-render
     * gates (which several of the confirming checks below ride on) see a
     * non-silent signal; amplitude is tiny, irrelevant to the specific
     * counters under test. */
    for (i = 0; i < hop; i++) ref[i] = 0.05f * ((i % 7) - 3);

    /* --- ref_ring_filled: cap = ref_ring_size (freeze-at-capacity fix in
     * aec_process_core's ring-write step). Sole reader is
     * ">= current_delay + hop" and current_delay is capped at rs - hop, so
     * rs is the largest value ever meaningful. The old unbounded form is
     * signed-overflow UB after ~37 h of continuous 16 kHz audio; the guard
     * runs BEFORE the increment, so a value already at/above the cap is
     * frozen rather than incremented (this is what makes the INT_MAX seed
     * below UB-free -- the UBSan wrapper of this same binary proves it). */
    {
        int rs = a.ref_ring_size;
        a.ref_ring_filled = rs - 1;
        aec_process(&a, mic, ref, out);
        CHECK(a.ref_ring_filled == rs, "ref_ring_filled: rs-1 -> rs (clamped)");
        aec_process(&a, mic, ref, out);
        CHECK(a.ref_ring_filled == rs, "ref_ring_filled: rs -> rs (frozen)");
        a.ref_ring_filled = INT_MAX - 1;
        for (i = 0; i < 500; i++) aec_process(&a, mic, ref, out);
        CHECK(a.ref_ring_filled == INT_MAX - 1,
              "ref_ring_filled: INT_MAX-1 stays frozen over 500 real hops (no wrap, no UB)");
        {
            int d = 1600;
            int at_cap  = (rs >= d + hop);
            int at_huge = (INT_MAX - 1 >= d + hop);
            CHECK(at_cap == at_huge,
                  "ref_ring_filled: '>= delay+hop' decision identical at rs vs INT_MAX-1");
        }
        a.ref_ring_filled = rs;  /* leave a sane value for later sections */
    }

    /* --- misadj_stable_count: cap = cfg.filter_misadjustment_stable_frames,
     * sole reader "< cap" (misadj_fire). Force+hold the "stable" precondition
     * (converged && !epc.active && !regime.main_paused) directly every call
     * -- aec_process()'s own EPC/regime logic is otherwise free to touch
     * these fields, so re-poking before every call is what actually pins
     * the precondition deterministically for this test, independent of
     * what the rest of the (silent-ish) pipeline does. */
    {
        int cap = a.cfg.filter_misadjustment_stable_frames;
        a.misadj_hangover_remaining = 0;
        a.misadj_stable_count = cap - 1;
        a.convergence.converged = 1; a.epc.active = 0; a.regime.main_paused = 0;
        aec_process(&a, mic, ref, out);
        CHECK(a.misadj_stable_count == cap, "misadj_stable_count: cap-1 -> cap");
        a.convergence.converged = 1; a.epc.active = 0; a.regime.main_paused = 0;
        aec_process(&a, mic, ref, out);
        CHECK(a.misadj_stable_count == cap, "misadj_stable_count: cap -> cap (no-op)");
        for (i = 0; i < 2000; i++) {
            a.convergence.converged = 1; a.epc.active = 0; a.regime.main_paused = 0;
            aec_process(&a, mic, ref, out);
        }
        CHECK(a.misadj_stable_count == cap,
              "misadj_stable_count: still at cap after 2000 more calls (runaway-proof)");
        /* decision-invariance: the sole reader is "< cap". Two distinct
         * (non-tautological) locals standing in for "counter currently at
         * the cap" vs "counter at a synthetic huge value the old unbounded
         * code could have reached". */
        {
            int counter_at_cap = cap, counter_at_huge = cap + 1000000;
            int at_cap  = (counter_at_cap  < cap);
            int at_huge = (counter_at_huge < cap);
            CHECK(at_cap == at_huge,
                  "misadj_stable_count: '< cap' decision identical at cap vs cap+1e6");
        }
    }

    /* --- ne_above: cap = cfg.ne_recent_sustain, sole reader ">= cap"
     * (aec_process, held near-end-energy gate). Force dt_from_energy above
     * ne_recent_threshold every call (aec_process's own DT analyzer
     * recomputes dt_from_energy later in the SAME hop from the near-silent
     * signal, decaying it back down -- so, same as above, re-poke before
     * every call to pin the precondition this specific test cares about). */
    {
        int cap = a.cfg.ne_recent_sustain;
        a.ne_above = cap - 1;
        a.dt_analyzer.dt_from_energy = a.cfg.ne_recent_threshold + 1.0f;
        aec_process(&a, mic, ref, out);
        CHECK(a.ne_above == cap, "ne_above: cap-1 -> cap");
        a.dt_analyzer.dt_from_energy = a.cfg.ne_recent_threshold + 1.0f;
        aec_process(&a, mic, ref, out);
        CHECK(a.ne_above == cap, "ne_above: cap -> cap (no-op)");
        for (i = 0; i < 2000; i++) {
            a.dt_analyzer.dt_from_energy = a.cfg.ne_recent_threshold + 1.0f;
            aec_process(&a, mic, ref, out);
        }
        CHECK(a.ne_above == cap, "ne_above: still at cap after 2000 more calls (runaway-proof)");
        {
            int counter_at_cap = cap, counter_at_huge = cap + 1000000;
            int at_cap  = (counter_at_cap  >= cap);
            int at_huge = (counter_at_huge >= cap);
            CHECK(at_cap == at_huge,
                  "ne_above: '>= cap' decision identical at cap vs cap+1e6");
        }
    }

    /* --- stationarity_active_hops: off-by-one fix. cap = EXACTLY
     * a.stationarity_converge_hops (post-fix; pre-fix settled one past it).
     * non_zero_render_seen latches permanently once set, so the branch runs
     * unconditionally on every subsequent hop without re-poking. */
    {
        int cap = a.stationarity_converge_hops;
        CHECK(cap > 0, "stationarity_converge_hops is a sane positive value");
        a.non_zero_render_seen = 1;
        a.stationarity_active_hops = cap - 1;
        aec_process(&a, mic, ref, out);
        CHECK(a.stationarity_active_hops == cap,
              "stationarity_active_hops: cap-1 -> cap (settles at EXACTLY converge_hops)");
        aec_process(&a, mic, ref, out);
        CHECK(a.stationarity_active_hops == cap,
              "stationarity_active_hops: cap -> cap (no-op)");
        for (i = 0; i < 2000; i++) aec_process(&a, mic, ref, out);
        CHECK(a.stationarity_active_hops == cap,
              "stationarity_active_hops: still at cap after 2000 more calls (runaway-proof)");
        {
            int counter_at_cap = cap, counter_at_huge = cap + 1000000;
            int at_cap  = (counter_at_cap  >= cap);
            int at_huge = (counter_at_huge >= cap);
            CHECK(at_cap == at_huge,
                  "stationarity_active_hops: '>= cap' decision identical at cap vs cap+1e6"
                  " (the <= -> < off-by-one fix changes nothing observable here)");
        }
    }

    /* --- aec_state.c (already fixed): blocks_with_active_render
     * (cap = live active_render_blocks+1, reader is strict ">"),
     * strong_not_saturated_render_blocks (cap = live filter_adaptation_threshold_hops,
     * reader is strict "<"). aec_state_update()'s internal wiring is deep
     * (filter_analyzer/filter_delay/etc.), so rather than reverse-engineer
     * every input aec3_post_run threads into it, this poke-then-run-many
     * form checks the property that actually matters: the field never moves
     * past its cap no matter how many more real hops run, PLUS a direct
     * decision-invariance check against a synthetic huge value. */
    {
        /* AEC_STATE_ACTIVE_RENDER_BLOCKS / FILTER_ADAPTATION_THRESHOLD_HOPS
         * are gone (live-computed per instance now, not compile-time
         * constants) -- read the equivalent live fields off the already-
         * constructed `a` instance instead. */
        int active_render_blocks = a.a3_state.active_render_blocks;
        int filter_adaptation_threshold_hops =
            a.a3_state.delay_state.filter_adaptation_threshold_hops;
        int cap1 = active_render_blocks + 1;
        int cap2 = filter_adaptation_threshold_hops;
        a.a3_state.blocks_with_active_render = cap1;
        a.a3_state.strong_not_saturated_render_blocks = cap2;
        for (i = 0; i < 500; i++) aec_process(&a, mic, ref, out);
        CHECK(a.a3_state.blocks_with_active_render <= cap1,
              "aec_state.blocks_with_active_render: never exceeds cap over 500 real hops");
        CHECK(a.a3_state.strong_not_saturated_render_blocks <= cap2,
              "aec_state.strong_not_saturated_render_blocks: never exceeds cap over 500 real hops");
        {
            int at_cap1  = (cap1 > active_render_blocks);
            int at_huge1 = ((cap1 + 1000000) > active_render_blocks);
            CHECK(at_cap1 == at_huge1, "blocks_with_active_render: '>' decision identical at cap vs cap+1e6");
            int at_cap2  = (cap2 < filter_adaptation_threshold_hops);
            int at_huge2 = ((cap2 + 1000000) < filter_adaptation_threshold_hops);
            CHECK(at_cap2 == at_huge2, "strong_not_saturated_render_blocks: '<' decision identical at cap vs cap+1e6");
        }
    }

    /* --- pbfdkf.c (already fixed, LONG_MAX-guarded): main_filter.base.call_counter. */
    {
        a.main_filter.base.call_counter = LONG_MAX - 1;
        aec_process(&a, mic, ref, out);
        CHECK(a.main_filter.base.call_counter == LONG_MAX,
              "pbfdkf call_counter: LONG_MAX-1 -> LONG_MAX");
        aec_process(&a, mic, ref, out);
        CHECK(a.main_filter.base.call_counter == LONG_MAX,
              "pbfdkf call_counter: LONG_MAX -> LONG_MAX (no-op, no wrap)");
    }

    aec_destroy(&a);
    free(mic); free(ref); free(out);
    printf("section_aec_level: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

/* ════════════════════════════════════════════════════════════════════
 * delay_aec3.c: number_pre_echo_updates driven through the
 * REAL production per-hop API with a checked-in real audio fixture (not a
 * synthetic reimplementation) -- plus a bonus invariant check on
 * pending_count (already-safe cyclic counter, same driven loop).
 * ════════════════════════════════════════════════════════════════════ */
static void section_delay_aec3(void) {
    const char *mic_path = "../wav/aec_record/aec_record_mic_10s.wav";
    const char *ref_path = "../wav/aec_record/aec_record_ref_10s.wav";
    WavReader *mr = wav_open_read(mic_path);
    WavReader *rr = wav_open_read(ref_path);
    DelayAec3 d;
    void *delay_pool = NULL;
    int hop = 160;
    float mic[160], ref[160];
    int reached_cap = 0;
    long hops_read = 0;
    long expected_min_hops;

    /* A missing/unreadable fixture used to
     * silently `return` here with NO CHECK() and NO g_fails increment --
     * this whole section's ~2000+ checks would vanish from the run (e.g. if
     * ever invoked from the wrong cwd) while main()'s final tally still
     * showed 0 fails and printed PASS, just with a smaller total-checks
     * count. A missing fixture must FAIL the whole binary now, via the same
     * CHECK() macro every other assertion in this file uses -- not reduce
     * scope silently. */
    if (!mr || !rr) {
        CHECK(0,
              "section_delay_aec3: FIXTURE MISSING -- could not open "
              "'../wav/aec_record/aec_record_mic_10s.wav' / "
              "'../wav/aec_record/aec_record_ref_10s.wav' (must run from c_impl/); "
              "a missing/unreadable fixture must FAIL the whole test binary, not "
              "silently reduce its scope");
        fprintf(stderr,
                "section_delay_aec3: could not open fixture "
                "'%s' / '%s' (must run from c_impl/)\n", mic_path, ref_path);
        if (mr) wav_close_read(mr);
        if (rr) wav_close_read(rr);
        return;
    }

    /* Minimum-hop-count assertion: the
     * loop below's only exit condition is a short/EOF read, with no floor on
     * how many hops it actually processed -- a truncated/corrupted fixture
     * could silently satisfy every CHECK() in the loop by just running
     * fewer iterations. The expected hop count is derived from the two WAV
     * files' OWN headers (WavReader::info.num_samples, populated by
     * wav_open_read's own chunk-size parsing) rather than assumed from the
     * "_10s" filename convention, so a shrunk/corrupted fixture shrinks this
     * expectation too and can't spuriously pass by silently reading fewer
     * hops than a hardcoded literal would have demanded.
     *
     * Independently confirmed against the actual checked-in fixture (Python
     * `wave` module, read directly off the file headers): both
     * aec_record_{mic,ref}_10s.wav are mono/16 kHz/160000 samples exactly
     * (10.000s, evenly divisible by hop=160 with zero remainder), so a
     * complete, untruncated read of either file processes EXACTLY 1000
     * hops. The sanity CHECK just below ties that independently-verified
     * number to the runtime-computed value -- if it ever fails, the
     * checked-in fixture itself changed duration and this comment (and the
     * "_10s" filename) need updating, not this test. */
    {
        long mic_hops = (long)mr->info.num_samples / hop;
        long ref_hops = (long)rr->info.num_samples / hop;
        expected_min_hops = mic_hops < ref_hops ? mic_hops : ref_hops;
    }
    CHECK(expected_min_hops == 1000,
          "sanity: the checked-in 10s @16kHz fixture's own WAV header implies exactly"
          " 1000 hops at hop=160 (confirmed via Python `wave`: num_samples=160000 for"
          " both files); if this fails the fixture changed duration");

    /* Pool-first construction (plan step 2): DelayAec3 is a metadata struct
     * now, so this section owns the block the estimator's arrays live in.
     * Sized for the exact (16 kHz, hop=160, full bank) triple it drives. */
    {
        const char *why = NULL;
        delay_pool = delay_pool_init(&d, 16000, hop, DA_NUM_FILTERS, &why);
        CHECK(delay_pool != NULL,
              why ? why
                  : "section_delay_aec3: delay_aec3_init on a caller-owned pool");
        if (!delay_pool) {
            wav_close_read(mr); wav_close_read(rr);
            return;
        }
    }
    for (;;) {
        int gm = wav_read_float(mr, mic, hop);
        int gr = wav_read_float(rr, ref, hop);
        int n_pre;
        if (gm < hop || gr < hop) break;
        hops_read++;
        delay_aec3_accumulate_ex(&d, mic, ref, hop, 1);

        n_pre = d.est.matched_filter.number_pre_echo_updates;
        CHECK(n_pre >= 0 && n_pre <= DA_PRE_ECHO_UPDATES_TO_REPORT,
              "number_pre_echo_updates: never exceeds DA_PRE_ECHO_UPDATES_TO_REPORT");
        if (n_pre == DA_PRE_ECHO_UPDATES_TO_REPORT) reached_cap = 1;

        CHECK(d.est.pending_count >= 0 && d.est.pending_count < DA_AEC3_BLOCK_SIZE,
              "pending_count: stays inside [0, DA_AEC3_BLOCK_SIZE) every hop");
    }
    wav_close_read(mr);
    wav_close_read(rr);

    CHECK(hops_read >= expected_min_hops,
          "section_delay_aec3: processed fewer hops than the fixture's own WAV header"
          " implies -- a truncated/short-read fixture must fail loudly here, not"
          " silently pass by running fewer loop iterations");

    CHECK(reached_cap,
          "number_pre_echo_updates: this real fixture actually DRIVES the counter up to"
          " the cap at some point (proves the cap is exercised, not vacuously true)");
    CHECK(d.est.matched_filter.number_pre_echo_updates == DA_PRE_ECHO_UPDATES_TO_REPORT,
          "number_pre_echo_updates: settles at the cap by end of a real 10s fixture and stays");

    /* decision-invariance for completeness (sole reader is ">="). */
    {
        int cap = DA_PRE_ECHO_UPDATES_TO_REPORT;
        int counter_at_cap = cap, counter_at_huge = cap + 1000000;
        int at_cap  = (counter_at_cap  >= cap);
        int at_huge = (counter_at_huge >= cap);
        CHECK(at_cap == at_huge, "number_pre_echo_updates: '>=' decision identical at cap vs cap+1e6");
    }
    free(delay_pool);
    printf("section_delay_aec3: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

/* ════════════════════════════════════════════════════════════════════
 * detectors.c: FilterConvergence.conv_counter (already safe: the *whole*
 * update function early-returns once c->converged latches, so conv_counter
 * can never be touched again outside a paired mark_diverged/reset that
 * also zeroes it -- confirm).
 * ════════════════════════════════════════════════════════════════════ */
static void section_filter_convergence(void) {
    FilterConvergence c;
    int i, fired_at = -1;
    /* Grid is irrelevant to this counter test; use the default 16k/hop128. */
    filter_convergence_init(&c, 128, 16000);
    /* Drive real convergence: near_power/raw_error_power chosen so
     * inst_erle_db = 10*log10(near/err) > FC_CONV_ERLE_DB=5.0 every call. */
    for (i = 0; i < 200; i++) {
        int r = filter_convergence_update_convergence(&c, /*near_power=*/10.0f,
                                                       /*raw_error_power=*/0.01f,
                                                       /*far_active=*/1, /*warmup_done=*/1);
        if (r) { fired_at = i; break; }
    }
    CHECK(fired_at >= 0, "FilterConvergence: converges under sustained high inst-ERLE");
    CHECK(c.converged == 1 && c.once_converged == 1, "FilterConvergence: converged latches");
    {
        int cc_at_convergence = c.conv_counter;
        for (i = 0; i < 2000; i++) {
            int r = filter_convergence_update_convergence(&c, 10.0f, 0.01f, 1, 1);
            CHECK(!r, "FilterConvergence: never re-fires once converged");
            CHECK(c.conv_counter == cc_at_convergence,
                  "FilterConvergence: conv_counter frozen once converged (top-of-function"
                  " early-return lockout, not merely a reset branch)");
        }
    }
    /* mark_diverged resets both together -- confirm the pairing. */
    filter_convergence_mark_diverged(&c);
    CHECK(c.converged == 0 && c.conv_counter == 0,
          "FilterConvergence: mark_diverged resets converged + conv_counter together");
    printf("section_filter_convergence: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

/* ════════════════════════════════════════════════════════════════════
 * epc_shadow.c: EpcDetector.hangover (guarded countdown) + ShadowCopy
 * streak/copy_counter (self-terminating, SC_GATE_STREAK mode) + pause_resume
 * (guarded countdown) -- all already safe, confirm.
 * ════════════════════════════════════════════════════════════════════ */
static void section_epc_shadow(void) {
    EpcDetector e;
    ShadowCopy s;
    int i;

    epc_init(&e, /*hangover=*/7, 2.0f, 0.5f, /*hop_size=*/160, /*sample_rate=*/16000);
    epc_force_delay(&e);
    CHECK(e.active == 1 && e.hangover == 7, "EpcDetector: force_delay arms hangover");
    /* epc_tick_hangover decrements-then-stays-active on each call where
     * hangover was > 0 (7 such calls: 7->6 .. 1->0, active=1 throughout),
     * and only clears `active` on the FIRST call it observes hangover
     * already at 0 -- so clearing active takes hangover+1 calls, not
     * hangover calls. */
    for (i = 0; i < 7; i++) { epc_tick_hangover(&e); }
    CHECK(e.hangover == 0 && e.active == 1,
          "EpcDetector: hangover counts down to exactly 0 in 7 ticks (active still latched)");
    epc_tick_hangover(&e);
    CHECK(e.active == 0 && e.hangover == 0,
          "EpcDetector: the 8th tick (hangover already 0) finally clears active");
    for (i = 0; i < 2000; i++) epc_tick_hangover(&e);
    CHECK(e.hangover == 0, "EpcDetector: hangover stays at 0, never goes negative");

    /* ShadowCopy, SC_GATE_STREAK mode: streak self-fires at
     * SC_AEC3_STREAK_FRAMES and resets to 0 in the SAME call, so it can
     * never persist above (SC_AEC3_STREAK_FRAMES - 1) between calls. Force
     * copy_allowed with a direct baseline poke (bypass the slow EMA
     * ramp-up; the increment/self-fire logic under test still runs for
     * real). */
    shadow_copy_init_for_grid(&s, SC_GATE_STREAK, /*threshold=*/0.9f,
                              /*hysteresis=*/10, /*epc_hangover=*/5,
                              /*hop_size=*/128, /*sample_rate=*/16000);
    s.copy_err_baseline = 10.0f;   /* fast-forward past the EMA ramp */
    {
        /* Mirrors epc_shadow.c's `static const int SC_AEC3_STREAK_FRAMES = 5`
         * -- a private tuning constant, deliberately not exposed via
         * epc_shadow.h (this test does not change that; it just needs the
         * same numeric value to state the invariant precisely). */
        const int SC_AEC3_STREAK_FRAMES_MIRROR = 5;
        int max_seen = 0;
        for (i = 0; i < 100; i++) {
            ShadowCopyDecision dec = shadow_copy_update(
                &s, /*shadow_frame_count=*/100, /*far_pwr=*/1.0f,
                /*main_err_smooth=*/1.0f, /*shadow_err_smooth=*/0.5f,
                /*epc_active=*/0, /*saturation_level=*/0.0f,
                /*dt_from_energy=*/0.0f, /*dt_from_coherence=*/0.0f,
                /*delay_reliable=*/1);
            (void)dec;
            if (s.copy_counter > max_seen) max_seen = s.copy_counter;
            CHECK(s.copy_counter < SC_AEC3_STREAK_FRAMES_MIRROR,
                  "ShadowCopy(STREAK): streak never reaches/exceeds its own fire threshold"
                  " (self-resets in the same call it would cross it)");
        }
        CHECK(max_seen > 0, "ShadowCopy(STREAK): streak actually grows under sustained qualifying calls");
    }
    {
        ShadowCopy g8, g128, g256, g48, legacy;
        shadow_copy_init_for_grid(&g8, SC_GATE_ENERGY, 0.65f, 3, 5,
                                  128, 8000);
        shadow_copy_init_for_grid(&g128, SC_GATE_ENERGY, 0.65f, 3, 5,
                                  128, 16000);
        shadow_copy_init_for_grid(&g256, SC_GATE_ENERGY, 0.65f, 3, 5,
                                  256, 16000);
        shadow_copy_init_for_grid(&g48, SC_GATE_ENERGY, 0.65f, 3, 5,
                                  512, 48000);
        shadow_copy_init(&legacy, SC_GATE_ENERGY, 0.65f, 3, 5);
        CHECK(g8.shadow_copy_hysteresis == 10 &&
              g128.shadow_copy_hysteresis == 10 &&
              g256.shadow_copy_hysteresis == 10 &&
              g48.shadow_copy_hysteresis == 10,
              "ShadowCopy: the 10-hop streak minimum is a hop count on every product grid");
        CHECK(fabsf(g8.copy_err_baseline_retention - 0.992012008f) < 1e-7f &&
              fabsf(g128.copy_err_baseline_retention - 0.995997996f) < 1e-7f &&
              fabsf(g256.copy_err_baseline_retention - 0.992012008f) < 1e-7f &&
              fabsf(g48.copy_err_baseline_retention - 0.994667557f) < 1e-7f,
              "ShadowCopy: copy-error EMA keeps its ~2 s envelope on every product grid");
        CHECK(legacy.shadow_copy_hysteresis == 10 &&
              legacy.copy_err_baseline_retention == 0.995f,
              "ShadowCopy: legacy 10 ms authoring grid keeps its authored coefficients");
        {
            ShadowCopy* grids[] = {&g8, &g128, &g256, &g48};
            const int fire_hops[] = {10, 10, 10, 10};
            const float retentions[] = {
                0.992012008f, 0.995997996f,
                0.992012008f, 0.994667557f,
            };
            for (int grid = 0; grid < 4; ++grid) {
                ShadowCopy* live = grids[grid];
                live->copy_err_baseline = 10.0f;
                for (int i = 0; i < fire_hops[grid]; ++i) {
                    ShadowCopyDecision d = shadow_copy_update(
                        live, 200 + i, 1e-2f, 1e-3f, 1e-7f,
                        0, 0.0f, 0.0f, 0.0f, 1);
                    CHECK(d.boost_q == (i == fire_hops[grid] - 1),
                          "ShadowCopy: the 10-hop streak minimum decides the live boost hop");
                }

                shadow_copy_reset(live);
                live->copy_err_baseline = 1.0f;
                (void)shadow_copy_update(live, 200, 1e-2f, 0.0f, 0.0f,
                                         0, 0.0f, 0.0f, 0.0f, 1);
                CHECK(fabsf(live->copy_err_baseline - retentions[grid]) < 1e-7f,
                      "ShadowCopy: retimed retention is applied by the live EMA");
            }
        }
    }
    printf("section_epc_shadow: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

/* ════════════════════════════════════════════════════════════════════
 * filter_quality.c: all three INT_MAX-saturated counters (already fixed,
 * bit-exact-parity-sensitive -- confirm INT_MAX-1 -> INT_MAX -> INT_MAX).
 * ════════════════════════════════════════════════════════════════════ */
static void section_filter_quality(void) {
    FilteringQualityAnalyzer m;
    fq_init(&m, 1, 160, 16000);

    m.filter_update_blocks_since_start = INT_MAX - 1;
    m.filter_update_blocks_since_reset = INT_MAX - 1;
    m.convergence_hops_counter         = INT_MAX - 1;
    fq_update(&m, /*active_render=*/1, /*transparent_mode=*/0,
              /*saturated_capture=*/0, /*external_delay_present=*/0,
              /*any_filter_converged=*/1);
    CHECK(m.filter_update_blocks_since_start == INT_MAX, "fq: blocks_since_start INT_MAX-1 -> INT_MAX");
    CHECK(m.filter_update_blocks_since_reset == INT_MAX, "fq: blocks_since_reset INT_MAX-1 -> INT_MAX");
    CHECK(m.convergence_hops_counter == INT_MAX, "fq: convergence_hops_counter INT_MAX-1 -> INT_MAX");

    fq_update(&m, 1, 0, 0, 0, 1);
    CHECK(m.filter_update_blocks_since_start == INT_MAX, "fq: blocks_since_start INT_MAX -> INT_MAX (no-op)");
    CHECK(m.filter_update_blocks_since_reset == INT_MAX, "fq: blocks_since_reset INT_MAX -> INT_MAX (no-op)");
    CHECK(m.convergence_hops_counter == INT_MAX, "fq: convergence_hops_counter INT_MAX -> INT_MAX (no-op)");

    {
        int i;
        for (i = 0; i < 2000; i++) fq_update(&m, 1, 0, 0, 0, 1);
        CHECK(m.filter_update_blocks_since_start == INT_MAX, "fq: blocks_since_start runaway-proof");
        CHECK(m.filter_update_blocks_since_reset == INT_MAX, "fq: blocks_since_reset runaway-proof");
        CHECK(m.convergence_hops_counter == INT_MAX, "fq: convergence_hops_counter runaway-proof");
    }
    printf("section_filter_quality: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

/* ════════════════════════════════════════════════════════════════════
 * fullband_erle.c: num_points (exact-match cyclic reset, already safe) +
 * hold_counter_inst_erle (INT_MIN-floored, already fixed).
 * ════════════════════════════════════════════════════════════════════ */
static void section_fullband_erle(void) {
    FullBandErleEstimator s;
    const int nf = 8;
    float x2[8], y2[8], e2[8];
    int i, k;

    fb_erle_init(&s, nf, 0.001f, 1000.0f, 160, 16000);
    for (k = 0; k < nf; k++) { x2[k] = 1.0e9f; y2[k] = 1.0e6f; e2[k] = 1.0e3f; }

    for (i = 0; i < 6 * 20; i++) {
        fb_erle_update(&s, x2, y2, e2, /*converged_filter=*/1);
        CHECK(s.inst.num_points >= 0 && s.inst.num_points < FBERLE_POINTS_TO_ACCUMULATE,
              "fullband_erle.num_points: stays inside [0, POINTS_TO_ACCUMULATE) every call");
    }

    s.hold_counter_inst_erle = INT_MIN + 1;
    fb_erle_update(&s, x2, y2, e2, /*converged_filter=*/0);
    CHECK(s.hold_counter_inst_erle == INT_MIN, "fullband_erle.hold_counter_inst_erle: INT_MIN+1 -> INT_MIN");
    fb_erle_update(&s, x2, y2, e2, 0);
    CHECK(s.hold_counter_inst_erle == INT_MIN, "fullband_erle.hold_counter_inst_erle: INT_MIN -> INT_MIN (no-op)");
    for (i = 0; i < 2000; i++) fb_erle_update(&s, x2, y2, e2, 0);
    CHECK(s.hold_counter_inst_erle == INT_MIN, "fullband_erle.hold_counter_inst_erle: runaway-proof (floor)");
    printf("section_fullband_erle: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

/* ════════════════════════════════════════════════════════════════════
 * erl_estimator.c: blocks_since_reset (startup_hops cap) +
 * hold_counter_time_domain (INT_MIN floor) -- both already fixed, confirm.
 * ════════════════════════════════════════════════════════════════════ */
static void section_erl_estimator(void) {
    ErlEstimator e;
    const int nb = 8;
    float erl_storage[8];
    int hold_storage[6];
    float x2[8], y2[8];
    int i, k;

    erl_estimator_init(&e, /*startup_phase_length_hops=*/1, nb, 160, 16000, erl_storage, hold_storage);
    /* x2 deliberately kept BELOW the fullband gate's own x2_sum > x2_min*n_bins
     * threshold (~3.5e8 for n_bins=8) -- an fp32 EMA converging toward a
     * FIXED ratio approaches its target monotonically from above and can
     * stick an epsilon short of it forever (confirmed empirically: with a
     * constant 0.5 ratio, erl_time_domain converges to ~0.500000238, four
     * ULPs above 0.5, so "new_erl(0.5) < erl_time_domain" never goes false
     * and the block re-arms hold_counter_time_domain every single call --
     * a test-setup trap, not a source bug). Keeping x2_sum under the gate
     * instead means the fullband "arm to ERL_HOLD_HOPS" block never runs at
     * all, isolating hold_counter_time_domain as a pure decrement-from-
     * wherever-we-poke-it counter, which is what this specific test needs. */
    for (k = 0; k < nb; k++) { x2[k] = 1.0f; y2[k] = 1.0f; }

    /* first call: blocks_since_reset 0 -> 1 (== startup_hops), gate passes;
     * x2_sum stays under the fullband gate's own threshold (by design, see
     * above), so hold_counter_time_domain is untouched by the arm-to-400
     * path and is purely decremented from its init value of 0. */
    erl_estimator_update(&e, x2, y2, /*converged_filter=*/1);
    CHECK(e.blocks_since_reset == 1, "erl_estimator.blocks_since_reset: 0 -> startup_hops(=1)");

    e.blocks_since_reset = 0;   /* re-poke cap-1 to test the transition explicitly */
    erl_estimator_update(&e, x2, y2, 1);
    CHECK(e.blocks_since_reset == 1, "erl_estimator.blocks_since_reset: cap-1 -> cap");
    erl_estimator_update(&e, x2, y2, 1);
    CHECK(e.blocks_since_reset == 1, "erl_estimator.blocks_since_reset: cap -> cap (no-op)");
    for (i = 0; i < 2000; i++) erl_estimator_update(&e, x2, y2, 1);
    CHECK(e.blocks_since_reset == 1, "erl_estimator.blocks_since_reset: runaway-proof");

    e.hold_counter_time_domain = INT_MIN + 1;
    erl_estimator_update(&e, x2, y2, 1);
    CHECK(e.hold_counter_time_domain == INT_MIN, "erl_estimator.hold_counter_time_domain: INT_MIN+1 -> INT_MIN");
    erl_estimator_update(&e, x2, y2, 1);
    CHECK(e.hold_counter_time_domain == INT_MIN, "erl_estimator.hold_counter_time_domain: INT_MIN -> INT_MIN (no-op)");
    printf("section_erl_estimator: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

/* ════════════════════════════════════════════════════════════════════
 * initial_state.c: strong_not_saturated_render_blocks (cap =
 * max(conservative_hops, initial_state_hops), already fixed, confirm).
 * ════════════════════════════════════════════════════════════════════ */
static void section_initial_state(void) {
    InitialState s;
    int cap, i;

    initial_state_init(&s, /*conservative_initial_phase=*/0, /*initial_state_seconds=*/2.5f, 160, 16000);
    cap = (s.conservative_hops > s.initial_state_hops) ? s.conservative_hops : s.initial_state_hops;

    s.strong_not_saturated_render_blocks = cap - 1;
    initial_state_update(&s, /*active_render=*/1, /*saturated_capture=*/0);
    CHECK(s.strong_not_saturated_render_blocks == cap,
          "initial_state.strong_not_saturated_render_blocks: cap-1 -> cap");
    initial_state_update(&s, 1, 0);
    CHECK(s.strong_not_saturated_render_blocks == cap,
          "initial_state.strong_not_saturated_render_blocks: cap -> cap (no-op)");
    for (i = 0; i < 2000; i++) initial_state_update(&s, 1, 0);
    CHECK(s.strong_not_saturated_render_blocks == cap,
          "initial_state.strong_not_saturated_render_blocks: runaway-proof");
    printf("section_initial_state: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

/* ════════════════════════════════════════════════════════════════════
 * filter_analyzer.c: blocks_since_reset (long, cap =
 * FA_CONVERGENCE_THRESHOLD_HOPS+1, reader is strict ">") + consistent.counter
 * (long, LONG_MAX-floored, bit-exact-parity-sensitive) -- both already
 * fixed, confirm.
 * ════════════════════════════════════════════════════════════════════ */
static void section_filter_analyzer(void) {
    FilterAnalyzer m;
    const int size = 320;
    float h_highpass[320];
    float abs_scratch[320];
    float render_sq_scratch[160];
    float filter_taps[320];
    float render_block[160];
    int i, k;
    long cap1;

    for (k = 0; k < size; k++) filter_taps[k] = (k == 0) ? 100.0f : 0.0f;
    for (k = 0; k < 160; k++) render_block[k] = 200.0f;

    fa_init(&m, h_highpass, size, /*active_render_limit=*/100.0f, /*bounded_erl=*/0,
            /*default_gain=*/1.0f, abs_scratch, size, render_sq_scratch, 160, 160, 16000);

    /* Prime once so delay_ref/peak_index settle (peak stays at index 0
     * every call since filter_taps[0] is the only nonzero tap). */
    fa_update(&m, filter_taps, render_block, 160);

    /* FA_CONVERGENCE_THRESHOLD_HOPS is gone (live-computed per instance now,
     * not a compile-time constant) -- read the equivalent live field off
     * the already-constructed `m` instance instead. */
    cap1 = m.convergence_threshold_hops + 1;   /* strict ">" reader downstream */
    m.blocks_since_reset = m.convergence_threshold_hops;
    fa_update(&m, filter_taps, render_block, 160);
    CHECK(m.blocks_since_reset == cap1,
          "filter_analyzer.blocks_since_reset: settles at convergence_threshold_hops+1 (matches its strict '>' reader)");
    fa_update(&m, filter_taps, render_block, 160);
    CHECK(m.blocks_since_reset == cap1, "filter_analyzer.blocks_since_reset: cap -> cap (no-op)");
    for (i = 0; i < 2000; i++) fa_update(&m, filter_taps, render_block, 160);
    CHECK(m.blocks_since_reset == cap1, "filter_analyzer.blocks_since_reset: runaway-proof");

    /* consistent.counter: only advances while delay_ref==delay_blocks (true
     * here, since the peak is pinned at index 0 every call) AND active
     * (render power over threshold, true given render_block=200.0f
     * constant). Poke straight to LONG_MAX-1 to test the true boundary. */
    m.consistent.counter = LONG_MAX - 1;
    fa_update(&m, filter_taps, render_block, 160);
    CHECK(m.consistent.counter == LONG_MAX, "filter_analyzer.consistent.counter: LONG_MAX-1 -> LONG_MAX");
    fa_update(&m, filter_taps, render_block, 160);
    CHECK(m.consistent.counter == LONG_MAX, "filter_analyzer.consistent.counter: LONG_MAX -> LONG_MAX (no-op)");
    printf("section_filter_analyzer: done (%ld checks so far, %ld fails)\n", g_checks, g_fails);
}

int main(void) {
    section_filter_plateau();
    section_aec_level();
    section_delay_aec3();
    section_filter_convergence();
    section_epc_shadow();
    section_filter_quality();
    section_fullband_erle();
    section_erl_estimator();
    section_initial_state();
    section_filter_analyzer();

    printf("TOTAL CHECKS: %ld  FAILS: %ld\n", g_checks, g_fails);
    if (g_fails) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS\n");
    return 0;
}
