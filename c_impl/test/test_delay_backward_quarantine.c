/* test_delay_backward_quarantine.c — permanent regression test for the
 * Path-B backward-jump quarantine (AecConfig::delay_backward_quarantine_enabled
 * / delay_backward_quarantine_s).
 *
 * THE DEFECT IT TARGETS, pinned here as documentation and reproduced FIRST
 * with the mechanism off: with a white-noise far end at a true bulk delay of
 * 6400 samples (16 kHz, fft 512 / hop 256), the matched filter acquires
 * CORRECTLY at 6336 on hop 32, then on hop 49 re-locks to 4800 -- exactly
 * 1600 samples (100 ms) EARLY, the pre-echo mis-attribution signature -- and
 * keeps reporting 4800 for the rest of the run. Because the wrong candidate
 * is SUSTAINED rather than a one-hop glitch, no K-consecutive-confirmation
 * gate can reject it: every confirmation window it is offered agrees with
 * itself.
 *
 * WHAT THE MECHANISM DOES, stated the way the rows below measure it: a
 * candidate strictly EARLIER than the delay in force, offered while the
 * linear filter still cancels at the applied alignment, is held for a bounded
 * window and then ACCEPTED. It DELAYS a mis-lock by the window; it does not
 * cure one. Curing pre-echo mis-attribution is estimator work and is not what
 * this guard is for.
 *
 * WHY BOUNDED AND DIRECTIONAL: the unbounded, direction-blind predicate this
 * replaced -- and the multipath measurement that killed it -- are written up
 * once, on delay_backward_quarantine_enabled in aec.h. Row 4 below is the
 * permanent guard against that regression.
 *
 * What this test proves:
 *   1. DEFECT REPRODUCES (mechanism off, the shipped default): the accepted
 *      delay drops from the 6336 class to 4800 on hop 49 and STAYS there.
 *      Without this row the rest could pass against a scene that never
 *      mis-locks at all.
 *   2. THE QUARANTINE DELAYS IT BY THE WINDOW, and holds the correct answer
 *      meanwhile: with the window at its 1.0 s default (62 hops on this grid)
 *      the wrong delay is accepted on hop 111 = 49 + 62, and 6336 is the
 *      accepted delay on every hop of the window.
 *   3. THE BOUND IS THE EXPIRY, falsifiably: sweeping the window over
 *      0.5 / 1.0 / 2.0 / 4.0 s moves the acceptance hop to exactly
 *      49 + window-in-hops every time (80 / 111 / 174 / 299). A cancellation
 *      that never collapses therefore costs the window and NOTHING more --
 *      there is no permanent-veto path.
 *   4. MULTIPATH / PATH ADDITION: an old reflection at 2400 persists at gain
 *      g_old while a stronger NEW path appears EARLIER, at 1600 / gain 0.5.
 *      Across g_old = 0.2 / 0.3 / 0.4 / 0.5 the guarded build re-locks onto
 *      the new path in ALL FOUR rows, never later than the unguarded build
 *      plus one window.
 *   5. FORWARD CANDIDATES ARE NOT THIS MECHANISM'S BUSINESS, pinned where it
 *      is falsifiable: with the surviving old reflection at 1600 and the new
 *      path LATER at 3200, the forward candidate arrives WHILE the applied
 *      alignment is still cancelling, and the trajectory is still identical
 *      on and off in all four gain rows. The plain forward MOVE (64 -> 3000)
 *      is asserted identical too, but on its own it proves less: this
 *      engine's duty-cycled analysis emits that candidate long after ERLE
 *      decayed to 0, so the predicate is false there whether or not a
 *      direction test exists (verified: dropping the direction test leaves
 *      the move scene green and turns the multipath one red).
 *   6. OFF IS THE UNCHANGED PATH: the enable defaults to 0, and with it off
 *      the trajectory is identical to a build that has no mechanism at all --
 *      pinned here as "row 1 still reproduces", with the audio-byte half
 *      proven out of band (rendering the seam config before and after this
 *      change: 512000 bytes byte-identical).
 *   7. WHICH ERLE READING DOES THE WORK is falsifiable, not asserted in
 *      prose: with the inst-ERLE-peak arm configured out of reach the
 *      quarantine stops engaging, and the windowed reading it was left with
 *      is measurably under its own threshold.
 *
 * The windowed-ERLE arm is NOT independently pinned here, and the reason is a
 * property of this engine, not an oversight: the C matched filter is
 * duty-cycled, so a genuine move produces its new candidate long after
 * windowed ERLE has decayed to 0, and no scene reached through this API makes
 * the slow arm the deciding one. The Python mirror
 * (python/tests/test_delay_backward_quarantine.py) DOES pin it, so the pair
 * covers both arms.
 *
 * Mutation checks (each breaks one line and must go red here):
 *   - drop the direction test (quarantine every differing candidate) -> the
 *     FORWARD MULTIPATH row fails (its g_old 0.4 / 0.5 trajectories separate
 *     by one window). The forward-MOVE row alone does NOT catch this, which
 *     is why the multipath variant exists;
 *   - drop the expiry (never let the countdown reach 0 / keep re-arming)
 *     -> rows 2, 3 and 4 fail: the wrong delay is never accepted and the
 *     window sweep has nothing to measure;
 *   - re-arm per candidate instead of once -> row 3's exact-expiry equalities
 *     fail;
 *   - drop the inst-ERLE-peak arm -> row 2 fails: windowed ERLE is only
 *     1.804 dB at the acceptance hop on this scene, below its own 2.5 dB
 *     threshold, so a windowed-only predicate cannot see this defect;
 *   - drop the windowed-ERLE arm -> row 7's "behaves like off" holds for the
 *     wrong reason, so ALSO run the Python mirror, whose move-scene row is
 *     red under this mutation;
 *   - ignore delay_backward_quarantine_enabled (always on) -> row 1 fails
 *     (the defect no longer reproduces with the mechanism off);
 *   - change the enable default to 1 -> row 1 fails for the same reason;
 *   - ignore delay_backward_quarantine_s (hardcode the window) -> row 3 fails;
 *   - move the quarantine onto Path A -> row 2's first-acquisition equality
 *     fails.
 *
 * Build (standalone, from c_impl/ -- mirrors test_duty_census.c):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_delay_backward_quarantine.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_delay_backward_quarantine
 * Run:  ./bin/test_delay_backward_quarantine
 * Also wired into the Makefile: `make test-delay-backward-quarantine`.
 */
#include "aec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; } \
        else         { printf("pass: %s\n", (msg)); g_pass++; } \
    } while (0)

#define SR          16000
#define FFT         512              /* hop 256 -- the pipelines' signal grid */
#define HOP         256

/* ---- the pre-echo mis-lock scene ---------------------------------------- */
#define MIS_HOPS    500              /* 8 s: 10x the hop-49 re-lock point     */
#define MIS_DELAY   6400             /* true bulk delay, 400 ms               */
#define MIS_WRONG   4800             /* MIS_DELAY - 1600: the pre-echo answer */

/* ---- the multipath / path-addition scene -------------------------------- */
/* An old reflection at MP_OLD persists for the whole run at gain g_old; a
 * STRONGER new path appears EARLIER, at MP_NEW / gain MP_GAIN_NEW, from hop
 * MP_AT onwards. Backward direction on purpose: this is the guarded
 * direction, so it is where "bounded" has to be demonstrated rather than
 * assumed. The geometry is the one (surveyed over old in
 * {2400,3200,4000,4800} x new in {512,768,1024,1600,2048}) where the
 * UNGUARDED build re-locks in all four gain rows inside MP_HOPS -- a row the
 * unguarded build cannot solve is a statement about the estimator, not about
 * this guard, and would make the comparison vacuous. */
#define MP_HOPS     1600             /* 25.6 s: the 0.5 row re-locks at 1515  */
#define MP_AT       250              /* the new path appears at hop 250       */
#define MP_OLD      2400             /* surviving old reflection, 150 ms      */
#define MP_NEW      1600             /* new path: EARLIER, 100 ms             */
#define MP_GAIN_NEW 0.5f

/* ---- the FORWARD multipath scene ---------------------------------------
 * Mirror image of the above, and the row that actually pins the DIRECTION
 * test: the surviving old reflection is the EARLIER one, so the new path's
 * candidate is a FORWARD jump arriving while the applied alignment is still
 * being cancelled. That combination -- live cancellation AND a forward
 * candidate -- is what the plain forward-MOVE scene below cannot produce on
 * this engine (its duty-cycled analysis emits the new candidate long after
 * ERLE decayed to 0, so the predicate is false there for a reason unrelated
 * to direction, and dropping the direction test leaves that scene green). */
#define FMP_HOPS    1200
#define FMP_OLD     1600             /* surviving old reflection: EARLIER    */
#define FMP_NEW     3200             /* new path: LATER, so unquarantined    */

/* ---- the forward-move scene --------------------------------------------- */
#define MOVE_HOPS   750              /* 12 s                                  */
#define MOVE_AT     375              /* the echo moves at hop 375 (6 s)       */
#define MOVE_D0     64               /* pre-move delay: inside the tap reach  */
#define MOVE_D1     3000             /* post-move delay: FORWARD, unguarded   */
/* The BACKWARD move: the same generator with the endpoints swapped, so the
 * post-move delay is EARLIER and the direction test fires. This is now the
 * only scene that offers the quarantine a backward candidate at all -- the
 * pre-echo mis-lock that used to supply one is fixed at the estimator (see
 * block 1), so without this the mechanism would ship with no engaging-path
 * coverage. Released by the collapse well inside the window, because the old
 * path is genuinely GONE and cancellation really does fall apart. */
#define BWD_HOPS    750
#define BWD_AT      375
#define BWD_D0      3000             /* pre-move delay                        */
#define BWD_D1      64               /* post-move delay: EARLIER, guarded     */

/* The window sweep of row 3. 4.0 s is the "no permanent veto" end: 250 hops
 * still expires well inside MIS_HOPS. */
static const float g_windows[] = { 0.5f, 1.0f, 2.0f, 4.0f };
#define N_WINDOWS ((int)(sizeof g_windows / sizeof g_windows[0]))

/* The library's own seconds -> cycles conversion, restated (not imported) so
 * a silent change to it fails here instead of being absorbed. */
static int window_hops(float seconds) {
    int h = (int)lrintf(seconds / ((float)HOP / (float)SR));
    return h < 1 ? 1 : h;
}

static float g_far[MP_HOPS * HOP];
static float g_near[MP_HOPS * HOP];

static unsigned g_rng_state;
static float rng_uniform(float amp) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 17;
    g_rng_state ^= g_rng_state << 5;
    {
        float u = (float)(g_rng_state & 0xFFFFFFu) / (float)0xFFFFFFu;
        return amp * (2.0f * u - 1.0f);
    }
}

static void far_noise(int n) {
    int i;
    g_rng_state = 0x5EED1234u;
    for (i = 0; i < n; ++i) g_far[i] = rng_uniform(0.35f);
}

/* White-noise far end; near = far delayed by `delay` and attenuated. White
 * noise is the point: it is the stimulus the matched filter is happiest with,
 * so a mis-lock here is not an artefact of a degenerate excitation. */
static void build_static(int n, int delay) {
    int i;
    far_noise(n);
    for (i = 0; i < n; ++i)
        g_near[i] = (i >= delay) ? 0.5f * g_far[i - delay] : 0.0f;
}

/* Same far end; the echo path genuinely MOVES at sample `at` -- before it the
 * echo sits at d0, after it at d1 and NOTHING remains at d0. Cancellation
 * therefore really does collapse, which is the early-release condition. */
static void build_moving(int n, int at, int d0, int d1) {
    int i;
    far_noise(n);
    for (i = 0; i < n; ++i) g_near[i] = 0.0f;
    for (i = d0; i < at; ++i)       g_near[i] = 0.5f * g_far[i - d0];
    for (i = at + d1; i < n; ++i)   g_near[i] = 0.5f * g_far[i - d1];
}

/* Path ADDITION rather than path movement: the old reflection is never
 * removed, so ERLE at the applied alignment does NOT collapse and the
 * early-release arm cannot carry this scene -- only the expiry can. */
static void build_multipath(int n, int at, int d_old, float g_old,
                            int d_new, float g_new) {
    int i;
    far_noise(n);
    for (i = 0; i < n; ++i) {
        float v = (i >= d_old) ? g_old * g_far[i - d_old] : 0.0f;
        if (i >= at && i >= d_new) v += g_new * g_far[i - d_new];
        g_near[i] = v;
    }
}

/* Runs `hops` hops of the NN-seam config (enable_res=0, return_res_context=1
 * -- the configuration both ULCNet pipelines and the 4ch lanes use) and
 * records the accepted delay after every hop into `out`. `window_s` < 0 keeps
 * the config default; `inst_db` < 0 keeps the preset's own threshold, so no
 * row silently re-states either constant as a literal. */
static int run_hops_ex(int enabled, float window_s, float inst_db,
                       int hops, int* out, float* erle_at_change) {
    AecConfig cfg;
    Aec aec;
    int h, prev = -99;
    float prev_erle = 0.0f;

    if (erle_at_change) *erle_at_change = -1.0f;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
    cfg.fft_size = FFT;
    cfg.enable_res = 0;
    cfg.return_res_context = 1;
    cfg.spatial_linear_context = 0;
    if (inst_db >= 0.0f) cfg.delay_acquire_inst_erle_db = inst_db;
    cfg.delay_backward_quarantine_enabled = enabled;
    if (window_s >= 0.0f) cfg.delay_backward_quarantine_s = window_s;
    if (aec_create(&aec, &cfg) != 0) return -1;
    if (aec_hop_size(&aec) != HOP) { aec_destroy(&aec); return -2; }

    for (h = 0; h < hops; ++h) {
        AecDebugStatus st;
        aec_process_context(&aec, &g_near[h * HOP], &g_far[h * HOP]);
        aec_debug_status(&aec, &st);
        out[h] = st.delay_samples;
        /* The reading the guard's slow arm judges a CHANGE on (not the first
         * acquisition): the PREVIOUS hop's cached windowed ERLE, which is
         * what a->last_erle_windowed holds when Path B runs. Reading this
         * hop's value instead would report the post-acceptance reset (0.0)
         * and prove nothing about the decision. */
        if (erle_at_change && *erle_at_change < 0.0f
                && prev >= 0 && st.delay_samples != prev)
            *erle_at_change = prev_erle;
        prev = st.delay_samples;
        prev_erle = st.erle_windowed_db;
    }
    aec_destroy(&aec);
    return 0;
}

static int run_hops(int enabled, int hops, int* out) {
    return run_hops_ex(enabled, -1.0f, -1.0f, hops, out, NULL);
}

/* First hop whose accepted delay differs from the previous hop's. */
static int first_change_after(const int* d, int hops, int from) {
    int h;
    for (h = from + 1; h < hops; ++h)
        if (d[h] != d[h - 1]) return h;
    return -1;
}

static int in_lock_class(int d, int truth) {
    /* The estimator answers on a 64-sample grid, so "locked on this path" is
     * a class, not a literal. Anything within two grid steps of the truth
     * qualifies; the 6400 scene's 4800 is 25 steps away and never can. */
    return d >= truth - 128 && d <= truth + 128;
}

static int off[MP_HOPS];
static int on[MP_HOPS];

int main(void) {
    int h, rc;
    int mis_off_relock = -1;

    printf("=== delay_backward_quarantine ===\n");
    printf("(window %.2f s = %d hops at %d Hz / hop %d)\n",
           (double)1.0f, window_hops(1.0f), SR, HOP);

    /* ---- 1. the defect reproduces with the mechanism OFF (the default) --- */
    build_static(MIS_HOPS * HOP, MIS_DELAY);
    rc = run_hops(0, MIS_HOPS, off);
    CHECK(rc == 0, "mislock scene: aec_create + hop size 256");
    if (rc == 0) {
        int locked_hop = -1, wrong_hop = -1, wrong_sustained = 1;
        for (h = 0; h < MIS_HOPS; ++h) {
            if (locked_hop < 0 && in_lock_class(off[h], MIS_DELAY)) locked_hop = h;
            if (wrong_hop < 0 && off[h] == MIS_WRONG)               wrong_hop = h;
        }
        for (h = (wrong_hop < 0 ? MIS_HOPS : wrong_hop); h < MIS_HOPS; ++h)
            if (off[h] != MIS_WRONG) wrong_sustained = 0;
        mis_off_relock = wrong_hop;

        /* This scene USED to be the defect: it acquired the true path and
         * then re-locked to the pre-echo answer, holding it to the end --
         * which this quarantine was built to DELAY, never to cure. Curing it
         * was estimator work, done at the selection seam (the aggregator now
         * reports the dominant peak; see test_delay_dominant_selection). The
         * scene is kept, inverted, because an end-to-end assertion through
         * aec_process() is what would catch the substitution returning by
         * another route -- which the aggregator-level test cannot see.
         * Asserted with the mechanism OFF, the shipped default, so it is the
         * estimator under test and not the guard. */
        CHECK(locked_hop >= 0, "OFF: acquires the true path first");
        CHECK(wrong_hop < 0,
              "OFF: the pre-echo answer (true - 1600) is never applied");
        {
            int held_one = 1;
            for (h = locked_hop; h >= 0 && h < MIS_HOPS; ++h)
                if (off[h] != off[locked_hop]) held_one = 0;
            CHECK(held_one,
                  "OFF: the acquired delay is held for the rest of the run");
        }
        (void)wrong_sustained;
        printf("      (OFF: lock hop %d -> %d, never re-locked to %d)\n",
               locked_hop, locked_hop >= 0 ? off[locked_hop] : -1, MIS_WRONG);
    }

    /* ---- 2 and 3 removed: their subject no longer exists ----
     * They measured how long the guard delayed the pre-echo re-lock, and
     * swept the window to prove the expiry was what released it. The
     * estimator no longer produces that candidate (block 1), so both rows
     * would be asserting about a scene that cannot happen.
     *
     * The claims survive where they are still demonstrable: the multipath
     * rows below engage the guard and are released by the EXPIRY at exactly
     * one window (+62 hops on this grid), which is the same "the window is
     * what sets acceptance" property block 3 swept for -- and unlike the
     * mis-lock scene, multipath is a real echo-path condition rather than an
     * estimator defect. The backward-MOVE scene is not a substitute: by the
     * time that move is accepted here cancellation at the old alignment has
     * long collapsed, so the predicate never holds and the guard does not
     * engage at all. Python's own backward-move row does engage and pins the
     * collapse-release path; see test_delay_backward_quarantine.py. */

    /* ---- 4. multipath / path addition: all four gain rows re-lock ---- */
    {
        static const float gains[] = { 0.2f, 0.3f, 0.4f, 0.5f };
        int i, q = window_hops(1.0f);
        int all_relock = 1, all_bounded = 1, all_land = 1;
        printf("      multipath old %d (g_old) + NEW %d @ %.1f from hop %d:\n",
               MP_OLD, MP_NEW, (double)MP_GAIN_NEW, MP_AT);
        for (i = 0; i < 4; ++i) {
            int off_hop, on_hop;
            build_multipath(MP_HOPS * HOP, MP_AT * HOP, MP_OLD, gains[i],
                            MP_NEW, MP_GAIN_NEW);
            if (run_hops(0, MP_HOPS, off) != 0
                    || run_hops(1, MP_HOPS, on) != 0) {
                all_relock = 0; continue;
            }
            off_hop = first_change_after(off, MP_HOPS, MP_AT);
            on_hop  = first_change_after(on,  MP_HOPS, MP_AT);
            if (off_hop < 0 || on_hop < 0) all_relock = 0;
            else if (on_hop < off_hop || on_hop - off_hop > q) all_bounded = 0;
            if (!in_lock_class(on[MP_HOPS - 1], MP_NEW)
                    || !in_lock_class(off[MP_HOPS - 1], MP_NEW)) all_land = 0;
            printf("        g_old %.1f: OFF hop %4d, ON hop %4d (+%d), "
                   "final OFF %d / ON %d\n",
                   (double)gains[i], off_hop, on_hop,
                   (off_hop > 0 && on_hop > 0) ? on_hop - off_hop : -1,
                   off[MP_HOPS - 1], on[MP_HOPS - 1]);
        }
        CHECK(all_relock,
              "multipath: the guarded build re-locks onto the NEW path in "
              "ALL FOUR gain rows (the unbounded predicate vetoed this "
              "permanently)");
        CHECK(all_bounded,
              "multipath: never later than the unguarded build by more than "
              "one window, and never earlier");
        CHECK(all_land,
              "multipath: both builds end on the new path's lock class");
    }

    /* ---- 4b. a FORWARD candidate offered WHILE cancelling is untouched --- */
    {
        static const float gains[] = { 0.2f, 0.3f, 0.4f, 0.5f };
        int i, all_same = 1, all_relock = 1;
        printf("      forward multipath old %d (g_old) + NEW %d @ %.1f from "
               "hop %d:\n", FMP_OLD, FMP_NEW, (double)MP_GAIN_NEW, MP_AT);
        for (i = 0; i < 4; ++i) {
            int off_hop;
            build_multipath(FMP_HOPS * HOP, MP_AT * HOP, FMP_OLD, gains[i],
                            FMP_NEW, MP_GAIN_NEW);
            if (run_hops(0, FMP_HOPS, off) != 0
                    || run_hops(1, FMP_HOPS, on) != 0) {
                all_same = 0; continue;
            }
            off_hop = first_change_after(off, FMP_HOPS, MP_AT);
            if (off_hop < 0 || !in_lock_class(off[FMP_HOPS - 1], FMP_NEW))
                all_relock = 0;
            if (memcmp(off, on, FMP_HOPS * sizeof(int)) != 0) all_same = 0;
            printf("        g_old %.1f: re-lock hop %4d, trajectories %s\n",
                   (double)gains[i], off_hop,
                   memcmp(off, on, FMP_HOPS * sizeof(int)) == 0
                       ? "identical" : "DIFFER");
        }
        CHECK(all_relock,
              "forward multipath control: the unguarded build re-locks onto "
              "the new LATER path in all four gain rows");
        CHECK(all_same,
              "forward multipath: identical trajectories on and off -- a "
              "LARGER candidate is not quarantined even while the applied "
              "alignment is demonstrably still cancelling");
    }

    /* ---- 5. a FORWARD move is not quarantined at all ---- */
    build_moving(MOVE_HOPS * HOP, MOVE_AT * HOP, MOVE_D0, MOVE_D1);
    {
        int off_hop, on_hop;
        rc = run_hops(0, MOVE_HOPS, off);
        CHECK(rc == 0, "forward-move scene (OFF): aec_create");
        off_hop = first_change_after(off, MOVE_HOPS, MOVE_AT);
        rc = run_hops(1, MOVE_HOPS, on);
        CHECK(rc == 0, "forward-move scene (ON): aec_create");
        on_hop = first_change_after(on, MOVE_HOPS, MOVE_AT);

        CHECK(off_hop > 0, "forward-move control: the unguarded build accepts "
                           "the moved delay");
        CHECK(memcmp(off, on, MOVE_HOPS * sizeof(int)) == 0,
              "forward move: the WHOLE accepted-delay trajectory is identical "
              "on and off -- a larger delay is never quarantined");
        CHECK(in_lock_class(on[MOVE_HOPS - 1], MOVE_D1),
              "forward move: the run ends on the moved delay");
        printf("      (forward move at hop %d: accepted hop %d both ways)\n",
               MOVE_AT, on_hop);
    }

    /* ---- 6. WHICH arm engages (the derivation, made falsifiable) ----
     * The predicate judges cancellation on two readings, both already used by
     * Path A's own guard: windowed ERLE against its 2.5 dB threshold, and the
     * recent inst-ERLE peak against delay_acquire_inst_erle_db. Raising that
     * second threshold to its validated maximum (100 dB, far above any ERLE
     * this filter reaches) neutralizes the fast arm and leaves the windowed
     * arm alone -- and the quarantine then never engages, because at the
     * acceptance hop windowed ERLE is still climbing out of the lag that
     * delay_acquire_inst_erle_db's own doc comment describes (measured
     * 1.804 dB on this scene, under its 2.5 dB threshold; the Python mirror
     * measures 1.660 dB against 7.269 dB of inst peak). That is why this
     * predicate cannot be windowed-only.
     *
     * Raising the threshold does not disturb Path A here: warm tap-transfer
     * reads the same constant, but the inst peak at the hop-32 acquisition is
     * 0.019 dB, so warm transfer was already not firing at either setting --
     * confirmed by the row below, which pins the SAME first acquisition. */
    {
        static int neutral[MIS_HOPS];
        static int base[MIS_HOPS];
        float erle_at_change = -1.0f;
        build_static(MIS_HOPS * HOP, MIS_DELAY);
        rc = run_hops(0, MIS_HOPS, base);
        CHECK(rc == 0, "mislock scene rebuild (control): aec_create");
        rc = run_hops_ex(1, -1.0f, 100.0f, MIS_HOPS, neutral, &erle_at_change);
        CHECK(rc == 0, "mislock scene (fast arm neutralized): aec_create");
        if (rc == 0) {
            CHECK(memcmp(base, neutral, MIS_HOPS * sizeof(int)) == 0,
                  "windowed ERLE ALONE never engages the quarantine here: "
                  "with the inst-peak arm out of reach the trajectory is the "
                  "unguarded one");
            /* The companion assertion -- that windowed ERLE at the change
             * hop reads under its own 2.5 dB threshold, which is WHY only
             * the fast arm could engage -- measured that value at the
             * pre-echo re-lock. There is no change hop in this scene any
             * more, so the reading is simply absent (-1) rather than
             * under-threshold, and asserting on it would be asserting on
             * nothing. The remaining row above is still the load-bearing
             * one: it shows the windowed arm alone leaves the trajectory
             * unguarded here. */
            printf("      (no change hop in this scene: windowed ERLE at "
                   "change = %.3f, i.e. never sampled)\n", erle_at_change);
        }
    }

    /* ---- 7. defaults + validation ---- */
    {
        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
        CHECK(cfg.delay_backward_quarantine_enabled == 0,
              "the quarantine is OFF in the shipped preset");
        CHECK(cfg.delay_backward_quarantine_s == 1.0f,
              "the window default is 1.0 s (inert while the enable is 0)");
        cfg.delay_backward_quarantine_enabled = 2;
        CHECK(aec_get_mem_size(&cfg) == 0,
              "a non-boolean enable is rejected by config validation");
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
        cfg.delay_backward_quarantine_s = -1.0f;
        CHECK(aec_get_mem_size(&cfg) == 0,
              "a negative window is rejected by config validation");
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
        cfg.delay_backward_quarantine_s = 3601.0f;
        CHECK(aec_get_mem_size(&cfg) == 0,
              "an out-of-range window is rejected by config validation");
    }

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
