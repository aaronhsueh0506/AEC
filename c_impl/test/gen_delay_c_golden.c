/* gen_delay_c_golden.c — write a C-REGRESSION golden for the delay estimator.
 *
 * The delay chain's matched-filter arithmetic is now hardcoded to the
 * float32/sliding-x2/NEON path and the duty-cycle analysis schedule is
 * always on (see delay_aec3.c's "matched-filter arithmetic" note and
 * aec.c's duty-cycle state machine) — both are INTENTIONAL, sampled-cost-
 * free divergences from the old Python float64 reference. The old
 * python/diag/gen_delay_golden.py golden (near/far audio + Python
 * LegacyDelayShim's expected outputs) is therefore no longer a meaningful
 * bit-exact target: comparing today's C against it will show routine
 * mismatches that are not bugs.
 *
 * This tool replaces that anchor with a same-format, C-produced-and-
 * consumed golden: it runs the CURRENT delay_aec3_accumulate() over the
 * same raw (unaligned) doubletalk case gen_delay_golden.py used, and
 * records THIS run's own outputs as "expected". test/parity_delay.c then
 * replays the golden and checks bit-exact reproducibility hop-by-hop. This
 * catches accidental regressions in delay_aec3.c (any change that alters
 * the recorded trace fails the check) — it no longer proves Python parity.
 * Regenerate intentionally (re-run this tool) whenever a deliberate change
 * to the delay estimator changes its output trace.
 *
 * Build + run (from c_impl/, standalone — does NOT link aec.c):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       src/delay_aec3.c test/gen_delay_c_golden.c -lm -o /tmp/gen_delay_golden
 *   /tmp/gen_delay_golden /tmp/delay_golden.bin \
 *       ../wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk
 *
 * M5 (multi-rate campaign, review F01): the delay chain's own block size
 * (DA_AEC3_BLOCK_SIZE=64, /4 decimation) is fixed regardless of input sample
 * rate -- the accumulate() API just buffers whatever hop it is handed into
 * that fixed-size block (see delay_aec3.h/.c), so this generator only needs
 * a per-rate HOP value, not a rebuild. Optional 3rd argv: hop size (default
 * 160, i.e. 16 kHz's 10ms hop). Pass 80 for 8 kHz / 480 for 48 kHz (10ms
 * hop at each rate) against that rate's own resampled WAV pair:
 *   /tmp/gen_delay_golden /tmp/delay_golden_8k.bin  <8k case stem>  80
 *   /tmp/gen_delay_golden /tmp/delay_golden_48k.bin <48k case stem> 480
 *
 * Binary layout (LE) — identical to gen_delay_golden.py so test/parity_delay.c
 * needs no changes:
 *   int32   hop, n_hops
 *   per hop:
 *     float32 near[hop]
 *     float32 far[hop]
 *     int32   estimated_delay
 *     int32   n_updates
 *     int32   is_solid
 *     float64 confidence
 */
#include "delay_aec3.h"
#include "wav_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HOP_DEFAULT 160
#define HOP_MAX     4096   /* generous ceiling (48 kHz's 480 is well under this) */

int main(int argc, char **argv) {
    const char *out_path = argc > 1 ? argv[1]
        : "/tmp/delay_golden.bin";
    const char *case_stem = argc > 2 ? argv[2]
        : "../wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk";
    int hop = argc > 3 ? atoi(argv[3]) : HOP_DEFAULT;
    if (hop <= 0 || hop > HOP_MAX) {
        fprintf(stderr, "bad hop %d (must be 1..%d)\n", hop, HOP_MAX);
        return 2;
    }

    char mic_path[1024], lpb_path[1024];
    snprintf(mic_path, sizeof(mic_path), "%s_mic.wav", case_stem);
    snprintf(lpb_path, sizeof(lpb_path), "%s_lpb.wav", case_stem);

    WavReader *mr = wav_open_read(mic_path);
    WavReader *rr = wav_open_read(lpb_path);
    if (!mr || !rr) {
        fprintf(stderr, "cannot open '%s' / '%s'\n", mic_path, lpb_path);
        return 2;
    }
    int n = mr->info.num_samples < rr->info.num_samples
          ? mr->info.num_samples : rr->info.num_samples;
    int n_hops = n / hop;

    FILE *f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_path); return 2; }
    int32_t hdr[2] = { hop, n_hops };
    fwrite(hdr, sizeof(int32_t), 2, f);

    DelayAec3 d;
    delay_aec3_init(&d, 16000);

    float near[HOP_MAX], far[HOP_MAX];
    int solid_hops = 0, max_nupd = 0;
    int32_t delay_min = 0, delay_max = 0;
    int have_delay_range = 0;

    for (int i = 0; i < n_hops; ++i) {
        wav_read_float(mr, near, hop);
        wav_read_float(rr, far, hop);

        delay_aec3_accumulate(&d, near, far, hop);

        int32_t est_delay = (int32_t)delay_aec3_estimated_delay(&d);
        int32_t nupd       = (int32_t)delay_aec3_n_updates(&d);
        int32_t solid      = (int32_t)delay_aec3_is_solid(&d);
        double  conf       = delay_aec3_confidence(&d);

        fwrite(near, sizeof(float), (size_t)hop, f);
        fwrite(far,  sizeof(float), (size_t)hop, f);
        int32_t rowhdr[3] = { est_delay, nupd, solid };
        fwrite(rowhdr, sizeof(int32_t), 3, f);
        fwrite(&conf, sizeof(double), 1, f);

        if (solid) solid_hops++;
        if (nupd > max_nupd) max_nupd = nupd;
        if (!have_delay_range) { delay_min = delay_max = est_delay; have_delay_range = 1; }
        if (est_delay < delay_min) delay_min = est_delay;
        if (est_delay > delay_max) delay_max = est_delay;
    }

    fclose(f);
    wav_close_read(mr);
    wav_close_read(rr);

    printf("wrote %s (%d hops, hop=%d) — C-regression golden (post fast-math+duty)\n",
           out_path, n_hops, hop);
    printf("  estimated_delay: min=%d max=%d\n", delay_min, delay_max);
    printf("  solid hops=%d  max n_updates=%d\n", solid_hops, max_nupd);
    return 0;
}
