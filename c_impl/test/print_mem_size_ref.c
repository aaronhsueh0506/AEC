/* print_mem_size_ref.c -- independent reference tool backing cli_delay_flags.sh's cross-check that
 * aec_wav --print-mem-size reports the number aec_get_mem_size() returns.
 *
 * It is deliberately not part of aec_wav.c / example/: it builds an AecConfig from the same CLI-shaped inputs and calls
 * aec_get_mem_breakdown() (which calls aec_get_mem_size()) from its own source file, so the cross-check runs against a code
 * path separate from aec_wav.c's --print-mem-size printf. A bug in the CLI's print statement (wrong field, transposed
 * arguments, a stale hardcoded number) diverges from this tool's number and fails the shell test; a bug in
 * aec_get_mem_size() / aec_get_mem_breakdown() itself would make both sides equally wrong and is caught instead by
 * test_delay_num_filters.c (hand-computed byte deltas per grid/n) and test_config_validation.c.
 *
 * Usage:
 *   print_mem_size_ref <sample_rate> [--fft-size N] [--delay-mode M]
 *                       [--delay-num-filters N] [--fixed-delay S]
 * (M in {matched,fixed,external}; unset options leave the balanced-preset default, as aec_wav.c's own CLI does.)
 *
 * On success prints exactly one line:
 *   total_bytes=N estimator_bytes=N ring_bytes=N
 * and exits 0. On a rejected config, prints nothing to stdout and exits 1; the shell test's fail-fast checks look at that exit
 * code, not at stdout, mirroring aec_wav's own contract for an illegal combination.
 */
#include "aec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_delay_mode(const char* s, AecDelayMode* out) {
    if (!strcmp(s, "matched"))  { *out = AEC_DELAY_MATCHED;          return 0; }
    if (!strcmp(s, "fixed"))    { *out = AEC_DELAY_FIXED;            return 0; }
    if (!strcmp(s, "external")) { *out = AEC_DELAY_EXTERNAL_ALIGNED; return 0; }
    return -1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <sample_rate> [--fft-size N] "
                        "[--delay-mode matched|fixed|external] "
                        "[--delay-num-filters N] [--fixed-delay S]\n", argv[0]);
        return 2;
    }
    int sr = atoi(argv[1]);
    int fft_size = 0;
    AecDelayMode delay_mode = AEC_DELAY_MATCHED;  int have_delay_mode = 0;
    int delay_num_filters = 0;                    int have_delay_num_filters = 0;
    int fixed_delay_samples = 0;                  int have_fixed_delay = 0;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (!strcmp(arg, "--fft-size") && i + 1 < argc) {
            fft_size = atoi(argv[++i]);
        } else if (!strcmp(arg, "--delay-mode") && i + 1 < argc) {
            if (parse_delay_mode(argv[++i], &delay_mode) != 0) {
                fprintf(stderr, "ERROR: unknown --delay-mode '%s'\n", argv[i]);
                return 2;
            }
            have_delay_mode = 1;
        } else if (!strcmp(arg, "--delay-num-filters") && i + 1 < argc) {
            delay_num_filters = atoi(argv[++i]); have_delay_num_filters = 1;
        } else if (!strcmp(arg, "--fixed-delay") && i + 1 < argc) {
            fixed_delay_samples = atoi(argv[++i]); have_fixed_delay = 1;
        } else {
            fprintf(stderr, "ERROR: unknown option '%s'\n", arg);
            return 2;
        }
    }

    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
    if (fft_size > 0)            cfg.fft_size = fft_size;
    if (have_delay_mode)         cfg.delay_mode = delay_mode;
    if (have_delay_num_filters)  cfg.delay_num_filters = delay_num_filters;
    if (have_fixed_delay)        cfg.fixed_delay_samples = fixed_delay_samples;

    AecMemBreakdown mb;
    if (!aec_get_mem_breakdown(&cfg, &mb)) {
        fprintf(stderr, "ERROR: aec_get_mem_breakdown rejected the config\n");
        return 1;
    }
    printf("total_bytes=%zu estimator_bytes=%zu ring_bytes=%zu\n",
           mb.total_bytes, mb.estimator_bytes, mb.ring_bytes);
    return 0;
}
