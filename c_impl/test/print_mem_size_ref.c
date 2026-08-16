/* print_mem_size_ref.c — independent reference tool backing
 * cli_delay_flags.sh's "aec_wav --print-mem-size's number equals
 * aec_get_mem_size()'s" cross-check (productization plan §3.4.6 RAM
 * acceptance tests 6/9).
 *
 * Deliberately NOT part of aec_wav.c / example/: this tool builds an
 * AecConfig from the same CLI-shaped inputs and calls
 * aec_get_mem_breakdown() (which itself calls aec_get_mem_size()) from its
 * OWN source file, so the shell test's cross-check runs against a genuinely
 * separate code path from aec_wav.c's --print-mem-size printf. A mutation
 * that breaks only the CLI's print statement -- wrong field, transposed
 * arguments, a stale hardcoded number -- diverges from this tool's number
 * and fails the shell test; a mutation that breaks aec_get_mem_size() /
 * aec_get_mem_breakdown() itself would make both sides equally wrong (that
 * class of regression is caught instead by test_delay_num_filters.c, which
 * pins hand-computed byte deltas per grid/n, and by test_config_validation.c).
 *
 * Usage:
 *   print_mem_size_ref <sample_rate> [--fft-size N] [--delay-mode M]
 *                       [--delay-num-filters N] [--fixed-delay S]
 * (M in {matched,fixed,external}; unset options leave the balanced-preset
 * default exactly as aec_wav.c's own CLI does.)
 *
 * On success prints exactly one line:
 *   total_bytes=N estimator_bytes=N ring_bytes=N
 * and exits 0. On a rejected config, prints nothing to stdout and exits 1 --
 * the shell test's fail-fast checks look for THAT exit code, not stdout
 * content, mirroring aec_wav's own contract for an illegal combination.
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
