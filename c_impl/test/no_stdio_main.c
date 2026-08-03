/* no_stdio_main.c — minimal-main ELF gate for AEC_NO_STDIO.
 *
 * Deliberately does NOT itself use stdio (no <stdio.h>, no printf/fprintf
 * anywhere in this file) so that `nm` run over the linked executable can
 * only attribute any stdio symbol it finds to libaec.a (built with
 * NO_STDIO=1), the audio_common archive it is linked against, or the
 * platform CRT — never to this harness. The only observable output is the
 * process exit code: exercise one real hop through the engine
 * (aec_create -> aec_process -> aec_destroy) and report success/failure
 * via the return value.
 *
 * Exit codes:
 *   0  -> aec_create + one hop of aec_process + aec_destroy all succeeded
 *   1  -> aec_create failed
 *   2  -> aec_hop_size() returned an unusable value
 *
 * Build/run: see the `audit-no-stdio` Makefile target (../Makefile), which
 * compiles this against a NO_STDIO=1 libaec.a + the resolved audio_common
 * archive, `nm`s the result, and runs it.
 */
#include "aec.h"
#include <string.h>

int main(void) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);

    Aec a;
    if (aec_create(&a, &cfg) != 0) return 1;

    int hop = aec_hop_size(&a);
    if (hop <= 0 || hop > 4096) { aec_destroy(&a); return 2; }

    float mic[4096];
    float ref[4096];
    float out[4096];
    memset(mic, 0, sizeof(mic));
    memset(ref, 0, sizeof(ref));

    aec_process(&a, mic, ref, out);

    aec_destroy(&a);
    return 0;
}
