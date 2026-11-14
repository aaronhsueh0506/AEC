/* no_stdio_main.c — minimal-main ELF gate for NO_STDIO=1.
 *
 * Deliberately does NOT itself use stdio (no <stdio.h>, no printf/fprintf
 * anywhere in this file) so that `nm` run over the linked executable can
 * only attribute any stdio symbol it finds to libaec.a (built with
 * NO_STDIO=1), the audiolib archive it is linked against, or the
 * platform CRT — never to this harness. The only observable output is the
 * process exit code: exercise one real hop through the engine
 * (aec_create -> aec_process -> aec_destroy) and report success/failure
 * via the return value.
 *
 * Exit codes:
 *   0  -> aec_create + one hop of aec_process + aec_destroy all succeeded
 *   1  -> aec_create failed
 *   2  -> aec_hop_size() returned an unusable value
 *   3  -> the RES gain stage (aec_res_internal.h) failed to initialise
 *
 * Build/run: see the `audit-no-stdio` Makefile target (../Makefile), which
 * compiles this against a NO_STDIO=1 libaec.a + the resolved audiolib
 * archive, `nm`s the result, and runs it.
 */
#include "aec.h"
#include "aec_internal.h"
#ifndef AEC_NO_ARRAY
#include "aec_res_internal.h"
#endif
#include <string.h>

int main(void) {
    AecConfig cfg;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);

    Aec *a = NULL;
    if ((a = aec_create(&cfg)) == NULL) return 1;

    int hop = aec_hop_size(a);
    if (hop <= 0 || hop > 4096) { aec_destroy(a); return 2; }

    float mic[4096];
    float ref[4096];
    float out[4096];
    memset(mic, 0, sizeof(mic));
    memset(ref, 0, sizeof(ref));

    aec_process(a, mic, ref, out);

    aec_destroy(a);

#ifndef AEC_NO_ARRAY
    {
        static Complex spec[1024];
        static float cn[1024], gr[1024];
        static uint32_t pool[4096];   /* 16 KiB, aligned up below */
        AecRes* r = NULL;
        void* base = (void*)(((uintptr_t)pool + 15u) & ~(uintptr_t)15u);
        size_t room = sizeof pool - 16u;
        if (aec_res_get_mem_size(&cfg) <= room)
            r = aec_res_init(base, room, &cfg);
        if (!r) return 3;
        {
            uint32_t rng = 1u;
            AecResInput in;
            in.error_spec = spec; in.near_spec = spec; in.r2 = cn; in.comfort_noise = cn;
            in.render_hop = cn; in.all_converged = 0; in.max_dt = 0.0f;
            in.max_saturation = 0.0f; in.clock_drift = 0;
            if (!aec_res_process(r, &in)) return 3;
            aec_cng_fill(&rng, spec, cn, gr, NULL, 0.3f, aec_res_n_bins(r));
        }
    }
#endif
    return 0;
}
