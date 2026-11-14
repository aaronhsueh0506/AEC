/* C++ header check: aec.h must be usable from C++ with the level enumerators and the public names of values the library
 * keeps internally. Built twice by `make test-header-cpp`: without AEC_HEADER_CPP_RUN it is compile-only (-fsyntax-only, the
 * API calls below); with it, the value checks run as a program that needs no library archive. */
#include "aec.h"

#ifdef AEC_HEADER_CPP_RUN
#include <cstdio>

#define CPP_EXPECT(cond, what) do { if (!(cond)) { std::fprintf(stderr, "test_header_cpp: FAILED: %s\n", (what)); bad = 1; } } while (0)

int main() {
    int bad = 0;
    CPP_EXPECT(AEC_LEVEL_MILD == 0 && AEC_LEVEL_BALANCED == 1 && AEC_LEVEL_AGGRESSIVE == 2 && AEC_LEVEL_NO_FLOOR == 3,
               "AEC_LEVEL_* are the CLI digits 0..3");
    CPP_EXPECT(AEC_DELAY_NUM_FILTERS_MAX == 5 && AEC_DELAY_FILTER_SIZE > 0 && AEC_DELAY_FILTER_INTRA_SHIFT > 0 &&
               AEC_DELAY_DOWN_SAMPLING_FACTOR > 0, "the public names of the delay geometry");
    CPP_EXPECT(AEC_CNG_SEED == 42u, "the public name of the comfort-noise seed");
    volatile float scale = AEC_PSD_SCALE_F;
    CPP_EXPECT(scale * AEC_PSD_SCALE_INV_F == 1.0f, "the PSD scale and its inverse multiply to exactly 1");
    if (!bad) std::printf("test_header_cpp: AEC_LEVEL_* digits, delay geometry names, AEC_CNG_SEED and PSD scale hold in C++\n");
    return bad;
}
#else
int aec_header_cpp_check(Aec* a, AecConfig* cfg) {
    AecLevel level = AEC_LEVEL_BALANCED;
    aec_config_from_level(cfg, AEC_LEVEL_AGGRESSIVE, 16000);
    AecDebugStatus st;
    aec_debug_status(a, &st);
    return aec_set_level(a, level) + aec_set_level(a, AEC_LEVEL_NO_FLOOR) +
           st.n_partitions + st.delay_period_hops +
           (aec_level_name(level)[0] == 'b');
}
#endif
