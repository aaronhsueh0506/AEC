/* C++ header check, compile only: aec.h, aec_internal.h and aec_debug.h must be usable from C++ with the level enumerators
 * and the public names of values the library keeps internally. */
#include "aec.h"
#include "aec_internal.h"
#include "aec_debug.h"

static_assert(AEC_LEVEL_MILD == 0 && AEC_LEVEL_BALANCED == 1 && AEC_LEVEL_AGGRESSIVE == 2 && AEC_LEVEL_NO_FLOOR == 3,
              "AEC_LEVEL_* are the CLI digits 0..3");

static_assert(AEC_DELAY_NUM_FILTERS_MAX == 5 && AEC_DELAY_FILTER_SIZE > 0 && AEC_DELAY_FILTER_INTRA_SHIFT > 0 &&
              AEC_DELAY_DOWN_SAMPLING_FACTOR > 0 && AEC_CNG_SEED == 42u && AEC_PSD_SCALE_F * AEC_PSD_SCALE_INV_F == 1.0f,
              "the public names of the delay geometry, the comfort-noise seed and the PSD scale");

int aec_header_cpp_check(Aec* a, AecConfig* cfg) {
    AecLevel level = AEC_LEVEL_BALANCED;
    float db = 0.0f;
    aec_config_from_level(cfg, AEC_LEVEL_AGGRESSIVE, 16000);
    AecDebugStatus st;
    aec_debug_status(a, &st);
    return aec_set_level(a, level) + aec_set_level(a, AEC_LEVEL_NO_FLOOR) +
           aec_level_floor_db(AEC_LEVEL_MILD, &db) + st.n_partitions + st.delay_period_hops +
           (aec_level_name(level)[0] == 'b');
}
