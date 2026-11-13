/* C++ header check, compile only: aec.h and aec_internal.h must be usable from C++ with the level enumerators. */
#include "aec.h"
#include "aec_internal.h"

static_assert(AEC_LEVEL_MILD == 0 && AEC_LEVEL_BALANCED == 1 && AEC_LEVEL_AGGRESSIVE == 2 && AEC_LEVEL_NO_FLOOR == 3,
              "AEC_LEVEL_* are the CLI digits 0..3");

int aec_header_cpp_check(Aec* a, AecConfig* cfg) {
    AecLevel level = AEC_LEVEL_BALANCED;
    float db = 0.0f;
    aec_config_from_level(cfg, AEC_LEVEL_AGGRESSIVE, 16000);
    return aec_set_level(a, level) + aec_set_level(a, AEC_LEVEL_NO_FLOOR) +
           aec_level_floor_db(AEC_LEVEL_MILD, &db);
}
