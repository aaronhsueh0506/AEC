/* C++ header check, compile only: aec.h must be usable from C++ with both spellings of the level. An enumerator
 * does not convert to an enum type of another declaration in C++, so every legacy AEC_PRESET_* name has to belong to
 * AecLevel itself. */
#include "aec.h"

static_assert(AEC_PRESET_MILD == AEC_LEVEL_MILD && AEC_PRESET_BALANCED == AEC_LEVEL_BALANCED &&
              AEC_PRESET_AGGRESSIVE == AEC_LEVEL_AGGRESSIVE && AEC_PRESET_NO_FLOOR == AEC_LEVEL_NO_FLOOR,
              "legacy AEC_PRESET_* values must equal AEC_LEVEL_*");

int aec_header_cpp_check(Aec* a, AecConfig* cfg) {
    AecLevel level = AEC_PRESET_BALANCED;
    AecPreset preset = AEC_PRESET_BALANCED;
    aec_config_from_level(cfg, AEC_PRESET_AGGRESSIVE, 16000);
    return aec_set_level(a, level, 0.0f) + aec_set_level(a, preset, 0.0f) +
           aec_set_level(a, AEC_PRESET_NO_FLOOR, 0.0f);
}
