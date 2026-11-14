/** wav_io.h -- thin shim onto the shared WAV reader/writer in audiolib/common/include/wav_io.h.
 * Keeps `#include "wav_io.h"` resolving. The writer style is the shared header's default (AEC style): PCM16 output by
 * default, AEC_OUT_FLOAT=1 selects raw float32, PCM16 quantization rounds half away from zero. */
#ifndef AEC_WAV_IO_SHIM_H
#define AEC_WAV_IO_SHIM_H
/* The guard is deliberately not WAV_IO_H, which belongs to the shared header included below. */

/* Locate the shared header by explicit relative path: the sibling checkout (SE/audiolib/common)
 * first, then two levels deeper (Audio_ALG/lib/aec/c_impl/example). */
#if defined(__has_include)
#  if __has_include("../../../audiolib/common/include/wav_io.h")
#    include "../../../audiolib/common/include/wav_io.h"
#  elif __has_include("../../../../../audiolib/common/include/wav_io.h")
#    include "../../../../../audiolib/common/include/wav_io.h"
#  else
#    error "wav_io.h: cannot locate audiolib/common/include/wav_io.h -- expected it as a sibling of this repo (SE/audiolib/common) or two levels up from an Audio_ALG submodule checkout (Audio_ALG/lib/aec -> SE/audiolib/common)"
#  endif
#else
#  error "wav_io.h: compiler lacks __has_include -- add an explicit #include for audiolib/common/include/wav_io.h"
#endif

#endif
