/**
 * wav_io.h - AEC thin shim onto the shared, hardened WAV I/O (F06 remediation)
 *
 * The actual reader/writer implementation moved to the single canonical
 * audio_common/include/wav_io.h (hardened parser + shared writer -- see
 * that file's header comment for the full rationale: fmt-chunk-size
 * validation, format/channels/bits/sample_rate/block_align/byte_rate
 * checks, RIFF odd-chunk pad-byte handling, file-size bounds checks on
 * every chunk_sz, and float32 NaN/Inf sanitize on read).
 *
 * This shim exists so that aec_wav.c's (and this repo's test-harness
 * sources') `#include "wav_io.h"` keeps resolving with zero source
 * changes, and to pin AEC's historical writer behavior:
 *   - PCM16 output by default; the AEC_OUT_FLOAT=1 env var switches to
 *     raw, unquantized IEEE float32 output (a test-only path, see
 *     wav_open_write / wav_write_float in the canonical header).
 *   - PCM16 quantization is round-half-away-from-zero.
 * (1 == WAV_IO_WRITER_AEC in audio_common/include/wav_io.h; duplicated as
 * a literal here because that symbolic name isn't defined until AFTER the
 * #include below -- see the canonical header's WAV_IO_WRITER_STYLE doc.)
 */
#ifndef AEC_WAV_IO_SHIM_H
#define AEC_WAV_IO_SHIM_H
/* NOTE: deliberately NOT guarded as WAV_IO_H -- that guard belongs to the
 * canonical audio_common/include/wav_io.h included below. Reusing it here
 * would make the canonical #include below a silent no-op (its own
 * #ifndef WAV_IO_H would see the guard already "defined" by this file and
 * skip its entire body), exactly like the #include_next bug documented
 * below -- just via guard-name collision instead of search-path restart. */

#ifndef WAV_IO_WRITER_STYLE
#define WAV_IO_WRITER_STYLE 1  /* WAV_IO_WRITER_AEC */
#endif

/* Locate audio_common/include/wav_io.h with an explicit relative path,
 * resolved with __has_include the same way this repo's own Makefile
 * resolves AC_DIR (`$(wildcard ../../audio_common ../../../../audio_common)`
 * from c_impl/ -- one directory deeper here since this file lives in
 * c_impl/example/, not c_impl/ itself): the first candidate is the normal
 * sibling-repo checkout (SE/AEC next to SE/audio_common), the second is
 * this repo vendored two levels deeper as an Audio_ALG submodule
 * (Audio_ALG/lib/aec/c_impl/example -> SE/audio_common).
 *
 * #include_next was tried and rejected: when the compiler resolves
 * "wav_io.h" via the *implicit* "same directory as the including file"
 * step (true for aec_wav.c, which lives in this same example/ directory)
 * rather than via an explicit -I entry, #include_next restarts the search
 * from the top of the include path instead of continuing past it --
 * verified directly with this toolchain (clang emits "#include_next in
 * file found relative to primary source file ... will search from start
 * of include path"), which re-finds this very shim and, thanks to the
 * header guard, makes the #include_next a silent no-op -- the canonical
 * header's contents never actually appear. The explicit relative-path
 * __has_include below has no such ambiguity and was verified against both
 * the sibling and the nested checkout layout.
 */
#if defined(__has_include)
#  if __has_include("../../../audio_common/include/wav_io.h")
#    include "../../../audio_common/include/wav_io.h"
#  elif __has_include("../../../../../audio_common/include/wav_io.h")
#    include "../../../../../audio_common/include/wav_io.h"
#  else
#    error "wav_io.h: cannot locate audio_common/include/wav_io.h -- expected it as a sibling of this repo (SE/audio_common) or two levels up from an Audio_ALG submodule checkout (Audio_ALG/lib/aec -> SE/audio_common)"
#  endif
#else
#  error "wav_io.h: compiler lacks __has_include -- add an explicit #include for audio_common/include/wav_io.h"
#endif

#endif // AEC_WAV_IO_SHIM_H
