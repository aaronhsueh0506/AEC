/* cli_level.h -- strict --level argument of the example and test programs. */
#ifndef AEC_EXAMPLE_CLI_LEVEL_H
#define AEC_EXAMPLE_CLI_LEVEL_H

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>

#include "aec.h"

/* Parses the whole string as a decimal AecLevel number (0 mild, 1 balanced, 2 aggressive,
 * 3 no_floor). Returns 0 and writes *out, or -1 (nothing written) for NULL, an empty or padded
 * string, trailing characters, or a value outside 0..3. */
static inline int aec_cli_parse_level(const char* text, AecLevel* out) {
    char* end = NULL;
    long v;
    if (!text || *text == '\0' || isspace((unsigned char)*text)) return -1;
    errno = 0;
    v = strtol(text, &end, 10);
    if (errno != 0 || *end != '\0' || v < 0 || v > 3) return -1;
    *out = (AecLevel)v;
    return 0;
}

#endif
