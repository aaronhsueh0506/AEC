/* res_gain_of.h -- the G_res pointer of the last completed hop, read through aec_get_residual(); NULL when the view is refused
 * or the instance is below AEC_RESIDUAL_FULL. */
#ifndef AEC_TEST_RES_GAIN_OF_H
#define AEC_TEST_RES_GAIN_OF_H

#include <string.h>
#include "aec.h"

static inline const float* res_gain_of(const Aec* a) {
    AecResidualView v;
    memset(&v, 0, sizeof v);
    v.struct_size = sizeof v;
    return aec_get_residual(a, &v) == AEC_OK ? v.res_gain : NULL;
}

#endif
