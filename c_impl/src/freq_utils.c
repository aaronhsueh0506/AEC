/* freq_utils.c — C port of python/modules/freq_utils.py. */
#include "freq_utils.h"

#include <math.h>

int hz_to_bin(float hz, int n_bins, int sr) {
    int fft_size = (n_bins - 1) * 2;
    /* int(round(hz * fft_size / sr)) — round-half-to-even via lrintf */
    return (int)lrintf(hz * fft_size / (float)sr);
}

float bin_to_hz(int bin_idx, int n_bins, int sr) {
    int fft_size = (n_bins - 1) * 2;
    return bin_idx * (float)sr / fft_size;
}
