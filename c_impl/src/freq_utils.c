/* freq_utils.c — C port of python/modules/freq_utils.py. */
#include "freq_utils.h"

#include <math.h>

int hz_to_bin(double hz, int n_bins, int sr) {
    int fft_size = (n_bins - 1) * 2;
    /* int(round(hz * fft_size / sr)) — round-half-to-even via lrint */
    return (int)lrint(hz * fft_size / (double)sr);
}

double bin_to_hz(int bin_idx, int n_bins, int sr) {
    int fft_size = (n_bins - 1) * 2;
    return bin_idx * (double)sr / fft_size;
}
