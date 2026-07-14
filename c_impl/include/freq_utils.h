/* freq_utils.h — frequency <-> FFT bin conversion (C port of
 * python/modules/freq_utils.py).
 *
 * Express AEC config constants in Hz and convert here, so a port stays correct
 * across fft_size / sample_rate. n_bins is the spectrum size (= fft_size/2 + 1);
 * for fft_size=512 @ 16 kHz this is 257 bins, 31.25 Hz/bin.
 *
 * Parity: Python int(round(.)) is round-half-to-even -> lrintf() here.
 * Float32-by-design (init-only helper; converted for uniformity as part of
 * the f32 campaign, drift accepted).
 */
#ifndef FREQ_UTILS_H
#define FREQ_UTILS_H

int   hz_to_bin(float hz, int n_bins, int sr);
float bin_to_hz(int bin_idx, int n_bins, int sr);

#endif /* FREQ_UTILS_H */
