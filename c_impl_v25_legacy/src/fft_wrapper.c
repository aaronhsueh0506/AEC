/**
 * fft_wrapper.c - FFT Wrapper Implementation using KISS FFT
 *
 * Real-to-complex FFT using KISS FFT's complex FFT internally
 */

#include "fft_wrapper.h"
#include "kiss_fft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal FFT handle structure
struct FftHandle {
    int fft_size;
    int n_freqs;            // fft_size/2 + 1

    kiss_fft_cfg fft_cfg;   // Forward FFT config
    kiss_fft_cfg ifft_cfg;  // Inverse FFT config

    kiss_fft_cpx* fft_in;   // Complex input buffer [fft_size]
    kiss_fft_cpx* fft_out;  // Complex output buffer [fft_size]
};

FftHandle* fft_create(int fft_size) {
    if (fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) {
        // fft_size must be power of 2
        return NULL;
    }

    FftHandle* h = (FftHandle*)calloc(1, sizeof(FftHandle));
    if (!h) return NULL;

    h->fft_size = fft_size;
    h->n_freqs = fft_size / 2 + 1;

    // Allocate KISS FFT configs
    h->fft_cfg = kiss_fft_alloc(fft_size, 0, NULL, NULL);  // Forward
    h->ifft_cfg = kiss_fft_alloc(fft_size, 1, NULL, NULL); // Inverse

    if (!h->fft_cfg || !h->ifft_cfg) {
        fft_destroy(h);
        return NULL;
    }

    // Allocate work buffers
    h->fft_in = (kiss_fft_cpx*)calloc(fft_size, sizeof(kiss_fft_cpx));
    h->fft_out = (kiss_fft_cpx*)calloc(fft_size, sizeof(kiss_fft_cpx));

    if (!h->fft_in || !h->fft_out) {
        fft_destroy(h);
        return NULL;
    }

    return h;
}

void fft_destroy(FftHandle* h) {
    if (!h) return;

    if (h->fft_cfg) kiss_fft_free(h->fft_cfg);
    if (h->ifft_cfg) kiss_fft_free(h->ifft_cfg);
    if (h->fft_in) free(h->fft_in);
    if (h->fft_out) free(h->fft_out);

    free(h);
}

int fft_get_size(const FftHandle* h) {
    return h ? h->fft_size : 0;
}

int fft_get_n_freqs(const FftHandle* h) {
    return h ? h->n_freqs : 0;
}

void fft_forward(FftHandle* h, const float* real_in, Complex* complex_out) {
    if (!h || !real_in || !complex_out) return;

    int n = h->fft_size;

    // Copy real input to complex buffer (imaginary = 0)
    for (int i = 0; i < n; i++) {
        h->fft_in[i].r = real_in[i];
        h->fft_in[i].i = 0.0f;
    }

    // Perform FFT
    kiss_fft(h->fft_cfg, h->fft_in, h->fft_out);

    // Copy first n_freqs bins to output
    for (int k = 0; k < h->n_freqs; k++) {
        complex_out[k].r = h->fft_out[k].r;
        complex_out[k].i = h->fft_out[k].i;
    }
}

void fft_inverse(FftHandle* h, const Complex* complex_in, float* real_out) {
    if (!h || !complex_in || !real_out) return;

    int n = h->fft_size;
    int n_freqs = h->n_freqs;

    // Reconstruct full spectrum with conjugate symmetry
    // X[k] = conj(X[N-k]) for real signals
    for (int k = 0; k < n_freqs; k++) {
        h->fft_in[k].r = complex_in[k].r;
        h->fft_in[k].i = complex_in[k].i;
    }

    // Fill conjugate symmetric part
    for (int k = 1; k < n_freqs - 1; k++) {
        h->fft_in[n - k].r = complex_in[k].r;
        h->fft_in[n - k].i = -complex_in[k].i;  // Conjugate
    }

    // Perform IFFT
    kiss_fft(h->ifft_cfg, h->fft_in, h->fft_out);

    // KISS FFT doesn't scale, so divide by N
    float scale = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
        real_out[i] = h->fft_out[i].r * scale;
    }
}

void fft_magnitude(const Complex* spectrum, float* magnitude, int n_freqs) {
    if (!spectrum || !magnitude) return;

    for (int k = 0; k < n_freqs; k++) {
        float re = spectrum[k].r;
        float im = spectrum[k].i;
        magnitude[k] = sqrtf(re * re + im * im);
    }
}

void fft_power(const Complex* spectrum, float* power, int n_freqs) {
    if (!spectrum || !power) return;

    for (int k = 0; k < n_freqs; k++) {
        float re = spectrum[k].r;
        float im = spectrum[k].i;
        power[k] = re * re + im * im;
    }
}

void fft_phase(const Complex* spectrum, float* phase, int n_freqs) {
    if (!spectrum || !phase) return;

    for (int k = 0; k < n_freqs; k++) {
        phase[k] = atan2f(spectrum[k].i, spectrum[k].r);
    }
}

void fft_from_mag_phase(const float* magnitude, const float* phase,
                        Complex* spectrum, int n_freqs) {
    if (!magnitude || !phase || !spectrum) return;

    for (int k = 0; k < n_freqs; k++) {
        spectrum[k].r = magnitude[k] * cosf(phase[k]);
        spectrum[k].i = magnitude[k] * sinf(phase[k]);
    }
}

void fft_apply_gain(Complex* spectrum, const float* gain, int n_freqs) {
    if (!spectrum || !gain) return;

    for (int k = 0; k < n_freqs; k++) {
        spectrum[k].r *= gain[k];
        spectrum[k].i *= gain[k];
    }
}
