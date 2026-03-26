/**
 * fft_wrapper.h - FFT wrapper interface
 *
 * Abstracts FFT implementation (KISS FFT by default)
 * Provides real-to-complex FFT for audio processing
 */

#ifndef FFT_WRAPPER_H
#define FFT_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 16-byte alignment for buffer placement (static memory) */
#ifndef ALIGN16
#define ALIGN16(x) (((size_t)(x) + 15u) & ~(size_t)15u)
#endif

// Complex number structure
typedef struct {
    float r;  // Real part
    float i;  // Imaginary part
} Complex;

// Opaque FFT handle
typedef struct FftHandle FftHandle;

/** Create FFT handle (malloc version) */
FftHandle* fft_create(int fft_size);

/** Initialize FFT handle in pre-allocated memory (static version) */
FftHandle* fft_init(void* mem, size_t mem_size, int fft_size);

/** Get memory required for fft_init() */
size_t fft_get_mem_size(int fft_size);

/** Destroy FFT handle (no-op if created via fft_init) */
void fft_destroy(FftHandle* handle);

/**
 * Get number of frequency bins (fft_size/2 + 1)
 */
int fft_get_n_freqs(const FftHandle* handle);

/**
 * Forward FFT: real input -> complex output
 *
 * @param handle FFT handle
 * @param time_in Real input [fft_size]
 * @param freq_out Complex output [n_freqs]
 */
void fft_forward(FftHandle* handle, const float* time_in, Complex* freq_out);

/**
 * Inverse FFT: complex input -> real output
 *
 * @param handle FFT handle
 * @param freq_in Complex input [n_freqs]
 * @param time_out Real output [fft_size]
 */
void fft_inverse(FftHandle* handle, const Complex* freq_in, float* time_out);

/**
 * Compute magnitude spectrum from complex spectrum
 *
 * @param freq Complex spectrum [n_freqs]
 * @param magnitude Output magnitude [n_freqs]
 * @param n_freqs Number of frequency bins
 */
void fft_magnitude(const Complex* freq, float* magnitude, int n_freqs);

/**
 * Compute power spectrum from complex spectrum
 *
 * @param freq Complex spectrum [n_freqs]
 * @param power Output power (magnitude^2) [n_freqs]
 * @param n_freqs Number of frequency bins
 */
void fft_power(const Complex* freq, float* power, int n_freqs);

/**
 * Apply gain to complex spectrum (in-place)
 *
 * @param freq Complex spectrum [n_freqs]
 * @param gain Gain array [n_freqs]
 * @param n_freqs Number of frequency bins
 */
void fft_apply_gain(Complex* freq, const float* gain, int n_freqs);

#ifdef __cplusplus
}
#endif

#endif // FFT_WRAPPER_H
