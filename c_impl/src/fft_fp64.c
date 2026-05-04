/* fft_fp64.c — fp64 radix-2 FFT replacement for kiss_fft, providing same
 * fft_wrapper.h interface. Uses double precision internally to match
 * numpy's pocketfft (which is fp64-internal even for fp32 inputs).
 *
 * Single drop-in replacement for `c_impl/src/fft_wrapper.c`. Build only
 * one of them — compiling both produces duplicate symbols.
 */
#include "fft_wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI_FFT
#define M_PI_FFT 3.14159265358979323846
#endif

struct FftHandle {
    int     fft_size;
    int     n_freqs;
    int     log2n;
    int*    bit_rev;        /* [fft_size] reversed-bit indices */
    double* tw_re;          /* [fft_size/2] precomputed twiddles */
    double* tw_im;
    double* in_re;          /* [fft_size] input scratch */
    double* in_im;          /* [fft_size] */
    int     is_static;      /* 1 if placed via fft_init (no free in destroy) */
};

static int ilog2(int n) { int k = 0; while ((1 << k) < n) k++; return k; }

/* Fill twiddles + bit-rev table after struct fields are valid. */
static void fft_compute_tables(FftHandle* h) {
    for (int i = 0; i < h->fft_size; ++i) {
        int x = i, y = 0;
        for (int j = 0; j < h->log2n; ++j) { y = (y << 1) | (x & 1); x >>= 1; }
        h->bit_rev[i] = y;
    }
    for (int k = 0; k < h->fft_size / 2; ++k) {
        double ang = -2.0 * M_PI_FFT * (double)k / (double)h->fft_size;
        h->tw_re[k] = cos(ang);
        h->tw_im[k] = sin(ang);
    }
    memset(h->in_re, 0, h->fft_size * sizeof(double));
    memset(h->in_im, 0, h->fft_size * sizeof(double));
}

FftHandle* fft_create(int fft_size) {
    if (fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) return NULL;
    FftHandle* h = (FftHandle*)calloc(1, sizeof(FftHandle));
    if (!h) return NULL;
    h->fft_size = fft_size;
    h->n_freqs  = fft_size / 2 + 1;
    h->log2n    = ilog2(fft_size);
    h->is_static = 0;

    h->bit_rev = (int*)malloc(fft_size * sizeof(int));
    h->tw_re   = (double*)malloc((fft_size / 2) * sizeof(double));
    h->tw_im   = (double*)malloc((fft_size / 2) * sizeof(double));
    h->in_re   = (double*)calloc(fft_size, sizeof(double));
    h->in_im   = (double*)calloc(fft_size, sizeof(double));
    fft_compute_tables(h);
    return h;
}

size_t fft_get_mem_size(int fft_size) {
    if (fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) return 0;
    size_t total = 0;
    total += ALIGN16(sizeof(FftHandle));
    total += ALIGN16(fft_size * sizeof(int));        /* bit_rev */
    total += ALIGN16((fft_size / 2) * sizeof(double));  /* tw_re */
    total += ALIGN16((fft_size / 2) * sizeof(double));  /* tw_im */
    total += ALIGN16(fft_size * sizeof(double));     /* in_re */
    total += ALIGN16(fft_size * sizeof(double));     /* in_im */
    return total;
}

FftHandle* fft_init(void* mem, size_t mem_size, int fft_size) {
    if (!mem || fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) return NULL;
    if (mem_size < fft_get_mem_size(fft_size)) return NULL;

    uint8_t* ptr = (uint8_t*)mem;
    FftHandle* h = (FftHandle*)ptr;
    ptr += ALIGN16(sizeof(FftHandle));
    memset(h, 0, sizeof(FftHandle));

    h->fft_size  = fft_size;
    h->n_freqs   = fft_size / 2 + 1;
    h->log2n     = ilog2(fft_size);
    h->is_static = 1;

    h->bit_rev = (int*)ptr;    ptr += ALIGN16(fft_size * sizeof(int));
    h->tw_re   = (double*)ptr; ptr += ALIGN16((fft_size / 2) * sizeof(double));
    h->tw_im   = (double*)ptr; ptr += ALIGN16((fft_size / 2) * sizeof(double));
    h->in_re   = (double*)ptr; ptr += ALIGN16(fft_size * sizeof(double));
    h->in_im   = (double*)ptr; /* last field — no further increment needed */

    fft_compute_tables(h);
    return h;
}

void fft_destroy(FftHandle* h) {
    if (!h) return;
    if (h->is_static) return;  /* static path: caller owns the buffer */
    free(h->bit_rev); free(h->tw_re); free(h->tw_im);
    free(h->in_re); free(h->in_im);
    free(h);
}

int fft_get_size(const FftHandle* h)    { return h ? h->fft_size : 0; }
int fft_get_n_freqs(const FftHandle* h) { return h ? h->n_freqs  : 0; }

/* Iterative radix-2 Cooley-Tukey FFT (in-place on (re,im)).
 * inverse=0 → forward (sign -). inverse=1 → backward (sign +). */
static void radix2_fft(double* re, double* im, int n, int log2n,
                       const double* tw_re, const double* tw_im,
                       const int* bit_rev, int inverse) {
    /* bit-reverse permute */
    for (int i = 0; i < n; ++i) {
        int j = bit_rev[i];
        if (j > i) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    /* butterflies */
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        int twiddle_step = n / m;
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m2; ++j) {
                int tw_idx = j * twiddle_step;
                double w_re = tw_re[tw_idx];
                double w_im = inverse ? -tw_im[tw_idx] : tw_im[tw_idx];
                double t_re = w_re * re[k + j + m2] - w_im * im[k + j + m2];
                double t_im = w_re * im[k + j + m2] + w_im * re[k + j + m2];
                double u_re = re[k + j];
                double u_im = im[k + j];
                re[k + j]      = u_re + t_re;
                im[k + j]      = u_im + t_im;
                re[k + j + m2] = u_re - t_re;
                im[k + j + m2] = u_im - t_im;
            }
        }
    }
}

void fft_forward(FftHandle* h, const float* real_in, Complex* complex_out) {
    if (!h || !real_in || !complex_out) return;
    int n = h->fft_size;
    for (int i = 0; i < n; ++i) {
        h->in_re[i] = (double)real_in[i];
        h->in_im[i] = 0.0;
    }
    radix2_fft(h->in_re, h->in_im, n, h->log2n, h->tw_re, h->tw_im,
               h->bit_rev, 0);
    for (int k = 0; k < h->n_freqs; ++k) {
        complex_out[k].r = (float)h->in_re[k];
        complex_out[k].i = (float)h->in_im[k];
    }
}

void fft_inverse(FftHandle* h, const Complex* complex_in, float* real_out) {
    if (!h || !complex_in || !real_out) return;
    int n = h->fft_size;
    int n_freqs = h->n_freqs;
    for (int k = 0; k < n_freqs; ++k) {
        h->in_re[k] = (double)complex_in[k].r;
        h->in_im[k] = (double)complex_in[k].i;
    }
    /* conjugate-symmetric reflection */
    for (int k = 1; k < n_freqs - 1; ++k) {
        h->in_re[n - k] =  (double)complex_in[k].r;
        h->in_im[n - k] = -(double)complex_in[k].i;
    }
    radix2_fft(h->in_re, h->in_im, n, h->log2n, h->tw_re, h->tw_im,
               h->bit_rev, 1);
    double inv_n = 1.0 / (double)n;
    for (int i = 0; i < n; ++i) real_out[i] = (float)(h->in_re[i] * inv_n);
}

/* Unchanged helpers */
void fft_magnitude(const Complex* sp, float* mag, int K) {
    for (int k = 0; k < K; ++k) {
        float re = sp[k].r, im = sp[k].i;
        mag[k] = sqrtf(re * re + im * im);
    }
}
void fft_power(const Complex* sp, float* pw, int K) {
    for (int k = 0; k < K; ++k) {
        float re = sp[k].r, im = sp[k].i;
        pw[k] = re * re + im * im;
    }
}
void fft_phase(const Complex* sp, float* ph, int K) {
    for (int k = 0; k < K; ++k) ph[k] = atan2f(sp[k].i, sp[k].r);
}
void fft_from_mag_phase(const float* m, const float* p, Complex* sp, int K) {
    for (int k = 0; k < K; ++k) {
        sp[k].r = m[k] * cosf(p[k]);
        sp[k].i = m[k] * sinf(p[k]);
    }
}
void fft_apply_gain(Complex* sp, const float* g, int K) {
    for (int k = 0; k < K; ++k) { sp[k].r *= g[k]; sp[k].i *= g[k]; }
}
