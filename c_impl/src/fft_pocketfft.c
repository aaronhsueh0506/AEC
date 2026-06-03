/* fft_pocketfft.c — fft_wrapper.h implementation backed by numpy 1.26.4's
 * vendored pocketfft (c_impl/lib/pocketfft/pocketfft.c).
 *
 * This is a DROP-IN replacement for the legacy fp64 radix-2 fft_fp64.c. The
 * public fft_wrapper.h API is byte-for-byte identical so every consumer
 * (aec3_post, pbfdkf, res_filter, delay_est, erle, filter_state_bridge, ...)
 * links unchanged. The point of this file is BIT-EXACT parity with numpy:
 *
 *     fft_forward(x)  ==  np.fft.rfft(x.astype(float64)).astype(complex64)
 *     fft_inverse(X)  ==  np.fft.irfft(X.astype(complex128), n).astype(float32)
 *
 * It reproduces numpy's _pocketfft.c wrapper EXACTLY:
 *
 *   execute_real_forward (np.fft.rfft, fct=1.0):
 *     - upcast real fp32 input -> fp64 (numpy: NPY_DOUBLE FORCECAST)
 *     - rfftp_forward in place over the n real samples, producing the packed
 *       half-complex layout  [r0, r1,i1, r2,i2, ..., r_{n/2}]
 *     - unpack to complex bins: DC imag = 0, Nyquist imag = 0 (even n)
 *     - downcast each fp64 -> fp32 (callers do complex128 -> complex64 .astype)
 *
 *   execute_real_backward (np.fft.irfft, fct=1.0/n):
 *     - upcast complex fp32 input -> complex128 (numpy: NPY_CDOUBLE FORCECAST)
 *     - repack to half-complex [DC, re1,im1, ..., re_{n/2}] DROPPING the DC and
 *       Nyquist imaginary parts (numpy copies only npts-1 doubles from dptr+2,
 *       i.e. it stops before the Nyquist imag slot, and writes dptr[0] as DC)
 *     - rfftp_backward in place with fct = 1.0/n (numpy normalises the inverse)
 *     - downcast fp64 -> fp32
 *
 * Build with -ffp-contract=off (mandatory for Python<->C byte-equal parity).
 *
 * Compile EXACTLY ONE of fft_pocketfft.c / fft_fp64.c — they export the same
 * symbols and would otherwise produce duplicate-definition link errors.
 */
#include "fft_wrapper.h"
#include "pocketfft.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

struct FftHandle {
    int          fft_size;
    int          n_freqs;
    rfft_plan_t  plan;        /* pocketfft rfftp plan (heap-owned)            */
    double*      packed;      /* [fft_size] half-complex scratch buffer       */
    int          is_static;   /* 1 if placed via fft_init (no free in destroy)*/
};

/* --- construction --------------------------------------------------------- */

FftHandle* fft_create(int fft_size) {
    if (fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) return NULL;
    FftHandle* h = (FftHandle*)calloc(1, sizeof(FftHandle));
    if (!h) return NULL;
    h->fft_size  = fft_size;
    h->n_freqs   = fft_size / 2 + 1;
    h->is_static = 0;
    h->plan   = make_rfftp_plan((size_t)fft_size);
    h->packed = (double*)calloc((size_t)fft_size, sizeof(double));
    if (!h->plan || !h->packed) {
        if (h->plan)   destroy_rfftp_plan(h->plan);
        if (h->packed) free(h->packed);
        free(h);
        return NULL;
    }
    return h;
}

/* The pocketfft plan + twiddle tables live on the heap (their size is
 * length-dependent and the upstream allocator is malloc-based). For the
 * static-memory path we place the FftHandle + packed scratch inline and let
 * the plan itself remain heap-allocated; fft_destroy frees the plan in both
 * paths but only frees the inline buffers on the heap path. This matches the
 * fft_wrapper.h contract (static path: caller owns the placed buffer). */
size_t fft_get_mem_size(int fft_size) {
    if (fft_size <= 0 || (fft_size & (fft_size - 1)) != 0) return 0;
    size_t total = 0;
    total += ALIGN16(sizeof(FftHandle));
    total += ALIGN16((size_t)fft_size * sizeof(double)); /* packed */
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
    h->is_static = 1;
    h->packed    = (double*)ptr;
    memset(h->packed, 0, (size_t)fft_size * sizeof(double));

    h->plan = make_rfftp_plan((size_t)fft_size);
    if (!h->plan) return NULL;   /* plan is the only heap object on this path */
    return h;
}

void fft_destroy(FftHandle* h) {
    if (!h) return;
    if (h->plan) destroy_rfftp_plan(h->plan);
    if (h->is_static) return;        /* static path: caller owns the buffer */
    free(h->packed);
    free(h);
}

int fft_get_size(const FftHandle* h)    { return h ? h->fft_size : 0; }
int fft_get_n_freqs(const FftHandle* h) { return h ? h->n_freqs  : 0; }

/* --- forward: np.fft.rfft ------------------------------------------------- */

void fft_forward(FftHandle* h, const float* real_in, Complex* complex_out) {
    if (!h || !real_in || !complex_out) return;
    const int n  = h->fft_size;
    const int nf = h->n_freqs;
    double* p = h->packed;

    /* fp32 -> fp64 upcast (numpy FORCECAST to NPY_DOUBLE) */
    for (int i = 0; i < n; ++i) p[i] = (double)real_in[i];

    /* forward, unnormalised (numpy fct = 1.0). In place: p becomes the packed
     * half-complex [r0, r1,i1, r2,i2, ..., r_{n/2}]. */
    rfftp_forward(h->plan, p, 1.0);

    /* unpack packed half-complex -> n_freqs complex bins, fp64 -> fp32.
     *   bin 0      : (p[0], 0)
     *   bin k>0,<N : (p[2k-1], p[2k])
     *   bin N=n/2  : (p[n-1], 0)   (even n; DC/Nyquist imag are exactly 0) */
    complex_out[0].r = (float)p[0];
    complex_out[0].i = 0.0f;
    for (int k = 1; k < nf - 1; ++k) {
        complex_out[k].r = (float)p[2 * k - 1];
        complex_out[k].i = (float)p[2 * k];
    }
    if (nf > 1) {
        complex_out[nf - 1].r = (float)p[n - 1];
        complex_out[nf - 1].i = 0.0f;
    }
}

/* --- inverse: np.fft.irfft ------------------------------------------------ */

void fft_inverse(FftHandle* h, const Complex* complex_in, float* real_out) {
    if (!h || !complex_in || !real_out) return;
    const int n  = h->fft_size;
    const int nf = h->n_freqs;
    double* p = h->packed;

    /* Repack complex bins -> packed half-complex, fp32 -> complex128 upcast.
     * numpy drops the DC and Nyquist imaginary parts (it copies only
     * npts-1 doubles starting at re1, then writes DC real into slot 0):
     *   p[0]      = Re(bin 0)                       (DC real, imag dropped)
     *   p[2k-1]   = Re(bin k), p[2k] = Im(bin k)    (k = 1 .. n/2-1)
     *   p[n-1]    = Re(bin n/2)                      (Nyquist real, imag dropped)
     */
    p[0] = (double)complex_in[0].r;
    for (int k = 1; k < nf - 1; ++k) {
        p[2 * k - 1] = (double)complex_in[k].r;
        p[2 * k]     = (double)complex_in[k].i;
    }
    if (nf > 1) {
        p[n - 1] = (double)complex_in[nf - 1].r;
    }

    /* inverse, normalised by 1/n (numpy fct = 1.0/n applied in copy_and_norm) */
    rfftp_backward(h->plan, p, 1.0 / (double)n);

    for (int i = 0; i < n; ++i) real_out[i] = (float)p[i];
}

/* --- spectral helpers (unchanged from fft_fp64.c) ------------------------- */

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
