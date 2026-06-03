/*
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see LICENSE.md
 *
 *  Copyright (C) 2004-2018 Max-Planck-Society
 *  \author Martin Reinecke
 *
 * Vendored from numpy 1.26.4 (numpy/fft/_pocketfft.c). Only the pure-C
 * real-FFT (rfftp) path is kept — the cfftp / fftblue (Bluestein) /
 * Python-wrapper code is stripped because numpy's np.fft.rfft/irfft for
 * power-of-two lengths dispatch exclusively to the rfftp packed-FFT path.
 *
 * Public API (a strict subset of pocketfft's, sufficient to reproduce
 * np.fft.rfft / np.fft.irfft bit-for-bit):
 *
 *   rfft_plan_t *make_rfftp_plan(size_t length);
 *   void         destroy_rfftp_plan(rfft_plan_t *plan);
 *   int          rfftp_forward (rfft_plan_t *plan, double c[], double fct);
 *   int          rfftp_backward(rfft_plan_t *plan, double c[], double fct);
 *
 * Buffer layout matches numpy/pocketfft's half-complex packed convention:
 *   forward  in:  c[0..n-1] = real samples
 *            out: c[0]      = DC (real)
 *                 c[2k-1]   = Re(bin k), c[2k] = Im(bin k)  (k = 1 .. n/2-1)
 *                 c[n-1]    = Re(Nyquist) for even n  (no imag slot)
 *   backward is the inverse of that layout.
 *
 * fct is the (un)normalisation factor pocketfft applies inside copy_and_norm
 * at the end of the transform: numpy uses fct=1.0 for the forward transform
 * and fct=1.0/n for the inverse transform.
 */

#ifndef POCKETFFT_RFFT_H
#define POCKETFFT_RFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct rfftp_plan_i;
typedef struct rfftp_plan_i *rfft_plan_t;

/* Build a real-FFT plan for a transform of `length` real samples. Returns
 * NULL on allocation failure or length==0. */
rfft_plan_t make_rfftp_plan(size_t length);

/* Free a plan returned by make_rfftp_plan (NULL-safe). */
void destroy_rfftp_plan(rfft_plan_t plan);

/* Forward real transform, in place on the half-complex packed buffer c[].
 * Applies the normalisation factor `fct`. Returns 0 on success, -1 on a
 * scratch allocation failure. */
int rfftp_forward(rfft_plan_t plan, double c[], double fct);

/* Inverse real transform, in place on the half-complex packed buffer c[].
 * Applies the normalisation factor `fct`. Returns 0 on success, -1 on a
 * scratch allocation failure. */
int rfftp_backward(rfft_plan_t plan, double c[], double fct);

#ifdef __cplusplus
}
#endif

#endif /* POCKETFFT_RFFT_H */
