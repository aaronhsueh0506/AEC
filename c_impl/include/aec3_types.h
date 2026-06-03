/* aec3_types.h — shared numeric types for the v3.22 AEC3 post-filter port.
 *
 * The AEC3 post-filter (AecState / ResidualEchoEstimator / SuppressionGain /
 * CNG) runs in DOUBLE precision end-to-end in the Python reference: every PSD
 * is lifted by _PSD_SCALE = 32768^2 (~1e9) and the R^2 / ERLE / gain / CNG math
 * operates at that magnitude, where float32's ~7-digit mantissa loses the low
 * bits against the 1e-30 floors. For byte-equal parity the C port mirrors that:
 * post-filter internals are `aec3_real` (double), and `ComplexD` is used wher-
 * ever Python promotes complex64 to float64 (np.abs(x)**2 etc.).
 *
 * The PBFDKF filter spectra remain the existing fp32 `Complex` (matching numpy
 * complex64); they are squared INTO double here, matching numpy's promotion
 * order: (double)re*re + (double)im*im, then * PSD_SCALE.
 */
#ifndef AEC3_TYPES_H
#define AEC3_TYPES_H

typedef double aec3_real;

typedef struct {
    double r;  /* Real part */
    double i;  /* Imaginary part */
} ComplexD;

#endif /* AEC3_TYPES_H */
