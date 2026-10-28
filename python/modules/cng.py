"""Comfort-noise numeric core shared by the in-chain CNG and the public fill helper.

The in-chain comfort noise (``AEC._post_add_comfort_noise``) and :func:`cng_fill` use this one numeric
core, so an application that renders the context seam itself (``output_base_spec * res_gain`` plus
comfort noise, then IFFT/OLA) reproduces the full ``process()`` output bit for bit.

Schedule, identical to the C ``aec_cng_fill`` / internal CNG:

* one LCG draw per bin ``1 .. n-2`` in ascending order (DC and Nyquist are untouched, no draw):
  ``rng = (rng * 69069 + 1) & 0x7FFFFFFF``, ``ix = rng >> 26``;
* real part uses ``lut[ix]``, imaginary part ``lut[(ix + 8) & 31]``, ``lut[i] = sqrt(2) sin(2 pi i / 32)``;
* ``nf = sqrt(max(cn / PSD_SCALE, 0))``, ``ng = sqrt(max(1 - g_res^2, 0))`` (float32 throughout);
* with an NR gain ``ng = ng * max(g_nr, nr_floor)``;
* ``spec += ng * (nf * lut)`` -- product order ``ng * (nf * lut)``, per component.

A hard post-chain reset (``AEC.get_post_reset_serial()`` changed) reseeds the generator to
``AEC_CNG_SEED``; the consumer does that before the first draw of the hop it sees the new serial on.
"""
from __future__ import annotations

import numpy as np

# Start state of the comfort-noise generator, also the state after every hard post-chain reset.
AEC_CNG_SEED = 42

# int16^2 PSD convention of ``AecResContext.comfort_noise`` (and of the in-chain N2 tracker).
CNG_PSD_SCALE = (32768.0) ** 2

# sqrt(2) * sin(2 pi i / 32); the sqrt(2) compensates the overlap-add power loss of uncorrelated frames.
SQRT2_SIN_LUT = (
    np.sqrt(2.0) * np.sin(2.0 * np.pi * np.arange(32, dtype=np.float64) / 32.0)
).astype(np.float32)


def cng_draw_indices(rng: int, n_random: int):
    """Advance the LCG ``n_random`` times; return ``(new_rng, re_idx, im_idx)`` (int32 arrays)."""
    re_idx = np.empty(n_random, dtype=np.int32)
    im_idx = np.empty(n_random, dtype=np.int32)
    seed = int(rng)
    for k in range(n_random):
        seed = (seed * 69069 + 1) & 0x7FFFFFFF
        ix = seed >> 26   # top 5 bits, 0..31
        re_idx[k] = ix
        im_idx[k] = (ix + 8) & 31
    return seed, re_idx, im_idx


def cng_noise_terms(comfort_noise, gain, n_bins: int, re_idx, im_idx, g_nr=None, nr_floor=0.0):
    """Per-bin ``(ng, cn_re, cn_im)`` for bins ``1 .. n_bins-2`` (float32 arrays of length n_bins-2).

    ``cn_re = nf * lut[re_idx]``, ``cn_im = nf * lut[im_idx]``; the amplitude to multiply them with is ``ng``.
    """
    cn = np.asarray(comfort_noise)[1:n_bins - 1]
    nf = np.sqrt(np.maximum(cn / CNG_PSD_SCALE, 0.0)).astype(np.float32)
    g = np.asarray(gain, dtype=np.float32)[1:n_bins - 1]
    ng = np.sqrt(np.maximum(np.float32(1.0) - g * g, 0.0)).astype(np.float32)
    if g_nr is not None:
        nr = np.asarray(g_nr, dtype=np.float32)[1:n_bins - 1]
        ng = (ng * np.maximum(nr, np.float32(nr_floor))).astype(np.float32)
    cn_re = (nf * SQRT2_SIN_LUT[re_idx]).astype(np.float32)
    cn_im = (nf * SQRT2_SIN_LUT[im_idx]).astype(np.float32)
    return ng, cn_re, cn_im


def cng_fill(rng, spec, comfort_noise, g_res, g_nr=None, nr_floor=0.0) -> int:
    """Reference twin of the C ``aec_cng_fill``: add comfort noise to ``spec`` in place.

    ``rng`` is the generator state in (``AEC_CNG_SEED`` after a reset); the new state is returned. ``spec`` is
    a complex64 array of ``n`` bins (written in place, bins ``1 .. n-2``); ``comfort_noise`` is the context
    ``comfort_noise`` (int16^2 PSD), ``g_res`` the RES gain, ``g_nr`` an optional NR gain with floor ``nr_floor``
    (the amplitude is then ``sqrt(1 - g_res^2) * max(g_nr, nr_floor)``). Fewer than 3 bins: nothing happens.
    """
    n = int(spec.shape[0])
    if n < 3:
        return int(rng)
    new_rng, re_idx, im_idx = cng_draw_indices(rng, n - 2)
    ng, cn_re, cn_im = cng_noise_terms(comfort_noise, g_res, n, re_idx, im_idx, g_nr, nr_floor)
    spec.real[1:n - 1] += ng * cn_re
    spec.imag[1:n - 1] += ng * cn_im
    return new_rng
