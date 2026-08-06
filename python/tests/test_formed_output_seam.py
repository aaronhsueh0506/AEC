"""Structural tests for AecConfig.return_formed_output -- the lightweight
formed-output seam. Sibling to test_res_context_wola.py's
context.formed_output, but skips return_res_context's R2/CNG/suppression
computation entirely (Audio_ALG/AIAEC/dataset_gen/linear_aec.py's use case:
materializing linear_error for a whole dataset, where that extra work would
be pure waste since nothing downstream reads it).

The seam remains necessary without an output limiter because it includes
shadow/main selection, crossfade and WOLA formation; process() can expose the
raw main-filter hop when the selector chooses another candidate.
"""

import numpy as np
import pytest

from aec import AEC, AecConfig, AecPreset


def _make_config(sample_rate: int, fft_size: int, enable_shadow: bool) -> AecConfig:
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sample_rate,
        frame_size=fft_size,
    )
    cfg.enable_shadow = enable_shadow
    cfg.enable_res = False
    cfg.enable_cng = False
    return cfg


@pytest.mark.parametrize(
    "sample_rate,fft_size",
    [(16000, 256), (16000, 512), (48000, 1024)],
)
@pytest.mark.parametrize("enable_shadow", [False, True])
def test_formed_output_matches_res_context_formed_output(
    sample_rate: int, fft_size: int, enable_shadow: bool
) -> None:
    """get_formed_output() (return_formed_output=True, cheap) must be
    byte-identical, every hop, to context.formed_output (return_res_context=
    True, which additionally computes R2/suppression-gain/CNG context) --
    proving the lightweight path doesn't skip anything that changes the
    actual value, only the extra work this dataset-gen use case never
    needed. Also exercises the shadow-filter selection/crossfade path
    (enable_shadow=True) the review flagged as needing coverage.
    """
    cheap_cfg = _make_config(sample_rate, fft_size, enable_shadow)
    cheap_cfg.return_formed_output = True
    cheap = AEC(cheap_cfg)

    heavy_cfg = _make_config(sample_rate, fft_size, enable_shadow)
    heavy_cfg.return_res_context = True
    heavy = AEC(heavy_cfg)

    hop = cheap.hop_size
    rng = np.random.RandomState(0x464F524D + int(enable_shadow) + sample_rate)

    for _ in range(60):
        render = (0.18 * rng.uniform(-1.0, 1.0, hop)).astype(np.float32)
        capture = (
            0.45 * render + 0.025 * rng.uniform(-1.0, 1.0, hop)
        ).astype(np.float32)
        cheap.process(capture.copy(), render.copy())
        _standalone, ctx = heavy.process(capture.copy(), render.copy())
        np.testing.assert_array_equal(
            cheap.get_formed_output(),
            np.asarray(ctx.formed_output, dtype=np.float32),
        )


def test_no_output_limiter_output_is_never_peak_scaled() -> None:
    """A burst meeting the removed limiter's trigger condition is unscaled."""
    cfg = _make_config(16000, 256, enable_shadow=True)
    cfg.return_formed_output = True
    aec = AEC(cfg)
    hop = aec.hop_size
    rng = np.random.RandomState(0x11317E5)

    would_have_limited = False
    for i in range(80):
        # Quiet baseline, a loud burst, then quiet again -- the excursion the
        # removed limiter's attack/release smoothing used to ride.
        amp = 0.9 if 20 <= i < 30 else 0.02
        far = (amp * 0.3 * rng.uniform(-1.0, 1.0, hop)).astype(np.float32)
        near = (amp * rng.uniform(-1.0, 1.0, hop)).astype(np.float32)
        result = aec.process(near.copy(), far.copy())
        formed = aec.get_formed_output()
        assert aec._refined_filter_output_last_selected, (
            "this test's algebraic identity requires refined to stay "
            "selected throughout -- retune the synthetic signal if AEC3's "
            "UseRefinedOutput ever picks coarse here"
        )
        # No limiter gain, no scaling, no smoothing: bit-for-bit the same.
        np.testing.assert_array_equal(result, formed)

        near_peak = float(np.max(np.abs(near)))
        out_peak = float(np.max(np.abs(result)))
        if out_peak > near_peak > 1e-6:
            would_have_limited = True

    assert would_have_limited, (
        "signal must reach the removed limiter's trigger condition"
    )
    assert not hasattr(aec, "_limiter_gain"), (
        "the custom output limiter state must not return"
    )


def test_get_formed_output_requires_flag_and_a_prior_process_call() -> None:
    cfg = _make_config(16000, 256, enable_shadow=False)
    aec = AEC(cfg)  # return_formed_output left at its default (False)
    with pytest.raises(ValueError):
        aec.get_formed_output()

    cfg2 = _make_config(16000, 256, enable_shadow=False)
    cfg2.return_formed_output = True
    aec2 = AEC(cfg2)
    with pytest.raises(ValueError):
        aec2.get_formed_output()  # process() hasn't run yet
    hop = aec2.hop_size
    aec2.process(np.zeros(hop, dtype=np.float32), np.zeros(hop, dtype=np.float32))
    aec2.get_formed_output()  # now succeeds
