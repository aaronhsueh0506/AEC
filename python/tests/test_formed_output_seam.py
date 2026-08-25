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
from modules.orchestrator import _SEL_CAPTURE, _SEL_REFINED


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
    """The formed seam and the res-context seam are the same signal.

    They are now the same code path rather than two that happen to agree:
    the formed output is captured after _aec3_post() has run the FORM step,
    and AecConfig refuses to serve it without that chain. The earlier
    lightweight route ran the selector on its own and skipped AecState, so
    the capture fallback's quality verdict stayed at its constructed False
    for the whole stream and the two routes diverged wherever the energy
    term fired -- measured at 286 hops out of 400 once the echo path was
    removed mid-stream. Byte-identity is asserted here so a future
    reintroduction of a second route has to prove it, and the shadow path
    (enable_shadow=True) is covered because selection/crossfade only happens
    when a coarse candidate exists.
    """
    cfg = _make_config(sample_rate, fft_size, enable_shadow)
    cfg.return_res_context = True
    cfg.return_formed_output = True
    aec = AEC(cfg)

    hop = aec.hop_size
    rng = np.random.RandomState(0x464F524D + int(enable_shadow) + sample_rate)

    for _ in range(60):
        render = (0.18 * rng.uniform(-1.0, 1.0, hop)).astype(np.float32)
        capture = (
            0.45 * render + 0.025 * rng.uniform(-1.0, 1.0, hop)
        ).astype(np.float32)
        _standalone, ctx = aec.process(capture.copy(), render.copy())
        np.testing.assert_array_equal(
            aec.get_formed_output(),
            np.asarray(ctx.formed_output, dtype=np.float32),
        )


def test_formed_output_requires_the_chain_that_produces_it() -> None:
    """Without the post chain the quality verdict is a constant, so the
    fallback would degrade to a bare energy rule and this seam would stop
    agreeing with the one the board ships. The config refuses it."""
    cfg = _make_config(16000, 256, enable_shadow=False)
    assert cfg.enable_res is False and cfg.return_res_context is False
    with pytest.raises(ValueError, match="return_formed_output"):
        AecConfig.from_preset(
            AecPreset.BALANCED,
            sample_rate=16000,
            frame_size=256,
            enable_res=False,
            return_formed_output=True,
        )


def test_no_output_limiter_output_is_never_peak_scaled() -> None:
    """A burst meeting the removed limiter's trigger condition is unscaled."""
    cfg = _make_config(16000, 256, enable_shadow=True)
    cfg.return_res_context = True
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
        result, _ctx = aec.process(near.copy(), far.copy())
        assert aec._refined_filter_output_last_selected, (
            "this test's algebraic identity requires refined to stay "
            "selected throughout -- retune the synthetic signal if AEC3's "
            "UseRefinedOutput ever picks coarse here"
        )
        # No limiter gain, no scaling, no smoothing: what process() returns is
        # the linear residual itself, bit for bit. Deliberately NOT compared
        # against get_formed_output(): the formed seam carries the capture
        # fallback and the two are different signals by design on any hop
        # where it fires.
        np.testing.assert_array_equal(result, aec._last_raw_output)

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
    cfg2.return_res_context = True
    cfg2.return_formed_output = True
    aec2 = AEC(cfg2)
    with pytest.raises(ValueError):
        aec2.get_formed_output()  # process() hasn't run yet
    hop = aec2.hop_size
    aec2.process(np.zeros(hop, dtype=np.float32), np.zeros(hop, dtype=np.float32))
    aec2.get_formed_output()  # now succeeds


@pytest.mark.parametrize("usable,expect_capture", [(True, False), (False, True)])
def test_capture_fallback_requires_an_unusable_filter(
    monkeypatch: pytest.MonkeyPatch, usable: bool, expect_capture: bool
) -> None:
    """Energy alone must not replace a filter the analyzer has accepted."""
    cfg = _make_config(16000, 256, enable_shadow=False)
    cfg.return_res_context = True
    aec = AEC(cfg)
    hop = aec.hop_size
    err = np.full(hop, 0.10, dtype=np.float32)
    near = np.full(hop, 0.01, dtype=np.float32)

    monkeypatch.setattr(
        type(aec._aec3_state), "usable_linear_estimate", lambda _self: usable
    )
    aec._form_prev_output_time = None
    aec._form_last_selection = _SEL_REFINED
    aec._aec3_select_linear_filter_output(
        e_refined_time=err, near_end_block=near, e_coarse_time=err
    )
    assert (aec._form_last_selection == _SEL_CAPTURE) is expect_capture
