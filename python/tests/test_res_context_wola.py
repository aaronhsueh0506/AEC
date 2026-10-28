"""Structural tests for the external RES/NR frequency-domain seam."""

import numpy as np
import pytest

from aec import AEC, AecConfig, AecPreset


@pytest.mark.parametrize(
    "sample_rate,fft_size",
    [(16000, 256), (16000, 512), (48000, 1024)],
)
@pytest.mark.parametrize("enable_shadow", [False, True])
def test_res_context_is_reconstructing_wola(
    sample_rate: int, fft_size: int, enable_shadow: bool
) -> None:
    config = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sample_rate,
        frame_size=fft_size,
    )
    config.enable_shadow = enable_shadow
    config.enable_res = False
    config.return_res_context = True
    config.enable_cng = False
    config.enable_delay_est = False
    config.enable_highpass = False
    config.enable_saturation = False

    aec = AEC(config)
    hop = aec.hop_size
    index = np.arange(fft_size, dtype=np.float64)
    window = np.sqrt(
        0.5 * (1.0 - np.cos(2.0 * np.pi * index / float(fft_size)))
    ).astype(np.float32)
    previous = np.zeros(hop, dtype=np.float32)
    ola = np.zeros(fft_size, dtype=np.float32)
    rng = np.random.RandomState(0x574F4C41 + int(enable_shadow))

    for _ in range(40):
        render = (0.18 * rng.uniform(-1.0, 1.0, hop)).astype(np.float32)
        capture = (
            0.45 * render + 0.025 * rng.uniform(-1.0, 1.0, hop)
        ).astype(np.float32)
        _standalone_output, context = aec.process(capture, render)

        formed = np.asarray(context.formed_output, dtype=np.float32)
        expected = np.fft.rfft(
            np.concatenate((previous, formed)) * window,
            n=fft_size,
        ).astype(np.complex64)
        np.testing.assert_allclose(context.error_spec, expected, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            context.near_spec,
            context.error_spec + context.echo_spec,
            rtol=0,
            atol=1e-6,
        )

        time_frame = np.fft.irfft(context.error_spec, n=fft_size).astype(np.float32)
        ola += time_frame * window
        np.testing.assert_allclose(ola[:hop], previous, rtol=0, atol=1e-6)
        ola[:-hop] = ola[hop:]
        ola[-hop:] = 0.0
        previous = formed.copy()


def test_res_context_filter_converged_tracks_this_hop_not_last() -> None:
    """AecResContext.filter_converged must reflect THIS hop's update_convergence()
    result, not the value from before it ran. The difference only shows on the
    rising-edge hop (the single hop where convergence flips False -> True), since
    self._convergence.converged does not change on any other hop. Oracle:
    AEC.get_stats().filter_converged reads self._filter_converged, which once the
    hop's process() call has returned always reflects the post-update_convergence
    state -- the C twin's contract, since aec_get_res_context() is a separate pull
    made after aec_process() returns. The two must always agree.
    """
    config = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=16000, frame_size=512)
    config.return_res_context = True
    aec = AEC(config)
    hop = aec.hop_size
    rng = np.random.RandomState(0x53303500)
    saw_rising_edge = False
    prev_converged = False
    hops_after_edge = 0
    # The rising edge lands within about a hundred hops; the cap only bounds a scene that never converges.
    for _ in range(2000):
        far = (0.3 * rng.uniform(-1.0, 1.0, hop)).astype(np.float32)
        # Near-perfect echo coupling (no near-end, no noise) so the filter
        # actually reaches the ERLE>5dB/10-consecutive-frame convergence
        # rule within the loop and the rising edge is exercised for real.
        near = (0.5 * far).astype(np.float32)
        _output, context = aec.process(near, far)
        now_converged = bool(aec.get_stats().filter_converged)
        assert context.filter_converged == now_converged, (
            "AecResContext.filter_converged (%r) disagrees with this hop's "
            "post-update_convergence state (%r) -- S-05 regression"
            % (context.filter_converged, now_converged))
        if now_converged and not prev_converged:
            saw_rising_edge = True
        prev_converged = now_converged
        if saw_rising_edge:
            # The hops after the edge keep the per-hop agreement under test; nothing later can differ.
            hops_after_edge += 1
            if hops_after_edge > 20:
                break
    assert saw_rising_edge, (
        "scene never converged -- strengthen the echo coupling so this test "
        "actually exercises the S-05 rising edge")


# ----------------------------------------------------------------------------------------------------------
# The output-capture guard must never leak into the analysis seam. Every hop is checked, including hops where
# the guard fires: STFT pairing of error_spec with [previous formed hop | formed hop], unity IFFT/WOLA
# reconstruction of the previous formed hop, near_spec = error_spec + echo_spec. The rendering base is a separate
# field (output_base_spec) and follows the guard rule, which is decided here independently from the context.

from guard_scenes import GRIDS, clip_scene, expected_guard, sqrt_hann, synthetic_scene  # noqa: E402


def _guard_config(sample_rate, fft_size, enable_shadow, enable_res=False, enable_cng=False, bare=True):
    config = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=sample_rate, frame_size=fft_size)
    config.enable_shadow = enable_shadow
    config.enable_res = enable_res
    config.return_res_context = True
    config.enable_cng = enable_cng
    if bare:
        config.enable_delay_est = False
        config.enable_highpass = False
        config.enable_saturation = False
    return config


def _check_analysis_seam_and_base(aec, hops, fft_size):
    """Assert the analysis-seam contract and the base rule on every hop; return (fired, usable_hops)."""
    hop = aec.hop_size
    window = sqrt_hann(fft_size)
    previous = np.zeros(hop, dtype=np.float32)
    ola = np.zeros(fft_size, dtype=np.float32)
    fired = usable_hops = 0
    serial = aec.get_post_reset_serial()
    for near, far in hops:
        _out, ctx = aec.process(near, far)
        if ctx.post_reset_serial != serial:     # a hard post-chain reset restarts the formed-hop sequence
            serial = ctx.post_reset_serial
            previous = np.zeros(hop, dtype=np.float32)
            ola[:] = 0.0
        formed = np.asarray(ctx.formed_output, dtype=np.float32)
        expected = np.fft.rfft(np.concatenate((previous, formed)) * window, n=fft_size).astype(np.complex64)
        np.testing.assert_allclose(ctx.error_spec, expected, rtol=0, atol=1e-6)
        np.testing.assert_allclose(ctx.near_spec, ctx.error_spec + ctx.echo_spec, rtol=0, atol=1e-6)
        frame = np.fft.irfft(ctx.error_spec, n=fft_size).astype(np.float32)
        ola += frame * window
        np.testing.assert_allclose(ola[:hop], previous, rtol=0, atol=1e-6)
        ola[:-hop] = ola[hop:]
        ola[-hop:] = 0.0
        previous = formed.copy()

        # Rendering base: independent decision from the context fields.
        want = expected_guard(ctx)
        assert ctx.guard_active is want
        reference = ctx.near_spec if want else ctx.error_spec
        np.testing.assert_array_equal(ctx.output_base_spec, reference)
        assert ctx.output_base_spec.dtype == np.complex64 and ctx.output_base_spec.shape == ctx.error_spec.shape
        fired += int(want)
        usable_hops += int(bool(ctx.usable_linear))
    return fired, usable_hops


@pytest.mark.parametrize("sample_rate,fft_size", GRIDS)
@pytest.mark.parametrize("enable_shadow", [False, True])
@pytest.mark.parametrize("scene", ["uncorrelated", "burst", "farend_clip", "nearend_clip"])
def test_analysis_seam_pairs_on_every_hop_while_the_guard_fires(sample_rate, fft_size, enable_shadow, scene):
    aec = AEC(_guard_config(sample_rate, fft_size, enable_shadow))
    hop = aec.hop_size
    if scene.endswith("_clip"):
        hops = clip_scene(scene.split("_")[0], sample_rate, hop, 200)
    else:
        hops = synthetic_scene(scene, hop, sample_rate, 160)
    fired, usable_hops = _check_analysis_seam_and_base(aec, hops, fft_size)
    # The scene has to exercise the guard, otherwise the pairing above proves nothing about guard hops.
    assert fired >= 3, "scene %s never fires the guard (%d hops)" % (scene, fired)
    if scene in ("uncorrelated", "nearend_clip"):
        assert usable_hops == 0, "a never-usable scene must stay unusable"
    if scene == "burst":
        assert fired < len(hops), "burst scene must also contain hops without a guard"


@pytest.mark.parametrize("sample_rate,fft_size", GRIDS)
def test_analysis_seam_pairs_when_the_guard_switches_and_filter_becomes_usable(sample_rate, fft_size):
    """Cold start fires the guard, then the filter becomes usable (guard off): the pairing holds across the
    switch and across the unusable -> usable transition."""
    aec = AEC(_guard_config(sample_rate, fft_size, enable_shadow=True, enable_res=True, bare=False))
    hop = aec.hop_size
    hops = clip_scene("farend", sample_rate, hop, 420)
    fired, usable_hops = _check_analysis_seam_and_base(aec, hops, fft_size)
    assert fired >= 5
    assert 0 < usable_hops < len(hops)
