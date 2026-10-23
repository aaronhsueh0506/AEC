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
