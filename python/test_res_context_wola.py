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
