"""The PBFDKF partition constraint must enforce overlap-save support."""

import numpy as np
import pytest

from modules.filters import PBFDAF


@pytest.mark.parametrize("fft_size,sample_rate", [
    (256, 8000),
    (256, 16000),
    (512, 16000),
    (1024, 48000),
])
def test_td_constraint_keeps_only_the_causal_half(fft_size, sample_rate):
    hop = fft_size // 2
    filt = PBFDAF(
        fft_size, n_partitions=2, mu=0.3, delta=1e-6,
        hop_size=hop, sample_rate=sample_rate,
    )
    np.testing.assert_array_equal(filt._td_window[:hop], np.ones(hop, np.float32))
    np.testing.assert_array_equal(filt._td_window[hop:], np.zeros(hop, np.float32))
