import numpy as np
import pytest

from modules.residual.reverb_decay_estimator import ReverbDecayEstimator


def test_constructor_keeps_legacy_positional_argument_order():
    """Adding sample_rate must not reinterpret the former third argument."""
    est = ReverbDecayEstimator(8, 128, 0.77)
    assert est.decay(mild=False) == pytest.approx(0.77)


@pytest.mark.parametrize(
    "sample_rate, hop, expected_per_hop",
    [(8000, 64, 0.5), (8000, 128, 0.25),
     (16000, 128, 0.5), (16000, 256, 0.25),
     (48000, 384, 0.5), (48000, 768, 0.25)],
)
def test_partition_slope_is_stored_in_four_ms_units(
        sample_rate, hop, expected_per_hop):
    """The same physical exponential tail must not change with the grid.

    The fixture loses 3 dB of power every 8 ms.  An 8-ms hop therefore
    retains 0.5 power and a 16-ms hop retains 0.25, independently of sample
    rate.  The estimator stores an AEC3 4 ms value; its consumer then raises
    that value by hop/4 ms.
    """
    est = ReverbDecayEstimator(
        n_partitions=8, hop_size=hop, sample_rate=sample_rate,
        default_decay=0.83, use_adaptive=True,
    )
    energy = np.asarray(
        [2.0 ** (-i * hop / (0.008 * sample_rate)) for i in range(8)],
        dtype=np.float32,
    )
    for _ in range(200):
        est.update(
            energy, filter_quality=1.0, filter_delay_blocks=0,
            usable_linear_filter=True, stationary_signal=False,
        )
    live_per_hop = est.decay(mild=False) ** (hop / (0.004 * sample_rate))
    assert live_per_hop == pytest.approx(expected_per_hop, abs=1e-12)
