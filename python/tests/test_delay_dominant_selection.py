"""The product delay is the DOMINANT matched-filter peak, not the pre-echo one.

AEC3 runs its delay estimator inside a RenderDelayController and wants the
earliest echo-path onset, so upstream substitutes the pre-echo candidate for
the reported delay while still qualifying it with the dominant peak's
histogram. Our product path is different: LegacyDelayShim uses the reported
sample delay directly to align a short PBFDKF, and when the onset sits further
before the dominant echo than that filter spans, the echo it must cancel falls
outside the filter entirely -- while confidence still reads 1.0, because
confidence describes the dominant histogram and not the substituted delay.

These tests pin the selection seam only. The pre-echo computation stays: it is
the upstream reference and a diagnostic, it just no longer decides where the
production filter is aligned.

Run:
    python3 -m pytest python/tests/test_delay_dominant_selection.py
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.delay.delay_types import DelayQuality
from modules.delay.lag_aggregator import (
    DelaySelectionThresholds,
    MatchedFilterLagAggregator,
)
from modules.delay.matched_filter import LagEstimate


DOWN_SAMPLING_FACTOR = 4
MAX_FILTER_LAG = 2000

# Downsampled samples. The separation is 400 ds = 1600 raw = 100 ms at 16 kHz,
# far wider than the 832-sample (52 ms) PBFDKF the reported delay aligns -- the
# case where picking the earlier candidate puts the dominant echo outside the
# filter. Taken from the a6941aef far-end single-talk fixture.
DOMINANT_LAG = 1552
EARLY_LAG = 1152


def _aggregator(sample_rate, detect_pre_echo=True):
    return MatchedFilterLagAggregator(
        max_filter_lag=MAX_FILTER_LAG,
        thresholds=DelaySelectionThresholds(),
        delay_headroom_samples=32,
        down_sampling_factor=DOWN_SAMPLING_FACTOR,
        detect_pre_echo=detect_pre_echo,
        sample_rate=sample_rate,
    )


def _drive(aggregator, updates, dominant=DOMINANT_LAG, early=EARLY_LAG):
    """Feed one steady dual-path estimate `updates` times."""
    estimate = None
    for _ in range(updates):
        estimate = aggregator.aggregate(
            LagEstimate(lag=dominant, pre_echo_lag=early)
        )
    return estimate


def _assert_two_distinct_candidates(aggregator):
    """The premise. A run where the two agree proves nothing about which one
    is selected, so every assertion below is worthless without this."""
    dominant = aggregator.delay_at_highest_peak()
    pre_echo = aggregator._pre_echo.candidate()
    assert dominant > 0, 'the dominant aggregator produced no candidate'
    assert pre_echo > 0, 'the pre-echo aggregator produced no candidate'
    assert dominant != pre_echo, (
        'dominant and pre-echo candidates coincide (%d); this run cannot tell '
        'which one the aggregator reported' % dominant
    )
    return dominant, pre_echo


@pytest.mark.parametrize('sample_rate', (16000, 48000))
def test_reported_delay_is_the_dominant_peak(sample_rate):
    aggregator = _aggregator(sample_rate)
    estimate = _drive(aggregator, DelaySelectionThresholds().converged + 5)
    assert estimate is not None, 'no estimate crossed the threshold'
    dominant, pre_echo = _assert_two_distinct_candidates(aggregator)
    assert estimate.delay == dominant, (
        'reported %d; dominant peak is %d and the earlier pre-echo candidate '
        'is %d' % (estimate.delay, dominant, pre_echo)
    )


@pytest.mark.parametrize('sample_rate', (16000, 48000))
def test_quality_still_comes_from_the_dominant_histogram(sample_rate):
    """The confidence and the delay must describe the SAME candidate. Reading
    one from the dominant histogram while reporting the other is what let a
    wrong delay carry confidence 1.0."""
    aggregator = _aggregator(sample_rate)
    thresholds = DelaySelectionThresholds()
    early = _drive(aggregator, thresholds.initial + 1)
    assert early is not None and early.quality is DelayQuality.COARSE
    refined = _drive(aggregator, thresholds.converged)
    assert refined is not None and refined.quality is DelayQuality.REFINED
    dominant, _pre_echo = _assert_two_distinct_candidates(aggregator)
    assert refined.delay == dominant


def test_pre_echo_is_still_computed():
    """Retained for the AEC3 reference and for diagnostics; only its role in
    selecting the production delay is gone. A patch that deleted the
    aggregator entirely would pass every test above."""
    aggregator = _aggregator(16000)
    _drive(aggregator, DelaySelectionThresholds().converged + 5)
    assert aggregator._pre_echo is not None
    assert aggregator._pre_echo.candidate() > 0


def test_disabling_pre_echo_does_not_change_the_reported_delay():
    """With the substitution gone the flag is diagnostic, so both settings must
    now report the same delay for the same input."""
    with_pre_echo = _aggregator(16000, detect_pre_echo=True)
    without = _aggregator(16000, detect_pre_echo=False)
    updates = DelaySelectionThresholds().converged + 5
    assert _drive(with_pre_echo, updates).delay == _drive(without, updates).delay


def test_reset_reacquires_the_dominant_candidate():
    """No stale pre-echo candidate may leak across a hard reset."""
    aggregator = _aggregator(16000)
    _drive(aggregator, DelaySelectionThresholds().converged + 5)
    aggregator.reset(hard_reset=True)
    assert aggregator._pre_echo.candidate() == 0
    assert not aggregator.reliable_delay_found()
    estimate = _drive(aggregator, DelaySelectionThresholds().converged + 5)
    assert estimate is not None
    dominant, _pre_echo = _assert_two_distinct_candidates(aggregator)
    assert estimate.delay == dominant
