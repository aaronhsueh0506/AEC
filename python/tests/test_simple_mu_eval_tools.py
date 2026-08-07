"""State-isolation tests for the simple-mu evaluation helpers."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
for path in (REPO / "eval", REPO / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_interaction_worker_restores_process_state_on_failure(monkeypatch):
    import simple_mu_interaction_matrix as matrix
    from modules import orchestrator

    keys = tuple(matrix.CONSTANTS.values())
    original_constants = {key: getattr(orchestrator, key) for key in keys}

    probe = types.ModuleType("simple_mu_case_probe")
    probe.FROZEN = dict(original_constants)
    probe.case_paths = lambda stem: ("mic.wav", "ref.wav")
    probe.bulk_delay = lambda mic, ref, sr: 0
    probe.hop_energy = lambda x, hop, n_hops: np.ones(n_hops)
    probe.retimed_values = lambda sr, frame: {
        key: value * 0.5 for key, value in original_constants.items()
    }
    monkeypatch.setitem(sys.modules, "simple_mu_case_probe", probe)

    driver = types.ModuleType("eval_aec_challenge")
    driver._ENABLE_CNG = False

    def fail_run(*args, **kwargs):
        raise RuntimeError("injected failure")

    driver.run_ours = fail_run
    monkeypatch.setitem(sys.modules, "eval_aec_challenge", driver)
    monkeypatch.setattr(matrix.sf, "read",
                        lambda path: (np.zeros(64, dtype=np.float64), 16000))
    monkeypatch.setenv("AEC_CFG_OVERRIDE", "caller-grid")
    monkeypatch.delenv("NO_PREALIGN", raising=False)

    with pytest.raises(RuntimeError, match="injected failure"):
        matrix.run_one(("case_farend_singletalk", 256, (1, 1, 1, 1)))

    assert os.environ["AEC_CFG_OVERRIDE"] == "caller-grid"
    assert "NO_PREALIGN" not in os.environ
    assert driver._ENABLE_CNG is False
    assert {key: getattr(orchestrator, key) for key in keys} == original_constants
