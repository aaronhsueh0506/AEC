"""CLI presets and diagnostics for paths that skip residual suppression."""
import sys

import numpy as np
import pytest

from aec import AEC, AecConfig, AecMode
from modules import orchestrator


@pytest.mark.parametrize("value,name", [
    ("0", "mild"), ("1", "balanced"), ("2", "aggressive"),
    ("3", "no_floor"), ("mild", "mild"), ("balanced", "balanced"),
    ("aggressive", "aggressive"), ("no_floor", "no_floor"),
])
def test_engine_cli_accepts_numeric_and_named_presets(monkeypatch, value, name):
    captured = []
    monkeypatch.setattr(orchestrator, "process_wav_files",
                        lambda *args, **kwargs: captured.append(args[3]))
    monkeypatch.setattr(sys, "argv", ["aec.py", "mic.wav", "ref.wav",
                                    "out.wav", "--preset", value])
    orchestrator.main()
    expected = AecConfig.from_preset(name)
    assert len(captured) == 1
    actual = captured[0]
    assert actual.min_gain_split_floor_enabled == expected.min_gain_split_floor_enabled
    assert actual.min_gain_floor_far_active_db == expected.min_gain_floor_far_active_db
    assert actual.min_gain_floor_dt_db == expected.min_gain_floor_dt_db


@pytest.mark.parametrize("value", ["-1", "4", "01", "BALANCED"])
def test_engine_cli_rejects_unknown_presets(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["aec.py", "mic.wav", "ref.wav",
                                    "out.wav", "--preset", value])
    with pytest.raises(SystemExit):
        orchestrator.main()


@pytest.mark.parametrize("converged", [False, True])
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("shadow", [False, True])
def test_skipped_residual_diagnostics_do_not_reuse_previous_factor(buffered, shadow, converged):
    cfg = AecConfig.from_preset(
        "balanced", sample_rate=16000, frame_size=512, hop_size=256,
        mode=AecMode.FDAF if buffered else AecMode.PBFDKF,
        filter_length=1024 if buffered else 512,
        use_kalman=True, enable_res=False, return_res_context=False,
        enable_shadow=shadow,
    )
    engine = AEC(cfg)
    engine._erle_factor_prev = 0.75
    engine._convergence._converged = converged
    engine._convergence._once_converged = converged
    rng = np.random.RandomState(73)
    for _ in range(8):
        ref = rng.normal(0, 0.05, 256).astype(np.float32)
        mic = (0.4 * ref + rng.normal(0, 0.01, 256)).astype(np.float32)
        assert isinstance(engine.process(mic, ref), np.ndarray)
        diag = engine.get_diagnostics()
        assert diag["erle_factor"] == 0.0
        assert diag["dt_indicator"] == 0.0
        if buffered:
            assert diag["erle_windowed"] == 0.0


@pytest.mark.parametrize("shadow", [False, True])
def test_context_reports_current_residual_factor(shadow):
    cfg = AecConfig.from_preset(
        "balanced", sample_rate=16000, frame_size=512,
        enable_res=False, return_res_context=True,
        return_formed_output=True, enable_shadow=shadow,
    )
    engine = AEC(cfg)
    engine._erle_factor_prev = 0.75
    zero = np.zeros(engine.hop_size, dtype=np.float32)
    _, context = engine.process(zero, zero)
    diag = engine.get_diagnostics()
    assert context.erle_factor == diag["erle_factor"] == 0.0
    assert context.dt_indicator == diag["dt_indicator"]
    assert context.filter_converged is engine._filter_converged
    assert context.formed_output.tobytes() == engine.get_formed_output().tobytes()
