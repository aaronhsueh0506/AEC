"""Unit tests for ShadowCopyController.

Verifies the controller in isolation matches the original inline gate behavior
on hand-constructed scenarios. Used as a regression lock during further refactor.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AecConfig, ShadowCopyController, AecMode


def make_ctrl():
    cfg = AecConfig.from_preset('balanced', mode=AecMode.PBFDKF)
    return ShadowCopyController(cfg), cfg


def test_warmup_returns_idle():
    ctrl, _ = make_ctrl()
    for f in range(49):
        d = ctrl.update(shadow_frame_count=f, far_pwr=1.0,
                        main_err_smooth=1.0, shadow_err_smooth=1.0,
                        epc_active=False, saturation_level=0.0,
                        dt_from_energy=0.0)
        assert not d.pause_main and not d.boost_q and not d.reverse_copy


def test_baseline_tracks_in_stable_fs():
    ctrl, _ = make_ctrl()
    # stable FS: errors balanced, far active, no EPC
    for f in range(50, 200):
        ctrl.update(shadow_frame_count=f, far_pwr=1e-2,
                    main_err_smooth=2e-3, shadow_err_smooth=2e-3,
                    epc_active=False, saturation_level=0.0,
                    dt_from_energy=0.0)
    # baseline EMA should have moved toward 2e-3 from 1e-6
    assert ctrl.copy_err_baseline > 1e-4, ctrl.copy_err_baseline


def test_dt_blocks_pause_decision():
    ctrl, cfg = make_ctrl()
    # warm baseline
    for f in range(50, 100):
        ctrl.update(shadow_frame_count=f, far_pwr=1e-2,
                    main_err_smooth=1e-3, shadow_err_smooth=1e-3,
                    epc_active=False, saturation_level=0.0,
                    dt_from_energy=0.0)
    # shadow now winning, but DT > 0.3 should block pause
    pauses = 0
    for f in range(100, 150):
        d = ctrl.update(shadow_frame_count=f, far_pwr=1e-2,
                        main_err_smooth=1e-2, shadow_err_smooth=1e-4,
                        epc_active=False, saturation_level=0.0,
                        dt_from_energy=0.5)  # DT high
        pauses += int(d.pause_main)
    assert pauses == 0


def test_pause_triggers_on_streak():
    ctrl, cfg = make_ctrl()
    # warm baseline at moderate level so error_is_normal can be true
    for f in range(50, 200):
        ctrl.update(shadow_frame_count=f, far_pwr=1e-2,
                    main_err_smooth=1e-3, shadow_err_smooth=1e-3,
                    epc_active=False, saturation_level=0.0,
                    dt_from_energy=0.0)
    # main_err must stay below baseline*4 → error_is_normal = True.
    # Use main_err just slightly above shadow_err (still < threshold ratio).
    boosted = False
    paused_after = False
    for f in range(200, 230):
        d = ctrl.update(shadow_frame_count=f, far_pwr=1e-2,
                        main_err_smooth=1e-3, shadow_err_smooth=1e-4,
                        epc_active=False, saturation_level=0.0,
                        dt_from_energy=0.0)
        if d.boost_q:
            boosted = True
        if boosted:
            paused_after = paused_after or d.pause_main
    assert boosted, 'boost_q never fired despite sustained shadow advantage'
    assert paused_after, 'pause_main not asserted after boost_q'


def test_reverse_copy_when_main_winning():
    ctrl, _ = make_ctrl()
    for f in range(50, 200):
        ctrl.update(shadow_frame_count=f, far_pwr=1e-2,
                    main_err_smooth=1e-3, shadow_err_smooth=1e-3,
                    epc_active=False, saturation_level=0.0,
                    dt_from_energy=0.0)
    d = ctrl.update(shadow_frame_count=200, far_pwr=1e-2,
                    main_err_smooth=1e-4, shadow_err_smooth=1e-3,
                    epc_active=False, saturation_level=0.0,
                    dt_from_energy=0.0)
    assert d.reverse_copy


def test_epc_active_blocks_everything():
    ctrl, _ = make_ctrl()
    for f in range(50, 200):
        ctrl.update(shadow_frame_count=f, far_pwr=1e-2,
                    main_err_smooth=1e-3, shadow_err_smooth=1e-3,
                    epc_active=False, saturation_level=0.0,
                    dt_from_energy=0.0)
    d = ctrl.update(shadow_frame_count=200, far_pwr=1e-2,
                    main_err_smooth=1e-3, shadow_err_smooth=1e-4,
                    epc_active=True, saturation_level=0.0,
                    dt_from_energy=0.0)
    assert not d.boost_q and not d.reverse_copy


def test_reset_restores_init():
    ctrl, _ = make_ctrl()
    for f in range(50, 200):
        ctrl.update(shadow_frame_count=f, far_pwr=1e-2,
                    main_err_smooth=1e-3, shadow_err_smooth=1e-3,
                    epc_active=False, saturation_level=0.0,
                    dt_from_energy=0.0)
    assert ctrl.copy_err_baseline > 1e-5
    ctrl.reset()
    assert ctrl.copy_err_baseline == ShadowCopyController.BASELINE_INIT
    assert ctrl.copy_counter == 0
    assert not ctrl.main_paused


if __name__ == '__main__':
    import inspect
    mod = sys.modules[__name__]
    tests = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
             if n.startswith('test_')]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS  {name}')
        except AssertionError as e:
            print(f'FAIL  {name}: {e}')
            fails += 1
    sys.exit(1 if fails else 0)
