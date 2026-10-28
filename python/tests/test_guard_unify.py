"""One algorithm, two ways of calling it: full ``process()`` vs the context seam + external rendering.

Acceptance is through the public API only. The external side is ``tests/guard_scenes.py::ExternalRenderer``: it
reads ``output_base_spec``, ``res_gain``, ``comfort_noise`` and ``post_reset_serial`` from the context, applies
the public comfort-noise reference helper ``cng_fill`` and an independent IFFT / sqrt-Hann OLA, and has to
reproduce the delivered hops of the full-mode instance bit for bit -- including hops where the output-capture
guard fires, switches, and across hard post-chain resets.
"""
import numpy as np
import pytest

from aec import AEC, AecConfig, AecPreset, AEC_CNG_SEED, cng_fill
from modules import orchestrator as orch
from modules.enums import AecDelayMode
from modules.state import AecState

from guard_scenes import (
    GRIDS, ExternalRenderer, blind_clip, cabs_np_f32, clip_scene, delay_branch_stream, expected_guard, pairwise_sum_f32,
    spectral_energy_f32, synthetic_scene,
)


def _config(sr, fft, split, enable_cng):
    cfg = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=sr, frame_size=fft)
    cfg.enable_cng = enable_cng
    if split:
        cfg.enable_res = False
        cfg.return_res_context = True
    else:
        cfg.enable_res = True
        cfg.return_res_context = False
    return cfg


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


class _Pair:
    """A full-mode instance and a split instance (context + external rendering), fed identically."""

    def __init__(self, sr, fft, enable_cng):
        np.random.seed(0)
        self.full = AEC(_config(sr, fft, False, enable_cng))
        np.random.seed(0)
        self.split = AEC(_config(sr, fft, True, enable_cng))
        self.hop = self.full.hop_size
        self.renderer = ExternalRenderer(fft, self.hop, enable_cng)
        self.full_out, self.split_out, self.ctxs = [], [], []

    def step(self, near, far):
        full = self.full.process(near.copy(), far.copy())
        _raw, ctx = self.split.process(near.copy(), far.copy())
        self.full_out.append(np.asarray(full, np.float32))
        self.split_out.append(self.renderer.render(ctx))
        self.ctxs.append(ctx)
        assert self.full.get_post_reset_serial() == self.split.get_post_reset_serial() == ctx.post_reset_serial

    def reset(self):
        self.full.reset()
        self.split.reset()

    def assert_identical(self):
        a, b = np.concatenate(self.full_out), np.concatenate(self.split_out)
        bad = np.nonzero(_bits(a) != _bits(b))[0]
        assert bad.size == 0, "first differing hop %d (%d samples differ)" % (bad[0] // self.hop, bad.size)

    @property
    def fired(self):
        return sum(int(c.guard_active) for c in self.ctxs)

    @property
    def usable_hops(self):
        return sum(int(bool(c.usable_linear)) for c in self.ctxs)


def _hops_for(scene, sr, hop, n):
    if scene.endswith("_clip"):
        return clip_scene(scene.split("_")[0], sr, hop, n)
    return synthetic_scene(scene, hop, sr, n)


@pytest.mark.parametrize("enable_cng", [False, True])
@pytest.mark.parametrize("scene", ["uncorrelated", "burst", "echo", "delayed_echo", "farend_clip", "nearend_clip"])
@pytest.mark.parametrize("sr,fft", GRIDS)
def test_full_equals_context_plus_external_rendering(sr, fft, scene, enable_cng):
    pair = _Pair(sr, fft, enable_cng)
    for near, far in _hops_for(scene, sr, pair.hop, 420 if scene == "farend_clip" else 200):
        pair.step(near, far)
    pair.assert_identical()
    if scene in ("uncorrelated", "burst", "farend_clip", "nearend_clip"):
        assert pair.fired >= 3, "scene %s must fire the guard" % scene
    if scene == "echo":
        assert pair.fired <= 2 and pair.usable_hops > 50
    if scene in ("uncorrelated", "nearend_clip"):
        assert pair.usable_hops == 0              # never-usable clips are covered
    if scene == "farend_clip":
        assert pair.fired < len(pair.ctxs) and pair.usable_hops > 0   # guard on and off, unusable and usable
    if scene == "delayed_echo":
        assert pair.renderer.resets >= 1, "the first delay acquisition must hard-reset the post chain"


@pytest.mark.parametrize("sr,fft", GRIDS)
def test_split_context_carries_the_same_analysis_state_as_full_mode_with_context(sr, fft):
    """Same config except the RES / output ownership switch: every context field is bit-identical, so the
    split API does not change the analysis state (the seam fields, gain, comfort noise, R^2, guard)."""
    np.random.seed(0)
    full_cfg = _config(sr, fft, False, True)
    full_cfg.return_res_context = True
    full = AEC(full_cfg)
    np.random.seed(0)
    split = AEC(_config(sr, fft, True, True))
    hops = clip_scene("farend", sr, full.hop_size, 220)
    names = ("raw_output", "formed_output", "echo_spec", "near_spec", "error_spec", "output_base_spec",
             "res_gain", "comfort_noise", "r2", "far_spec")
    for near, far in hops:
        _o1, c1 = full.process(near.copy(), far.copy())
        _o2, c2 = split.process(near.copy(), far.copy())
        for name in names:
            np.testing.assert_array_equal(getattr(c1, name), getattr(c2, name), err_msg=name)
        assert (c1.guard_active, c1.usable_linear, c1.post_reset_serial) == \
               (c2.guard_active, c2.usable_linear, c2.post_reset_serial)


@pytest.mark.parametrize("sr,fft", GRIDS)
def test_forced_verdict_switching_keeps_full_and_split_identical(sr, fft, monkeypatch):
    """Usable / unusable switching every few hops on a scene whose residual is louder than the capture: the guard
    turns on and off, once per hop, in both API modes."""
    clock = {"hop": 0}
    monkeypatch.setattr(AecState, "usable_linear_estimate",
                        lambda _self: (clock["hop"] // 7) % 3 == 0)
    pair = _Pair(sr, fft, True)
    modes = set()
    for i, (near, far) in enumerate(synthetic_scene("uncorrelated", pair.hop, sr, 180)):
        clock["hop"] = i
        pair.step(near, far)
        modes.add((bool(pair.ctxs[-1].usable_linear), bool(pair.ctxs[-1].guard_active)))
    pair.assert_identical()
    assert (True, False) in modes and (False, True) in modes, modes
    assert (True, True) not in modes     # the guard never fires on a usable hop


@pytest.mark.parametrize("sr,fft", GRIDS)
def test_stream_reset_and_replay(sr, fft):
    """reset() mid-stream counts as one hard post-chain reset; the external renderer follows through the serial
    alone and stays bit-identical to the full-mode instance, also when the same stream is replayed."""
    pair = _Pair(sr, fft, True)
    hops = clip_scene("farend", sr, pair.hop, 90)
    for near, far in hops:
        pair.step(near, far)
    s0 = pair.full.get_post_reset_serial()
    pair.reset()
    assert pair.full.get_post_reset_serial() == (s0 + 1) & 0xFFFFFFFF
    assert pair.full.get_post_reset_serial() == pair.full.get_post_reset_serial()   # non-consuming
    for near, far in hops:                    # replay
        pair.step(near, far)
    pair.reset()
    pair.reset()                              # two resets between two hops: the serial moves by two
    for near, far in hops[:40]:
        pair.step(near, far)
    pair.assert_identical()
    assert pair.renderer.resets >= 2


@pytest.mark.parametrize("sr,fft", GRIDS)
def test_serial_counts_hard_resets_only(sr, fft):
    """A first delay acquisition (full recovery chain) moves the serial; a steady stream does not."""
    aec = AEC(_config(sr, fft, True, True))
    hops = synthetic_scene("delayed_echo", aec.hop_size, sr, 120)
    serials = []
    for near, far in hops:
        _o, ctx = aec.process(near, far)
        serials.append(ctx.post_reset_serial)
    changes = [i for i in range(1, len(serials)) if serials[i] != serials[i - 1]]
    assert len(changes) >= 1
    assert serials[0] == 0 and serials[-1] == len(changes)
    assert aec.get_post_reset_serial() == serials[-1]


# ---------------------------------------------------------------- delay branches vs the post-reset serial
# Mirror of test_guard_unify.c sections 5 / 6. Every delay event of the split instance is classified from the state that
# decides it (Path A first acquisition: warm tap transfer / DT-aware soft acquisition / hard; Path B shift: DT-aware soft
# realign / hard; bad-relock revert, read from the reject list and checked against its delay signature). The contract
# (spec section 2): the serial moves by exactly 1 on the hard branches and by 0 on every other one, and never without a
# hard event. The revert additionally keeps the full-mode comfort-noise LCG running, and a counterfactual renderer that
# clears its OLA tail and reseeds its CNG on the revert hop must NOT match process() there (so the identity can see it).

def _lcg_advance(seed, n):
    for _ in range(n):
        seed = (seed * 69069 + 1) % (1 << 31)
    return seed


class _DelayEvents:
    def __init__(self):
        self.pa_warm = self.pa_soft = self.pa_hard = self.pb_soft = self.pb_hard = self.reverts = 0
        self.serial_changes = 0
        self.bad = []
        self.revert_cf_checked = 0


def _run_delay_branches(hops, sr, fft, warm_transfer=True):
    pair = _Pair.__new__(_Pair)
    cfgs = []
    for split in (False, True):
        cfg = _config(sr, fft, split, True)
        cfg.delay_acquire_warm_transfer = warm_transfer
        cfgs.append(cfg)
    np.random.seed(0)
    pair.full = AEC(cfgs[0])
    np.random.seed(0)
    pair.split = AEC(cfgs[1])
    pair.hop = pair.full.hop_size
    pair.renderer = ExternalRenderer(fft, pair.hop, True)
    pair.full_out, pair.split_out, pair.ctxs = [], [], []
    a = pair.split
    cfg = a.config
    nb = fft // 2 + 1
    cf = ExternalRenderer(fft, pair.hop, True)    # counterfactual renderer, until the first revert
    cf_active = True
    ev = _DelayEvents()
    last_pb = None                                # (hop, old delay, new delay) of the last Path B shift
    for h, (near, far) in enumerate(hops):
        d0, s0 = a._current_delay, a.get_post_reset_serial()
        rej0 = len(getattr(a, "_delay_reject_values", ()))
        cng0 = pair.full._aec3_cng_seed
        span = a.filter.tap_span
        warm_pred = (cfg.delay_acquire_warm_transfer
                     and max(list(a._erle_slope_buf)[-15:] or [0.0]) > float(cfg.delay_acquire_inst_erle_db))
        pair.step(near, far)
        d1, s1 = a._current_delay, a.get_post_reset_serial()
        assert pair.full._current_delay == d1, "hop %d: full and split took different delay decisions" % h
        ne_soft = cfg.dt_aware_recovery_soft and a._ne_recent_frames > 0
        revert = len(getattr(a, "_delay_reject_values", ())) > rej0
        want, event = 0, True
        if revert:
            ev.reverts += 1
            if not (last_pb is not None and h - last_pb[0] == orch._DELAY_SHIFT_JUDGMENT_HOPS
                    and d0 == last_pb[2] and d1 == last_pb[1] and d1 != d0):
                ev.bad.append("hop %d: revert without its signature (delay %d -> %d, last shift %s)" % (h, d0, d1, last_pb))
            c1 = pair.full._aec3_cng_seed
            if not (c1 == cng0 or c1 == _lcg_advance(cng0, nb - 2)):
                ev.bad.append("hop %d: the full-mode comfort-noise LCG was reseeded on the revert" % h)
        elif d1 != d0 and d0 < 0:
            if warm_pred and 0 < d1 < span:
                ev.pa_warm += 1
            elif ne_soft:
                ev.pa_soft += 1
            else:
                ev.pa_hard += 1
                want = 1
        elif d1 != d0:
            if ne_soft:
                ev.pb_soft += 1
            else:
                ev.pb_hard += 1
                want = 1
            last_pb = (h, d0, d1)
        else:
            event = False
        delta = (s1 - s0) & 0xFFFFFFFF
        if delta:
            ev.serial_changes += 1
        if event and delta != want:
            ev.bad.append("hop %d: delay %d -> %d (revert %s, ne_soft %s) moved the serial by %d, contract %d"
                          % (h, d0, d1, revert, ne_soft, delta, want))
        if not event and delta:
            ev.bad.append("hop %d: the serial moved by %d without a hard delay event" % (h, delta))
        if cf_active:
            if revert:
                cf.serial = -1                    # act as if the serial had moved: clear the OLA tail, reseed the CNG
            out_cf = cf.render(pair.ctxs[-1])
            if revert:
                ev.revert_cf_checked += 1
                if np.array_equal(_bits(out_cf), _bits(pair.full_out[-1])):
                    ev.bad.append("hop %d: clearing / reseeding on the revert hop leaves the audio unchanged, so the "
                                  "identity check could not see a reset there" % h)
                cf_active = False
    pair.assert_identical()
    assert not ev.bad, "\n".join(ev.bad[:10])
    return ev


@pytest.mark.parametrize("sr,fft", GRIDS)
def test_dt_aware_soft_acquisition_and_soft_shifts_do_not_move_the_serial(sr, fft):
    ev = _run_delay_branches(delay_branch_stream("dt_soft", sr, fft, 14), sr, fft, warm_transfer=False)
    assert ev.pa_soft >= 1, "a DT-aware soft acquisition must be reached (%s)" % vars(ev)
    assert ev.pb_soft >= 2, "DT-aware soft shifts must be reached (%s)" % vars(ev)
    assert ev.serial_changes == 0


@pytest.mark.parametrize("sr,fft", GRIDS)
def test_hard_delay_events_move_the_serial_and_the_bad_relock_revert_does_not(sr, fft):
    ev = _run_delay_branches(delay_branch_stream("relock", sr, fft, 14), sr, fft)
    assert ev.pa_hard + ev.pa_warm >= 1 and ev.pb_hard >= 1, vars(ev)
    assert ev.serial_changes == ev.pa_hard + ev.pb_hard
    if sr == 16000:
        assert ev.reverts >= 1 and ev.revert_cf_checked == 1, "the bad-relock revert must be reached (%s)" % vars(ev)


@pytest.mark.parametrize("rel,sr,fft,need", [
    # warm tap-transfer acquisition, DT-aware soft shifts and a bad-relock revert in one clip
    ("aec_challenge_blind/farend_singletalk/Y91uE2tRg0SUB2a9XjT30w_farend_singletalk_with_movement_mic.wav", 16000, 256,
     ("pa_warm", "pb_soft", "reverts")),
    # hard acquisition, hard shift and a revert late in the clip
    ("aec_challenge_blind/farend_singletalk/IUp2c0A4yEyttmLfJ3t0Xw_farend_singletalk_mic.wav", 16000, 512,
     ("pa_hard", "pb_hard", "reverts")),
])
def test_delay_branches_on_blind_clips(rel, sr, fft, need):
    ev = _run_delay_branches(blind_clip(rel, sr, fft // 2), sr, fft)
    for k in need:
        assert getattr(ev, k) >= 1, "%s must be reached (%s)" % (k, vars(ev))


@pytest.mark.parametrize("warm", [True, False])
def test_soft_external_realign_does_not_move_the_serial(warm):
    """apply_external_realign (warm IR transfer or filter-only reset) leaves the OLA / CNG untouched, so it must not
    raise the post reset event, and full vs split stay bit-identical across it."""
    sr, fft, hop, delay = 16000, 512, 256, 300
    rng = np.random.default_rng(0x5EED)
    far_hist = (rng.standard_normal(4 * delay + 360 * hop) * 0.25).astype(np.float32)
    pad = 4 * delay

    def make(split):
        cfg = AecConfig(sample_rate=sr, frame_size=fft, hop_size=hop, enable_delay_est=False, fixed_delay_samples=-1,
                        enable_cng=True, enable_res=not split, return_res_context=split)
        a = AEC(cfg)
        assert a.config.delay_mode is AecDelayMode.EXTERNAL_ALIGNED
        return a

    full, split = make(False), make(True)
    renderer = ExternalRenderer(fft, hop, True)
    outs_f, outs_s = [], []
    n_pre = 250 if warm else 6
    align = 0
    for h in range(n_pre + 40):
        if h == n_pre:
            r_f = full.apply_external_realign(delay)
            r_s = split.apply_external_realign(delay)
            assert r_f == r_s == (1 if warm else 0)
            align = delay
        base = pad + h * hop
        mic = 0.5 * far_hist[base - delay:base - delay + hop]
        far = far_hist[base - align:base - align + hop]
        outs_f.append(np.asarray(full.process(mic.copy(), far.copy()), np.float32))
        _o, ctx = split.process(mic.copy(), far.copy())
        outs_s.append(renderer.render(ctx))
    assert full.get_post_reset_serial() == split.get_post_reset_serial() == 0
    assert renderer.resets == 0
    np.testing.assert_array_equal(_bits(np.concatenate(outs_f)), _bits(np.concatenate(outs_s)))


# ---------------------------------------------------------------- comfort-noise reference helper

def _lcg_oracle(seed, n):
    """Independent LCG of the C side: rng = (rng * 69069 + 1) & 0x7FFFFFFF, ix = rng >> 26."""
    out = []
    for _ in range(n):
        seed = (seed * 69069 + 1) % (1 << 31)
        out.append(seed >> 26)
    return seed, out


def test_cng_fill_matches_the_in_chain_arithmetic_and_schedule():
    """With no NR gain the public helper reproduces the in-chain comfort noise: same draws, same bins, same
    product order ng * (nf * lut). The expected values here come from an independent scalar evaluation."""
    n = 129
    rng_ = np.random.RandomState(5)
    cn = (rng_.uniform(0.0, 1.0, n) * 32768.0 ** 2 * 1e-6).astype(np.float32)
    g = rng_.uniform(0.0, 1.2, n).astype(np.float32)             # includes g > 1: clamps to zero noise
    spec0 = (rng_.randn(n) + 1j * rng_.randn(n)).astype(np.complex64)
    spec = spec0.copy()
    state = cng_fill(AEC_CNG_SEED, spec, cn, g)
    assert AEC_CNG_SEED == 42
    end_state, draws = _lcg_oracle(AEC_CNG_SEED, n - 2)
    assert state == end_state                                    # exactly one draw per bin 1 .. n-2
    lut = np.array([np.sqrt(2.0) * np.sin(2.0 * np.pi * i / 32.0) for i in range(32)], dtype=np.float32)
    want = spec0.copy()
    f = np.float32
    for k in range(1, n - 1):
        ix = draws[k - 1]
        nf = f(np.sqrt(f(max(f(cn[k] / f(32768.0 ** 2)), f(0.0)))))
        t = f(f(1.0) - f(g[k] * g[k]))
        ng = f(np.sqrt(t)) if t > 0 else f(0.0)
        want[k] = np.complex64(complex(f(spec0[k].real + f(ng * f(nf * lut[ix]))),
                                       f(spec0[k].imag + f(ng * f(nf * lut[(ix + 8) & 31])))))
    np.testing.assert_array_equal(_bits(spec.real), _bits(want.real))
    np.testing.assert_array_equal(_bits(spec.imag), _bits(want.imag))
    assert spec[0] == spec0[0] and spec[-1] == spec0[-1]          # DC and Nyquist untouched
    # Too few bins: nothing happens and the state is returned unchanged.
    tiny = np.zeros(2, np.complex64)
    assert cng_fill(7, tiny, cn[:2], g[:2]) == 7 and not tiny.any()


def test_cng_fill_nr_gain_scales_the_amplitude():
    n = 33
    rng_ = np.random.RandomState(9)
    cn = (rng_.uniform(0.5, 1.0, n) * 32768.0 ** 2 * 1e-5).astype(np.float32)
    g = np.full(n, 0.5, np.float32)
    nr = rng_.uniform(0.0, 1.0, n).astype(np.float32)
    a = np.zeros(n, np.complex64)
    b = np.zeros(n, np.complex64)
    s1 = cng_fill(AEC_CNG_SEED, a, cn, g, nr, 0.25)
    s2 = cng_fill(AEC_CNG_SEED, b, cn, g)
    assert s1 == s2                                              # the NR gain never changes the draw schedule
    ratio = np.maximum(nr, np.float32(0.25))[1:-1]
    np.testing.assert_allclose(np.abs(a[1:-1]), np.abs(b[1:-1]) * ratio, rtol=1e-5)


# ---------------------------------------------------------------- guard decision at the threshold

def test_magnitude_kernel_replica_equals_numpy_abs():
    """The guard reads np.abs(complex64); it has to be the C kernel's scaled hypot, else the decision could differ
    from the C side at the threshold."""
    r = np.random.RandomState(3)
    z = ((r.randn(200000) + 1j * r.randn(200000)) * 10.0 ** r.uniform(-4, 0, 200000)).astype(np.complex64)
    z[:4] = [0, 1e-30, 3 - 4j, -2j]
    np.testing.assert_array_equal(_bits(cabs_np_f32(z)), _bits(np.abs(z)))


def test_pairwise_replica_equals_numpy_for_the_legal_grids():
    for n in (129, 257, 513):
        for seed in range(6):
            a = (np.random.RandomState(100 * n + seed).randn(n) ** 2).astype(np.float32)
            assert pairwise_sum_f32(a) == np.sum(a, dtype=np.float32), (n, seed)


@pytest.mark.parametrize("sr,fft", GRIDS)
def test_guard_decision_at_the_float32_threshold(sr, fft):
    """E energy equal to Y energy (strict '>' => no fire), and the same spectra with one bin moved by +/- 1 ulp.
    Expected decisions come from the independent pairwise-sum replica, not from the code under test."""
    aec = AEC(_config(sr, fft, True, False))
    n = fft // 2 + 1
    decisions = set()
    flips = []
    equal_cases = 0
    for seed in range(2):
        rng_ = np.random.RandomState(seed)
        e = ((rng_.randn(n) + 1j * rng_.randn(n)) * 0.01).astype(np.complex64)
        # Y = E mirrored in the real part: identical per-bin magnitudes, hence identical sums; echo = Y - E.
        y = (-e.real + 1j * e.imag).astype(np.complex64)
        echo = (y - e).astype(np.complex64)
        assert np.array_equal(e + echo, y)                      # exact, so Y really is the mirror
        base, active = aec._post_output_base(e, echo, False)
        assert not active and base is e                          # equal energies: strict '>' keeps E
        assert spectral_energy_f32(e) == spectral_energy_f32(y)
        for direction in (np.inf, -np.inf):
            # Bump bins 1..m of E by one ulp each (echo fixed, so Y follows): a single bin is absorbed by the
            # float32 sum, a growing prefix walks the energy difference across 0 / +1 ulp of the sum.
            e2 = e.copy()
            for m in range(1, n - 1):
                # direction > 0 moves |Re E| up (and |Re Y| down, as echo is fixed); direction < 0 the reverse.
                e2.real[m] = np.nextafter(e.real[m], np.float32(direction * np.sign(e.real[m])))
                y2 = (e2 + echo).astype(np.complex64)
                energy_e, energy_y = spectral_energy_f32(e2), spectral_energy_f32(y2)
                want = bool(energy_e > energy_y)
                base, active = aec._post_output_base(e2, echo, False)
                assert active is want, (seed, m, direction, energy_e, energy_y)
                assert (base is not e2) is want
                if want:
                    np.testing.assert_array_equal(base, y2)
                    flips.append((energy_e, energy_y))
                decisions.add(want)
                # A usable verdict never fires, whatever the energies are.
                assert aec._post_output_base(e2, echo, True)[1] is False
                if energy_e == energy_y:
                    equal_cases += 1
    assert decisions == {True, False}, "the fixture must straddle the threshold"
    assert equal_cases >= 1, "the fixture must contain exactly equal energies (strict '>' keeps E)"
    # The firing side is a genuine ulp-level win: every firing case exceeds by at most a few float32 ulps.
    assert all(ee > ey and ee <= ey * np.float32(1.0 + 1e-5) for ee, ey in flips)


def test_guard_can_be_disabled_by_config():
    cfg = _config(16000, 256, True, False)
    cfg.output_capture_when_linear_unusable = False
    aec = AEC(cfg)
    e = np.full(129, 1.0 + 0j, np.complex64)
    echo = np.full(129, -1.0 + 0j, np.complex64)               # Y = 0: any enabled guard would fire
    assert aec._post_output_base(e, echo, False)[1] is False
