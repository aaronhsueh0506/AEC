"""Three-state delay mode (``AecConfig.delay_mode``) — config surface,
deprecated-field translation, and the behavioural identity of each mode.

Why this test exists: alignment used to be decided by TWO fields whose
relationship was implicit — ``enable_delay_est`` plus an undocumented
"``fixed_delay_samples >= 0`` silently overrides it" rule inside the
orchestrator. Nothing named the third state at all (far pre-aligned by the
caller), and nothing rejected a combination the selected behaviour could not
honour: ``AecConfig(fixed_delay_samples=1600)`` looked like it enabled a
fixed delay while ``enable_delay_est`` still read True. ``delay_mode`` is now
the single source of truth, ``enable_delay_est`` is a deprecated mirror
folded into it by one translation step, and every illegal combination
raises.

What is asserted:
  1. THE DEFAULT IS UNCHANGED — ``AecConfig()`` resolves to MATCHED / n=5 /
     fixed=-1, and produces output bit-identical to an explicit
     ``delay_mode=MATCHED``. (The cross-commit half of the byte-exact gate —
     "identical to the pre-delay_mode implementation" — is proven out of
     band against the stable baseline build; what a permanent test CAN pin
     is that the new spelling never diverges from the default path.)
  2. TRANSLATION IS EXACT (the can-fail core) — every row of the mapping
     table in ``AecConfig._resolve_delay_mode`` is checked, AND the legacy
     spelling is proven to produce bit-identical AUDIO to the explicit
     spelling for both non-MATCHED modes. Breaking one row of the table
     therefore fails here even if the resolved enum happens to survive.
  3. ILLEGAL COMBINATIONS RAISE — the full mode x field matrix, including
     ``delay_num_filters=0`` (never a silent "off" switch, in any mode).
  4. THE MODES ARE ACTUALLY DIFFERENT — FIXED applies the caller's delay
     with no estimator built; EXTERNAL_ALIGNED builds neither estimator nor
     ring. Without this a translation that resolved to the right enum but
     wired the wrong branch would pass everything above.
  5. LATE MUTATION STILL TRANSLATES — ``cfg.enable_delay_est = False`` after
     construction is honoured at AEC() time, mirroring the C port (which
     resolves inside ``aec_create()``, not inside ``aec_config_defaults()``).
  6. THE RING IS SIZED PER MODE (plan §3.2.8-9, step 3) — ``MATCHED`` keeps
     the search ring (``max(delay_buffer_ms, max_delay_ms + 4096)``, which
     also gates which estimates the controller may accept), ``FIXED`` carves
     exactly ``fixed_delay_samples + hop`` and is INERT to those two knobs,
     and ``EXTERNAL_ALIGNED`` carves nothing and leaves no other
     delay-attached state. All three mirror C's ``aec_ref_ring_samples()``.
  7. THE EXACT-FIT RING STILL SERVES THE RIGHT SAMPLES — the far the engine
     actually consumes is byte-equal to the caller's far delayed by exactly
     ``fixed_delay_samples``, across hundreds of wraps (the split read is
     now the common path, not an edge case), through ``reset()``'s refill,
     and a real synthetic echo is cancelled only when the delay is right.

Mutation checks (each breaks one line and must go red here):
  - drop the ``fixed_delay_samples >= 0`` arm of the translation (always
    resolve to EXTERNAL_ALIGNED) -> "legacy fixed spelling" rows fail;
  - drop the ``enable_delay_est`` guard (translate unconditionally) ->
    the MATCHED default resolves to EXTERNAL_ALIGNED and rows 1/4 fail;
  - drop the mirror rewrite -> the idempotency/round-trip rows fail;
  - relax any ``raise`` in ``_resolve_delay_mode`` -> the matrix in 3 fails;
  - drop the ``+ hop`` from ``ref_ring_samples``'s FIXED arm -> the ring
    sizing rows and the wrap/refill alignment rows fail;
  - shift the ring read offset by one hop/sample -> every "served far is
    far[t-N]" row fails;
  - hand EXTERNAL_ALIGNED a ring -> the no-ring rows fail.

Run:
    python3 -m pytest python/tests/test_delay_mode.py
"""
from __future__ import annotations

import dataclasses
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aec import AEC, AecConfig
from modules.config import ref_ring_samples
from modules.enums import AecDelayMode


_SR = 16000
_FIXED = 1600          # 100 ms at 16 kHz — inside every ring, no estimator needed


def _synth(sample_rate: int, seconds: float, delay_samples: int, seed: int = 11):
    """White far-end plus a pure delayed/attenuated copy as the near-end."""
    rng = np.random.default_rng(seed)
    total = int(seconds * sample_rate)
    far = rng.standard_normal(total + delay_samples).astype(np.float32) * 0.2
    near = np.zeros_like(far)
    near[delay_samples:] = far[:-delay_samples] * 0.5
    return near, far


def _run(cfg: AecConfig, near, far, seconds: float = 1.5):
    np.random.seed(42)   # CNG determinism
    aec = AEC(cfg)
    hop = int(cfg.hop_size)
    n_hops = int(seconds * cfg.sample_rate) // hop
    out = [aec.process(near[i * hop:(i + 1) * hop], far[i * hop:(i + 1) * hop])
           for i in range(n_hops)]
    return np.concatenate(out), aec


class DelayModeDefaultTests(unittest.TestCase):
    """The shipped default path must be indistinguishable from before."""

    def test_default_is_matched_n5_unset_fixed(self) -> None:
        cfg = AecConfig()
        self.assertIs(cfg.delay_mode, AecDelayMode.MATCHED)
        self.assertEqual(cfg.delay_num_filters, 5)
        self.assertEqual(cfg.fixed_delay_samples, -1)
        # The deprecated mirror is rewritten from the resolved mode.
        self.assertIs(cfg.enable_delay_est, True)

    def test_explicit_matched_is_bit_identical_to_the_default(self) -> None:
        near, far = _synth(_SR, 1.5, 800)
        a, _ = _run(AecConfig(), near, far)
        b, _ = _run(AecConfig(delay_mode=AecDelayMode.MATCHED), near, far)
        self.assertTrue(np.array_equal(a, b),
                        "explicit MATCHED diverged from the default path")

    def test_from_preset_keeps_the_default_mode(self) -> None:
        for preset in ("mild", "balanced", "aggressive"):
            cfg = AecConfig.from_preset(preset)
            self.assertIs(cfg.delay_mode, AecDelayMode.MATCHED, preset)
            self.assertEqual(cfg.fixed_delay_samples, -1, preset)


class DelayModeTranslationTests(unittest.TestCase):
    """Every row of the deprecated-field mapping table."""

    def test_mapping_table(self) -> None:
        rows = [
            # (enable_delay_est, fixed, delay_mode in, expected resolved)
            (True,  -1, AecDelayMode.MATCHED, AecDelayMode.MATCHED),
            (False, -1, AecDelayMode.MATCHED, AecDelayMode.EXTERNAL_ALIGNED),
            (False, _FIXED, AecDelayMode.MATCHED, AecDelayMode.FIXED),
            (False, _FIXED, AecDelayMode.FIXED, AecDelayMode.FIXED),
            (False, -1, AecDelayMode.EXTERNAL_ALIGNED,
             AecDelayMode.EXTERNAL_ALIGNED),
            (True,  _FIXED, AecDelayMode.FIXED, AecDelayMode.FIXED),
            (True,  -1, AecDelayMode.EXTERNAL_ALIGNED,
             AecDelayMode.EXTERNAL_ALIGNED),
        ]
        for est, fixed, mode_in, expected in rows:
            with self.subTest(est=est, fixed=fixed, mode_in=mode_in.name):
                cfg = AecConfig(enable_delay_est=est,
                                fixed_delay_samples=fixed,
                                delay_mode=mode_in)
                self.assertIs(cfg.delay_mode, expected)
                # Mirror rewritten from the resolved mode, both directions.
                self.assertEqual(cfg.enable_delay_est,
                                 expected is AecDelayMode.MATCHED)

    def test_coerce_accepts_int_and_string_spellings(self) -> None:
        for spelling in (1, "fixed", "FIXED", " Fixed ", AecDelayMode.FIXED):
            with self.subTest(spelling=spelling):
                cfg = AecConfig(delay_mode=spelling,
                                fixed_delay_samples=_FIXED)
                self.assertIs(cfg.delay_mode, AecDelayMode.FIXED)

    def test_coerce_rejects_garbage(self) -> None:
        for bad in (3, -1, "matched_filter", True, None, 2.0):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    AecConfig(delay_mode=bad)

    def test_resolution_is_idempotent_across_replace_roundtrips(self) -> None:
        cfg = AecConfig(enable_delay_est=False, fixed_delay_samples=_FIXED)
        self.assertIs(cfg.delay_mode, AecDelayMode.FIXED)
        again = dataclasses.replace(cfg)
        self.assertIs(again.delay_mode, AecDelayMode.FIXED)
        self.assertEqual(again.fixed_delay_samples, _FIXED)
        self.assertFalse(again.enable_delay_est)
        third = AecConfig(**dataclasses.asdict(again))
        self.assertIs(third.delay_mode, AecDelayMode.FIXED)

    def test_legacy_spelling_is_bit_identical_to_explicit_external(self) -> None:
        near, far = _synth(_SR, 1.5, 800)
        legacy, _ = _run(AecConfig(enable_delay_est=False), near, far)
        explicit, _ = _run(
            AecConfig(delay_mode=AecDelayMode.EXTERNAL_ALIGNED), near, far)
        self.assertTrue(np.array_equal(legacy, explicit))

    def test_legacy_spelling_is_bit_identical_to_explicit_fixed(self) -> None:
        near, far = _synth(_SR, 1.5, _FIXED)
        legacy, _ = _run(AecConfig(enable_delay_est=False,
                                   fixed_delay_samples=_FIXED), near, far)
        explicit, _ = _run(AecConfig(delay_mode=AecDelayMode.FIXED,
                                     fixed_delay_samples=_FIXED), near, far)
        self.assertTrue(np.array_equal(legacy, explicit))

    def test_late_mutation_of_the_deprecated_mirror_is_honoured(self) -> None:
        """``cfg.enable_delay_est = False`` after construction still works.

        Mirrors the C port, which resolves inside aec_create()/aec_init()
        rather than aec_config_defaults(), so a caller that pokes the field
        on an already-built config (a shape several call sites in this repo
        use) gets the mode it asked for rather than silently keeping
        MATCHED.
        """
        cfg = AecConfig()
        cfg.enable_delay_est = False
        aec = AEC(cfg)
        self.assertIs(cfg.delay_mode, AecDelayMode.EXTERNAL_ALIGNED)
        self.assertIsNone(aec.delay_est)
        self.assertFalse(aec._delay_active)


class DelayModeIllegalCombinationTests(unittest.TestCase):
    """Illegal mode x field combinations are REJECTED, never normalised."""

    def test_fixed_delay_outside_fixed_mode_raises(self) -> None:
        for mode in (AecDelayMode.MATCHED, AecDelayMode.EXTERNAL_ALIGNED):
            with self.subTest(mode=mode.name):
                with self.assertRaises(ValueError):
                    AecConfig(delay_mode=mode, fixed_delay_samples=_FIXED)
                with self.assertRaises(ValueError):
                    AecConfig(delay_mode=mode, fixed_delay_samples=0)

    def test_fixed_mode_without_a_fixed_delay_raises(self) -> None:
        with self.assertRaises(ValueError):
            AecConfig(delay_mode=AecDelayMode.FIXED)
        with self.assertRaises(ValueError):
            AecConfig(delay_mode=AecDelayMode.FIXED, fixed_delay_samples=-1)

    def test_fixed_delay_beyond_the_ring_bound_raises(self) -> None:
        AecConfig(delay_mode=AecDelayMode.FIXED,
                  fixed_delay_samples=120 * _SR)          # exactly at the bound
        with self.assertRaises(ValueError):
            AecConfig(delay_mode=AecDelayMode.FIXED,
                      fixed_delay_samples=120 * _SR + 1)

    def test_non_default_bank_size_outside_matched_raises(self) -> None:
        for mode, fixed in ((AecDelayMode.FIXED, _FIXED),
                            (AecDelayMode.EXTERNAL_ALIGNED, -1)):
            for n in (1, 2, 3, 4):
                with self.subTest(mode=mode.name, n=n):
                    with self.assertRaises(ValueError):
                        AecConfig(delay_mode=mode, fixed_delay_samples=fixed,
                                  delay_num_filters=n)
            # the default is accepted in every mode
            AecConfig(delay_mode=mode, fixed_delay_samples=fixed,
                      delay_num_filters=5)

    def test_zero_bank_size_is_never_a_silent_off_switch(self) -> None:
        for mode, fixed in ((AecDelayMode.MATCHED, -1),
                            (AecDelayMode.FIXED, _FIXED),
                            (AecDelayMode.EXTERNAL_ALIGNED, -1)):
            with self.subTest(mode=mode.name):
                with self.assertRaises(ValueError):
                    AecConfig(delay_mode=mode, fixed_delay_samples=fixed,
                              delay_num_filters=0)

    def test_legacy_implicit_override_is_gone(self) -> None:
        """The old "fixed>=0 silently overrides enable_delay_est" rule.

        It used to construct fine and quietly run in fixed-delay mode with
        ``enable_delay_est`` still reading True. It now raises: MATCHED
        cannot honour a fixed delay, so the caller must say which they meant.
        """
        with self.assertRaises(ValueError):
            AecConfig(fixed_delay_samples=_FIXED)          # enable_delay_est=True


class DelayModeBehaviourTests(unittest.TestCase):
    """The three modes must actually wire different machinery."""

    def test_matched_builds_an_estimator_and_a_ring(self) -> None:
        near, far = _synth(_SR, 1.5, 800)
        _, aec = _run(AecConfig(), near, far)
        self.assertIsNotNone(aec.delay_est)
        self.assertTrue(aec._delay_active)
        self.assertGreater(aec._ref_ring_size, 0)

    def test_fixed_applies_the_delay_with_no_estimator(self) -> None:
        near, far = _synth(_SR, 1.5, _FIXED)
        cfg = AecConfig(delay_mode=AecDelayMode.FIXED,
                        fixed_delay_samples=_FIXED)
        _, aec = _run(cfg, near, far)
        self.assertIsNone(aec.delay_est)
        self.assertTrue(aec._delay_active)
        # The applied delay is the caller's, unchanged, for the whole run.
        self.assertEqual(aec._current_delay, _FIXED)

    def test_fixed_ring_covers_a_delay_beyond_max_delay_ms(self) -> None:
        """The ring covers a delay past the max_delay_ms budget -- exactly.

        max_delay_ms/delay_buffer_ms size the MATCHED SEARCH ring and are
        inert here (plan step 3): what the ring must hold is this one
        immutable delay plus the hop being read, and nothing else.
        """
        big = 40000   # 2.5 s at 16 kHz, past the 1024 ms max_delay_ms default
        cfg = AecConfig(delay_mode=AecDelayMode.FIXED, fixed_delay_samples=big)
        aec = AEC(cfg)
        self.assertEqual(aec._ref_ring_size, big + cfg.hop_size)

    def test_external_aligned_builds_neither_estimator_nor_ring(self) -> None:
        near, far = _synth(_SR, 1.5, 800)
        cfg = AecConfig(delay_mode=AecDelayMode.EXTERNAL_ALIGNED)
        _, aec = _run(cfg, near, far)
        self.assertIsNone(aec.delay_est)
        self.assertFalse(aec._delay_active)
        self.assertFalse(hasattr(aec, '_ref_ring'))
        # ...and no other delay-attached state left dangling: the applied
        # delay is spelled 0 (the contract value, and what the C port's
        # aec_debug_status reports), not left unset. It used to be absent
        # entirely, which made get_stats() raise AttributeError in this mode.
        self.assertEqual(aec._current_delay, 0)
        self.assertEqual(aec.get_stats().delay_samples, 0)

    def test_fixed_and_external_are_not_the_same_audio(self) -> None:
        """Guards a translation that resolves right but wires one branch."""
        near, far = _synth(_SR, 1.5, _FIXED)
        fixed_out, _ = _run(AecConfig(delay_mode=AecDelayMode.FIXED,
                                      fixed_delay_samples=_FIXED), near, far)
        ext_out, _ = _run(AecConfig(delay_mode=AecDelayMode.EXTERNAL_ALIGNED),
                          near, far)
        self.assertFalse(np.array_equal(fixed_out, ext_out))

    def test_reset_reseeds_fixed_delay_not_minus_one(self) -> None:
        near, far = _synth(_SR, 1.5, _FIXED)
        cfg = AecConfig(delay_mode=AecDelayMode.FIXED,
                        fixed_delay_samples=_FIXED)
        _, aec = _run(cfg, near, far)
        aec.reset()
        self.assertEqual(aec._current_delay, _FIXED)
        self.assertEqual(aec._ref_ring_filled, 0)


# ── plan §3.2.8-9: the ring is sized per mode ────────────────────────────────

# (sample_rate, fft_size) x fixed delay rows worth distinguishing: the
# degenerate 0, a delay below one hop, one exactly one hop, an exact hop
# multiple, a NON-multiple (the case that makes the read wrap at an offset
# that moves), and one past the max_delay_ms search budget.
_RING_ROWS = [
    (16000, 256, 0),
    (16000, 256, 100),
    (16000, 256, 128),
    (16000, 256, 1536),
    (16000, 256, 1600),
    (16000, 256, 40000),
    (16000, 512, 1600),
    (48000, 1024, 4801),
]


def _cfg(sr, fft, **kw):
    return AecConfig(sample_rate=sr, frame_size=fft, hop_size=fft // 2, **kw)


class AlignmentRingSizingTests(unittest.TestCase):
    """``config.ref_ring_samples`` -- one helper, mirrored by C's
    ``aec_ref_ring_samples()``.

    Before plan step 3 all three modes that had a ring shared ONE size,
    ``max(delay_buffer_ms, max_delay_ms + 4096)``, whatever the mode knew
    about its own delay. FIXED now carves the tight bound it can prove
    (``fixed + hop``), and EXTERNAL_ALIGNED carves nothing.
    """

    def test_matched_ring_is_still_the_search_budget(self) -> None:
        """Unchanged, and deliberately so: this size also gates which
        estimates the controller may accept (``new_delay <= size - hop``),
        so shrinking it would change the DEFAULT path's audio."""
        for sr, fft in ((16000, 256), (16000, 512), (48000, 1024)):
            with self.subTest(sr=sr, fft=fft):
                cfg = _cfg(sr, fft)
                want = max(int(cfg.delay_buffer_ms * sr / 1000),
                           int(cfg.max_delay_ms * sr / 1000) + 4096)
                self.assertEqual(ref_ring_samples(cfg, cfg.hop_size), want)
                self.assertEqual(AEC(cfg)._ref_ring_size, want)

    def test_fixed_ring_is_exactly_fixed_plus_hop(self) -> None:
        for sr, fft, fixed in _RING_ROWS:
            with self.subTest(sr=sr, fft=fft, fixed=fixed):
                cfg = _cfg(sr, fft, delay_mode=AecDelayMode.FIXED,
                           fixed_delay_samples=fixed)
                want = fixed + cfg.hop_size
                self.assertEqual(ref_ring_samples(cfg, cfg.hop_size), want)
                self.assertEqual(AEC(cfg)._ref_ring_size, want)

    def test_fixed_ring_ignores_the_search_knobs(self) -> None:
        """The claim that would still hold under the OLD formula for large
        delays, and fails the moment a ``max()`` against the search budget
        comes back: those two fields must not reach this mode at all."""
        for sr, fft, fixed in _RING_ROWS:
            with self.subTest(sr=sr, fft=fft, fixed=fixed):
                base = _cfg(sr, fft, delay_mode=AecDelayMode.FIXED,
                            fixed_delay_samples=fixed)
                wide = dataclasses.replace(base, max_delay_ms=60000.0,
                                           delay_buffer_ms=120000.0)
                self.assertEqual(ref_ring_samples(wide, wide.hop_size),
                                 ref_ring_samples(base, base.hop_size))
                self.assertEqual(AEC(wide)._ref_ring_size,
                                 AEC(base)._ref_ring_size)

    def test_fixed_ring_is_dramatically_smaller_than_the_matched_ring(self) -> None:
        """The headline of step 3, as a ratio rather than a hardcoded size."""
        cfg = _cfg(16000, 256, delay_mode=AecDelayMode.FIXED,
                   fixed_delay_samples=400)          # 25 ms
        matched = ref_ring_samples(_cfg(16000, 256), 128)
        self.assertLess(ref_ring_samples(cfg, 128) * 20, matched)

    def test_external_aligned_wants_no_ring(self) -> None:
        for sr, fft in ((16000, 256), (16000, 512), (48000, 1024)):
            with self.subTest(sr=sr, fft=fft):
                cfg = _cfg(sr, fft, delay_mode=AecDelayMode.EXTERNAL_ALIGNED)
                self.assertEqual(ref_ring_samples(cfg, cfg.hop_size), 0)
                self.assertFalse(hasattr(AEC(cfg), '_ref_ring'))

    def test_helper_rejects_a_nonpositive_hop(self) -> None:
        for hop in (0, -1):
            with self.subTest(hop=hop):
                with self.assertRaises(ValueError):
                    ref_ring_samples(AecConfig(), hop)


def _capture_aligned_far(aec):
    """Record the far hop the engine actually consumed, per process() call.

    ``_render_activity.update(far_end)`` is the single call site immediately
    downstream of the ring write/read block, so its argument IS the aligned
    far -- the Python equivalent of C's ``AecLinearContext.aligned_far_hop``,
    which Python has no public seam for.
    """
    seen = []
    original = aec._render_activity.update

    def spy(far_end):
        seen.append(np.array(far_end, dtype=np.float32, copy=True))
        return original(far_end)

    aec._render_activity.update = spy
    return seen


class FixedRingAlignmentTests(unittest.TestCase):
    """An exact-fit ring wraps constantly, so the split (two-part) read is
    now the common path rather than a rare edge case. These walk hundreds of
    wraps and demand the served far be byte-equal to the caller's own far
    delayed by exactly ``fixed_delay_samples``."""

    def _drive(self, sr, fft, fixed, hops):
        cfg = _cfg(sr, fft, delay_mode=AecDelayMode.FIXED,
                   fixed_delay_samples=fixed, enable_res=False)
        np.random.seed(3)
        aec = AEC(cfg)
        hop = cfg.hop_size
        seen = _capture_aligned_far(aec)
        rng = np.random.default_rng(5)
        far = (rng.standard_normal(hops * hop).astype(np.float32) * 0.05)
        near = np.zeros_like(far)
        if fixed:
            near[fixed:] = far[:-fixed] * 0.5
        else:
            near[:] = far * 0.5
        for h in range(hops):
            aec.process(near[h * hop:(h + 1) * hop],
                        far[h * hop:(h + 1) * hop])
        return aec, far, seen, hop

    def test_aligned_far_is_the_caller_far_delayed_by_fixed(self) -> None:
        for sr, fft, fixed in _RING_ROWS:
            if fixed >= 8000:
                continue          # covered by the C wrap scenarios; slow here
            hops = 260
            with self.subTest(sr=sr, fft=fft, fixed=fixed):
                aec, far, seen, hop = self._drive(sr, fft, fixed, hops)
                ring = aec._ref_ring_size
                self.assertEqual(ring, fixed + hop)
                # Honest-RAW window: the ring cannot serve the offset until
                # the samples already written cover it, so hops
                # 0..ceil(fixed/hop)-1 carry the caller's RAW far and hop
                # ceil(fixed/hop) is the FIRST aligned one (fixed == 0 needs
                # no read at all). Both sides of that boundary are asserted:
                # skipping the first aligned hop would hide an off-by-one that
                # serves raw audio for one hop past the switch.
                raw_hops = 0 if fixed == 0 else -(-fixed // hop)
                laps = hops * hop / ring
                self.assertGreater(laps, 4.0, "ring never wrapped: weak test")
                for h in range(raw_hops):
                    np.testing.assert_array_equal(
                        seen[h], far[h * hop:(h + 1) * hop],
                        f"hop {h}: pre-fill far is not the caller's raw hop")
                for h in range(raw_hops, hops):
                    end = (h + 1) * hop - fixed
                    want = far[end - hop:end]
                    np.testing.assert_array_equal(
                        seen[h], want,
                        f"hop {h}: served far is not far[t-{fixed}]")

    def test_fixed_delay_cancels_a_known_echo(self) -> None:
        """Correctness, not just bookkeeping: a synthetic echo at a known
        delay is actually cancelled once compensated, and not at all when
        the caller's fixed delay is wrong.

        Runs the SHIPPED chain (RES on). ``enable_res=False`` is not a
        neutral simplification here: it also starves the AEC3 ERLE feed
        (``last_erle_windowed`` is only cached under
        ``enable_res or return_res_context``), which stalls this
        configuration at ~3 dB on every delay mode alike -- a property of
        that diagnostic config, not of the alignment path.
        """
        sr, hop, true_delay = 16000, 128, 1600
        rng = np.random.default_rng(7)
        n = sr * 4
        far = rng.standard_normal(n).astype(np.float32) * 0.1
        near = np.zeros_like(far)
        near[true_delay:] = far[:-true_delay] * 0.5

        def erle_db(fixed):
            cfg = _cfg(sr, 256, delay_mode=AecDelayMode.FIXED,
                       fixed_delay_samples=fixed)
            np.random.seed(1)
            aec = AEC(cfg)
            out = np.concatenate([
                aec.process(near[i * hop:(i + 1) * hop],
                            far[i * hop:(i + 1) * hop])
                for i in range(n // hop)])
            half = len(out) // 2       # steady state only
            return 10.0 * np.log10(
                float(np.sum(near[half:len(out)] ** 2))
                / max(float(np.sum(out[half:] ** 2)), 1e-20))

        # +4000 samples = 250 ms past the real path, well outside the
        # 52 ms PBFDKF span, so the filter cannot absorb it as a residual.
        right = erle_db(true_delay)
        wrong = erle_db(true_delay + 4000)
        self.assertGreater(right, 25.0,
                           f"compensated echo not cancelled (ERLE {right:.1f} dB)")
        self.assertGreater(right - wrong, 20.0,
                           f"a wrong fixed delay cancelled just as well "
                           f"({right:.1f} vs {wrong:.1f} dB) -- the ring read "
                           f"is not actually applying the delay")

    def test_reset_refills_the_ring_and_realigns(self) -> None:
        """The exact-fit ring has no slack left to hide an off-by-one
        refill: after reset() the RAW-far window must be exactly as long as
        it was at init, then the served far must be correct again."""
        sr, fft, fixed, hops = 16000, 256, 1600, 60
        aec, far, seen, hop = self._drive(sr, fft, fixed, hops)
        fill = -(-fixed // hop) + 1
        aec.reset()
        self.assertEqual(aec._ref_ring_filled, 0)
        seen.clear()
        rng = np.random.default_rng(9)
        far2 = rng.standard_normal(hops * hop).astype(np.float32) * 0.05
        near2 = np.zeros_like(far2)
        near2[fixed:] = far2[:-fixed] * 0.5
        for h in range(hops):
            aec.process(near2[h * hop:(h + 1) * hop],
                        far2[h * hop:(h + 1) * hop])
        # Inside the refill window the engine serves RAW far again...
        for h in range(fill - 1):
            np.testing.assert_array_equal(
                seen[h], far2[h * hop:(h + 1) * hop],
                f"refill hop {h}: expected RAW far while the ring is short")
        # ...and afterwards the delayed far, from the refilled ring only.
        for h in range(fill, hops):
            end = (h + 1) * hop - fixed
            np.testing.assert_array_equal(
                seen[h], far2[end - hop:end],
                f"post-refill hop {h}: served far is not far[t-{fixed}]")


class ExternalAlignedPassthroughTests(unittest.TestCase):
    """EXTERNAL_ALIGNED must hand the linear filter the caller's own far,
    sample for sample, on every hop -- there is no ring to go through."""

    def test_far_is_passed_through_sample_exact(self) -> None:
        for sr, fft in ((16000, 256), (48000, 1024)):
            with self.subTest(sr=sr, fft=fft):
                cfg = _cfg(sr, fft,
                           delay_mode=AecDelayMode.EXTERNAL_ALIGNED,
                           enable_res=False)
                np.random.seed(2)
                aec = AEC(cfg)
                hop = cfg.hop_size
                seen = _capture_aligned_far(aec)
                rng = np.random.default_rng(13)
                hops = 120
                far = rng.standard_normal(hops * hop).astype(np.float32) * 0.05
                near = np.zeros_like(far)
                near[:] = far * 0.5
                for h in range(hops):
                    aec.process(near[h * hop:(h + 1) * hop],
                                far[h * hop:(h + 1) * hop])
                for h in range(hops):
                    np.testing.assert_array_equal(
                        seen[h], far[h * hop:(h + 1) * hop],
                        f"hop {h}: external-aligned far was modified")


if __name__ == '__main__':
    unittest.main()
