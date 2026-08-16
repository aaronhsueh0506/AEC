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

Mutation checks (each breaks one line and must go red here):
  - drop the ``fixed_delay_samples >= 0`` arm of the translation (always
    resolve to EXTERNAL_ALIGNED) -> "legacy fixed spelling" rows fail;
  - drop the ``enable_delay_est`` guard (translate unconditionally) ->
    the MATCHED default resolves to EXTERNAL_ALIGNED and rows 1/4 fail;
  - drop the mirror rewrite -> the idempotency/round-trip rows fail;
  - relax any ``raise`` in ``_resolve_delay_mode`` -> the matrix in 3 fails.

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
        """The ring grows to fixed + 4096 even past the max_delay_ms budget."""
        big = 40000   # 2.5 s at 16 kHz, past the 1024 ms max_delay_ms default
        cfg = AecConfig(delay_mode=AecDelayMode.FIXED, fixed_delay_samples=big)
        aec = AEC(cfg)
        self.assertGreaterEqual(aec._ref_ring_size, big + 4096)

    def test_external_aligned_builds_neither_estimator_nor_ring(self) -> None:
        near, far = _synth(_SR, 1.5, 800)
        cfg = AecConfig(delay_mode=AecDelayMode.EXTERNAL_ALIGNED)
        _, aec = _run(cfg, near, far)
        self.assertIsNone(aec.delay_est)
        self.assertFalse(aec._delay_active)
        self.assertFalse(hasattr(aec, '_ref_ring'))

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


if __name__ == '__main__':
    unittest.main()
