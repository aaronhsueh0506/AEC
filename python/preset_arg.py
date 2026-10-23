"""Shared ``--preset`` argument parsing for the CLI entry points.

The preset is given either as the C ``AecPreset`` value (0 = mild, 1 = balanced,
2 = aggressive, 3 = no_floor) or by name. Both normalise to the name, so
downstream consumers (``scores.json``, ``AecPreset(...)`` lookups, corpus
contracts) always see the string.
"""
import argparse

PRESET_NAMES = ('mild', 'balanced', 'aggressive', 'no_floor')   # index == C AecPreset value
PRESET_METAVAR = '{0,1,2,3,mild,balanced,aggressive,no_floor}'


def parse_preset_arg(text):
    """argparse ``type=``: '0'/'1'/'2'/'3' or a preset name -> the preset name."""
    if text in PRESET_NAMES:
        return text
    if text in ('0', '1', '2', '3'):
        return PRESET_NAMES[int(text)]
    raise argparse.ArgumentTypeError(
        "invalid preset %r (expected 0|1|2|3 or one of %s)"
        % (text, '|'.join(PRESET_NAMES)))
