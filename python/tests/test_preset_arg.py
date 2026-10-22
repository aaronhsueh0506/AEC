"""--preset accepts the C AecPreset value (0/1/2/3) or the name, and always yields the name."""
import argparse
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preset_arg import PRESET_METAVAR, PRESET_NAMES, parse_preset_arg  # noqa: E402


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', type=parse_preset_arg, choices=PRESET_NAMES,
                   metavar=PRESET_METAVAR, default=None)
    return p


@pytest.mark.parametrize('text,name', [
    ('0', 'mild'), ('1', 'balanced'), ('2', 'aggressive'), ('3', 'no_floor'),
    ('mild', 'mild'), ('balanced', 'balanced'), ('aggressive', 'aggressive'),
    ('no_floor', 'no_floor'),
])
def test_value_and_name_normalise_to_the_name(text, name):
    assert _parser().parse_args(['--preset', text]).preset == name


@pytest.mark.parametrize('text', ['4', '-1', '01', '', 'BALANCED'])
def test_everything_else_is_rejected(text):
    with pytest.raises(SystemExit):
        _parser().parse_args(['--preset', text])


def test_numbers_follow_the_c_enum_order():
    """c_impl/include/aec.h: AEC_PRESET_MILD=0, BALANCED=1, AGGRESSIVE=2, NO_FLOOR=3."""
    here = os.path.dirname(os.path.abspath(__file__))
    header = os.path.join(here, '..', '..', 'c_impl', 'include', 'aec.h')
    with open(header, encoding='utf-8') as fh:
        text = fh.read()
    for value, name in enumerate(PRESET_NAMES):
        m = re.search(r'AEC_PRESET_%s\s*=\s*(\d+)' % name.upper(), text)
        assert m and int(m.group(1)) == value
