#!/usr/bin/env python3
"""Regenerate the v3.21 800-case bench report from the saved scores JSON.

The bench (`v3_21_800case_bench.py`) saves `scores_800case.json` BEFORE it
writes the markdown report.  If the running process was launched with an older
in-memory copy of `_write_report` (e.g. the flag-table KeyError fix landed
after launch), the report step crashes but the JSON survives.  This script
reloads the JSON and calls the current (fixed) `_write_report` to produce the
report — no re-rendering, no re-scoring.

Usage:
    cd /path/to/AEC
    python3 python/v3_21_800case_report_from_json.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_21_800case_bench import _write_report, JSON_PATH, DOC_PATH


def main() -> None:
    if not JSON_PATH.exists():
        print(f'[error] {JSON_PATH} not found — bench has not saved scores yet.')
        sys.exit(1)
    json_data = json.loads(JSON_PATH.read_text(encoding='utf-8'))
    # JSON is {stem: result_dict}; results list is just the values, byte-identical
    # to what the bench passed to _write_report.
    results = list(json_data.values())
    print(f'[regen] Loaded {len(results)} cases from {JSON_PATH}')
    _write_report(results, be_ok=True)
    print(f'[regen] Report written to {DOC_PATH}')


if __name__ == '__main__':
    main()
