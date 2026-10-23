"""AEC algorithm modules.

* ``enums``, ``dataclasses``, ``config`` -- enums, state/event records, ``AecConfig`` with ``from_preset()``
* ``preprocessing``, ``filters``, ``detectors``, ``epc`` -- front end, PBFDAF/PBFDKF, detectors, path-change handling
* ``delay/``, ``state/``, ``residual/``, ``render/``, ``filter/`` -- delay estimation, AecState, residual echo suppression, render analysis, filter bridges
* ``orchestrator`` -- ``AEC``, ``process_wav_files``, ``main``
* ``p52_regime_classifier`` -- analysis-only acoustic regime classifier
"""
