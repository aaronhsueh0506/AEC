#!/usr/bin/env python3
"""Sweep dt_erle_gate threshold on 10-case quick test."""
import subprocess, re, os

base_dir = os.path.dirname(__file__)
aec_file = os.path.join(base_dir, "aec.py")
test_dir = os.path.join(base_dir, "../wav/aec_quick_test/")

# Read original content
with open(aec_file, 'r') as f:
    original = f.read()

for gate_thresh in [0.7, 0.5, 0.4]:
    # Modify gate threshold
    modified = original.replace(
        "dt_erle_gate = np.clip((dt_indicator - 0.7) / 0.3, 0.0, 1.0)",
        f"dt_erle_gate = np.clip((dt_indicator - {gate_thresh}) / 0.3, 0.0, 1.0)"
    )
    with open(aec_file, 'w') as f:
        f.write(modified)

    # Run eval
    subprocess.run(
        ["python3", "eval_aec_challenge.py", test_dir,
         "--preset", "balanced", "--filter", "512", "--parallel"],
        cwd=base_dir, capture_output=True
    )
    # Run AECMOS
    result = subprocess.run(
        ["python3", "eval_aecmos_local.py", test_dir],
        cwd=base_dir, capture_output=True, text=True
    )

    # Parse MEAN lines
    means = [line.strip() for line in result.stdout.split('\n') if 'MEAN' in line]
    # FS overall echo (line 0, col 2), DT overall echo (line 8, col 2), DT overall deg (line 9, col 2), NE deg (line 7, col 2)
    def get_ours(line):
        parts = line.split()
        return float(parts[2]) if len(parts) >= 3 else 0

    if len(means) >= 10:
        fs_echo = get_ours(means[0])
        ne_deg = get_ours(means[7])
        dt_echo = get_ours(means[8])
        dt_deg = get_ours(means[9])
        print(f"gate={gate_thresh:.1f}  FS_echo={fs_echo:.3f}  DT_echo={dt_echo:.3f}  DT_deg={dt_deg:.3f}  NE_deg={ne_deg:.3f}")
    else:
        print(f"gate={gate_thresh:.1f}  PARSE ERROR ({len(means)} MEAN lines)")

# Restore original
with open(aec_file, 'w') as f:
    f.write(original)
print("\nRestored original aec.py (gate=0.7)")
