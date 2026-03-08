
"""
F1 Australian GP Podium Predictor
===================================

"""

import sys
import subprocess
import os

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

steps = [
    ("01_build_dataset.py", "Step 1/3 -- Building dataset..."),
    ("02_eda.py",            "Step 2/3 -- Running EDA..."),
    ("03_model.py",          "Step 3/3 -- Training model & predicting..."),
]

print("=" * 55)
print("  F1 Australian GP Podium Predictor -- Full Pipeline")
print("=" * 55)

for script, msg in steps:
    print(f"\n>> {msg}")
    # Don't capture output -- let scripts print directly to console.
    # This avoids the cp1252 pipe encoding issue on Windows.
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, script)]
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {result.returncode}")
        sys.exit(1)

print("\n" + "=" * 55)
print("  [DONE] All steps completed successfully!")
print("  Outputs:")
print("    data/aus_gp_dataset.csv     -- Full feature dataset")
print("    data/predictions_2026.csv   -- 2026 race predictions")
print("    plots/eda_overview.png      -- EDA visualizations")
print("    plots/model_results.png     -- Model results & prediction")
print("=" * 55)