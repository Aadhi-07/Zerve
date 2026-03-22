import subprocess, sys

# Install lightgbm to /tmp (the only writable location in this environment)
result = subprocess.run(
    ["pip", "install", "lightgbm", "--target=/tmp/lgb_pkg", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else "No stdout")
print(result.stderr[-500:] if result.stderr else "No stderr")

# Add to path so downstream blocks can import it
if "/tmp/lgb_pkg" not in sys.path:
    sys.path.insert(0, "/tmp/lgb_pkg")

# Verify it's importable
import lightgbm as lgb
print(f"LightGBM version: {lgb.__version__}")
