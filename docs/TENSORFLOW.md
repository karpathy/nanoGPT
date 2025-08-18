# TensorFlow installation (optional)

TensorBoard runs fine without TensorFlow; the “TensorFlow installation not found — running with reduced feature set” message is informational.
If you want TensorFlow available in this environment, use the tf dependency group we provide.

## Install

- Apple Silicon (macOS, arm64):
  uv sync --group tf

- Linux / Windows:
  uv sync --group tf

Notes:
- On Python 3.13, we depend on TF nightlies (tf-nightly / tensorflow-macos-nightly) due to limited stable wheel availability.
- On Apple Silicon, tensorflow-metal is also installed for Metal acceleration.

## Verify

- macOS:
  uv run python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"

- Linux/Windows:
  uv run python -c "import tensorflow as tf; print(tf.__version__)"
