"""Launcher shim for the compiled train module.

When train.py is shipped as a .so, `python train.py` fails (no file) and
`python -m train` fails (C extensions have no code object for -m). This
launcher lets users invoke the compiled training loop via `python run_train.py`.
"""
import train  # noqa: F401  -- importing runs the module-level training loop
