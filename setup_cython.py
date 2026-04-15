"""
Build .so extensions from existing .py files via Cython, in-place.

Usage:
    python setup_cython.py build_ext --inplace

This is the obfuscation build. The goal is to ship .so binaries instead of
.py source. We do not modify the source files. Files that fail to compile
are reported and skipped.
"""
import sys
from pathlib import Path

from setuptools import setup
from Cython.Build import cythonize

REPO_ROOT = Path(__file__).resolve().parent

TARGETS = [
    "model.py",
    "configurator.py",
    "sample.py",
    "train.py",
]

def main():
    ext_modules = cythonize(
        TARGETS,
        language_level=3,
        compiler_directives={
            "always_allow_keywords": True,
        },
        build_dir="build/cython",
    )
    setup(
        name="nanogpt-obfuscated",
        ext_modules=ext_modules,
        script_args=sys.argv[1:] or ["build_ext", "--inplace"],
    )

if __name__ == "__main__":
    main()
