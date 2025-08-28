from __future__ import annotations

import stat
from pathlib import Path

import pytest

from ml_playground.cli import main


def _make_exec_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.mark.integration
def test_convert_creates_ollama_export_with_fake_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Integration-style test: run `convert bundestag_char` through the CLI, but monkeypatch
    the exporter to use a temporary out_dir, export_dir, and fake external tools.

    This avoids heavyweight deps while verifying:
      - CLI dispatches to the exporter
      - Exporter resolves paths from config
      - Conversion and quantization steps result in expected artifacts
    """
    # Arrange: temp directories
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = tmp_path / "export"

    # Fake checkpoint expected by exporter
    ckpt = out_dir / "ckpt_last.pt"
    ckpt.write_bytes(b"dummy-checkpoint")

    # Fake external tools: write simple Python scripts that create the expected outputs
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    # convert tool: args: <ckpt_in> <fp16_out>
    convert_py = _make_exec_script(
        bin_dir / "convert_to_gguf.py",
        """#!/usr/bin/env python3
import sys, pathlib
# create the output file to simulate conversion
outp = pathlib.Path(sys.argv[2])
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_bytes(b'FP16-GGUF')
""",
    )

    # quant tool: args: <fp16_in> <quant_out> <preset>
    quant_py = _make_exec_script(
        bin_dir / "quantize.py",
        """#!/usr/bin/env python3
import sys, pathlib
# create the output file to simulate quantization
outp = pathlib.Path(sys.argv[2])
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_bytes(b'Q-GGUF')
""",
    )

    # Patch exporter TOML reader to inject our temp paths and fake tool binaries
    from ml_playground.experiments.bundestag_char import ollama_export as exp

    real_read = exp._read_toml_dict

    def _patched_read_toml_dict(path: Path):  # type: ignore[no-untyped-def]
        raw = real_read(path)
        raw.setdefault("train", {}).setdefault("runtime", {})["out_dir"] = str(out_dir)
        raw.setdefault("export", {}).setdefault("ollama", {})
        raw["export"]["ollama"]["enabled"] = True
        raw["export"]["ollama"]["export_dir"] = str(export_dir)
        raw["export"]["ollama"]["convert_bin"] = str(convert_py)
        raw["export"]["ollama"]["quant_bin"] = str(quant_py)
        return raw

    monkeypatch.setattr(
        "ml_playground.experiments.bundestag_char.ollama_export._read_toml_dict",
        _patched_read_toml_dict,
        raising=True,
    )

    # Act: run CLI
    main(["convert", "bundestag_char"])

    # Assert: expected files created in our export dir
    model_path = export_dir / "model.gguf"
    modelfile_path = export_dir / "Modelfile"
    readme_path = export_dir / "README.md"
    sums_path = export_dir / "SHA256SUMS.txt"

    assert model_path.exists(), "model.gguf should be created by fake quant tool"
    assert modelfile_path.exists(), "Modelfile should be written"
    assert readme_path.exists(), "README.md should be written"
    assert sums_path.exists(), "SHA256SUMS.txt should be written"

    # Modelfile should reference the local model
    contents = modelfile_path.read_text(encoding="utf-8")
    assert "FROM ./model.gguf" in contents
