from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, NoReturn, cast


@dataclass(frozen=True)
class OllamaExportConfig:
    enabled: bool
    export_dir: Path
    model_name: str
    quant: str
    template: Optional[Path]
    convert_bin: Optional[str]
    quant_bin: Optional[str]


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _is_writable_directory(path: Path) -> bool:
    # Strict: let filesystem errors propagate to callers
    path.mkdir(parents=True, exist_ok=True)
    test_file = path / ".__write_test__"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink(missing_ok=True)
    return True


def _which(name: str) -> Optional[str]:
    # shutil.which should not raise; return value indicates presence
    return shutil.which(name)




def _inputs_stamp(cfg: OllamaExportConfig, ckpt: Path) -> dict[str, Any]:
    stamp: dict[str, Any] = {
        "ckpt_path": str(ckpt),
        "ckpt_sha256": _sha256_of_file(ckpt) if ckpt.exists() else "",
        "quant": cfg.quant,
        "model_name": cfg.model_name,
        "convert_bin": cfg.convert_bin or "",
        "quant_bin": cfg.quant_bin or "",
        "template_path": str(cfg.template) if cfg.template else "",
        "template_sha256": _sha256_of_file(cfg.template)
        if cfg.template and cfg.template.exists()
        else "",
    }
    return stamp


def _load_stamp(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    # Strict: invalid stamp is an error
    return json.loads(path.read_text(encoding="utf-8"))


def _same_stamp(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a == b


def _fail(msg: str, code: int = 2) -> NoReturn:
    print(msg)
    raise SystemExit(code)


def convert(
    export_cfg: OllamaExportConfig,
    out_dir: Path,
    ckpt_best_filename: str = "ckpt_best.pt",
    ckpt_last_filename: str = "ckpt_last.pt",
) -> None:
    """Convert and quantize using injected config and resolved runtime paths.

    Experiments must not read config files; callers (CLI) inject a fully-validated
    export config and runtime out_dir/filenames here.
    """
    print(f"[export] start (injected): out_dir={out_dir} cwd={os.getcwd()}")
    print(
        f"[export] enabled={export_cfg.enabled} export_dir={export_cfg.export_dir} model_name={export_cfg.model_name} quant={export_cfg.quant}"
    )

    if not export_cfg.enabled:
        _fail(
            "export.ollama is disabled. Enable it by injecting an enabled OllamaExportConfig.",
            code=2,
        )

    # Ensure export dir writable (also creates it)
    try:
        if not _is_writable_directory(export_cfg.export_dir):
            _fail(f"export: directory not writable: {export_cfg.export_dir}")
    except Exception as e:
        _fail(
            f"export: failed to prepare export directory {export_cfg.export_dir}: {e}"
        )
    print(f"[export] export_dir ready: {export_cfg.export_dir}")

    # Resolve checkpoint from injected runtime
    candidates = [out_dir / ckpt_best_filename, out_dir / ckpt_last_filename]
    ckpt_path: Optional[Path] = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        _fail(
            f"export: no checkpoint found. Expected one of: {', '.join(str(p) for p in candidates)}"
        )
    print(f"[export] checkpoint={ckpt_path}")

    # Paths
    export_dir = export_cfg.export_dir
    fp16_path = export_dir / "model-fp16.gguf"
    model_path = export_dir / "model.gguf"
    modelfile_path = export_dir / "Modelfile"
    readme_path = export_dir / "README.md"
    sums_path = export_dir / "SHA256SUMS.txt"
    template_dst = export_dir / "template.txt"
    stamp_path = export_dir / ".export_inputs.json"

    # Idempotency: if outputs exist and stamp unchanged, exit 0
    inputs_now = _inputs_stamp(export_cfg, ckpt_path)
    prev = _load_stamp(stamp_path)
    if (
        model_path.exists()
        and modelfile_path.exists()
        and prev
        and _same_stamp(prev, inputs_now)
    ):
        print("[export] outputs up-to-date; nothing to do (idempotent).")
        print(f"[export] existing: {model_path}, {modelfile_path}")
        return

    # Tool resolution
    convert_bin = export_cfg.convert_bin or _which("convert-hf-to-gguf.py")
    quant_bin = export_cfg.quant_bin or _which("quantize")
    print(f"[export] tools: convert_bin={convert_bin} quant_bin={quant_bin}")

    if not convert_bin:
        _fail(
            "export: conversion tool not found. Configure convert_bin or ensure it is on PATH.",
            code=2,
        )
    if not quant_bin:
        _fail(
            "export: quantization tool not found. Configure quant_bin or ensure it is on PATH.",
            code=2,
        )
    conv_bin: str = cast(str, convert_bin)
    q_bin: str = cast(str, quant_bin)

    # Run conversion to FP16 GGUF
    try:
        if fp16_path.exists():
            fp16_path.unlink()
        cmd_convert: list[str] = [conv_bin, str(ckpt_path), str(fp16_path)]
        print(f"[export] running: {' '.join(cmd_convert)}")
        subprocess.run(cmd_convert, check=True)
    except FileNotFoundError:
        _fail(f"export: conversion tool not executable: {conv_bin}")
    except subprocess.CalledProcessError as e:
        _fail(f"export: conversion failed with exit code {e.returncode}")
    except Exception as e:
        _fail(f"export: conversion failed: {e}")

    if not fp16_path.exists():
        _fail(
            "export: FP16 GGUF conversion did not produce expected file: model-fp16.gguf"
        )
    else:
        print(f"[export] created {fp16_path}")

    # Run quantization
    try:
        if model_path.exists():
            model_path.unlink()
        cmd_quant: list[str] = [
            q_bin,
            str(fp16_path),
            str(model_path),
            str(export_cfg.quant),
        ]
        print(f"[export] running: {' '.join(cmd_quant)}")
        subprocess.run(cmd_quant, check=True)
    except FileNotFoundError:
        _fail(f"export: quantization tool not executable: {q_bin}")
    except subprocess.CalledProcessError as e:
        _fail(f"export: quantization failed with exit code {e.returncode}")
    except Exception as e:
        _fail(f"export: quantization failed: {e}")

    if not model_path.exists():
        _fail("export: quantization did not produce expected file: model.gguf")
    else:
        print(f"[export] created {model_path}")

    # Write Modelfile
    lines = ["FROM ./model.gguf"]
    if export_cfg.template and export_cfg.template.exists():
        shutil.copyfile(export_cfg.template, template_dst)
        lines.append("TEMPLATE file ./template.txt")
        print(f"[export] copied template to {template_dst}")
    _write_text(modelfile_path, "\n".join(lines) + "\n")

    # Write README
    readme = (
        f"# Ollama Export\n\n"
        f"Create the model in Ollama:\n\n"
        f"    ollama create {export_cfg.model_name} -f Modelfile\n\n"
        f"Run the model:\n\n"
        f"    ollama run {export_cfg.model_name}\n"
    )
    _write_text(readme_path, readme)

    # Write checksums
    sums = [
        f"{_sha256_of_file(model_path)}  model.gguf",
        f"{_sha256_of_file(modelfile_path)}  Modelfile",
    ]
    _write_text(sums_path, "\n".join(sums) + "\n")

    # Persist stamp for idempotency
    stamp_path.write_text(json.dumps(inputs_now, indent=2), encoding="utf-8")
    print(f"[export] wrote stamp {stamp_path}")

    print(f"export: wrote {model_path}")
    print(f"export: wrote {modelfile_path}")
    print(f"[export] wrote: {sums_path}")
    print("[export] done")
