from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, NoReturn, cast

import tomllib


@dataclass(frozen=True)
class OllamaExportConfig:
    enabled: bool
    export_dir: Path
    model_name: str
    quant: str
    template: Optional[Path]
    convert_bin: Optional[str]
    quant_bin: Optional[str]


def _read_toml_dict(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".__write_test__"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _which(name: str) -> Optional[str]:
    try:
        return shutil.which(name)
    except Exception:
        return None


def _resolve_ckpt(cfg_raw: dict[str, Any], exp_root: Path) -> Path:
    # Prefer train.runtime; fallback to sample.runtime
    train_rt = (cfg_raw.get("train") or {}).get("runtime") or {}
    sample_rt = (cfg_raw.get("sample") or {}).get("runtime") or {}
    out_dir_val = train_rt.get("out_dir") or sample_rt.get("out_dir")
    if not out_dir_val:
        raise SystemExit(
            "export: could not resolve out_dir from config [train.runtime] or [sample.runtime]."
        )
    out_dir = Path(out_dir_val)
    if not out_dir.is_absolute():
        # If path looks repo-root relative (e.g., starts with 'ml_playground/...'), resolve from repo root
        parts = out_dir.parts
        if parts and parts[0] == "ml_playground":
            repo_root = (
                exp_root.parent.parent.parent
            )  # project root containing 'ml_playground'
            out_dir = (repo_root / out_dir).resolve()
        else:
            # Resolve relative to experiment root
            out_dir = (exp_root / out_dir).resolve()
    # Determine checkpoint filenames
    best_name = train_rt.get("ckpt_best_filename", "ckpt_best.pt")
    last_name = train_rt.get("ckpt_last_filename", "ckpt_last.pt")
    candidates = [out_dir / best_name, out_dir / last_name]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(
        f"export: no checkpoint found. Expected one of: {', '.join(str(p) for p in candidates)}"
    )


def _load_export_cfg(cfg_path: Path, raw: dict[str, Any]) -> OllamaExportConfig:
    exp_root = cfg_path.parent
    exp_block = (raw.get("export") or {}).get("ollama") or {}
    enabled = bool(exp_block.get("enabled", False))
    # default export dir to exp_root/export if not provided
    export_dir_val = exp_block.get("export_dir")
    if export_dir_val:
        export_dir = Path(str(export_dir_val))
        if not export_dir.is_absolute():
            export_dir = (exp_root / export_dir).resolve()
    else:
        export_dir = (exp_root / "export").resolve()
    model_name = str(exp_block.get("model_name") or "bundestag-char")
    quant = str(exp_block.get("quant") or "q4_K_M")
    template = exp_block.get("template") or ""
    template_path = Path(template).resolve() if template else None
    convert_bin = str(exp_block.get("convert_bin") or "").strip() or None
    quant_bin = str(exp_block.get("quant_bin") or "").strip() or None
    return OllamaExportConfig(
        enabled=enabled,
        export_dir=export_dir,
        model_name=model_name,
        quant=quant,
        template=template_path,
        convert_bin=convert_bin,
        quant_bin=quant_bin,
    )


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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _same_stamp(a: dict[str, Any], b: dict[str, Any]) -> bool:
    try:
        return a == b
    except Exception:
        return False


def _fail(msg: str, code: int = 2) -> NoReturn:
    print(msg)
    raise SystemExit(code)


def convert_from_toml(cfg_path: Path) -> None:
    exp_root = cfg_path.parent
    print(f"[export] start: cfg={cfg_path} cwd={os.getcwd()}")
    try:
        raw = _read_toml_dict(cfg_path)
    except Exception as e:
        _fail(f"export: failed to read config at {cfg_path}: {e}")
        return

    print(f"[export] experiment_root={exp_root}")
    export_cfg = _load_export_cfg(cfg_path, raw)
    print(
        f"[export] enabled={export_cfg.enabled} export_dir={export_cfg.export_dir} model_name={export_cfg.model_name} quant={export_cfg.quant}"
    )

    if not export_cfg.enabled:
        _fail(
            "export.ollama is disabled or missing. Enable it in the experiment's config.toml under [export.ollama].",
            code=2,
        )

    # Ensure export dir writable (also creates it)
    if not _is_writable_directory(export_cfg.export_dir):
        _fail(f"export: directory not writable: {export_cfg.export_dir}")
    else:
        print(f"[export] export_dir ready: {export_cfg.export_dir}")

    # Resolve checkpoint
    print("[export] resolving checkpoint from config...")
    ckpt_path = _resolve_ckpt(raw, exp_root)
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
            "export: conversion tool not found. Configure [export.ollama].convert_bin with a path to llama.cpp's convert-hf-to-gguf.py or ensure it is on PATH.",
            code=2,
        )
    if not quant_bin:
        _fail(
            "export: quantization tool not found. Configure [export.ollama].quant_bin with a path to llama.cpp's quantize or ensure it is on PATH.",
            code=2,
        )
    conv_bin: str = cast(str, convert_bin)
    q_bin: str = cast(str, quant_bin)

    # Run conversion to FP16 GGUF (best-effort: expect tool signature 'convert <in> <out>')
    try:
        # Clean prior fp16 artifact to avoid confusion
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
        try:
            shutil.copyfile(export_cfg.template, template_dst)
            lines.append("TEMPLATE file ./template.txt")
            print(f"[export] copied template to {template_dst}")
        except Exception:
            # If template fails to copy, continue without TEMPLATE
            print(
                "[export] warning: failed to copy template; continuing without TEMPLATE"
            )
            pass
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
    try:
        sums = [
            f"{_sha256_of_file(model_path)}  model.gguf",
            f"{_sha256_of_file(modelfile_path)}  Modelfile",
        ]
        _write_text(sums_path, "\n".join(sums) + "\n")
    except Exception:
        print("[export] warning: failed to write SHA256SUMS.txt")
        pass

    # Persist stamp for idempotency
    try:
        stamp_path.write_text(json.dumps(inputs_now, indent=2), encoding="utf-8")
        print(f"[export] wrote stamp {stamp_path}")
    except Exception:
        print("[export] warning: failed to write idempotency stamp")
        pass

    print(f"export: wrote {model_path}")
    print(f"export: wrote {modelfile_path}")
    print(f"export: wrote {readme_path}")
    print(f"export: wrote {sums_path}")
