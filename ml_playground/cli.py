from __future__ import annotations
import argparse
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping, get_origin, get_args, Union, Literal
from ml_playground.config import (
    TrainExperiment,
    SampleExperiment,
)
from ml_playground.trainer import train
from ml_playground.sampler import sample
from ml_playground.cli_config import load_config


# -----------------------------
# Strict config loaders (TOML 1.0 via tomllib)
# -----------------------------

def _read_toml(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def _fail_unknown_keys(d: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(d.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {sorted(unknown)}; allowed: {sorted(allowed)}")


def _ensure_section(d: Mapping[str, Any], section: str, where: str) -> Mapping[str, Any]:
    if section not in d:
        raise ValueError(f"Missing required section [{section}] in {where}")
    sec = d[section]
    if not isinstance(sec, dict):
        raise ValueError(f"Section [{section}] must be a table in {where}")
    return sec


def _coerce_and_norm_path(value: Any, base_dir: Path, where: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"Expected path-like for {where}, got {type(value).__name__}")
    p = Path(value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _require_positive(name: str, val: int) -> None:
    if not isinstance(val, int) or val <= 0:
        raise ValueError(f"{name} must be a positive integer, got {val!r}")


# Basic runtime type checks against dataclass field annotations

def _is_instance_of(value: Any, typ: Any) -> bool:
    origin = get_origin(typ)
    if origin is Union:
        return any(_is_instance_of(value, t) for t in get_args(typ))
    if origin is Literal:
        return value in get_args(typ)
    # Primitive types
    if typ is int:
        return isinstance(value, int)
    if typ is float:
        return isinstance(value, (int, float))
    if typ is bool:
        return isinstance(value, bool)
    if typ is str:
        return isinstance(value, str)
    if typ is Path:
        return isinstance(value, (Path, str))
    # Optional[...] already handled via Union
    # For any other annotations, be permissive (dataclass constructor will likely accept correct mappings)
    return True


def _assert_types(tbl: Mapping[str, Any], cls: Any, where: str) -> None:
    fields = getattr(cls, '__dataclass_fields__', {})
    for name, f in fields.items():
        if name in tbl:
            typ = f.type
            val = tbl[name]
            # Special-case Literals
            origin = get_origin(typ)
            if origin is None and str(typ).startswith('typing.Literal'):
                if val not in get_args(typ):
                    raise TypeError(f"Invalid literal for {where}.{name}: {val!r}; allowed: {get_args(typ)}")
            else:
                if not _is_instance_of(val, typ):
                    raise TypeError(f"Type mismatch for {where}.{name}: expected {typ}, got {type(val).__name__}")


def load_train_config(path: Path) -> TrainExperiment:
    cfg = load_config(path)
    if cfg.train is None:
        raise Exception("Config must contain [train] block")
    return cfg.train


def load_sample_config(path: Path) -> SampleExperiment:
    cfg = load_config(path)
    if cfg.sample is None:
        raise Exception("Config must contain [sample] block")
    return cfg.sample





def _load_env_files() -> None:
    """Load .env files from CWD without extra dependencies.

    - Does not override already-set environment variables.
    - Supports simple KEY=VALUE lines; ignores comments and blank lines.
    """

    def _parse_set(path: Path) -> None:
        try:
            if not path.exists():
                return
            for line in path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                key, val = s.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            # Best-effort only; silently ignore parsing errors
            pass

    _parse_set(Path.cwd() / ".env")


def main(argv: list[str] | None = None) -> None:
    # Load .env variables early so downstream modules can see them
    _load_env_files()
    parser: ArgumentParser = configureArguments()
    # Add --delete-existing (-D) to ArgumentParser directly
    parser.add_argument(
        "--delete-existing",
        "-D",
        action="store_true",
        default=False,
        help="Delete the output directory (out_dir) before starting (prepare/train/loop commands only).",
    )
    args = parser.parse_args(argv)

    # Resolve config.toml from experiment where applicable
    def _resolve_config_from_experiment(experiment: str) -> Path:
        base = Path(__file__).resolve().parent / "experiments" / experiment / "config.toml"
        return base

    # If command provides an experiment, resolve its config.toml automatically
    if hasattr(args, "experiment"):
        exp = getattr(args, "experiment")
        if isinstance(exp, str):
            cfg_path = _resolve_config_from_experiment(exp)
            # For train/sample/loop: config is required and must exist
            if args.cmd in {"train", "sample", "loop"}:
                if not cfg_path.exists():
                    raise SystemExit(
                        f"Config not found for experiment '{exp}'. Expected at: {cfg_path}"
                    )
                setattr(args, "config", cfg_path)
            else:
                # For prepare: config is optional; set if exists
                if cfg_path.exists():
                    setattr(args, "config", cfg_path)

    # Special: If the experiment is "bundestag_finetuning_mps" OR the config explicitly declares that dataset in [prepare]
    def is_bundestag_finetuning_mps(experiment: str | None, config: Path | None) -> bool:
        if experiment == "bundestag_finetuning_mps":
            return True
        if config is not None and config.exists():
            import tomllib

            try:
                with open(config, "rb") as f:
                    d = tomllib.load(f)
                # Only route to this integration if [prepare].dataset explicitly matches
                return d.get("prepare", {}).get("dataset") == "bundestag_finetuning_mps"
            except Exception:
                pass
        return False

    # Special: If the experiment is "gemma_finetuning_mps" OR the config file contains the integration-specific block
    def is_gemma_finetuning_mps(experiment: str | None, config: Path | None) -> bool:
        if experiment == "gemma_finetuning_mps":
            return True
        if config is not None and config.exists():
            import tomllib

            try:
                with open(config, "rb") as f:
                    d = tomllib.load(f)
                # Check for Gemma-specific config structure
                if (
                    "prepare" in d
                    and "train" in d
                    and ("hf_model" in d["train"] or "peft" in d["train"])
                    and d.get("prepare", {}).get("dataset") == "gemma_finetuning_mps"
                ):
                    return True
            except Exception:
                pass
        return False

    # Integration always handles bundestag_finetuning_mps configs
    from ml_playground.experiments.bundestag_finetuning_mps import (
        bundestag_finetuning_mps as integ,
    )
    from ml_playground.experiments.speakger import gemma_finetuning_mps as gemma_integ

    import shutil
    import tomllib

    # Helper to find out_dir from config TOML file, with fallback to runtime fields
    def _find_out_dir(config_path, section=None):
        if config_path is None:
            return None
        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            if section and section in data and "out_dir" in data[section]:
                return Path(data[section]["out_dir"])
            # Try [train][runtime][out_dir] if present
            if "train" in data and "runtime" in data["train"]:
                if "out_dir" in data["train"]["runtime"]:
                    return Path(data["train"]["runtime"]["out_dir"])
            if "sample" in data and "runtime" in data["sample"]:
                if "out_dir" in data["sample"]["runtime"]:
                    return Path(data["sample"]["runtime"]["out_dir"])
        except Exception:
            pass
        return None

    if args.cmd == "prepare":
        # Unified prepare: supports both legacy PREPARERS and integration TOML-based pipelines
        try:
            from ml_playground.datasets import PREPARERS as _PREPARERS  # type: ignore
        except Exception:  # pragma: no cover - allow running without datasets package
            from ml_playground.experiments import PREPARERS as _PREPARERS  # type: ignore

        # If a config is provided and matches an integration, call its prepare_from_toml
        if (
            getattr(args, "config", None) is not None
            and isinstance(args.config, Path)
            and args.config.exists()
        ):
            if is_bundestag_finetuning_mps(
                getattr(args, "experiment", None), getattr(args, "config", None)
            ):
                # delete dataset_dir for prepare if requested
                if args.delete_existing:
                    try:
                        with open(args.config, "rb") as f:
                            d = tomllib.load(f)
                        ds_dir = d.get("prepare", {}).get("dataset_dir")
                        if ds_dir:
                            p = Path(ds_dir)
                            if p.exists():
                                print(
                                    f"[ml_playground] Deleting dataset_dir {p} as requested."
                                )
                                shutil.rmtree(p)
                    except Exception:
                        pass
                integ.prepare_from_toml(args.config)
                return
            if is_gemma_finetuning_mps(
                getattr(args, "experiment", None), getattr(args, "config", None)
            ):
                if args.delete_existing:
                    try:
                        with open(args.config, "rb") as f:
                            d = tomllib.load(f)
                        ds_dir = d.get("prepare", {}).get("dataset_dir")
                        if ds_dir:
                            p = Path(ds_dir)
                            if p.exists():
                                print(
                                    f"[ml_playground] Deleting dataset_dir {p} as requested."
                                )
                                shutil.rmtree(p)
                    except Exception:
                        pass
                gemma_integ.prepare_from_toml(args.config)
                return

        # Legacy path: use registered preparers by dataset name
        if args.delete_existing:
            out_dir = _find_out_dir(
                getattr(args, "config", None), section="runtime"
            ) or getattr(args, "out_dir", None)
            if out_dir and out_dir.exists():
                print(
                    f"[ml_playground] Deleting output directory {out_dir} as requested."
                )
                shutil.rmtree(out_dir)
        prepare = _PREPARERS.get(args.experiment)
        if prepare is None:
            raise SystemExit(
                f"Unknown experiment: {getattr(args, 'experiment', None)}. Provide a known experiment name or a TOML config for an integration dataset."
            )
        prepare()
        return

    # For integration commands: route each one to the proper integration entrypoint
    if is_bundestag_finetuning_mps(
        getattr(args, "experiment", None), getattr(args, "config", None)
    ):
        if args.delete_existing:
            out_dir = _find_out_dir(getattr(args, "config", None)) or getattr(
                args, "out_dir", None
            )
            if out_dir and out_dir.exists():
                print(
                    f"[ml_playground] Deleting output directory {out_dir} as requested."
                )
                shutil.rmtree(out_dir)
        if args.cmd == "loop":
            print(
                "[ml_playground] Routing to integration: bundestag_finetuning_mps (PEFT pipeline: prepare → train → sample)"
            )
            # Debug: show prepare.* inferred from TOML
            try:
                with open(args.config, "rb") as _f:
                    _d = tomllib.load(_f)
                _prep = _d.get("prepare", {})
                print(
                    f"[ml_playground][debug] prepare.dataset={_prep.get('dataset')}, "
                    f"raw_dir={_prep.get('raw_dir')}, dataset_dir={_prep.get('dataset_dir')}"
                )
            except Exception:
                pass
            integ.loop(args.config)
        elif args.cmd == "train":
            print(
                "[ml_playground] Routing to integration: bundestag_finetuning_mps (train only)"
            )
            try:
                with open(args.config, "rb") as _f:
                    _d = tomllib.load(_f)
                _prep = _d.get("prepare", {})
                print(
                    f"[ml_playground][debug] prepare.dataset={_prep.get('dataset')}, "
                    f"raw_dir={_prep.get('raw_dir')}, dataset_dir={_prep.get('dataset_dir')}"
                )
            except Exception:
                pass
            integ.train_from_toml(args.config)
        elif args.cmd == "sample":
            print(
                "[ml_playground] Routing to integration: bundestag_finetuning_mps (sample only)"
            )
            try:
                with open(args.config, "rb") as _f:
                    _d = tomllib.load(_f)
                _prep = _d.get("prepare", {})
                print(
                    f"[ml_playground][debug] prepare.dataset={_prep.get('dataset')}, "
                    f"raw_dir={_prep.get('raw_dir')}, dataset_dir={_prep.get('dataset_dir')}"
                )
            except Exception:
                pass
            integ.sample_from_toml(args.config)
        else:
            raise SystemExit(
                f"[ml_playground] Unsupported command '{args.cmd}' for this integration."
            )
        return

    # For Gemma integration commands: route each one to the proper integration entrypoint
    if is_gemma_finetuning_mps(
        getattr(args, "experiment", None), getattr(args, "config", None)
    ):
        if args.delete_existing:
            out_dir = _find_out_dir(getattr(args, "config", None)) or getattr(
                args, "out_dir", None
            )
            if out_dir and out_dir.exists():
                print(
                    f"[ml_playground] Deleting output directory {out_dir} as requested."
                )
                shutil.rmtree(out_dir)
        if args.cmd == "loop":
            print(
                "[ml_playground] Routing to integration: gemma_finetuning_mps (PEFT pipeline: prepare → train → sample)"
            )
            # Debug: show prepare.* inferred from TOML
            try:
                with open(args.config, "rb") as _f:
                    _d = tomllib.load(_f)
                _prep = _d.get("prepare", {})
                print(
                    f"[ml_playground][debug] prepare.dataset={_prep.get('dataset')}, "
                    f"raw_dir={_prep.get('raw_dir')}, dataset_dir={_prep.get('dataset_dir')}"
                )
            except Exception:
                pass
            gemma_integ.loop(args.config)
        elif args.cmd == "train":
            print(
                "[ml_playground] Routing to integration: gemma_finetuning_mps (train only)"
            )
            try:
                with open(args.config, "rb") as _f:
                    _d = tomllib.load(_f)
                _prep = _d.get("prepare", {})
                print(
                    f"[ml_playground][debug] prepare.dataset={_prep.get('dataset')}, "
                    f"raw_dir={_prep.get('raw_dir')}, dataset_dir={_prep.get('dataset_dir')}"
                )
            except Exception:
                pass
            gemma_integ.train_from_toml(args.config)
        elif args.cmd == "sample":
            print(
                "[ml_playground] Routing to integration: gemma_finetuning_mps (sample only)"
            )
            try:
                with open(args.config, "rb") as _f:
                    _d = tomllib.load(_f)
                _prep = _d.get("prepare", {})
                print(
                    f"[ml_playground][debug] prepare.dataset={_prep.get('dataset')}, "
                    f"raw_dir={_prep.get('raw_dir')}, dataset_dir={_prep.get('dataset_dir')}"
                )
            except Exception:
                pass
            gemma_integ.sample_from_toml(args.config)
        else:
            raise SystemExit(
                f"[ml_playground] Unsupported command '{args.cmd}' for this integration."
            )
        return

    # Generic pipeline (default)
    if args.cmd == "loop":
        try:
            from ml_playground.datasets import PREPARERS as _PREPARERS  # type: ignore
        except Exception:  # pragma: no cover
            from ml_playground.experiments import PREPARERS as _PREPARERS  # type: ignore

        if args.delete_existing:
            out_dir = _find_out_dir(getattr(args, "config", None)) or getattr(
                args, "out_dir", None
            )
            if out_dir and out_dir.exists():
                print(
                    f"[ml_playground] Deleting output directory {out_dir} as requested."
                )
                shutil.rmtree(out_dir)
        prepare = _PREPARERS.get(args.experiment)
        if prepare is None:
            raise SystemExit(f"Unknown experiment: {args.experiment}")
        # 1) prepare
        prepare()
        # 2) load configs (strict) before training to fail fast
        try:
            train_cfg: TrainExperiment = load_train_config(args.config)
        except Exception as e:
            raise SystemExit(str(e))
        try:
            sample_cfg: SampleExperiment = load_sample_config(args.config)
        except Exception as e:
            raise SystemExit(str(e))
        # 3) train
        train(train_cfg)
        # Copy dataset meta.pkl into out_dir to satisfy strict sampling requirements
        try:
            data_cfg = train_cfg.data
            if data_cfg.meta_pkl is not None:
                src_meta = data_cfg.dataset_dir / data_cfg.meta_pkl
                dst_meta = train_cfg.runtime.out_dir / "meta.pkl"
                if src_meta.exists():
                    shutil.copy2(src_meta, dst_meta)
        except Exception as e:
            # Non-fatal for loop flow, but note that sampling will fail without meta.pkl
            print(f"[loop] Warning: failed to copy required meta.pkl into out_dir: {e}")
        # 4) sample
        sample(sample_cfg)
        return


    if args.cmd == "train":
        try:
            train_cfg: TrainExperiment = load_train_config(args.config)
        except Exception as e:
            raise SystemExit(str(e))
        train(train_cfg)
        return

    if args.cmd == "sample":
        try:
            sample_cfg: SampleExperiment = load_sample_config(args.config)
        except Exception as e:
            raise SystemExit(str(e))
        sample(sample_cfg)
        return



def configureArguments():
    p = argparse.ArgumentParser("ml_playground")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Prepare now always takes an experiment name; integration configs are auto-resolved
    prep_parser = sub.add_parser(
        "prepare",
        help="Prepare experiment by name (or via integration config resolved from experiments/<experiment>/config.toml)",
    )
    prep_parser.add_argument(
        "experiment",
        help="Experiment name (e.g., shakespeare, bundestag_char, bundestag_tiktoken, speakger, bundestag_qwen15b_lora_mps)",
    )

    # Train and sample now take only an experiment; config is auto-resolved
    train_parser = sub.add_parser("train", help="Train for the given experiment (config.toml auto-resolved)")
    train_parser.add_argument("experiment", help="Experiment name")

    sample_parser = sub.add_parser(
        "sample",
        help="Sample for the given experiment (config.toml auto-resolved; tries best/last checkpoints)",
    )
    sample_parser.add_argument("experiment", help="Experiment name")

    # Loop takes only an experiment
    loop_parser = sub.add_parser(
        "loop", help="Run prepare -> train -> sample for the given experiment (config.toml auto-resolved)"
    )
    loop_parser.add_argument(
        "experiment",
        help="Experiment name",
    )

    return p


if __name__ == "__main__":
    main()
