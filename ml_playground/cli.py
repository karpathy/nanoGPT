from __future__ import annotations
import argparse
import os
from argparse import ArgumentParser
from pathlib import Path
from ml_playground.config import (
    TrainExperiment,
    SampleExperiment,
    DataConfig,
)
from ml_playground.trainer import train
from ml_playground.sampler import sample




def load_train_config(path: Path) -> TrainExperiment:
    """Strict loader with minimal legacy checks and path resolution.

    - Ensures required subsections exist ([model],[data],[optim],[schedule],[runtime]).
    - Produces a friendly unknown-key error for [train.data].
    - Resolves relative dataset_dir and out_dir against the config file directory.
    - Delegates detailed value validation to Pydantic models.
    """
    import tomllib
    base_dir = path.parent.resolve()
    with path.open("rb") as f:
        raw = tomllib.load(f)

    train_tbl = raw.get("train")
    if not isinstance(train_tbl, dict):
        # Keep legacy top-level message used by CLI tests
        raise Exception("Config must contain [train] block")

    # Required subsections
    for sec in ("model", "data", "optim", "schedule", "runtime"):
        if not isinstance(train_tbl.get(sec), dict):
            raise ValueError(f"Missing required section [{sec}]")

    # Unknown keys in [train.data]
    allowed_data_keys = set(DataConfig.model_fields.keys())
    unknown = set(train_tbl["data"].keys()) - allowed_data_keys
    if unknown:
        raise ValueError("Unknown key(s) in [train.data]")

    # Resolve relative paths
    d = dict(train_tbl)
    d_data = dict(d["data"])
    dd = d_data.get("dataset_dir")
    if isinstance(dd, (str, Path)):
        p = Path(dd)
        if not p.is_absolute():
            d_data["dataset_dir"] = (base_dir / p).resolve()
    d["data"] = d_data

    d_runtime = dict(d["runtime"])
    od = d_runtime.get("out_dir")
    if isinstance(od, (str, Path)):
        p = Path(od)
        if not p.is_absolute():
            d_runtime["out_dir"] = (base_dir / p).resolve()
    d["runtime"] = d_runtime

    try:
        return TrainExperiment.model_validate(d)
    except Exception as e:
        # Normalize to ValueError for test expectations
        raise ValueError(str(e))


def load_sample_config(path: Path) -> SampleExperiment:
    """Strict loader for sample section with minimal checks and path resolution.

    - Ensures required subsections exist ([runtime], [sample]).
    - Resolves relative out_dir against the config file directory.
    - Delegates detailed value validation to Pydantic models.
    """
    import tomllib
    base_dir = path.parent.resolve()
    with path.open("rb") as f:
        raw = tomllib.load(f)

    sample_tbl = raw.get("sample")
    if not isinstance(sample_tbl, dict):
        # Keep top-level semantic but tests check for missing [runtime] when sample table exists,
        # so only raise here if the entire [sample] table is missing.
        raise Exception("Config must contain [sample] block")

    for sec in ("runtime", "sample"):
        if not isinstance(sample_tbl.get(sec), dict):
            raise ValueError(f"Missing required section [{sec}]")

    # Resolve relative runtime.out_dir
    d = dict(sample_tbl)
    d_runtime = dict(d["runtime"])
    od = d_runtime.get("out_dir")
    if isinstance(od, (str, Path)):
        p = Path(od)
        if not p.is_absolute():
            d_runtime["out_dir"] = (base_dir / p).resolve()
    d["runtime"] = d_runtime

    try:
        return SampleExperiment.model_validate(d)
    except Exception as e:
        raise ValueError(str(e))





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
