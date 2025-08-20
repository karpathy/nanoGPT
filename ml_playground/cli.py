from __future__ import annotations
import argparse
from argparse import ArgumentParser
from pathlib import Path
import shutil
import tomllib

from ml_playground.config import (
    TrainExperiment,
    SampleExperiment,
    DataConfig,
    RuntimeConfig,
)
from ml_playground import datasets
from ml_playground.sampler import sample
from ml_playground.trainer import train


def load_train_config(path: Path) -> TrainExperiment:
    """Strict loader with minimal legacy checks and path resolution.

    - Ensures required subsections exist ([model],[data],[optim],[schedule],[runtime]).
    - Produces a friendly unknown-key error for [train.data].
    - Resolves relative dataset_dir and out_dir against the config file directory.
    - Delegates detailed value validation to Pydantic models.
    """
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

    # Resolve relative paths and prune unsupported sections/keys
    allowed_sections = {"model", "data", "optim", "schedule", "runtime"}
    d = {k: train_tbl[k] for k in allowed_sections}

    # [train.data]: resolve dataset_dir relative to file and keep strict unknown-key check above
    d_data = dict(d["data"])
    dd = d_data.get("dataset_dir")
    if isinstance(dd, (str, Path)):
        p = Path(dd)
        if not p.is_absolute():
            d_data["dataset_dir"] = (base_dir / p).resolve()
    d["data"] = d_data

    # [train.runtime]: drop unknown keys then resolve out_dir
    allowed_runtime_keys = set(RuntimeConfig.model_fields.keys())
    d_runtime_all = dict(d["runtime"])
    d_runtime = {k: v for k, v in d_runtime_all.items() if k in allowed_runtime_keys}
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


def main(argv: list[str] | None = None) -> None:
    parser: ArgumentParser = configureArguments()
    args = parser.parse_args(argv)

    # Resolve config.toml from experiment where applicable
    def _resolve_config_from_experiment(experiment: str) -> Path:
        base = (
            Path(__file__).resolve().parent / "experiments" / experiment / "config.toml"
        )
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

    if args.cmd == "prepare":
        _PREPARERS = datasets.PREPARERS
        prepare = _PREPARERS.get(args.experiment)
        if prepare is None:
            raise SystemExit(f"Unknown experiment: {args.experiment}")
        prepare()
        return

    # Generic pipeline (default)
    if args.cmd == "loop":
        # Use the PREPARERS registry as-is (tests patch this directly)
        _PREPARERS = datasets.PREPARERS

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
        # Route speakger to Gemma PEFT integration trainer (JSONL/PEFT pipeline)
        if getattr(args, "experiment", None) == "speakger":
            # Lazy import: heavy optional deps (transformers/peft) are only loaded on this path
            from ml_playground.experiments.speakger import gemma_finetuning_mps as _sg

            # Ensure dataset is prepared before training (idempotent)
            try:
                _sg.prepare_from_toml(args.config)
            except Exception:
                # Let the trainer surface a clearer error if preparation failed
                pass
            _sg.train_from_toml(args.config)
            return
        try:
            train_cfg_single: TrainExperiment = load_train_config(args.config)
        except Exception as e:
            raise SystemExit(str(e))
        train(train_cfg_single)
        return

    if args.cmd == "sample":
        # Route speakger to Gemma PEFT integration sampler (JSONL/PEFT pipeline)
        if getattr(args, "experiment", None) == "speakger":
            # Lazy import: heavy optional deps (transformers/peft) are only loaded on this path
            from ml_playground.experiments.speakger import gemma_finetuning_mps as _sg

            _sg.sample_from_toml(args.config)
            return
        try:
            sample_cfg_single: SampleExperiment = load_sample_config(args.config)
        except Exception as e:
            raise SystemExit(str(e))
        # Best-effort: copy dataset meta.pkl into out_dir for strict sampling
        try:
            with args.config.open("rb") as f:
                raw = tomllib.load(f)
            data_tbl = raw.get("train", {}).get("data", {}) or {}
            ds = data_tbl.get("dataset_dir")
            meta_name = data_tbl.get("meta_pkl", "meta.pkl")
            dst_meta = sample_cfg_single.runtime.out_dir / "meta.pkl"
            if not dst_meta.exists() and isinstance(meta_name, (str, Path)):
                # Ensure destination directory exists
                sample_cfg_single.runtime.out_dir.mkdir(parents=True, exist_ok=True)
                candidates: list[Path] = []
                if isinstance(ds, (str, Path)):
                    ds_path = Path(ds)
                    # 1) Try dataset_dir as provided (absolute or CWD-relative)
                    candidates.append(ds_path)
                    # 2) If not absolute, also try relative to the config directory
                    if not ds_path.is_absolute():
                        candidates.append((args.config.parent / ds_path).resolve())
                for cand in candidates:
                    src_meta = Path(cand) / str(meta_name)
                    if src_meta.exists():
                        shutil.copy2(src_meta, dst_meta)
                        break
        except Exception:
            # Non-fatal; sampler will enforce presence
            pass
        # Also best-effort: copy meta.json if present so sampler can use JSON hints
        try:
            with args.config.open("rb") as f:
                raw = tomllib.load(f)
            data_tbl = raw.get("train", {}).get("data", {}) or {}
            ds = data_tbl.get("dataset_dir")
            dst_json = sample_cfg_single.runtime.out_dir / "meta.json"
            if not dst_json.exists():
                json_candidates: list[Path] = []
                if isinstance(ds, (str, Path)):
                    ds_path = Path(ds)
                    json_candidates.append(ds_path)
                    if not ds_path.is_absolute():
                        json_candidates.append((args.config.parent / ds_path).resolve())
                for cand in json_candidates:
                    src_json = Path(cand) / "meta.json"
                    if src_json.exists():
                        sample_cfg_single.runtime.out_dir.mkdir(
                            parents=True, exist_ok=True
                        )
                        shutil.copy2(src_json, dst_json)
                        break
        except Exception:
            # Non-fatal; sampler has a deterministic fallback
            pass
        sample(sample_cfg_single)
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
    train_parser = sub.add_parser(
        "train", help="Train for the given experiment (config.toml auto-resolved)"
    )
    train_parser.add_argument("experiment", help="Experiment name")

    sample_parser = sub.add_parser(
        "sample",
        help="Sample for the given experiment (config.toml auto-resolved; tries best/last checkpoints)",
    )
    sample_parser.add_argument("experiment", help="Experiment name")

    # Loop takes only an experiment
    loop_parser = sub.add_parser(
        "loop",
        help="Run prepare -> train -> sample for the given experiment (config.toml auto-resolved)",
    )
    loop_parser.add_argument(
        "experiment",
        help="Experiment name",
    )

    return p


if __name__ == "__main__":
    main()
