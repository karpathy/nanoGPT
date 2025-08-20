from __future__ import annotations
import argparse
import os
from argparse import ArgumentParser
from pathlib import Path
from ml_playground.config import load_toml, AppConfig
from ml_playground.trainer import train
from ml_playground.sampler import sample


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

    # Special: If the dataset is "bundestag_finetuning_mps" OR the config explicitly declares that dataset in [prepare]
    def is_bundestag_finetuning_mps(dataset: str | None, config: Path | None) -> bool:
        if dataset == "bundestag_finetuning_mps":
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

    # Special: If the dataset is "gemma_finetuning_mps" OR the config file contains the integration-specific block
    def is_gemma_finetuning_mps(dataset: str | None, config: Path | None) -> bool:
        if dataset == "gemma_finetuning_mps":
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
                getattr(args, "dataset", None), getattr(args, "config", None)
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
                getattr(args, "dataset", None), getattr(args, "config", None)
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
        prepare = _PREPARERS.get(args.dataset)
        if prepare is None:
            raise SystemExit(
                f"Unknown dataset: {getattr(args, 'dataset', None)}. Provide a known dataset name or a TOML config for an integration dataset."
            )
        prepare()
        return

    # For integration commands: route each one to the proper integration entrypoint
    if is_bundestag_finetuning_mps(
        getattr(args, "dataset", None), getattr(args, "config", None)
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
        getattr(args, "dataset", None), getattr(args, "config", None)
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
        prepare = _PREPARERS.get(args.dataset)
        if prepare is None:
            raise SystemExit(f"Unknown dataset: {args.dataset}")
        # 1) prepare
        prepare()
        # 2) train
        loop_cfg: AppConfig = load_toml(args.config)
        if loop_cfg.train is None or loop_cfg.sample is None:
            raise SystemExit(
                "Config for loop must contain both [train] and [sample] blocks"
            )
        train(loop_cfg.train)
        # Copy dataset meta.pkl into out_dir for correct sampling (especially for char-level datasets)
        try:
            data_cfg = loop_cfg.train.data
            if data_cfg.meta_pkl is not None:
                src_meta = data_cfg.dataset_dir / data_cfg.meta_pkl
                dst_meta = loop_cfg.train.runtime.out_dir / "meta.pkl"
                if src_meta.exists():
                    shutil.copy2(src_meta, dst_meta)
        except Exception as e:
            # non-fatal; sampling will fallback if meta not present
            print(f"[loop] Warning: could not copy meta.pkl: {e}")
        # 3) sample
        sample(loop_cfg.sample)
        return

    cfg: AppConfig = load_toml(args.config)

    if args.cmd == "train":
        if cfg.train is None:
            raise SystemExit("Config must contain [train] block")
        train(cfg.train)
        return

    if args.cmd == "sample":
        if cfg.sample is None:
            raise SystemExit("Config must contain [sample] block")
        sample(cfg.sample)
        return


def configureArguments():
    p = argparse.ArgumentParser("ml_playground")
    sub = p.add_subparsers(dest="cmd", required=True)
    prep_parser = sub.add_parser(
        "prepare",
        help="Prepare dataset by name (legacy) or via TOML for integration datasets",
    )
    prep_parser.add_argument(
        "dataset",
        help="Dataset name (e.g., shakespeare, bundestag_char, bundestag_tiktoken, bundestag_finetuning_mps, gemma_finetuning_mps)",
    )
    prep_parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="Optional TOML config path for integration datasets (bundestag_finetuning_mps, gemma_finetuning_mps)",
    )
    sub.add_parser("train", help="Train from TOML config").add_argument(
        "config", type=Path
    )
    sub.add_parser(
        "sample",
        help="Sample using TOML config (tries ckpt_best.pt, ckpt_last.pt, then legacy ckpt.pt in out_dir)",
    ).add_argument("config", type=Path)
    loop_parser = sub.add_parser(
        "loop", help="Run prepare -> train -> sample in one go"
    )
    loop_parser.add_argument(
        "dataset",
        choices=[
            "shakespeare",
            "bundestag_char",
            "bundestag_tiktoken",
            "bundestag_finetuning_mps",
            "gemma_finetuning_mps",
        ],
        help="Dataset name",
    )
    loop_parser.add_argument(
        "config",
        type=Path,
        help="TOML config path containing [train] and [sample] blocks",
    )
    return p


if __name__ == "__main__":
    main()
