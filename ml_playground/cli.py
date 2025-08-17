from __future__ import annotations
import argparse
from pathlib import Path
from ml_playground.config import load_toml, AppConfig
from ml_playground.trainer import train
from ml_playground.sampler import sample


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser("ml_playground")
    sub = p.add_subparsers(dest="cmd", required=True)

    pprep = sub.add_parser(
        "prepare", help="Prepare dataset by name (internal preparers)"
    )
    pprep.add_argument(
        "dataset", choices=["shakespeare", "bundestag_char"], help="Dataset name"
    )

    ptrain = sub.add_parser("train", help="Train from TOML config")
    ptrain.add_argument("config", type=Path)

    psample = sub.add_parser(
        "sample",
        help="Sample using TOML config (tries ckpt_best.pt, ckpt_last.pt, then legacy ckpt.pt in out_dir)",
    )
    psample.add_argument("config", type=Path)

    ploop = sub.add_parser("loop", help="Run prepare -> train -> sample in one go")
    ploop.add_argument(
        "dataset", choices=["shakespeare", "bundestag_char"], help="Dataset name"
    )
    ploop.add_argument(
        "config",
        type=Path,
        help="TOML config path containing [train] and [sample] blocks",
    )

    args = p.parse_args(argv)

    if args.cmd == "prepare":
        from ml_playground.datasets import PREPARERS  # type: ignore

        fn = PREPARERS.get(args.dataset)
        if fn is None:
            raise SystemExit(f"Unknown dataset: {args.dataset}")
        fn()
        return

    if args.cmd == "loop":
        from ml_playground.datasets import PREPARERS  # type: ignore
        import shutil

        fn = PREPARERS.get(args.dataset)
        if fn is None:
            raise SystemExit(f"Unknown dataset: {args.dataset}")
        # 1) prepare
        fn()
        # 2) train
        cfg: AppConfig = load_toml(args.config)
        if cfg.train is None or cfg.sample is None:
            raise SystemExit(
                "Config for loop must contain both [train] and [sample] blocks"
            )
        train(cfg.train)
        # Copy dataset meta.pkl into out_dir for correct sampling (especially for char-level datasets)
        try:
            data_cfg = cfg.train.data
            if data_cfg.meta_pkl is not None:
                src_meta = data_cfg.dataset_dir / data_cfg.meta_pkl
                dst_meta = cfg.train.runtime.out_dir / "meta.pkl"
                if src_meta.exists():
                    shutil.copy2(src_meta, dst_meta)
        except Exception as e:
            # non-fatal; sampling will fallback if meta not present
            print(f"[loop] Warning: could not copy meta.pkl: {e}")
        # 3) sample
        sample(cfg.sample)
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


if __name__ == "__main__":
    main()
