# Debug script to understand data table format
from pathlib import Path
import tempfile
from ml_playground.checkpoint import CheckpointManager, Checkpoint

# Create a temporary directory
with tempfile.TemporaryDirectory() as tmp_dir:
    manager = CheckpointManager(Path(tmp_dir), atomic=True, keep_last=2, keep_best=2)

    # Simulate the data table format from pytest-bdd
    # Based on the documentation, data tables are passed as list of lists
    metrics_table = [["metric"], ["1.0"], ["0.9"], ["1.1"]]

    print("Data table format:", metrics_table)

    # Test the parsing logic
    for row in metrics_table[1:]:  # Skip header row
        metric = float(row[0])
        iter_num = metrics_table.index(row) - 1  # -1 because we skip header
        print(f"Processing row: {row}, metric: {metric}, iter_num: {iter_num}")

        # Create dummy checkpoint
        ckpt = Checkpoint(
            model={"w": 1},
            optimizer={"o": 1},
            model_args={
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 4,
                "block_size": 8,
                "bias": False,
                "vocab_size": 16,
                "dropout": 0.0,
            },
            iter_num=iter_num,
            best_val_loss=metric,
            config={"ok": True},
            ema=None,
        )

        manager.save_checkpoint(
            ckpt,
            "ckpt_best.pt",
            metric=metric,
            iter_num=iter_num,
            logger=None,
            is_best=True,
        )

    # Check what files were created
    files = list(Path(tmp_dir).glob("ckpt_best_*.pt"))
    print("Files created:", [f.name for f in files])
