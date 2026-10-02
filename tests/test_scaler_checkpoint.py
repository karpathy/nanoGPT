"""Exercise native checkpoint save/resume with a real CPU GradScaler."""
from contextlib import contextmanager
import os
from pathlib import Path
import pickle
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ScalerCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data = self.root / "data"
        self.out = self.root / "out"
        self.data.mkdir()
        tokens = (np.arange(100) % 8).astype(np.uint16)
        for split in ("train", "val"):
            tokens.tofile(self.data / f"{split}.bin")
        with (self.data / "meta.pkl").open("wb") as handle:
            pickle.dump({"vocab_size": 8}, handle)

    def tearDown(self):
        self.directory.cleanup()

    def run_training(self, resume=False, dtype="float16"):
        config = self.root / "config.py"
        config.write_text(
            f"dataset = {str(self.data)!r}\n"
            f"out_dir = {str(self.out)!r}\n"
            f"init_from = {'resume' if resume else 'scratch'!r}\n"
            f"dtype = {dtype!r}\n"
            "device = 'cpu'\ncompile = False\n"
            "n_layer = 1\nn_head = 1\nn_embd = 8\nblock_size = 4\n"
            "batch_size = 2\ngradient_accumulation_steps = 1\n"
            "eval_iters = 1\neval_interval = 1\nmax_iters = 2\n"
            "learning_rate = 0.001\ndecay_lr = False\n"
            "dropout = 0.0\nwandb_log = False\n"
        )
        scalers = []

        def cpu_scaler(*, enabled):
            # Keep native scale/backward/unscale/step/update behavior. Only the
            # device selection changes so checkpoint continuity can run on CPU.
            scaler = torch.amp.GradScaler("cpu", enabled=enabled, init_scale=32.0, growth_interval=100)
            scalers.append(scaler)
            return scaler

        sys.path.insert(0, str(REPO))
        try:
            with working_directory(REPO), patch.object(sys, "argv", ["train.py", str(config)]), \
                 patch.dict(os.environ, {"RANK": "-1"}), \
                 patch("torch.cuda.amp.GradScaler", side_effect=cpu_scaler):
                runpy.run_path(str(REPO / "train.py"), run_name="__main__")
        finally:
            sys.path.pop(0)
        return scalers[0], torch.load(self.out / "ckpt.pt", map_location="cpu", weights_only=False)

    def test_checkpoint_saves_enabled_scaler_state(self):
        scaler, checkpoint = self.run_training()
        self.assertIn("scaler", checkpoint)
        self.assertEqual(checkpoint["scaler"]["scale"], 32.0)
        self.assertEqual(checkpoint["scaler"]["_growth_tracker"], checkpoint["iter_num"])
        self.assertEqual(scaler.state_dict()["_growth_tracker"], 3)

    def test_resume_restores_scale_and_growth_tracker(self):
        _, checkpoint = self.run_training()
        state = torch.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=1000).state_dict()
        state["_growth_tracker"] = 777
        checkpoint["scaler"] = state
        torch.save(checkpoint, self.out / "ckpt.pt")
        start_iter = checkpoint["iter_num"]
        scaler, _ = self.run_training(resume=True)
        expected_steps = 3 - start_iter
        self.assertEqual(scaler.state_dict()["scale"], state["scale"])
        self.assertEqual(scaler.state_dict()["_growth_tracker"], 777 + expected_steps)

    def test_old_and_disabled_checkpoints_remain_loadable(self):
        _, checkpoint = self.run_training(dtype="float32")
        checkpoint.pop("scaler", None)
        torch.save(checkpoint, self.out / "ckpt.pt")
        scaler, resumed = self.run_training(resume=True, dtype="float32")
        self.assertFalse(scaler.is_enabled())
        self.assertEqual(resumed.get("scaler", {}), {})


if __name__ == "__main__":
    unittest.main()

