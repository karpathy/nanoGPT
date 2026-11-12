import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parent


class TestConfigurator(unittest.TestCase):
    def _run(self, argv, *, out_dir="out", wandb_run_name="gpt2", batch_size=12):
        ns = {
            "out_dir": out_dir,
            "wandb_run_name": wandb_run_name,
            "batch_size": batch_size,
        }
        with patch.object(sys, "argv", ["train.py", *argv]):
            code = (REPO / "configurator.py").read_text()
            exec(code, ns)
        return ns

    def test_override_keeps_equals_in_string_values(self):
        ns = self._run(["--out_dir=/tmp/run=v2", "--wandb_run_name=lr=6e-4"])
        self.assertEqual(ns["out_dir"], "/tmp/run=v2")
        self.assertEqual(ns["wandb_run_name"], "lr=6e-4")

    def test_plain_numeric_override_still_works(self):
        ns = self._run(["--batch_size=32"])
        self.assertEqual(ns["batch_size"], 32)
        self.assertIsInstance(ns["batch_size"], int)


if __name__ == "__main__":
    unittest.main()
