"""CPU integration tests for train.py's evaluation-only mode.

Run with: python -m unittest discover -s tests -v
"""
import contextlib
import io
import os
from pathlib import Path
import pickle
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


class EvalOnlyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        data = self.work / 'data' / 'tiny'
        data.mkdir(parents=True)
        tokens = np.arange(64, dtype=np.uint16) % 16
        for split in ('train', 'val'):
            tokens.tofile(data / f'{split}.bin')
        with (data / 'meta.pkl').open('wb') as f:
            pickle.dump({'vocab_size': 16}, f)
        shutil.copyfile(ROOT / 'configurator.py', self.work / 'configurator.py')
        previous_cwd = Path.cwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, previous_cwd)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def run_training(self, **overrides):
        config = dict(
            dataset='tiny', out_dir='out', device='cpu', dtype='float32',
            compile=False, wandb_log=False, n_layer=1, n_head=1, n_embd=8,
            block_size=4, batch_size=2, gradient_accumulation_steps=1,
            dropout=0.0, eval_iters=1, eval_interval=1, max_iters=1,
            decay_lr=False, always_save_checkpoint=True,
        )
        config.update(overrides)
        argv = ['train.py'] + [f'--{k}={v}' for k, v in config.items()]
        output = io.StringIO()
        original_step = torch.optim.AdamW.step
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output):
            with patch.object(torch.optim.AdamW, 'step', autospec=True,
                              side_effect=original_step) as step:
                runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
        return step.call_count, output.getvalue()

    def test_fresh_eval_only(self):
        updates, output = self.run_training(eval_only=True)
        self.assertEqual(updates, 0)
        self.assertEqual(output.count('val loss'), 1)
        self.assertFalse((self.work / 'out' / 'ckpt.pt').exists())

    def test_resumed_eval_only(self):
        updates, _ = self.run_training()
        self.assertGreater(updates, 0)
        checkpoint = self.work / 'out' / 'ckpt.pt'
        saved = checkpoint.read_bytes()
        self.assertEqual(torch.load(checkpoint, weights_only=False)['iter_num'], 1)
        for interval in (1, 2):
            with self.subTest(eval_interval=interval):
                checkpoint.write_bytes(saved)
                before = checkpoint.stat().st_mtime_ns
                updates, output = self.run_training(
                    init_from='resume', eval_only=True, eval_interval=interval)
                self.assertEqual(updates, 0)
                self.assertEqual(output.count('val loss'), 1)
                self.assertEqual(checkpoint.read_bytes(), saved)
                self.assertEqual(checkpoint.stat().st_mtime_ns, before)


if __name__ == '__main__':
    unittest.main()
