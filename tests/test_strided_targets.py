"""Next-token targets are commonly a non-contiguous sequence slice."""
import unittest

import torch

from model import GPT, GPTConfig


class StridedTargetTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT(GPTConfig(
            block_size=6, vocab_size=11, n_layer=1, n_head=2, n_embd=8, dropout=0.0,
        ))

    def test_shifted_next_token_targets_match_explicit_log_probability(self):
        tokens = torch.randint(11, (2, 7))
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        self.assertFalse(targets.is_contiguous())
        logits, loss = self.model(inputs, targets)
        selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        oracle = (torch.logsumexp(logits, dim=-1) - selected).mean()
        torch.testing.assert_close(loss, oracle)

    def test_masked_noncontiguous_targets_reduce_only_valid_tokens(self):
        backing = torch.randint(11, (3, 12))
        targets = backing[:, ::2]
        targets[0, 1] = -1
        targets[2, 3] = -1
        logits, loss = self.model(torch.randint(11, (3, 6)), targets)
        valid = targets != -1
        selected = logits.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        oracle = (torch.logsumexp(logits, dim=-1) - selected)[valid].mean()
        torch.testing.assert_close(loss, oracle)

    def test_gradients_equal_contiguous_target_reference(self):
        tokens = torch.randint(11, (2, 7))
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        _, loss = self.model(inputs, targets)
        loss.backward()
        strided_grads = {
            name: parameter.grad.clone()
            for name, parameter in self.model.named_parameters()
        }
        self.model.zero_grad(set_to_none=True)
        _, reference = self.model(inputs.contiguous(), targets.contiguous())
        reference.backward()
        for name, parameter in self.model.named_parameters():
            torch.testing.assert_close(parameter.grad, strided_grads[name])


if __name__ == "__main__":
    unittest.main()

