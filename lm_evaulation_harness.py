from contextlib import nullcontext
from typing import Iterable

import torch

import lm_eval
from lm_eval.base import BaseLM
from lm_eval.evaluator import evaluate, make_table

from model import GPT
from tokenizer import Tokenizer


class NanoGPTModel(BaseLM):

    def __init__(self, model: GPT, tokenizer: Tokenizer, device="cuda", temperature=0.8, top_k=200,
                 max_gen_tokens=128, batch_size=1, eot_token=50256):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_gen_tokens = max_gen_tokens
        self._batch_size = batch_size
        self._eot_token = eot_token
        self._temperature = temperature
        self._top_k = top_k

    @property
    def eot_token_id(self):
        return self._eot_token

    @property
    def max_length(self):
        return self._model.config.block_size

    @property
    def max_gen_toks(self):
        return self._max_gen_tokens

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self._tokenizer.encode(string)

    def tok_decode(self, tokens: Iterable[int]):
        return self._tokenizer.decode(list(tokens))

    def _model_generate(self, context, max_length, eos_token_id):
        return self._model.generate(context, max_length, temperature=self._temperature, top_k=self._top_k, eos_token=eos_token_id)

    def _model_call(self, inps):
        targets = torch.zeros_like(inps) - 1
        return self._model(inps, targets=targets)[0]


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'out'  # ignored if init_from is not 'resume'
    max_new_tokens = 500  # number of tokens generated in each sample
    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
    tasks = ["lambada_openai"]  # examples: --tasks='["lambada_openai"]'
    exec(open('configurator.py').read())  # overrides from command line or config file
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model, tokenizer = GPT.init_from(init_from, out_dir=out_dir, device=device)
    model.eval()
    model.to(device)

    with torch.no_grad():
        with ctx:
            results = evaluate(
                lm=NanoGPTModel(model, tokenizer, device=device, max_gen_tokens=max_new_tokens, temperature=temperature, top_k=top_k),
                task_dict=lm_eval.tasks.get_task_dict(tasks)
            )
            print(make_table(results))
