import contextlib
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from utils import *

# FP8 Transformer Engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


def train(
    cfg_path: str,
    bsz: int = 8,
    n_steps: int = 128,
    grad_acc_steps: int = 8,
    pt_compile: bool = False,
    compile_mode: str = 'default',
    use_fp8: bool = False,
    profile: bool = False,
    output_dir: str = 'outputs/'
):
    '''
    :param       cfg_path: Model configuration file path
    :param            bsz: Batch size
    :param        n_steps: Number of training steps
    :param grad_acc_steps: Number of gradient accumulation steps
    :param     pt_compile: Enable PyTorch compile
    :param   compile_mode: Set PyTorch compile mode. Options: "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    :param        use_fp8: Enable FP8
    :param        profile: Enable profiling
    :param     output_dir: Profiling output saving directory
    '''
    torch.manual_seed(3985)

    cfg_m, model_cls, blk_cls = get_model_config(cfg_path, use_fp8)
    model = model_cls(**asdict(cfg_m)).to('cuda')
    print(f'Loaded {model_cls} model.', end=' ')
    cfg_m.estimate_flops_per_token(model, bsz)

    data_loader = create_data_loader(bsz, n_steps, cfg_m)
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f'{output_dir}/{Path(cfg_path).stem}_single_trace.json'

    if pt_compile:
        print(f'Compiling in {compile_mode} mode')
        model = torch.compile(model, mode=compile_mode)

    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')

    loop_iter = configure_train_loop(data_loader, profile, output_path, cfg_m, bsz, use_fp8)
    model.train()
    
    if use_fp8:
        # FP8
        fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
        fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')
        
        @contextlib.contextmanager
        def maybe_fp8_ctx():
            with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
                yield
    else:
        maybe_fp8_ctx = nullcontext

    for step_idx, data_batch in loop_iter:
        input_BT, label_BT = map(lambda t: t.pin_memory().to('cuda'), data_batch)

        with torch.amp.autocast('cuda', torch.bfloat16):
            with maybe_fp8_ctx():
                weight_cache = use_fp8 and (step_idx % grad_acc_steps == 0)
                logits_BTV = model(input_BT, is_first_microbatch=weight_cache)
                loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.flatten())
                loss /= grad_acc_steps
        loss.backward()

        if (step_idx + 1) % grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)


if __name__ == '__main__':
    import fire
    fire.Fire(train)
