"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
import json
from google.cloud import storage
from datetime import datetime, timezone


def retrieve_current_best_val_loss(bucket_name, metadata_file):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Retrieve the current metadata
    metadata_blob = bucket.blob(metadata_file)
    if metadata_blob.exists():
        metadata_content = json.loads(metadata_blob.download_as_text())
        best_val_loss = metadata_content.get("best_val_loss", float("inf"))
    else:
        metadata_content = {}
        best_val_loss = float("inf")

    print(f"Current best_val_loss: {best_val_loss}")
    return best_val_loss, metadata_content


def save_checkpoint_to_cloud(
    bucket_name, metadata_file, metadata_content, model_file, model_content
):
    upload_from_string(bucket_name, metadata_file, metadata_content)
    upload_model(bucket_name, model_file, model_content)
    print(f"Checkpoint and metadata uploaded to {bucket_name}")


def upload_model(bucket_name, model_file, model_content):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    torch.save(model_content, "temp.pt")
    model_blob = bucket.blob(model_file)
    model_blob.upload_from_filename("temp.pt")


def upload_from_string(bucket_name, file_name, content):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(json.dumps(content))


def download_checkpoint(bucket_name, model_file):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download the checkpoint
    model_blob = bucket.blob(model_file)
    model_blob.download_to_filename("temp.pt")

    print(f"Checkpoint downloaded from {bucket_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load("temp.pt", map_location=device)


def extract_dropouts_values(model):
    dropout_values = []
    for _i, layer in enumerate(model.transformer.h):
        _mlp_dropout = torch.sigmoid(layer.mlp.dropout.log_drop_p).item()
        _resid_dropout = torch.sigmoid(
            layer.attn.resid_dropout.log_drop_p
        ).item()
        dropout_values.append(
            {
                "layer": _i,
                "mlp_dropout": _mlp_dropout,
                "resid_dropout": _resid_dropout,
            }
        )

    return dropout_values

def get_dtype(device):    
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
        else:
            return "float16"
    else:
        return "float32"

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
bucket_name = "fiebre-fantasy-sakana"
metadata_filename = "better-transformer/metadata.json"
model_filename = "better-transformer/ckpt.pt"
out_dir = "out"
eval_interval = 250
log_interval = 200
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "enwik8"  # "shakespeare_char" #'openwebtext'
gradient_accumulation_steps = 2
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 64
# model
n_layer = 1
n_head = 1
n_embd = 128
dropout = 0.2  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# concrete dropout
concrete_dropout = False
init_p = None  # If none, it starts randomly.
temperature = 0.1
weight_reg_weight = 1e-5  # weight for the weight regularization loss
dropout_reg_weight = 1e-4  # weight for the dropout regularization loss
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = get_dtype(device)

compile = False  # use PyTorch 2.0 to compile the model to be faster
save_to_cloud = False
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    print("Running on single GPU")
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss, metadata_content = retrieve_current_best_val_loss(
    bucket_name, metadata_filename
)

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    concrete_dropout=concrete_dropout,
    init_p=init_p,
    temperature=temperature,
    weight_reg_weight=weight_reg_weight,
    dropout_reg_weight=dropout_reg_weight,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    # print(f"Resuming training from {out_dir}")
    print(f"Resuming training from {bucket_name}/{model_filename}")
    # resume training from a checkpoint.
    checkpoint = download_checkpoint(bucket_name, model_filename)
    # ckpt_path = os.path.join(out_dir, "ckpt.pt")
    # checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    print(f"resuming from iteration {iter_num}, best_val_loss: {best_val_loss}")
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
    )
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, splits=["train", "val"]):
    out = {}
    model.eval()
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss, dropout_loss, bpc = model(X, Y)
            losses[k] = bpc.item()

        out[split] = losses.mean()

    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(model)
        print(
            f"step {iter_num}: train bpc {losses['train']:.4f}, val bpc {losses['val']:.4f}"
        )
        if save_to_cloud:
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    metadata = {
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss.item(),
                    }
                    print(f"Saving checkpoint to {bucket_name}/{model_filename}")
                    save_checkpoint_to_cloud(
                        bucket_name,
                        metadata_filename,
                        metadata,
                        model_filename,
                        checkpoint,
                    )

            upload_from_string(
                bucket_name,
                file_name=f"better-transformer/{run_id}/val_{iter_num}.json",
                content={"iter_num": iter_num, "val_loss": losses["val"].item()},
            )
            upload_model(
                bucket_name,
                f"better-transformer/{run_id}/model_{iter_num}.pt",
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "config": config,
                },
            )

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss, dropout_loss, bpc = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        bpc = bpc.item()
        try:
            dropout_loss = dropout_loss.item()
        except AttributeError: # no concrete dropout
            dropout_loss = 0.0

        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, bpc {bpc:.3f}, dropout loss {dropout_loss:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
        try:
            dropout_values = extract_dropouts_values(raw_model)
        except AttributeError: # no concrete dropout
            dropout_values = []
            

        dropout_metadata = {
            "iter_num": iter_num,
            "dropout_values": dropout_values,
            "loss": lossf,
            "bpc": bpc,
            "dropout_loss": dropout_loss,
            "weight_reg_weight": weight_reg_weight,
            "dropout_reg_weight": dropout_reg_weight,
            "init_p": init_p,
            "temperature": temperature,
            "time": t0,
            "concrete_dropout": concrete_dropout,
            "dropout": dropout,  # constant dropout, where applicable
        }
        if save_to_cloud:
            upload_from_string(
                bucket_name,
                f"better-transformer/{run_id}/train_{iter_num}.json",
                dropout_metadata,
            )

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        loss = estimate_loss(model, splits=["test"])
        print(f"final test loss: {loss['test']:.4f}")
        if save_to_cloud:
            upload_from_string(
                bucket_name,
                f"better-transformer/{run_id}/final_test.json",
                {"iter_num": iter_num, "test_loss": loss["test"].item()},
            )

        break


if ddp:
    destroy_process_group()
