"""
验证可复现性：用相同种子跑两次，比较前100步的 loss 是否完全一致。
"""
import subprocess, sys

def run_100steps(run_id):
    cmd = [
        '/root/miniconda3/bin/python', 'train.py',
        '--dataset=shakespeare_char',
        '--n_layer=4', '--n_head=4', '--n_embd=128',
        '--batch_size=16', '--block_size=64',
        '--gradient_accumulation_steps=1',
        '--max_iters=100',
        '--eval_interval=9999',  # skip eval to speed up
        '--log_interval=1',
        '--compile=False',
        '--wandb_log=False',
        '--seed=42',
        '--out_dir=/root/autodl-tmp/ckpt',
        '--decay_lr=False',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/root/projects/nanoGPT')
    lines = result.stdout.strip().split('\n')
    losses = []
    for line in lines:
        if line.startswith('iter '):
            # e.g. "iter 0: loss 4.1234, time ..."
            parts = line.split()
            loss_val = float(parts[3].rstrip(','))
            losses.append(loss_val)
    return losses

print("=== Run 1 (seed=42) ===")
losses1 = run_100steps(1)
print(f"  First 5 losses: {losses1[:5]}")
print(f"  Last 5 losses:  {losses1[-5:]}")
print(f"  Total steps recorded: {len(losses1)}")

print("\n=== Run 2 (seed=42) ===")
losses2 = run_100steps(2)
print(f"  First 5 losses: {losses2[:5]}")
print(f"  Last 5 losses:  {losses2[-5:]}")
print(f"  Total steps recorded: {len(losses2)}")

print("\n=== Reproducibility Check ===")
if len(losses1) == 0 or len(losses2) == 0:
    print("ERROR: No loss values captured!")
    sys.exit(1)

n = min(len(losses1), len(losses2))
identical = all(abs(losses1[i] - losses2[i]) < 1e-6 for i in range(n))
if identical:
    print(f"✓ PASS: All {n} steps have identical loss values!")
else:
    diffs = [(i, losses1[i], losses2[i]) for i in range(n) if abs(losses1[i]-losses2[i]) >= 1e-6]
    print(f"✗ FAIL: {len(diffs)} steps differ. First diff: step {diffs[0][0]}, {diffs[0][1]:.6f} vs {diffs[0][2]:.6f}")
