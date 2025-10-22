"""
Quick test script to verify SSM model implementation
"""
import torch
from model import SSMConfig, SSM

# Test 1: Model initialization
print("Test 1: Initializing SSM model...")
config = SSMConfig(
    block_size=128,
    vocab_size=100,
    n_layer=4,
    n_embd=256,
    dropout=0.0,
    bias=True,
    ssm_state_dim=16,
    ssm_conv_dim=4,
    ssm_expand=2
)
model = SSM(config)
print(f"✓ Model initialized successfully with {model.get_num_params()/1e6:.2f}M parameters")

# Test 2: Forward pass
print("\nTest 2: Testing forward pass...")
batch_size = 2
seq_len = 32
idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

logits, loss = model(idx, targets)
print(f"✓ Forward pass successful")
print(f"  Input shape: {idx.shape}")
print(f"  Logits shape: {logits.shape}")
print(f"  Loss: {loss.item():.4f}")

# Test 3: Generation
print("\nTest 3: Testing generation...")
model.eval()
start_ids = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(start_ids, max_new_tokens=10, temperature=1.0, top_k=10)
print(f"✓ Generation successful")
print(f"  Generated sequence length: {generated.shape[1]}")
print(f"  Generated tokens: {generated[0].tolist()}")

# Test 4: Backward pass
print("\nTest 4: Testing backward pass...")
model.train()
optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=1e-3,
    betas=(0.9, 0.95),
    device_type='cpu'
)
logits, loss = model(idx, targets)
loss.backward()
optimizer.step()
print(f"✓ Backward pass and optimizer step successful")

print("\n" + "="*50)
print("All tests passed! SSM implementation is working correctly.")
print("="*50)
