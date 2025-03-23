import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import numpy as np
import matplotlib.pyplot as plt
# PARAMS
temperature = 0.2
top_k=1
seed = 1337
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
out_dir = 'out-transcendence-gpt'
exec(open('configurator.py').read()) 

# Torch configs
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)


model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

data_dir = os.path.join('data', dataset)

test_data = torch.load(os.path.join(data_dir, 'test.pt'))[0]

correct_predictions = 0

SET_TOKEN = 3
NO_SET_TOKEN = 4

confusion_matrix = np.zeros((2, 2), dtype=int)

# run generation
with torch.no_grad():
    with ctx:
        num_samples = len(test_data)

        for i in range(num_samples):
            input_sequence = test_data[i, :-2].unsqueeze(0).to(device)

            actual_next_token = test_data[i, -2]
            
            predicted_next_token = model.generate(input_sequence, 1, temperature=temperature, top_k=top_k)[0][-1]
            
            actual_idx = 1 if actual_next_token == SET_TOKEN else 0
            pred_idx = 1 if predicted_next_token == SET_TOKEN else 0
            
            confusion_matrix[actual_idx, pred_idx] += 1

tn = confusion_matrix[0, 0]  # NO_SET correctly predicted as NO_SET
fp = confusion_matrix[0, 1]  # NO_SET incorrectly predicted as SET
fn = confusion_matrix[1, 0]  # SET incorrectly predicted as NO_SET
tp = confusion_matrix[1, 1]  # SET correctly predicted as SET


accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Confusion Matrix:")
print(f"TN={tn} (Token {NO_SET_TOKEN} predicted as {NO_SET_TOKEN})")
print(f"FP={fp} (Token {NO_SET_TOKEN} predicted as {SET_TOKEN})")
print(f"FN={fn} (Token {SET_TOKEN} predicted as {NO_SET_TOKEN})")
print(f"TP={tp} (Token {SET_TOKEN} predicted as {SET_TOKEN})")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ["NO_SET", "SET"])
plt.yticks(tick_marks, ["NO_SET", "SET"])
plt.xlabel('Predicted Token')
plt.ylabel('Actual Token')

thresh = confusion_matrix.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
# Uncomment below to save image on disk
# plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
# print(f"Confusion matrix saved to {os.path.join(out_dir, 'confusion_matrix.png')}")


