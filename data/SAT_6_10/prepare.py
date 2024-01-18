"""
Prepare the SAT Solver dataset for proposition-level language modeling.
Mapping: Each Proposition -> their id, Negation -> '-', clauses separator -> '0', Original SAT Instance separator -> '[SEP]' (Only one per instance), Result: 'SAT' 'UNSAT'
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

MAX_PROP = 30
DATASET_FILE = 'SAT_N6_10_Clause_Trace.txt'

input_file_path = os.path.join(os.path.dirname(__file__), DATASET_FILE)
if not os.path.exists(input_file_path):
    # download from HuggingFace
    url = 'https://huggingface.co/datasets/leyanpan/sat-solver/resolve/main/SAT_N6_10_Clause_Trace.txt?download=true'
    r = requests.get(url)
    with open(input_file_path, 'wb') as f:
        f.write(r.content)

with open(input_file_path, 'r') as f:
    lines = f.readlines()
print(f"Number of SAT Solution Instances: {len(lines):,}")

token_set = []
# Propositions
for i in range(1, MAX_PROP + 1):
    token_set.append(str(i))
# Negation
token_set.append('-')
# Separators
token_set.append('0')
token_set.append('[SEP]')

# Result
token_set.append('SAT')
token_set.append('UNSAT')

# Padding
token_set.append('[PAD]')

# # Parentheses
# token_set.append('(')
# token_set.append(')')

# # Conflict Clause Marker
# token_set.append('<CC>')
# token_set.append('</CC>')


# get all the unique characters that occur in this text
vocab_size = len(token_set)
print("all the unique tokens:", ' '.join(token_set))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(token_set) }
itos = { i:ch for i,ch in enumerate(token_set) }
def encode(s):
    return [stoi[c] for c in s.split()] # encoder: take a string, output a list of integers
def decode(l):
    return ' '.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(lines)
train_data = lines[:int(n*0.9)]
val_data = lines[int(n*0.9):]

# encode both to integers
train_ids = [encode(line.replace('-', '- ')) for line in train_data]
val_ids = [encode(line.replace('-', '- ')) for line in val_data]


print(f"train has {len(train_ids):,} samples")
print(f"val has {len(val_ids):,} samples")

# pad all sequences to the same length (the longest sequence in the dataset)
pad_id = stoi['[PAD]']
max_len = max([len(x) for x in train_ids + val_ids])
print(f"max sequence length: {max_len}")
padded_train_ids = [x + [pad_id] * (max_len - len(x)) for x in train_ids]
padded_val_ids = [x + [pad_id] * (max_len - len(x)) for x in val_ids]

# export to bin files
padded_train_ids = np.array(padded_train_ids, dtype=np.int16)
padded_val_ids = np.array(padded_val_ids, dtype=np.int16)
print('Train data shape:' + str(padded_train_ids.shape))
print('Validation data shape:' + str(padded_val_ids.shape))
np.save(os.path.join(os.path.dirname(__file__), 'train.npy'), padded_train_ids)
np.save(os.path.join(os.path.dirname(__file__), 'val.npy'), padded_val_ids)

# Generate classification dataset
train_formulas = [encode(line.split('[SEP]')[0].replace('-', '- ')) for line in train_data]
val_formulas = [encode(line.split('[SEP]')[0].replace('-', '- ')) for line in val_data]

# pad all formulas to the same length (the longest formula in the dataset)
max_len = max([len(x) for x in train_formulas + val_formulas])
print(f"max formula length: {max_len}")
padded_train_formulas = [x + [pad_id] * (max_len - len(x)) for x in train_formulas]
padded_val_formulas = [x + [pad_id] * (max_len - len(x)) for x in val_formulas]

train_labels = [stoi['UNSAT'] if 'UNSAT' in line else stoi['SAT'] for line in train_data]
val_labels = [stoi['UNSAT'] if 'UNSAT' in line else stoi['SAT'] for line in val_data]
np.savez(os.path.join(os.path.dirname(__file__), 'train_class.npz'), X=padded_train_formulas, y=train_labels)
np.savez(os.path.join(os.path.dirname(__file__), 'val_class.npz'), X=padded_val_formulas, y=val_labels)

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

