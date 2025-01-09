import os
import torch
from transformers import GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList, GenerationConfig, AutoModelForCausalLM
from sat_dataset import SATTokenizer


class SATStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens=['SAT', 'UNSAT', '[EOS]']):
        self.stops = [tokenizer.encode(token)[0] for token in stop_tokens]
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for row in input_ids:
            if not any(stop_id in row for stop_id in self.stops):
                # If any row does not contain a stop token, continue generation
                return False
        # If all rows contain at least one stop token, stop generation
        return True
    
def get_interval_values(logits, intervals):
    """
    Extracts the logits at specific time-step 'intervals' for each batch element.

    logits:    [batch_size, seq_len, vocab_size]
    intervals: [batch_size] (each entry is an integer index into seq_len)

    Returns:   [batch_size, vocab_size]
    """
    batch_size, seq_len, vocab_size = logits.shape
    return logits[torch.arange(batch_size), intervals, :]

def is_old_tokenizer(tokenizer: SATTokenizer):
    return "-" in tokenizer.vocab

def get_dataset_path(dir, split='train', ext='txt'):
    raw_path = os.path.join(dir, f'{split}.{ext}')
    if not os.path.exists(raw_path):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(dir)
        try:
            os.system(f'python prepare.py')
        finally:
            os.chdir(cur_path)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'File {raw_path} not found and could not be downloaded.')
    return raw_path

def line_sat(line, sep=' '):
    if sep + 'UNSAT' in line:
        return False
    elif sep + 'SAT' in line:
        return True
    return None

def load_model_and_tokenizer(model_dir, padding_side="left"):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = SATTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = padding_side
    return model, tokenizer

# Very inelegant way to load a config file
def load_conf_file(args, key='config'):
    if hasattr(args, key) and getattr(args, key) is not None:
        with open(getattr(args, key), 'r') as f:
            conf_code = f.read()
            exec(conf_code, vars(args))
            if '__builtins__' in vars(args):
                del vars(args)['__builtins__']
        return args
    
def get_context_size(model):
    # List of known attribute names for context size across different models
    known_attributes = ['max_position_embeddings', 'n_positions', 'n_ctx']
    
    # Attempt to find and return the value of the first matching attribute
    for attr in known_attributes:
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
        
    if hasattr(model, 'config') and hasattr(model.config, 'block_size'):
        return model.config.block_size
    # Return None if none of the attributes are found
    raise AttributeError("Context size attribute not found in model configuration for attribute names: " + ", ".join(known_attributes))
    
def pad_max_len(input_ids, tokenizer, device):
    input = {"input_ids": input_ids}
    res = tokenizer.pad(input, padding=True, return_tensors="pt")
    return [input_id.to(device) for input_id in res['input_ids']]

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)