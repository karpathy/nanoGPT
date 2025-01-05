import os
import torch
import argparse
import tqdm

# We'll still use GenerationConfig, StoppingCriteriaList from Hugging Face for convenience
from transformers import GenerationConfig, StoppingCriteriaList

# Import your custom GPT, GPTConfig classes (adjust the import as needed)
# e.g., from gpt import GPT, GPTConfig
from model_ext import GPT, GPTConfig

from sat_dataset import SATTokenizer
from utils import (
    line_sat,
    SATStoppingCriteria,
    is_old_tokenizer,
    load_conf_file,
    get_context_size,
)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run a GPT-based SAT model for dataset completion and evaluate SAT/UNSAT prediction accuracy."
    )

    # Required arguments
    parser.add_argument("model_dir", type=str, default=None,
                        help="Path to the model directory containing ckpt.pt.")
    parser.add_argument("dataset", type=str, default=None,
                        help="Path to the dataset directory.")

    # Optional arguments
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to a config file (JSON/YAML) if needed.")
    parser.add_argument("-l", "--max_len", type=int, default=850,
                        help="Maximum length of generated completions.")
    parser.add_argument("-n", "--num_samples", type=int, default=None,
                        help="Number of samples to generate. If None, the entire file is used.")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size for generation.")
    parser.add_argument("-f", "--file_name", type=str, default='test.txt',
                        help="Which file to evaluate in the dataset directory.")
    parser.add_argument("-o", "--out_file", type=str, default=None,
                        help="Path to output file for completions.")

    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if CUDA is available.")
    parser.add_argument("--stop_crit", action="store_true",
                        help="Use a stopping criterion during generation.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Print debug information.")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()
    load_conf_file(args)  # merges any config file contents into args

    # Decide if we can use GPU
    args.use_cuda = not args.cpu and torch.cuda.is_available()

    return args


def load_custom_gpt_model(model_dir, device='cpu'):
    """
    Loads a custom GPT model from a directory containing 'ckpt.pt'.
    This follows the code snippet you provided for custom initialization.
    """
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Create GPTConfig from the checkpoint's stored args
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Remove unwanted prefixes (e.g. _orig_mod.)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            new_key = k[len(unwanted_prefix):]
            state_dict[new_key] = state_dict.pop(k)

    # Load parameters
    model.load_state_dict(state_dict)
    return model


def batch_generate_completions(
    input_file,
    model,
    tokenizer,
    batch_size,
    max_length,
    num_samples=None,
    stop_criteria=None,
    debug=False
):
    completions = []
    true_labels = []
    pred_labels = []

    old_tk = is_old_tokenizer(tokenizer)

    with open(input_file, 'r') as file:
        if old_tk:
            lines = [line.strip().replace("-", "- ") for line in file]
        else:
            lines = [line.strip() for line in file]

    # Model context size
    context_size = get_context_size(model)
    # Min of model's context vs. desired max_length
    gen_config = GenerationConfig(
        max_length=min(max_length, context_size),
        num_return_sequences=1,
        # Because we are not using a HF GPT2 tokenizer,
        # set pad/eos to something consistent if needed:
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.pad_token_id
    )

    if num_samples is None:
        num_samples = len(lines)

    for i in tqdm.tqdm(range(0, num_samples, batch_size)):
        batch_lines = lines[i : i + batch_size]
        # Grab everything up to [SEP], inclusive
        batch_prompts = [
            ln[: ln.find("[SEP]") + len("[SEP]")] for ln in batch_lines
        ]
        batch_true_labels = [line_sat(ln) for ln in batch_lines]
        true_labels.extend(batch_true_labels)

        # Tokenize
        tokenized = tokenizer(
            batch_prompts, return_tensors="pt",
            padding=True, truncation=True
        )
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        # Generate
        if attention_mask is not None:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                stopping_criteria=stop_criteria
            )
        else:
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=gen_config,
                stopping_criteria=stop_criteria
            )

        # Decode + parse predictions
        for out in outputs:
            comp = tokenizer.decode(out, skip_special_tokens=True)
            # Only keep up to "SAT" if present
            idx_sat = comp.find("SAT")
            if idx_sat != -1:
                comp = comp[: idx_sat + len("SAT")]
            else:
                # If "SAT" not found, keep nothing or partial text
                comp = comp[: -1]

            if debug:
                print(comp)

            completions.append(comp)
            pred_labels.append(line_sat(comp))

    return completions, true_labels, pred_labels


if __name__ == "__main__":
    args = parse_args()

    # Validate dataset and model_dir
    if not args.dataset:
        raise ValueError("Please specify --dataset to evaluate.")
    if not args.model_dir:
        raise ValueError("Please specify --model_dir containing ckpt.pt")

    # Input file to evaluate
    input_fn = os.path.join(args.dataset, args.file_name)

    # If no out_file given, create a default
    if args.out_file is None:
        base_model = os.path.basename(args.model_dir)
        base_data = os.path.basename(args.dataset)
        args.out_file = os.path.join("preds", f"{base_model}_{base_data}_{args.file_name}")
    print(f"Writing to {args.out_file}.")

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load the custom GPT model
    device = "cuda" if args.use_cuda else "cpu"
    model = load_custom_gpt_model(args.model_dir, device=device)
    if args.debug:
        print(model.summary())
        for name, param in model.named_parameters():
            print(name, param.shape, param.numel())
    model.to(device)

    # Instantiate a fresh SATTokenizer instead of loading from directory
    tokenizer = SATTokenizer()

    # Optionally enable stopping criteria
    stop_criteria = None
    if args.stop_crit:
        sat_stop = SATStoppingCriteria(tokenizer)
        stop_criteria = StoppingCriteriaList([sat_stop])

    # # Generate
    # completions, true_labels, pred_labels = batch_generate_completions(
    #     input_file=input_fn,
    #     model=model,
    #     tokenizer=tokenizer,
    #     batch_size=args.batch_size,
    #     max_length=args.max_len,
    #     num_samples=args.num_samples,
    #     stop_criteria=stop_criteria,
    #     debug=args.debug
    # )
    with open(input_fn, 'r') as f:
        lines = [line.strip() for line in f]

    args.num_samples = min(args.num_samples, len(lines)) if args.num_samples is not None else len(lines)
    
    true_labels = [line_sat(line) for line in lines[:args.num_samples]]
    completions = []
    pred_labels = []

    for line in lines[:args.num_samples]:
        prompt = line[:line.find("[SEP]") + len("[SEP]")] 
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(device)
        stop_tokens = tokenizer.encode("SAT UNSAT")
        output = model.generate(input_ids=input_ids, max_new_tokens=args.max_len, stop=stop_tokens)
        completion = tokenizer.decode(output[0].tolist())
        completions.append(completion)
        pred_labels.append(line_sat(completion))
        if args.debug:
            print(f"Completion: {completion}")
            print("---------------")
    # Evaluate
    num_failed = 0
    if true_labels and pred_labels:
        for i in range(args.num_samples):
            # If line_sat returned None => no SAT/UNSAT found => fail
            if pred_labels[i] is None:
                num_failed += 1
                pred_labels[i] = not true_labels[i]

        # Compute metrics (pos_label=False means "UNSAT" is the positive class)
        f1 = f1_score(true_labels, pred_labels, pos_label=False)
        acc = accuracy_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels, pos_label=False)
        recall = recall_score(true_labels, pred_labels, pos_label=False)
        completion_acc = 1 - num_failed / len(true_labels)

        print(f"Completion: {completion_acc}")
        print(f"F1 Score: {f1}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {recall}")
    else:
        print("No labels to evaluate. Check your data file for lines containing [SEP].")

    # Write completions to file
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        for comp in completions:
            f.write(comp + "\n")