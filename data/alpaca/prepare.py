"""
准备 Stanford Alpaca 数据集用于 GPT-2 124M 的 SFT。

数据格式：每条样本由 prompt(指令+输入) 和 response(回答) 组成，
我们用 GPT-2 BPE tokenizer 编码，并生成与 token 序列等长的 mask：
  - mask=0 表示该 token 属于 prompt（不参与 loss）
  - mask=1 表示该 token 属于 response + eos（参与 loss）

输出文件（uint16 token / uint8 mask，一维拼接）：
  train.bin, train_labels.bin, val.bin, val_labels.bin

用法：
  python data/alpaca/prepare.py
"""
import os
import json
import requests
import tiktoken
import numpy as np

# Alpaca 官方 prompt 模板（与 stanford_alpaca 仓库 train.py 中一致）
PROMPT_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
PROMPT_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_prompt(instruction, input_text):
    if input_text:
        return PROMPT_TEMPLATE_WITH_INPUT.format(instruction=instruction, input=input_text)
    return PROMPT_TEMPLATE_NO_INPUT.format(instruction=instruction)


def main():
    here = os.path.dirname(__file__)

    # 1) 下载 alpaca_data.json（52,002 条指令数据）
    input_file_path = os.path.join(here, 'alpaca_data.json')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json'
        print(f"downloading {data_url} ...")
        resp = requests.get(data_url, timeout=120)
        resp.raise_for_status()
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(resp.text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"loaded {len(data)} examples")

    # 2) 9:1 切分训练/验证
    #    Alpaca 训练用 1~3 epoch 即可，验证集仅用于监控，不必太大
    n = len(data)
    n_train = int(n * 0.97)
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"train {len(train_data)}, val {len(val_data)}")

    # 3) 用 GPT-2 BPE 编码
    enc = tiktoken.get_encoding("gpt2")
    eos_id = enc.eot_token  # 50256，<|endoftext|>，作为每条样本的结束符

    def build(split_data):
        tokens = []
        masks = []
        n_resp_tokens = 0
        for ex in split_data:
            instruction = ex['instruction'].strip()
            input_text = (ex.get('input') or '').strip()
            output = ex['output'].strip()

            prompt = format_prompt(instruction, input_text)
            # 用 encode_ordinary：不插入 special token，纯 BPE
            prompt_ids = enc.encode_ordinary(prompt)
            response_ids = enc.encode_ordinary(output)
            response_ids.append(eos_id)

            tokens.extend(prompt_ids)
            masks.extend([0] * len(prompt_ids))
            tokens.extend(response_ids)
            masks.extend([1] * len(response_ids))
            n_resp_tokens += len(response_ids)
        return tokens, masks, n_resp_tokens

    train_tokens, train_masks, train_resp = build(train_data)
    val_tokens, val_masks, val_resp = build(val_data)

    print(f"train: {len(train_tokens):,} tokens, response tokens {train_resp:,} ({100*train_resp/len(train_tokens):.1f}%)")
    print(f"val:   {len(val_tokens):,} tokens, response tokens {val_resp:,} ({100*val_resp/len(val_tokens):.1f}%)")

    # 4) 落盘：tokens 用 uint16，mask 用 uint8
    np.array(train_tokens, dtype=np.uint16).tofile(os.path.join(here, 'train.bin'))
    np.array(train_masks, dtype=np.uint8).tofile(os.path.join(here, 'train_labels.bin'))
    np.array(val_tokens, dtype=np.uint16).tofile(os.path.join(here, 'val.bin'))
    np.array(val_masks, dtype=np.uint8).tofile(os.path.join(here, 'val_labels.bin'))
    print("done. files saved to", here)


if __name__ == '__main__':
    main()
