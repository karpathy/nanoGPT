# Sadly some old embedded GPUs like Jetson Nano can't get tiktokenizer even if you try
# to compile it.
try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    print("tiktoken not available, using fallback tokenizer")
    tiktoken = None
    tiktoken_available = False

def get_tokenizer(model_name="gpt2"):
    if tiktoken_available:
        return tiktoken.get_encoding("gpt2") if model_name == "gpt2" else tiktoken.encoding_for_model(model_name)
    else:
        # GPT-2 BPE fallback using encoder.py + vocab.json and merges.txt
        import os
        import json
        import re
        import requests

        base_dir = os.path.dirname(__file__)
        encoder_path = os.path.join(base_dir, 'encoder.py')
        models_dir = os.path.join(base_dir, 'models')
        model_path = os.path.join(models_dir, model_name)
        vocab_path = os.path.join(model_path, 'encoder.json')
        merges_path = os.path.join(model_path, 'vocab.bpe')

        os.makedirs(model_path, exist_ok=True)

        if not os.path.exists(encoder_path):
            data_url = 'https://raw.githubusercontent.com/openai/gpt-2/master/src/encoder.py'
            with open(encoder_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)

        if not os.path.exists(vocab_path):
            data_url = 'https://huggingface.co/gpt2/resolve/main/vocab.json'
            with open(vocab_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)

        if not os.path.exists(merges_path):
            data_url = 'https://huggingface.co/gpt2/resolve/main/merges.txt'
            with open(merges_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)

        # Add current directory to path before importing encoder.py dynamically
        import sys
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)

        from encoder import get_encoder

        tokenizer = get_encoder(model_name=model_name, models_dir=models_dir)
        # Patch for API compatibility with tiktoken
        tokenizer.encode_ordinary = tokenizer.encode

        return tokenizer
