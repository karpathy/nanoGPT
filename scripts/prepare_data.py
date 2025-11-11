"""Data preparation script for nanoGPT training pipeline."""
import argparse
import os
import pickle
from pathlib import Path
import numpy as np

try:
    import tiktoken
except ImportError:
    print("Installing tiktoken...")
    os.system("pip install tiktoken")
    import tiktoken


def prepare_data(input_dir: str, output_dir: str, val_split: float = 0.1):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Reading text files from {input_path}")
    
    all_text = []
    text_files = list(input_path.glob("*.txt"))
    
    if not text_files:
        print(f"❌ No .txt files found in {input_path}")
        return False
    
    for text_file in text_files:
        print(f"  Reading {text_file.name}...")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
            all_text.append(text)
    
    combined_text = "\n\n".join(all_text)
    print(f"✅ Total characters: {len(combined_text):,}")
    
    n = len(combined_text)
    train_text = combined_text[:int(n * (1 - val_split))]
    val_text = combined_text[int(n * (1 - val_split)):]
    
    print(f"📊 Train characters: {len(train_text):,}")
    print(f"📊 Val characters: {len(val_text):,}")
    
    print("\n🔤 Encoding text with GPT-2 tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    
    train_ids = enc.encode_ordinary(train_text)
    val_ids = enc.encode_ordinary(val_text)
    
    print(f"✅ Train tokens: {len(train_ids):,}")
    print(f"✅ Val tokens: {len(val_ids):,}")
    
    print("\n💾 Saving binary files...")
    
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_path = output_path / "train.bin"
    val_path = output_path / "val.bin"
    
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    
    print(f"✅ Saved {train_path}")
    print(f"✅ Saved {val_path}")
    
    meta = {
        'vocab_size': enc.n_vocab,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
    }
    
    meta_path = output_path / "meta.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"✅ Saved {meta_path}")
    
    if train_path.exists() and val_path.exists():
        print("\n✨ Data preparation complete!")
        return True
    else:
        print("\n❌ Failed to create binary files")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()
    
    success = prepare_data(args.input_dir, args.output_dir, args.val_split)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
