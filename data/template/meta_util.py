
import pickle
import argparse
from collections import Counter

def load_meta(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def view_tokens(meta_path):
    meta = load_meta(meta_path)
    print(f"Vocabulary Size: {meta['vocab_size']}")
    print("String to Index Mapping:")
    for k, v in list(meta['stoi'].items())[:]:  # Display only the first 10 for brevity
        print(f"{k}: {v}")
    print("Index to String Mapping:")
    for k, v in list(meta['itos'].items())[:]:  # Display only the first 10 for brevity
        print(f"{k}: {v}")

def merge_metas(meta_path1, meta_path2, output_path):
    meta1 = load_meta(meta_path1)
    meta2 = load_meta(meta_path2)
    
    # Combine and resolve conflicts if any
    stoi = {**meta1['stoi'], **meta2['stoi']}
    itos = {**meta1['itos'], **meta2['itos']}
    
    # Check for conflicts and duplicates - simplistic approach, needs refinement for real conflicts
    stoi_counter = Counter(stoi.values())
    conflict_ids = [id for id, count in stoi_counter.items() if count > 1]
    if conflict_ids:
        print(f"Conflicts detected in token IDs: {conflict_ids}. Manual resolution required.")
        return
    
    # Assuming no conflicts or resolved
    vocab_size = len(stoi)
    meta = {'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}
    with open(output_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Merged meta saved to {output_path}.")

def create_meta_from_text(text_file, output_path, special_chars={'<ukn>': 0}):
    with open(text_file, 'r') as f:
        tokens = f.read().split('\n')
    
    stoi = {token: i for i, token in enumerate(tokens, start=len(special_chars))}
    stoi.update(special_chars)  # Add special characters with predefined IDs
    itos = {i: token for token, i in stoi.items()}
    vocab_size = len(stoi)
    
    meta = {'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}
    with open(output_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Meta created from text and saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser(description="Utility for handling token metadata.")
    parser.add_argument("--view", type=str, help="Path to the meta.pkl file to view.")
    parser.add_argument("--merge", nargs=2, help="Paths to the two meta.pkl files to merge.")
    parser.add_argument("--create", nargs=2, help="Path to the input text file and the output meta.pkl file for creation.")
    
    args = parser.parse_args()
    
    if args.view:
        view_tokens(args.view)
    elif args.merge:
        merge_metas(args.merge[0], args.merge[1], "merged_meta.pkl")
    elif args.create:
        create_meta_from_text(args.create[0], args.create[1])

if __name__ == "__main__":
    main()

