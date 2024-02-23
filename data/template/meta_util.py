import pickle
import argparse


def load_meta(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def view_tokens(meta_path):
    meta = load_meta(meta_path)
    print(f"Vocabulary Size: {meta['vocab_size']}")
    print("String to Index Mapping:")
    for k, v in list(meta["stoi"].items())[:]:
        print(f"{k}: {v}")
    print("Index to String Mapping:")
    for k, v in list(meta["itos"].items())[:]:
        print(f"{k}: {v}")


def merge_metas(meta_path1, meta_path2, output_path):
    meta1 = load_meta(meta_path1)
    meta2 = load_meta(meta_path2)

    # Start with the stoi and itos from the first meta file
    stoi = meta1["stoi"].copy()
    itos = meta1["itos"].copy()

    # Update with tokens from the second meta, resolving conflicts by prioritizing the first meta
    for token, id in meta2["stoi"].items():
        if token not in stoi:
            # If the token is not in stoi, add it
            new_id = (
                max(itos.keys()) + 1
            )  # Assign a new ID that is the current max ID + 1
            stoi[token] = new_id
            itos[new_id] = token
        # If the token is already in stoi, it's skipped, thereby prioritizing meta1's mapping

    vocab_size = len(stoi)
    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(output_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Merged meta saved to {output_path}, prioritizing {meta_path1}.")


def create_meta_from_text(text_file, output_path, special_chars={"<ukn>": 0}):
    with open(text_file, "r") as f:
        tokens = f.read().split("\n")

    stoi = {token: i for i, token in enumerate(tokens, start=len(special_chars))}
    stoi.update(special_chars)  # Add special characters with predefined IDs
    itos = {i: token for token, i in stoi.items()}
    vocab_size = len(stoi)

    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(output_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Meta created from text and saved to {output_path}.")


def main():
    parser = argparse.ArgumentParser(description="Utility for handling token metadata.")

    parser.add_argument("--view", type=str, help="Path to the meta.pkl file to view.")

    parser.add_argument(
        "--merge", nargs=2, help="Paths to the two meta.pkl files to merge."
    )

    parser.add_argument(
        "--create",
        nargs=2,
        help="Path to the input text file and the output meta.pkl file for creation.",
    )

    args = parser.parse_args()

    if args.view:
        view_tokens(args.view)
    elif args.merge:
        merge_metas(args.merge[0], args.merge[1], "merged_meta.pkl")
    elif args.create:
        create_meta_from_text(args.create[0], args.create[1])


if __name__ == "__main__":
    main()
