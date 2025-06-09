import os

INPUT_FOLDER = "output"
OUTPUT_FILE = "data/calls/input.txt"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
    for filename in sorted(os.listdir(INPUT_FOLDER)):
        if filename.endswith(".txt"):
            filepath = os.path.join(INPUT_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as in_file:
                for line in in_file:
                    line = line.strip()
                    if line.startswith("CALLER:") or line.startswith("OPERATOR:"):
                        out_file.write(line + "\n")
                out_file.write("\n")  # Blank line between calls
print(f"✅ Transcripts merged into {OUTPUT_FILE}")
