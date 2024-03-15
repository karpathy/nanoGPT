import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate Zhuyin sound combinations with or without tones.')
parser.add_argument('--tones', action='store_true', help='Include tones in the combinations')
args = parser.parse_args()

# Define the lists of Zhuyin symbols for initials, medials, finals, and tones
initials = ["ㄅ", "ㄆ", "ㄇ", "ㄈ", "ㄉ", "ㄊ", "ㄋ", "ㄌ", "ㄍ", "ㄎ", "ㄏ",
            "ㄐ", "ㄑ", "ㄒ", "ㄓ", "ㄔ", "ㄕ", "ㄖ", "ㄗ", "ㄘ", "ㄙ", ""]
medials = ["", "ㄧ", "ㄨ", "ㄩ"]
finals = ["ㄚ", "ㄛ", "ㄜ", "ㄝ", "ㄞ", "ㄟ", "ㄠ", "ㄡ", "ㄢ", "ㄣ", "ㄤ", "ㄥ", "ㄦ", "",
          "ㄧㄚ", "ㄧㄝ", "ㄧㄠ", "ㄧㄡ", "ㄧㄢ", "ㄧㄣ", "ㄧㄤ", "ㄧㄥ",
          "ㄨㄚ", "ㄨㄛ", "ㄨㄞ", "ㄨㄟ", "ㄨㄢ", "ㄨㄣ", "ㄨㄤ", "ㄨㄥ",
          "ㄩㄝ", "ㄩㄢ", "ㄩㄣ", "ㄩㄥ"]
tones = ["", "ˊ", "ˇ", "ˋ", "˙"]

# Create combinations of sounds
combinations = []
if args.tones:
    for tone in tones:
        for initial in initials:
            for medial in medials:
                for final in finals:
                    # Ensure valid combinations based on Mandarin phonology rules
                    if (medial or final) and (initial or medial):  # Must have either a medial or a final, and either an initial or a medial
                        combination = initial + medial + final + tone
                        combinations.append(combination)
else:
    for initial in initials:
        for medial in medials:
            for final in finals:
                # Ensure valid combinations based on Mandarin phonology rules
                if (medial or final) and (initial or medial):  # Must have either a medial or a final, and either an initial or a medial
                    combination = initial + medial + final
                    combinations.append(combination)

# Write combinations to a text file
file_path = f"all_zhuyin_sound_combinations_tones-{args.tones}.txt"
with open(file_path, "w", encoding="utf-8") as file:
    for combination in combinations:
        file.write(f"{combination}\n")

