
# Define the Hangul Unicode block range
HANGUL_START = 0xAC00
HANGUL_END = 0xD7A3

# Number of initial consonants, medial vowels, and final consonants
NUM_CHOSEONG = 19
NUM_JUNGSEONG = 21
NUM_JONGSEONG = 28

# Lists of the initial consonants, medial vowels, and final consonants
CHOSEONG_LIST = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]
JUNGSEONG_LIST = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]
JONGSEONG_LIST = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

def decompose_hangul(syllable):
    if not HANGUL_START <= ord(syllable) <= HANGUL_END:
        return "Input is not a valid Hangul syllable."

    syllable_index = ord(syllable) - HANGUL_START
    choseong_index = syllable_index // (NUM_JUNGSEONG * NUM_JONGSEONG)
    jungseong_index = (syllable_index % (NUM_JUNGSEONG * NUM_JONGSEONG)) // NUM_JONGSEONG
    jongseong_index = syllable_index % NUM_JONGSEONG

    # Output the indices for easier parsing
    print(f"{choseong_index},{jungseong_index},{jongseong_index}")

def decompose_hangul_hr(syllable):
    # Human readable output
    # Verify the input is a Hangul syllable
    if not HANGUL_START <= ord(syllable) <= HANGUL_END:
        return "Input is not a valid Hangul syllable."

    # Calculate the components of the syllable
    syllable_index = ord(syllable) - HANGUL_START
    choseong_index = syllable_index // (NUM_JUNGSEONG * NUM_JONGSEONG)
    jungseong_index = (syllable_index % (NUM_JUNGSEONG * NUM_JONGSEONG)) // NUM_JONGSEONG
    jongseong_index = syllable_index % NUM_JONGSEONG

    # Retrieve the actual characters
    choseong = CHOSEONG_LIST[choseong_index]
    jungseong = JUNGSEONG_LIST[jungseong_index]
    jongseong = JONGSEONG_LIST[jongseong_index]

    # Print the components and their types
    print(f"Initial Consonant (Choseong): {choseong} - Position: Initial - Type: Consonant")
    print(f"Medial Vowel (Jungseong): {jungseong} - Position: Medial - Type: Vowel")
    if jongseong:
        print(f"Final Consonant (Jongseong): {jongseong} - Position: Final - Type: Consonant")
    else:
        print("No Final Consonant (Jongseong)")

# Example usage
syllable = '한'
print(f"Decomposing the syllable: {syllable}")
decompose_hangul(syllable)

def reconstruct_hangul(choseong_index, jungseong_index, jongseong_index):
    # Calculate the Unicode code point for the syllable
    syllable_code = HANGUL_START + (choseong_index * NUM_JUNGSEONG * NUM_JONGSEONG) + (jungseong_index * NUM_JONGSEONG) + jongseong_index
    # Convert the code point to the actual Hangul syllable
    return chr(syllable_code)

# Example input, replace with the output from the first script
choseong_index = 0
jungseong_index = 0
jongseong_index = 4

# Reconstruct and print the original Hangul syllable
original_syllable = reconstruct_hangul(choseong_index, jungseong_index, jongseong_index)
print(f"Reconstructed Hangul syllable: {original_syllable}")

