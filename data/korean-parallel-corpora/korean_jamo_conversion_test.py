from jamo import h2j, j2hcj, j2h, is_jamo

def korean_to_phonetic(text):
    """Converts Korean text to its phonetic representation."""
    # Convert Hangul to individual jamos
    decomposed_text = h2j(text)
    # Convert jamos back to Hangul compatibility jamos (for readability)
    phonetic_text = j2hcj(decomposed_text)
    return phonetic_text

# Example usage
korean_text = "안 녕 하 세 요"
phonetic_text = korean_to_phonetic(korean_text)
print("Original:", korean_text)
print("Phonetic:", phonetic_text.split(" "))

# Test string
phonetic_text="ㅇㅏㄴ ㄴㅕㅇ ㅎㅏ ㅅㅔ ㅇㅛ"

reconstructed_list = []
for pho in phonetic_text.split(" "):
    if len(pho) == 0:
        # if '' then skip
        continue
    elif is_jamo(pho[0]):
        # if is jamo then add after conversion
        # print(reconstructed_list)
        # print(pho)
        reconstructed_list.append(j2h(*pho))
    else:
        # if special space character reconstruct back to spaces
        reconstructed_list.append(pho.replace('▁', ' '))

# Reconstructed phrase
print(''.join(reconstructed_list))

