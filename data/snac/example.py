import argparse
from snac_converter import SpeechTokenizer, preprocess_audio_to_24khz, load_mp3_as_tensor

parser = argparse.ArgumentParser(description="Encode and decode audio using SNAC")
parser.add_argument('input', help="Input file path or directory (for encode)")
args = parser.parse_args()

snac_model = SpeechTokenizer('cuda')

# Do the snac encoder
temp_wav_path = "temp.wav"
preprocess_audio_to_24khz(args.input, temp_wav_path)
audio_snac = load_mp3_as_tensor(temp_wav_path).to(snac_model.device)
print("check")
codes = snac_model.encode(audio_snac)
print("check2")
print(codes)
