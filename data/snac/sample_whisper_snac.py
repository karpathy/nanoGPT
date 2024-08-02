import argparse
import tempfile
import json
import os
from pydub import AudioSegment
from snac_converter import SpeechTokenizer, preprocess_audio_to_24khz, load_mp3_as_tensor

def save_audio_temp(audio_segment, format="mp3"):
    """Save the specific audio segment temporarily"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
    temp_file.close()

    # Export audio segment to this temp file
    audio_segment.export(temp_file.name, format=format)

    return temp_file.name

parser = argparse.ArgumentParser(description="Encode and decode audio using SNAC")
parser.add_argument('input', help="Input file path or directory (for encode)")
parser.add_argument('whisper', help="Input file path or directory for whisper outputs")

args = parser.parse_args()

snac_model = SpeechTokenizer('cuda')

audio = AudioSegment.from_mp3(args.input)

with open(args.whisper, 'r') as file:
    data = json.load(file)

for entry in data['transcription']:
    print(f"From: {entry['timestamps']['from']} to: {entry['timestamps']['to']}, word: {entry['text']}")
    # Using pydub to extract segments of audio wanted
    begin = entry['offsets']['from']
    end = entry['offsets']['to']
    # Temporarily solution when word.start == word.end
    # if (word.start == word.end):
    #     end += 1
    audio_section = audio[begin:end]
    # Save the audio segment to a temp file
    temp_path = save_audio_temp(audio_section)

    # Do the snac encoder
    temp_wav_path = "temp.wav"
    preprocess_audio_to_24khz(temp_path, temp_wav_path)
    audio_snac = load_mp3_as_tensor(temp_wav_path)
    print(f"loaded tensor:{audio_snac}")
    audio_snac = audio_snac.to(snac_model.device)
    print(f"Tensor moved to device: {audio_snac}")
    # audio_snac = load_mp3_as_tensor(temp_wav_path).to(snac_model.device)
    print(os.path.exists(temp_wav_path))
    print(temp_wav_path)
    codes = snac_model.encode(audio_snac)
    code_list = [c.tolist() for c in codes]
    print(codes)
