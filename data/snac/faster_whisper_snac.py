import argparse
import tempfile
import json
import os
from tqdm import tqdm
from pydub import AudioSegment
from snac_converter import SpeechTokenizer, preprocess_audio_to_24khz, load_mp3_as_tensor
from faster_whisper import WhisperModel

def save_audio_temp(audio_segment, format="mp3"):
    """Save the specific audio segment temporarily"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
    temp_file.close()

    # Export audio segment to this temp file
    audio_segment.export(temp_file.name, format=format)

    return temp_file.name

def save_to_json(codes, word, path):
    """Save snac_code and corresponding word to json file"""
    codes_list = [c.tolist() for c in codes]
    combined_list = [item for sublist in codes_list for item in sublist]
    with open(path, 'a') as f:
        object = {
            'snac_code': combined_list,
            'word': word,
        }
        json.dump(object, f, indent=4)
    print(f"Encoded audio saved to {path}")

def process_directory(model, snac_model, input_dir, output_dir):
    """Process all MP3 files in a directory and create individual txt files for each."""
    os.makedirs(output_dir, exist_ok=True)
    mp3_files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]
    for filename in tqdm(mp3_files, desc="Processing MP3 files"):
        input_path = os.path.join(input_dir, filename)
        output_base = os.path.splitext(filename)[0]
        json_output_path = os.path.join(output_dir, f"{output_base}.json")
        audio_processor(model, input_path, snac_model, json_output_path)

def audio_processor(model, path, snac_model, out_path):
    """Given the .mp3 file, output json file containing snac_code and corresponding word"""
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # args.input is the path to the mp3 file
    segments, info = model.transcribe(path, word_timestamps=True)
    audio = AudioSegment.from_mp3(path)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        for word in segment.words:
            print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
            # Using pydub to extract segments of audio wanted
            begin = word.start * 1000
            end = word.end * 1000
            # Temporarily solution when word.start == word.end
            print(word.start == word.end)
            if (word.start == word.end):
                end += 1
            audio_section = audio[begin:end]
            print(word.start == word.end)

            # Save the audio segment to a temp file
            temp_path = save_audio_temp(audio_section)

            # Do the snac encoder
            temp_wav_path = "temp.wav"
            preprocess_audio_to_24khz(temp_path, temp_wav_path)
            audio_snac = load_mp3_as_tensor(temp_wav_path).to(snac_model.device)
            codes = snac_model.encode(audio_snac)
            save_to_json(codes, word.word, out_path)
            os.remove(temp_path)

def main():
    parser = argparse.ArgumentParser(description="Encode and decode audio using SNAC")
    parser.add_argument('input', help="Input file path or directory (for encode)")
    parser.add_argument('output', help="Output file path or directory (for encode)")
    parser.add_argument('--directory', action='store_true', help="Process all MP3 files in the input directory")
    args = parser.parse_args()
    model_size = "large-v3"

    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    snac_model = SpeechTokenizer('cuda')

    if args.directory:
        process_directory(model, snac_model, args.input, args.output)
    else:
        audio_processor(model, args.input, snac_model, args.output + '.json')    

if __name__ == "__main__":
    main()
