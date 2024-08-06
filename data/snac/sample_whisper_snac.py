import argparse
import tempfile
import json
import sys
import os
from pydub import AudioSegment
from rich import print
from rich.progress import Progress, track
from snac_converter import (
    SpeechTokenizer,
    preprocess_audio_to_24khz,
    load_mp3_as_tensor,
)


def save_audio_temp(audio_segment, format="mp3"):
    """Save the specific audio segment temporarily"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
    audio_segment.export(temp_file.name, format=format)
    return temp_file.name


def append_to_json_file(file_path, data):
    """Append data to a JSON file incrementally"""
    if os.path.exists(file_path):
        with open(file_path, "r+") as file:
            existing_data = json.load(file)
            existing_data.append(data)
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_path, "w") as file:
            json.dump([data], file, indent=4)


def flatten_tensors(tensors):
    """Flatten the tensors using the specified pattern"""
    flattened = []
    separator_token = 4097
    i = 0

    while i < len(tensors[0][0]):
        if i < len(tensors[0][0]):
            flattened.append(tensors[0][0][i].item())
        if 2 * i + 1 < len(tensors[1][0]):
            flattened.extend(
                [tensors[1][0][2 * i].item(), tensors[1][0][2 * i + 1].item()]
            )
        if 4 * i + 3 < len(tensors[2][0]):
            flattened.extend(
                [
                    tensors[2][0][4 * i].item(),
                    tensors[2][0][4 * i + 1].item(),
                    tensors[2][0][4 * i + 2].item(),
                    tensors[2][0][4 * i + 3].item(),
                ]
            )
        flattened.append(separator_token)
        i += 1

    return flattened


parser = argparse.ArgumentParser(description="Encode and decode audio using SNAC")
parser.add_argument("input", help="Input file path or directory (for encode)")
parser.add_argument("whisper", help="Input file path or directory for whisper outputs")
parser.add_argument("output", help="Output file path for the new JSON")

args = parser.parse_args()

snac_model = SpeechTokenizer("cuda")

audio = AudioSegment.from_mp3(args.input)
audio_duration = len(audio)  # Duration of the audio in milliseconds

with open(args.whisper, "r") as file:
    data = json.load(file)

with Progress() as progress:
    task = progress.add_task(
        "[cyan]Processing transcription entries...", total=len(data["transcription"])
    )

    for entry in data["transcription"]:

        print( f"[green]From:[/green] {entry['timestamps']['from']} [green]to:[/green] {entry['timestamps']['to']}, [green]word:[/green] [yellow]{entry['text']}[/yellow]")

        begin = entry["offsets"]["from"]
        end = entry["offsets"]["to"]

        # Skip entries where 'from' and 'to' are equal
        if begin == end:
            print(
                f"[yellow]Skipping entry with equal 'from' and 'to' values:[/yellow] {entry}"
            )
            progress.update(task, advance=1)
            continue

        # Check if the end time exceeds the audio duration
        if end > audio_duration:
            print(
                f"[red]Entry with end time {end} exceeding audio duration {audio_duration}:[/red] {entry}"
            )
            print(f"[red]Exiting [/red]")
            sys.exit()

        audio_section = audio[begin:end]
        temp_path = save_audio_temp(audio_section)

        temp_wav_path = "temp.wav"
        preprocess_audio_to_24khz(temp_path, temp_wav_path)

        try:
            # Load and process the audio segment
            audio_snac = load_mp3_as_tensor(temp_wav_path)
            audio_snac = audio_snac.to(snac_model.device)
            codes = snac_model.encode(audio_snac)
            code_list = [c.tolist() for c in codes]

            # Flatten the tensors using the specified pattern
            sequential_snac_tokens = flatten_tensors(codes)

            # Print token length
            snac_token_len = len(sequential_snac_tokens)
            text_len = len(entry["text"])

            print(f"Snac token Length [bold]{snac_token_len}[/bold]")
            print(f"Text char Length [bold]{text_len}[/bold]")

            # Collect results
            result = {
                "snac_tokens": code_list,
                "sequential_snac_tokens": sequential_snac_tokens,
                "snac_token_len": snac_token_len,
                "text": entry["text"],
                "text_len": text_len,
                "from": entry["timestamps"]["from"],
                "to": entry["timestamps"]["to"],
            }

            # Append result to JSON file
            append_to_json_file(args.output, result)

        except Exception as e:
            print(
                f"[red]Error processing segment from {entry['timestamps']['from']} to {entry['timestamps']['to']}:[/red] {e}"
            )
        finally:
            # Ensure temporary files are deleted
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

        progress.update(task, advance=1)

print(f"[blue]Results saved to {args.output}[/blue]")
