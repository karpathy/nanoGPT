import csv
import argparse
import mido
from mido import MidiFile, MidiTrack, Message, bpm2tempo

def create_midi_from_csv(csv_file_path, midi_file_path, note_columns):
    # MIDI setup
    ticks_per_beat = 480  # Standard ticks per beat, adjust as needed
    tempo = bpm2tempo(120)  # 120 BPM; adjust as needed for your tempo
    sixteenth_note_duration = ticks_per_beat // 4  # Duration of a 16th note in ticks

    # Create a new MIDI file & track
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # Add tempo to the MIDI file
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    # Read the CSV file and create MIDI notes
    with open(csv_file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            start_time = 0  # Initialize start time for the first note in the chord
            for note_str in row[:note_columns]:  # Parse each note in the row up to note_columns
                if note_str == '-1':
                    continue  # Skip this note
                note = int(note_str)
                # Add note on message
                track.append(Message('note_on', note=note, velocity=64, time=start_time))
                start_time = 0  # Subsequent notes in the chord have no delay

            # Insert a note off message for all notes except those with -1
            for note_str in row[:note_columns]:
                if note_str == '-1':
                    continue  # Skip this note
                note = int(note_str)
                track.append(Message('note_off', note=note, velocity=64, time=sixteenth_note_duration))

    # Save the MIDI file
    mid.save(midi_file_path)
    print(f'MIDI file saved as {midi_file_path}')

if __name__ == '__main__':
    # Setup argparse
    parser = argparse.ArgumentParser(description='Convert CSV file of notes to MIDI.')
    parser.add_argument('-i', '--csv_file_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('-c', '--note_columns', type=int, default=4, help='Number of note columns in the CSV file')
    parser.add_argument('-o', '--output',  type=str, default='output.mid', help='Output MIDI file path')

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    create_midi_from_csv(args.csv_file_path, args.output, args.note_columns)

