# JS Bach Midi Dataset

Scripts for processing [czhuang's JSB-Chorales-dataset](https://github.com/czhuang/JSB-Chorales-dataset/tree/master),
this collection adopts the train, validation, and test division as outlined by
Boulanger-Lewandowski in 2012.

This dataset has 382 chorales by Johann Sebastian Bach, whose music in is the
public domain, at 1/16th note resolution.

Each chord is represented by four integers, signifying the note's position on a
piano keyboard; a value of '-1' indicates that no note is being played.


## Process

### Step 1: Obtain Dataset and Tokenize as char

First, download the dataset and and create `.bin` files for training:

```sh
bash get_dataset.sh
prepare.py midi_12.csv --method char
```

### Step 2: Training Transformer Network on midi_jsbach

Use the run_experiments.py script to train the model on the prepared dataset.

```sh
python3 run_experiments.py --config explorations/bach.json --out out
```

### Step 3: Run sample.py and Save Output to a File

Generate samples from the trained model and redirect the output to a CSV file.

```sh
python3 sample.py --out_dir out --num_samples 1 --temperature 0.8 --sample_file ./data/midi_jsbach/generated_music.csv
```

### Step 4: Open File with Vim or Editor to trim file to just CSV Contents

Edit the generated CSV file to clean up any non-CSV content.

```sh
vim generated_music.csv # or preferred editor
cp generated_music.csv ./data/midi_jsbach
```

### Step 5: Utilize the sample_result_to_midi.sh Script to Process to MIDI and Play Results

Convert the cleaned CSV file into MIDI format and play the file.

```sh
bash sample_result_to_midi.sh generated_music.csv
```

### (Optional) Step 6: Change sample.py Parameters and Generate More MIDI Files

Repeat step 3 with different settings for and explore different results:

```sh
python3 sample.py --out_dir out --num_samples 1 --temperature 1.2 --sample_file ./data/midi_jsbach/generated_music.csv
```

## References:

[Arxiv Link: Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription](https://arxiv.org/abs/1206.6392)

Boulanger-Lewandowski, N., Vincent, P., & Bengio, Y. (2012). Modeling Temporal
Dependencies in High-Dimensional Sequences: Application to Polyphonic Music
Generation and Transcription. Proceedings of the 29th International Conference
on Machine Learning (ICML-12), 1159–1166.

