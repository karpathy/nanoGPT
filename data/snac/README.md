SNAC
====

This is an open source audio tokenization method.

Essentially it turns each "frame" of audio into multiple values per tensor
(resenbling a little bit the wavelet transform window).

## How SNAC works

For example, for 24KHz, one frame could be:

```
4000
300 400
133 3000 1234 1337
```

Since each frame has a fixed length, this can ordered as a sequence.

In this directory, we simply do the smallest row first then append the next two:

```
4000 300 400 133 3000 1234 1337
```

To mark new frames, we simply add a separation token between sequential outputs:

```
4000 300 400 133 3000 1234 1337 `4097` 100 2000 3000 3999 4000 4001 4002 `4097`
```

Each of these numbers counts as a single token, and spaces are skipped for
maximum context length capability.

## Usage


**Step 1.**

The following will download an open source audio dataset (tiny-sherlock-audio)
and process it into tokens.

```
bash get_dataset.sh
```

**Step 2.**

Train the model on the sherlock audio (simply modify the tokenization stage for
different datasets):
```
bash train.sh
```

**Step 3.**

Create an mp3 output to test out:
```
bash sample.sh
```

Then listen to the mp3 created should be called `output.mp3`.

## Discussion

Above steps replicated earlier
[findings](https://www.youtube.com/watch?v=sbz3w9nFV0E) of voice timbre
emulation.

Resulting mp3 from the above steps shows that simple and smaller models -- in
this case the base 6 layer 384 embeddings 6 heads nanoGPT -- can achieve voice
timbre emulation.

## Acknowledgements and Shoutouts

Many thanks to Srinivas Billa for sharing this approach, and shoutout to AbdulMajed for featuring and discussing the Audio-NanoGPT approach on the 1littlecoder podcast:
    * https://www.youtube.com/watch?v=sbz3w9nFV0E
