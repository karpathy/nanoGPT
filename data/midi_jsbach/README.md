# JS Bach Midi Dataset

Scripts for processing [czhuang's JSB-Chorales-dataset](https://github.com/czhuang/JSB-Chorales-dataset/tree/master),
this collection adopts the train, validation, and test division as outlined by
Boulanger-Lewandowski in 2012.

This dataset has 382 chorales by Johann Sebastian Bach, whose music in is the
public domain, at 1/16th note resolution.

Each chord is represented by four integers, signifying the note's position on a
piano keyboard; a value of '-1' indicates that no note is being played.


## References:

[Arxiv Link: Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription](https://arxiv.org/abs/1206.6392)

Boulanger-Lewandowski, N., Vincent, P., & Bengio, Y. (2012). Modeling Temporal
Dependencies in High-Dimensional Sequences: Application to Polyphonic Music
Generation and Transcription. Proceedings of the 29th International Conference
on Machine Learning (ICML-12), 1159–1166.

