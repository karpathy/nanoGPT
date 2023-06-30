

Summary
=======


Trainning commands
------------------

(For training on your own input file) run this command first to analyze tokens within input.txt and generate corresponding token files (train.bin and val.bin). These will be by default under the custom folder. You need to 
move or copy these files to under custom_char folder.
```python data/custom/prepare.py input.txt```

Then, run this command to commence training cycles. Adjust parameters as needed.

```python train.py config/train_custom.py --device=cuda --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0```


Generating inference commands
-----------------------------

After training process finishes, run this command to obtain samples of generated text:

```python sample.py --out_dir=out-custom-char```



Results
-------


<insert table with the following format>


The tables below record run time for different parameter combinations
as well as samples of the generated text. The BLEU score (4-gram) reflects how similar
the generated text is compared with the reference input on a scale of 0 to 1.

| Iterations | Block Size | Time | Result | BLEU score |
| --- | --- | --- | --- |
| 2000 | 64 | 1:45 | [output 1](out/2kb64t4l5.txt) | 0.145 |
| 4000 | 64 | 3:24 | [output 2](out/4kb64t6m5l4.txt) | 0.174 |
| 10000 | 64 | 8:16 | [output 3](out/10kb64t1551l3.txt) | 0.244 |
| 40000 | 64 | 32:25 | [output 4](out/40kb64t32l26.txt) | 0.164 |

| Iterations | Block Size  | Time | Result | BLEU score
| --- | --- | --- | --- |
| 10000 | 64 | 8.16 | [output 5](out/10kb64t1551l3.txt) | 0.244 |
| 10000 | 128 | 14:20 | [output 6](out/10kb128t14l27.txt) | 0.202 |
| 10000 | 256 | 59:00 | [output 7](out/10kb256t59l17.txt) | 0.169 |









