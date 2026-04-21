


Sensitivity Analysis (On llama60m): 

In this experiment, I wish that I can do a parameter sweep on learning rate of adamw and muon,  use log uniform sampling. Taking range of adam [1e-5, 1e-2], and range of muon [1e-4, 1e-1]

And then do a parameter sweep on linesearch c1: range [1e-2, 1], use uniform. (scheduler use cosine annealing), then run both linesearch + adam and linesearch + muon

Save runing wall clock time, validation loss and training loss, and save checkpoint.

Save path = /scratch.global/chen8596/sensitivity_analysis

Please write a file that separates with the optuna logic, maybe reuse
train_llama.py and train_linesearch_llama.py and train_llama_muon.py and train_linesearch_llama_muon.py

In the end, create a file that collect the result and produce a plot.

