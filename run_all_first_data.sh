#!/bin/bash

# python train.py config/transcendence_gpt_30_train_30_test.py
# python eval_model.py config/transcendence_gpt_30_train_30_test.py > results/results_30_train_30_test.txt

python train.py config/transcendence_gpt_1_train_30_test.py
python eval_model.py config/transcendence_gpt_1_train_30_test.py > results/results_1_train_30_test.txt

# python train.py config/transcendence_gpt_30_train_1_test.py
# python eval_model.py config/transcendence_gpt_30_train_1_test.py > results/results_30_train_1_test.txt

python train.py config/transcendence_gpt_1_train_1_test.py
python eval_model.py config/transcendence_gpt_1_train_1_test.py > results/results_1_train_1_test.txt