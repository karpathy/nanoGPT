from config.train_shakespeare_char import *
import subprocess

process_num = 3
iter_steps = 10 
variables = ("learning_rate", "beta1", "beta2", "weight_decay")
iter_scale = {"learning_rate": 10, "beta1": 0.9, "beta2": 0.9, "weight_decay": 0.9}

for v in variables:
    for i in range(iter_steps):
        exec(open('config/train_shakespeare_char.py').read())
        globals()[v] *= i * iter_scale[v]
        subprocess.run(["python",
                        "train.py",
                        "config/train_shakespeare_char.py",
                        "--{}={}".format(v,globals()[v])])
