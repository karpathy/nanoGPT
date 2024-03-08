from config.train_shakespeare_char import *
import subprocess
import itertools

process_num = 3
iter_steps = 5 
variables = {"learning_rate": globals()["learning_rate"],
             "beta1": globals()["beta1"],
             "beta2": globals()["beta2"],
             "weight_decay": globals()["weight_decay"]}

iter_scale = {"learning_rate": 10,
              "beta1": 0.01,
              "beta2": 0.01,
              "weight_decay": 0.1}

hyperparameter_space = {
    "learning_rate": [variables["learning_rate"] * (iter_scale["learning_rate"] ** f) for f in range(iter_steps)],
    "beta1": [variables["beta1"] - iter_scale["beta1"] * f for f in range(iter_steps)],
    "beta2": [variables["beta2"] - iter_scale["beta2"] * f for f in range(iter_steps)],
    "weight_decay": [variables["weight_decay"] * (iter_scale["weight_decay"] ** f) for f in range(iter_steps)],
}

combinations = list(itertools.product(*(hyperparameter_space[name] for name in hyperparameter_space)))

for combo in combinations:

    p = dict(zip(hyperparameter_space.keys(), combo))

    wandb_run_name = \
        "{}  | ".format(globals()["opt_type"]) + \
        "lr: {:.2e} | ".format(p["learning_rate"]) + \
        "weight-decay: {:.2e}  | ".format(p["weight_decay"]) + \
        "beta1: {} beta2: {}  | ".format(p["beta1"], p["beta2"])

    cmd = ["python",
           "train.py",
           "config/train_shakespeare_char.py",
           "--wandb_run_name={}".format(wandb_run_name)]

    for n, v in p.items():
        cmd.append("--{}={}".format(n,v))

    subprocess.run(cmd)
