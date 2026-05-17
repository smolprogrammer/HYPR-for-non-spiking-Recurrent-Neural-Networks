import subprocess
from itertools import product

DATA_DIR = "/home/datasets"
EXPERIMENT = "lstm_smnist_hypr.yaml"
RESULT_DIR = "/home/results"
REPO_NAME = "aim_repo"

search_space = {
    "training.learning_rate": [0.002, 0.001, 0.0005],
    "dataset.batch_size":     [64, 128, 256],
    "model.hidden_size":      [128, 256, 512],
    "model.num_layers":       [1, 2, 3],
    "training.epochs":        [10, 20, 30],
    "hypr_args.num_chunks":  [1, 2],
}

keys = list(search_space.keys())
values = list(search_space.values())

for combo in product(*values):
    params = dict(zip(keys, combo))

    cmd = ["python", "main.py", f"experiment={EXPERIMENT}", f"data_dir={DATA_DIR}", f"result_dir={RESULT_DIR}", f"repo_name={REPO_NAME}"]

    cmd += [f"{k}={v}" for k, v in params.items()]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd)