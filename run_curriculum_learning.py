import subprocess

prev_csv_dir = ""
prev_output_dir = ""

def run_experiments_command(config, csv_dir, output_dir):
    global prev_csv_dir
    global prev_output_dir
    command = ["python3", "run_experiments.py"]
    command.extend(["--config", f"explorations/{config}"])
    command.append("--value_only")
    command.extend(["--csv_ckpt_dir", f"csv_logs/{csv_dir}"])
    command.extend(["--output_dir", output_dir])
    if prev_csv_dir and prev_output_dir:
        command.extend(["--use-best-val-loss-from", f"csv_logs/{prev_csv_dir}", prev_output_dir])
    prev_csv_dir = csv_dir
    prev_output_dir = output_dir
    return command

def main():
    # Adjustable csv, ouput, and config names
    csv_dirs = ["openwebtext_csv", "wikitext103_csv", "xsum_csv", "billsum_csv", "cnn_dailymail_csv"]
    output_dirs = ["out_openwebtext", "out_wikitext103", "out_xsum", "out_billsum", "out_cnn_dailymail"]
    configs = ["openwebtext.json", "wikitext103.json", "xsum.json", "billsum.json", "cnn_dailymail.json"]
    for csv, output, config in zip(csv_dirs, output_dirs, configs):
        subprocess.run(run_experiments_command(config, csv, output))

if __name__ == "__main__":
    main()
