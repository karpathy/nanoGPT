# Plots (MEA benchmarks)

This folder contains:

- `plot_mea_results.py`: generates the figures used in `../MEA.md`
- `data/*.json`: benchmark outputs captured on an A100-80GB
- `mea_attention_scaling.png`, `mea_train_smoke_scaling.png`: rendered figures

Re-generate the images:

```bash
python -m venv .venv
.venv/bin/pip install -U pip matplotlib seaborn pandas
.venv/bin/python plot_mea_results.py
```

