# OpenWebText (sample)

This dataset prep script creates a **small sample** of OpenWebText suitable for *real (non-toy)* single-GPU experiments.

It avoids downloading and caching the full OpenWebText dataset (~54GB HF cache) used by `data/openwebtext/prepare.py`.

## Prepare

```bash
pip install datasets tiktoken tqdm
python data/openwebtext_sample/prepare.py --subset 'train[:1%]' --val_frac 0.01
```

This writes:

- `data/openwebtext_sample/train.bin`
- `data/openwebtext_sample/val.bin`

