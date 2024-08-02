# NanoGPT Training Experiment

This README provides an overview of the experimental setup, modifications, execution, and results of training a GPT-2 model using the NanoGPT codebase on an AWS instance.

## Experiment Setup

### AWS Instance

- **AMI**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.0 (Ubuntu 20.04) 20240730
- **Instance Type**: g5.8xlarge (single GPU)
- **Storage**: 500 GB

### Environment

- **Operating System**: Ubuntu 20.04
- **Framework**: PyTorch 2.3.0

## Modifications to NanoGPT Code

### Token Reduction

Reduced the amount of tokens to approximately 4 billion. The changes can be validated in the following scripts:

- `data/openwebtext/prepare.py`
- `data/openwebtext/count_tokens.py`

### Training Configuration

Updated the training configuration to adapt to the new training size:

- **File**: `config/train_gpt2.py`
- **Changes**:
  ```python
  max_iters = 6000
  lr_decay_iters = 6000
  ```

## Execution

The training was executed using the following command:

```sh
torchrun --standalone --nproc_per_node=1 train.py config/train_gpt2.py 2>&1 | tee -a training.log
```

## Results

The training results are summarized as follows:

```
wandb: Run summary:
wandb:       iter 6000
wandb:         lr 6e-05
wandb:        mfu 13.70159
wandb: train/loss 3.2568
wandb:   val/loss 3.26934
```

### Key Metrics

- **Iterations**: 6000
- **Learning Rate**: 6e-05
- **Model Flop Utilization (MFU)**: 13.70159%
- **Training Loss**: 3.2568
- **Validation Loss**: 3.26934

## Conclusion

This experiment involved training a GPT-2 model using the NanoGPT codebase on an AWS g5.8xlarge instance. The modifications included reducing the token count to approximately 4 billion and updating the training configuration. The training was successfully executed, and the results indicate a reasonable training and validation loss, with a final learning rate of 6e-05 and MFU of 13.70159%.

For further details, refer to the training log (`training.log`) and the modified scripts in the `data/openwebtext` and `config` directories.
