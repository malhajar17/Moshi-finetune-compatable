# Moshi Fine-tuning on FlexAI

Fine-tune the Moshi 7B audio language model on FlexAI using LoRA and distributed training.

## 🚀 Quick Start

### Step 1: Authenticate with HuggingFace

```bash
huggingface-cli login
```

### Step 2: Download Dataset from HuggingFace

Download the DailyTalkContiguous dataset to your local machine:

```bash
python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download(repo_id='kyutai/DailyTalkContiguous', \
  repo_type='dataset', \
  local_dir='./DailyTalkContiguous', \
  resume_download=True)"
```

This will create a `DailyTalkContiguous/` directory with:
- `dailytalk.jsonl` - Metadata file with audio paths
- `data_stereo/` - Stereo audio files (left: Moshi, right: user)

### Step 3: Push Dataset to FlexAI Storage

Upload the dataset to FlexAI as a checkpoint:

```bash
flexai checkpoint push dailytalk-contiguous \
  --source-path ./DailyTalkContiguous
```

This makes the dataset available at `/input-checkpoint` during training.

### Step 4: Run Training on FlexAI

Launch distributed training on 1x H100 GPUs:

```bash
flexai training run moshi-finetune \
  --accels 1 --nodes 1 \
  --repository-url https://github.com/malhajar17/Moshi-finetune-compatable \
  --checkpoint dailytalk-contiguous \
  --env FORCE_TORCHRUN=1 \
  --env NCCL_NVLS_ENABLE=0 \
  --env TORCH_COMPILE_DISABLE=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --affinity "cluster=k8s-training-smc-001" \
  --env HF_HUB_CACHE=/output/.cache \
  --env HF_HUB_DISABLE_XET=1 \
  --requirements-path requirements.txt \
  --runtime nvidia-25.03 \
  -- python -m torch.distributed.run \
     --nproc-per-node 1 \
     --nnodes 1 \
     train.py \
     example/moshi_7B.yaml
```

### Step 5: Monitor Training

Watch the training logs in real-time:

```bash
flexai training logs moshi-finetune
```



## ⚙️ Configuration

Training configuration in `example/moshi_7B.yaml`:

| Setting | Value | Description |
|---------|-------|-------------|
| `train_data` | `/input-checkpoint` | Dataset path (FlexAI mount) |
| `run_dir` | `/output-checkpoint` | Output directory for checkpoints |
| `lora.rank` | 128 | LoRA adapter rank |
| `batch_size` | 1 | Batch size per GPU |
| `max_steps` | 2000 | Total training steps |
| `optim.lr` | 2e-6 | Learning rate |
| `ckpt_freq` | 100 | Checkpoint save frequency |

## 💾 Retrieving Checkpoints

After training, download your checkpoints:

```bash
# List available checkpoints
flexai training checkpoints moshi-finetune

# Download specific checkpoint
flexai checkpoint pull <CHECKPOINT_ID> --destination ./trained-moshi
```

## Troubleshooting

**"No files ending with '.jsonl'"**: Dataset checkpoint is empty. Make sure `dailytalk.jsonl` exists and re-upload.

**"KeyError: CUDA_VISIBLE_DEVICES"**: Add `--env CUDA_VISIBLE_DEVICES=0,1,2,3`

**"CUDA failure"**: Add `--env NCCL_NVLS_ENABLE=0`

**NumPy warnings**: Non-fatal, training continues normally.
