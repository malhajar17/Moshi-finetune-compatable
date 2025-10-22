# Moshi Fine-tuning on FlexAI# Minimal Moshi-Finetune Setup



Fine-tune the Moshi 7B audio language model on FlexAI using LoRA and distributed training.This is a minimal setup for fine-tuning Moshi models, extracted from the original [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) repository.



## Quick Start## 🚀 Quick Start



### 1. Authenticate with HuggingFace### 1. Install Dependencies

```bash```bash

huggingface-cli logincd /workspace/fcs-experiments-private

```pip install -r code/moshi-finetune/requirements.txt

```

### 2. Download Dataset

```bash### 2. Prepare Configuration

python3 -c "from huggingface_hub import snapshot_download; \Edit `example/moshi_7B.yaml` to set your paths:

  snapshot_download(repo_id='kyutai/DailyTalkContiguous', \- `data.train_data`: Path to your training data (`.jsonl` file)

  repo_type='dataset', \- `run_dir`: Directory where checkpoints will be saved

  local_dir='./DailyTalkContiguous', \

  resume_download=True)"### 3. Run Training

``````bash

# Single GPU

### 3. Upload to FlexAItorchrun --nproc-per-node 1 -m train example/moshi_7B.yaml

```bash

flexai checkpoint push dailytalk-contiguous --source-path ./DailyTalkContiguous# Multiple GPUs (8)

```torchrun --nproc-per-node 8 --master_port $RANDOM -m train example/moshi_7B.yaml

```

### 4. Run Training

```bash## 📊 Example Configuration

flexai training run moshi-finetune \

  --accels 4 --nodes 1 \The `example/moshi_7B.yaml` contains optimized settings for training:

  --repository-url https://github.com/malhajar17/Moshi-finetune-compatable \- LoRA rank: 128

  --checkpoint dailytalk-contiguous \- Batch size: 16

  --env FORCE_TORCHRUN=1 \- Learning rate: 2e-6

  --env NCCL_NVLS_ENABLE=0 \- Duration: 100 seconds

  --env TORCH_COMPILE_DISABLE=1 \- Max steps: 2000

  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \

  --affinity "cluster=k8s-training-smc-001" \## 🔧 Data Format

  --env HF_HUB_CACHE=/output/.cache \

  --env HF_HUB_DISABLE_XET=1 \Your training data should be a `.jsonl` file where each line contains:

  --requirements-path requirements.txt \```json

  --runtime nvidia-25.03 \{"path": "relative/path/to/audio.wav", "duration": 24.5}

  -- python -m torch.distributed.run --nproc-per-node 4 --nnodes 1 train.py example/moshi_7B.yaml```

```

Audio files should be stereo:

### 5. Monitor- Left channel: Moshi-generated audio

```bash- Right channel: User input audio

flexai training logs moshi-finetune

```Each audio file should have a corresponding `.json` transcript file.



## Configuration## 📚 Full Documentation



Training config is in `example/moshi_7B.yaml`:For complete documentation, dataset preparation, and advanced configuration, see the original [moshi-finetune repository](https://github.com/kyutai-labs/moshi-finetune).

- Dataset path: `/input-checkpoint` (FlexAI checkpoint mount)

- Output path: `/output-checkpoint` (where checkpoints are saved)## 🎯 What's Different in This Minimal Setup

- LoRA rank: 128

- Batch size: 1- **Simplified dependencies**: Only core requirements without torch (uses pre-installed torch 2.8)

- Max steps: 2000- **Minimal moshi package**: Resolves dependency conflicts without pytest issues

- **Clean structure**: Only essential files for training

## Checkpoints- **Relative paths**: Works from the experiments directory

Checkpoints are saved to `/output-checkpoint` every 100 steps.

Retrieve with:
```bash
flexai training checkpoints moshi-finetune
flexai checkpoint pull <CHECKPOINT_ID> --destination ./trained-moshi
```

## Troubleshooting

**"No files ending with '.jsonl'"**: Dataset checkpoint is empty. Make sure `dailytalk.jsonl` exists and re-upload.

**"KeyError: CUDA_VISIBLE_DEVICES"**: Add `--env CUDA_VISIBLE_DEVICES=0,1,2,3`

**"CUDA failure"**: Add `--env NCCL_NVLS_ENABLE=0`

**NumPy warnings**: Non-fatal, training continues normally.
