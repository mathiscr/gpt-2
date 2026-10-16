# GPT-2 (124M) from scratch in PyTorch: multi-GPU training on 2× RTX 3090

A from-scratch reproduction of GPT-2 small (124M parameters), trained on **10B tokens** on two consumer GPUs with DistributedDataParallel.

> **Transparency note.** This project follows Andrej Karpathy's ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) tutorial. The model and training recipe are his; what I did on top of it was to re-engineer the code and make it run efficiently on my own hardware, learning a lot about efficient PyTorch along the way (see below).

## Results

| | |
|---|---|
| Model | GPT-2 small, 124M parameters (12 layers, 12 heads, d=768, context 1024) |
| Training data | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), 10B-token sample (`sample-10BT`), GPT-2 BPE tokenizer |
| Hardware | 2× NVIDIA RTX 3090 (DDP) |
| Throughput | ~125k tokens/s |
| Training time | ~20 hours |
| Final validation loss | 3.11 |


![Validation loss](assets/val_loss.png)
![Training loss](assets/train_loss.png)

Validation loss reaches 3.11 after 10B tokens. Training and validation losses stay close throughout, so there is no sign of overfitting (the run is about one pass over the data). The curve flattens towards the end as the cosine schedule brings the learning rate to its minimum.

## What I did

- **Efficient PyTorch training loop**: `torch.compile`, bf16 autocast, TF32 matmuls, fused AdamW, FlashAttention through `F.scaled_dot_product_attention`, gradient accumulation to reach a large effective batch size, gradient clipping.
- **Multi-GPU with DDP**: gradient synchronization only on the last micro-step of each accumulation window, metrics averaged across ranks, all logging and checkpointing restricted to the master process.
- **Rewritten data pipeline**: the tutorial's hand-rolled shard loader is replaced by a standard `Dataset` + `DataLoader` + `DistributedSampler`. Token shards are memory-mapped `.npy` files, a window index is computed with a cumulative sum and `searchsorted`, and worker processes load batches in parallel.
- **Reorganized codebase**: model, data, distributed setup, training loop and configuration (dataclasses) live in separate modules.
- **Experiment tracking**: loss, learning rate, gradient norm and throughput logged to Weights & Biases, text samples traced with Weave, atomic checkpoints (write to a temporary file, then rename).

## Training setup

- Optimizer: AdamW (β = 0.9, 0.95), weight decay applied only to matrices (2D parameters)
- Schedule: linear warmup, then cosine decay to a minimum learning rate (peak 6e-4)
- Batch: about 0.5M tokens per step (micro-batch × accumulation steps × GPUs)
- Weight tying between token embedding and output head, scaled initialization of residual projections

## Repository structure

```
src/gpt-2/
├── train.py         # main script to run
├── config.py        # GPTConfig and TrainConfig dataclasses
├── model.py         # GPT model (attention, MLP, blocks, optimizer setup)
├── trainer.py       # training / evaluation / sampling / checkpointing
├── data.py          # Dataset, DataLoader and DistributedSampler
├── distributed.py   # DDP setup and teardown
└── fineweb.py       # downloads FineWeb-Edu and writes the token shards
```

## Usage

The project uses uv to handle the local environment.
```bash
uv sync
```

The project uses `wandb` and `weave` for experiment tracking. On the first run, W&B asks whether you want to log in or skip visualization.

### Data

The training data is the 10B-token sample of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), tokenized with the GPT-2 BPE tokenizer:

```bash
cd src/gpt-2
uv run python fineweb.py
```

The script streams the dataset, tokenizes it in parallel and writes shards of 100M tokens as `.npy` files (`edufineweb_val_000000.npy`, `edufineweb_train_000001.npy`, ...) in `edu_fineweb10B/`. The first shard is the validation set. `max_shards` caps the number of shards (21, about 2B tokens, in the current script): set it to `None` to tokenize the full 10B sample.

### Training

```bash
# for 2 GPUs
uv run torchrun --standalone --nproc_per_node=2 train.py
```

Hyperparameters live in `config.py`.

## Experiments in progress

The model supports config flags for several modern changes: RoPE instead of learned position embeddings, ReLU² activation and RMSNorm.
These can be adds by adding the flags --flag
Exemple:
```bash
uv run torchrun --standalone --nproc_per_node=2 train.py --name other_name --activation relu2
```

## Acknowledgments

Andrej Karpathy for the tutorial and the [nanoGPT](https://github.com/karpathy/nanoGPT) / [build-nanogpt](https://github.com/karpathy/build-nanogpt) repositories, whose recipe this project is based on.
