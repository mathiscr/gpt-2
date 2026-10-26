import math
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    activation: str = "gelu" # gelu | relu2
    norm: str = "LayerNorm" # LayerNorm | RMSNorm
    use_rope: bool = False
    use_muon: bool = False


@dataclass
class TrainConfig:
    # data
    data_root: str = "edu_fineweb10B"
    num_workers_train: int = 2
    num_workers_val: int = 0

    # batch
    B: int = 16
    T: int = 1024  # seq len
    total_batch_size: int = 524288  # 2 ** 19

    # sampler
    num_return_sequences: int = 5
    max_length: int = 30
    sample_every: int = 10000
    prompt: str = "Hello, I'm "

    # optimization
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 1907
    weight_decay: float = 0.1
    eval_every: int = 100
    grad_clip:float = 1.0
    eval_steps: int = 20


    # checkpoint
    checkpoint_every: int = 2000
    checkpoint_dir: str = "checkpoint"

    project: str = "gpt2-from-scratch"
    name: str = "baseline-500M"
    group: str = "500Mtokens"
    use_wandb: bool = False
    seed: int = 42

    @property
    def min_lr(self) -> float:
        return self.min_lr_ratio * self.max_lr

    def grad_accum_steps(self, ddp_world_size: int) -> int:
        assert self.total_batch_size % (self.B * self.T * ddp_world_size) == 0
        return self.total_batch_size // (self.B * self.T * ddp_world_size)

    def get_lr(self, it):
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        if it > self.max_steps:
            return self.min_lr
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
