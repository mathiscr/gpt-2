import torch

from config import GPTConfig, TrainConfig
from distributed import setup_ddp, destroy_ddp
from trainer import Trainer


def main():
    info = setup_ddp()
    config = TrainConfig()
    gpt_config = GPTConfig()

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")

    try:
        Trainer(gpt_config, config, info).fit()
    finally:
        destroy_ddp(info)


if __name__ == "__main__":
    main()
