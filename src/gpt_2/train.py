import argparse
from dataclasses import fields, replace

import torch

from config import GPTConfig, TrainConfig
from distributed import setup_ddp, destroy_ddp
from trainer import Trainer

def build_configs():
    parser = argparse.ArgumentParser()
    for cls in (GPTConfig, TrainConfig):
        for f in fields(cls):
            if f.type in (bool, "bool"):
                # donne --use_rope et --no-use_rope
                parser.add_argument(
                    f"--{f.name}", action=argparse.BooleanOptionalAction, default=None
                )
            elif f.type in (int, float, str, "int", "float", "str"):
                t = {"int": int, "float": float, "str": str}.get(f.type, f.type)
                parser.add_argument(f"--{f.name}", type=t, default=None)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    def override(cls):
        names = {f.name for f in fields(cls)}
        return replace(cls(), **{k: v for k, v in args.items() if k in names})

    return override(GPTConfig), override(TrainConfig)

def main():
    info = setup_ddp()
    gpt_config, config = build_configs()


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
