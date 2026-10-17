import os
import time
from dataclasses import asdict
from itertools import islice

import tiktoken
import torch
import wandb
import weave
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from data import build_loader, infinite_loader
from model import GPT


class Trainer:
    def __init__(self, gpt_config, config, info):
        self.gpt_config = gpt_config
        self.config = config
        self.info = info

        self.grad_accum_steps = config.grad_accum_steps(info.ddp_world_size)

        if info.master_process:
            print(f"total desired batch size: {config.total_batch_size}")
            print(f"calculated accum grad steps: {self.grad_accum_steps}")

        self.train_loader = infinite_loader(
            build_loader(
                config.B,
                gpt_config.block_size,
                "train",
                info.ddp_world_size,
                info.ddp_rank,
                num_workers=config.num_workers_train,
                data_root=config.data_root,
            )
        )
        self.val_loader = build_loader(
            config.B,
            gpt_config.block_size,
            "val",
            info.ddp_world_size,
            info.ddp_rank,
            num_workers=config.num_workers_val,
            data_root=config.data_root,
        )

        self.model = GPT(gpt_config)
        self.model.to(info.device)
        self.base_model = self.model
        self.model = torch.compile(self.model)
        if info.ddp:
            self.model = DDP(self.model, device_ids=[info.ddp_local_rank])

        self.enc = tiktoken.get_encoding("gpt2")
        self.optimizer = self.base_model.configure_optimizer(
            weight_decay=config.weight_decay,
            lr=config.max_lr,
            device=info.device,
            master_process=info.master_process,
        )

        if info.master_process:
            wandb.init(
                project=config.project,
                name=config.name,
                group=config.group,
                config={
                    **asdict(gpt_config),
                    **asdict(config),
                    "min_lr": config.min_lr,
                    "grad_accum_steps": self.grad_accum_steps,
                    "world_size": info.ddp_world_size,
                },
            )
            weave.init(config.project)

            @weave.op()
            def generate_samples(
                step: int, prompt: str, num_return_sequences: int, max_length: int
            ) -> list[str]:
                tokens = torch.tensor(self.enc.encode(prompt), dtype=torch.long)
                x = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(info.device)
                sample_rng = torch.Generator(device=info.device)
                sample_rng.manual_seed(self.config.seed)
                while x.size(1) < max_length:
                    with torch.no_grad():
                        with torch.autocast(
                            device_type=info.device_type,
                            dtype=torch.bfloat16,
                        ):
                            logits, _ = self.base_model(x)
                        probs = F.softmax(logits[:, -1, :], dim=1)
                        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                        x = torch.cat((x, torch.gather(topk_indices, -1, ix)), dim=1)
                return [
                    self.enc.decode(x[i, :max_length].tolist())
                    for i in range(num_return_sequences)
                ]

            self.generate_samples = generate_samples

    @torch.no_grad()
    def evaluate(self, step):
        self.model.eval()
        val_loss_acum = 0
        val_loss_steps = self.config.eval_steps
        for x, y in islice(self.val_loader, val_loss_steps):
            x, y = x.to(self.info.device), y.to(self.info.device)
            with torch.autocast(
                device_type=self.info.device_type,
                dtype=torch.bfloat16,
            ):
                logits, loss = self.model(x, y)
            loss = loss / val_loss_steps
            val_loss_acum += loss.detach()
        if self.info.ddp:
            dist.all_reduce(val_loss_acum, op=dist.ReduceOp.AVG)
        if self.info.master_process:
            print(f"validation loss: {val_loss_acum.item():.4f}")
            wandb.log({"val/loss": val_loss_acum.item()}, step=step)

    def sample(self, step):
        samples = self.generate_samples(
            step,
            self.config.prompt,
            self.config.num_return_sequences,
            self.config.max_length,
        )
        for s in samples:
            print(">", s)

    def train_step(self, step):
        t0 = time.time()
        self.model.train()
        for opt in self.optimizer:
            opt.zero_grad(set_to_none=True)
        loss_accum = 0
        for micro_step in range(self.grad_accum_steps):
            x, y = next(self.train_loader)
            x, y = x.to(self.info.device), y.to(self.info.device)
            with torch.autocast(
                device_type=self.info.device_type,
                dtype=torch.bfloat16,
            ):
                logits, loss = self.model(x, y)
            loss = loss / self.grad_accum_steps
            loss_accum += loss.detach()
            if self.info.ddp:
                self.model.require_backward_grad_sync = (
                    micro_step == self.grad_accum_steps - 1
                )
            loss.backward()
        if self.info.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        lr = self.config.get_lr(step)
        for opt in self.optimizer:
            for param_group in opt.param_groups:
                param_group["lr"] = lr
            opt.step()
        if self.info.device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_processed = (
            self.config.B
            * self.gpt_config.block_size
            * self.grad_accum_steps
            * self.info.ddp_world_size
        )
        tokens_per_sec = tokens_processed / (t1 - t0)
        if self.info.master_process:
            print(
                f"step {step:4d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm {norm:.4f}  | dt {dt:.2f} ms, tok/s {tokens_per_sec:.2f}"
            )
            wandb.log(
                {
                    "train/loss": loss_accum.item(),
                    "train/lr": lr,
                    "train/grad_norm": norm.item(),
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/dt_ms": dt,
                },
                step=step,
            )

    def fit(self):
        for step in range(self.config.max_steps):
            last_step = step == self.config.max_steps - 1

            # evaluate val loss
            if step % self.config.eval_every == 0 or last_step:
                self.evaluate(step)

            if (
                self.info.master_process
                and step > 0
                and step % self.config.sample_every == 0
            ):
                self.sample(step)

            self.train_step(step)

            if step > 0 and step % self.config.checkpoint_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(self.config.max_steps - 1)
        if self.info.master_process:
            torch.save(
                self.base_model.state_dict(),
                f"gpt2_model_weights_{self.config.name}.pt",
            )
            print("weights saved")
            wandb.finish()

    def save_checkpoint(self, step):
        if not self.info.master_process:
            return
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.config.checkpoint_dir, f"checkpoint_last_{self.config.name}.pt"
        )
        tmp = path + ".tmp"
        torch.save(
            {
                "model": self.base_model.state_dict(),
                "optimizer": [opt.state_dict() for opt in self.optimizer],
                "step": step,
                "gpt_config": asdict(self.gpt_config),
            },
            tmp,
        )
        os.replace(tmp, path)
        print(f"checkpoint saved: {path}")
