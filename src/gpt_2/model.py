from torch import nn
from torch.nn import functional as F
import torch
import inspect


def apply_rotation(x, cos, sin):
    _, _, T, hs = x.shape
    x1, x2 = x[..., : hs // 2], x[..., hs // 2 :]
    cos, sin = cos[:T], sin[:T]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    out = torch.cat([y1, y2], dim=-1)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                "The dimension of the embedding must be divisible by the number of heads"
            )
        # q k v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.config = config

        if config.use_rope:
            head_size = self.n_embd // self.n_head
            freqs = (
                1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
            ).unsqueeze(0)
            positions = torch.arange(0, config.block_size).float().unsqueeze(1)
            angles = positions @ freqs
            cos = angles.cos()
            sin = angles.sin()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape  # batch_size, sequence_length, embedding_dim

        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)
        # nh: number of head, hs: head size
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.config.use_rope:
            q, k = (
                apply_rotation(q, self.cos, self.sin),
                apply_rotation(k, self.cos, self.sin),
            )

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # C = nh * hs
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.config = config

    def forward(self, x):
        x = self.c_fc(x)
        if self.config.activation == "gelu":
            x = F.gelu(x)
        elif self.config.activation == "relu2":
            x = F.relu(x).square()
        else:
            raise ValueError(
                f"{self.config.activation} is not a supported activation function"
            )
        x = self.c_proj(x)
        return x


NORMS = {
    "LayerNorm": nn.LayerNorm,
    "RMSNorm": nn.RMSNorm,
}


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = NORMS[config.norm](config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = NORMS[config.norm](config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=NORMS[config.norm](config.n_embd),
            )
        )
        if config.use_rope == False:
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing between input and output
        self.transformer.wte.weight = self.lm_head.weight

        # parameters initialization
        self.apply(self._init_weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                "Length of the sequence (T) must be lesser than the block size"
            )
        # calculation an forward of the embeddings of position and tokens
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb
        if self.config.use_rope == False:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
            x = x + pos_emb
        # forward of the blocks
        for block in self.transformer.h:
            x = block(x)
        # final layer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, self.config.vocab_size), targets.view(B * T)
            )
        return logits, loss

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def configure_optimizer(self, weight_decay, lr, device, master_process):
        param_dict = {n: p for n, p in self.named_parameters()}
        param_dict = {n: p for n, p in param_dict.items() if p.requires_grad}
        muon_dict = (
            {
                n: p
                for n, p in param_dict.items()
                if p.dim() == 2 and "wte" not in n and "wpe" not in n
            }
            if self.config.use_muon
            else {}
        )
        decay_param = [
            p for n, p in param_dict.items() if p.dim() >= 2 and n not in muon_dict
        ]
        muon_param = muon_dict.values()
        nodecay_param = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_param = [
            {"params": decay_param, "weight_decay": weight_decay},
            {"params": nodecay_param, "weight_decay": 0.0},
        ]
        num_decay_param = sum(p.numel() for p in decay_param)
        num_nodecay_param = sum(p.numel() for p in nodecay_param)
        if master_process:
            print(f"num decay param: {len(decay_param)} with {num_decay_param}")
            print(f"num nodecay para: {len(nodecay_param)} with {num_nodecay_param}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        if master_process:
            print("using fused in AdamW", use_fused)
        optimizer_adamw = torch.optim.AdamW(
            optim_param, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        if self.config.use_muon:
            optimizer_muon = torch.optim.Muon(
                muon_param,
                lr=lr,
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw"
            )
            return [optimizer_adamw, optimizer_muon]
        return [optimizer_adamw]
