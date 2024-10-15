from .sqrll import sqrll_kernel
from dataclasses import dataclass
import torch


def rms_norm(x, weight, eps, dim=-1):
    x = x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + eps)
    if dim == -1:
        x = x * weight
    elif dim == -3:
        x = x * weight[:, None, None]
    return x
    
class RmsNorm(torch.nn.Module):
    def __init__(self, n_embed, eps=1e-6, dim=-1):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n_embed))
        self.dim = dim

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps, self.dim)

    

class SqrllLayer(torch.nn.Module):
    def __init__(self, n_in, n_mem, n_out):
        super().__init__()
        self.wr = torch.nn.Linear(n_in, n_mem)
        self.wi = torch.nn.Linear(n_in, n_mem, bias=False)
        self.wig = torch.nn.Linear(n_in, n_mem)
        self.wog = torch.nn.Linear(n_in, n_mem)
        self.wo = torch.nn.Linear(n_mem, n_out, bias=False)

    def forward(self, x, mem=None):
        ig = self.wig(x).sigmoid()
        og = self.wog(x).sigmoid()
        r = self.wr(x).sigmoid()
        x = self.wi(x) * ig

        y = sqrll_kernel(x, r, mem)
        mem = y[:,-1].detach().clone()
        
        y = torch.nn.functional.softsign(y)
        y = y * og
        y = self.wo(y)

        return y, mem


class SqrllFFN(torch.nn.Module):
    def __init__(self, n_embed, n_ffn):
        super().__init__()
        self.wi = torch.nn.Linear(n_embed, n_ffn, bias=False)
        self.wg = torch.nn.Linear(n_embed, n_ffn)
        self.wo = torch.nn.Linear(n_ffn, n_embed, bias=False)

    def forward(self, x):
        y = self.wi(x) * self.wg(x).sigmoid()
        y = self.wo(y)
        return y
    

class SqrllResid(torch.nn.Module):
    def __init__(self, n_embed, n_mem, n_ffn=0, dropout=0.1):
        super().__init__()
        self.norm = RmsNorm(n_embed)
        self.sqrll = SqrllLayer(n_embed, n_mem, n_embed)
        self.dropout = torch.nn.Dropout(p=dropout)
        if n_ffn:
            self.ffnorm = RmsNorm(n_embed)
            self.ffn = SqrllFFN(n_embed, n_mem)
            self.ffdrop = torch.nn.Dropout(p=dropout)

    def forward(self, x, mem=None):
        y = self.norm(x)
        y, mem = self.sqrll(y, mem)
        x = x + self.dropout(y)
        if hasattr(self, 'ffn'):
            y = self.ffnorm(x)
            y = self.ffn(y)
            x = x + self.ffdrop(y)
        return x, mem




@dataclass
class SqrllConfig:
    n_tokens_in: int = 256
    n_vector_in: int = 0
    n_out: int = 256
    n_embed: int = 1024
    n_mem: int = 1024
    n_layer: int = 16
    n_ffn: int = 1024
    ffn_rate: int = 1
    dropout: float = 0.1



class SqrllStack(torch.nn.Module):
    def __init__(self, cfg: SqrllConfig):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            SqrllResid(
                cfg.n_embed, 
                cfg.n_mem,
                n_ffn=(cfg.n_ffn if (l+1)%cfg.ffn_rate==0 else 0),
                dropout=cfg.dropout,
            )
            for l in range(cfg.n_layer)
        ])

    def forward(self, x, mem=None):
        nexmem = []
        for i, block in enumerate(self.blocks):
            m = mem[i] if mem is not None else None
            x, m = block(x, m)
            nexmem += [m]
        return x, nexmem
    

class SqrLLM(torch.nn.Module):
    def __init__(self, cfg: SqrllConfig):
        super().__init__()
        self.config = cfg
        if cfg.n_tokens_in:
            self.w_in_t = torch.nn.Embedding(cfg.n_tokens_in, cfg.n_embed)
        if cfg.n_vector_in:
            self.w_in_v = torch.nn.Linear(cfg.n_vector_in, cfg.n_embed)
        self.sqrll = SqrllStack(cfg)
        self.norm = RmsNorm(cfg.n_embed)

        # TODO gated MLP output layer?
        self.w_out = torch.nn.Linear(cfg.n_embed, cfg.n_out, bias=False)

    def forward(self, in_toks=None, in_vecs=None, in_raw=None, mem=None, return_embed=False):
        x = []
        if in_toks is not None:
            x += [self.w_in_t(in_toks)]
        if in_vecs is not None:
            x += [self.w_in_v(in_vecs)]
        if in_raw is not None:
            x += [in_raw]
        x = sum(x)
        x, mem = self.sqrll(x, mem)
        x = self.norm(x)
        if return_embed:
            return self.w_out(x), x, mem
        return self.w_out(x), mem
    
    def save(self, filename, metadata=None):
        model_dict = {
            'config': self.config,
            'weights': self.state_dict(),
            'meta': metadata,
        }
        torch.save(model_dict, filename)

    @staticmethod
    def load(filename, **overrides):
        model_dict = torch.load(filename)
        cfg = model_dict['config']
        for k, v in overrides.items():
            setattr(cfg, k, v)
        model = SqrLLM(cfg)
        model.load_state_dict(model_dict['weights'])
        return model
    
