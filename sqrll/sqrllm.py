from .sqrll import sqrll_kernel
import torch


def rms_norm(x, weight, eps):
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * weight
    
class RmsNorm(torch.nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n_embed))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)
    

class SqrllLayer(torch.nn.Module):
    def __init__(self, n_in, n_mem, n_out):
        super().__init__()
        self.wr = torch.nn.Linear(n_in, n_mem)
        self.wi = torch.nn.Linear(n_in, n_mem, bias=False)
        self.wig = torch.nn.Linear(n_in, n_mem)
        self.wog = torch.nn.Linear(n_in, n_mem)
        self.wo = torch.nn.Linear(n_mem, n_out, bias=False)

    def forward(self, x, mem=None):
        og = self.wog(x).sigmoid()
        r = self.wr(x).sigmoid()
        x = self.wi(x) * self.wig(x).sigmoid()

        y = sqrll_kernel(x, r, mem)
        mem = y[:,-1].detach().clone()
        
        y = torch.nn.functional.softsign(y)
        y = y * og
        y = self.wo(y)

        return y, mem


class SqrllFFN(torch.nn.Module):
    def __init__(self, n_embed, n_ffn, dropout=0.1):
        super().__init__()
        self.norm = RmsNorm(n_embed)
        self.wi = torch.nn.Linear(n_embed, n_ffn, bias=False)
        self.wg = torch.nn.Linear(n_embed, n_ffn)
        self.wo = torch.nn.Linear(n_ffn, n_embed)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        y = self.norm(x)
        y = self.wi(y) * self.wg(y).sigmoid()
        y = self.wo(y)
        y = self.dropout(y)
        return x + y
    

class SqrllResid(torch.nn.Module):
    def __init__(self, n_embed, n_mem, n_ffn=0, dropout=0.1):
        super().__init__()
        self.norm = RmsNorm(n_embed)
        self.sqrll = SqrllLayer(n_embed, n_mem, n_embed)
        self.dropout = torch.nn.Dropout(p=dropout)
        if n_ffn:
            self.ffn = SqrllFFN(n_embed, n_mem, dropout=dropout)

    def forward(self, x, mem=None):
        y = self.norm(x)
        y, mem = self.sqrll(y, mem)
        x = x + self.dropout(y)
        if hasattr(self, 'ffn'):
            x = self.ffn(x)
        return x, mem


class SqrllStack(torch.nn.Module):
    def __init__(
            self, 
            n_embed = 1024,
            n_mem = 1024,
            n_layer = 16,
            n_ffn = 1024,
            ffn_rate = 4,
            dropout = 0.1,
            ):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            SqrllResid(
                n_embed, 
                n_mem,
                n_ffn=(n_ffn if (l+1)%ffn_rate==0 else 0),
                dropout=dropout,
            )
            for l in range(n_layer)
        ])

    def forward(self, x, mem=None):
        nexmem = []
        for i, block in enumerate(self.blocks):
            m = mem[i] if mem is not None else None
            x, m = block(x, m)
            nexmem += [m]
        return x, nexmem
    

class SqrLLM(torch.nn.Module):
    def __init__(
            self, 
            n_in = 256,
            n_out = 256,
            n_embed = 1024,
            n_mem = 1024,
            n_layer = 16,
            n_ffn = 1024,
            ffn_rate = 4,
            dropout = 0.1,
            ):
        super().__init__()
        self.w_in = torch.nn.Embedding(n_in, n_embed)
        self.sqrll = SqrllStack(
            n_embed, 
            n_mem, 
            n_layer, 
            n_ffn,
            ffn_rate,
            dropout=dropout,
        )
        self.norm = RmsNorm(n_embed)
        self.w_out = torch.nn.Linear(n_embed, n_out)

    def forward(self, x, mem=None):
        x = self.w_in(x)
        x, mem = self.sqrll(x, mem)
        x = self.norm(x)
        return self.w_out(x), mem
    


class StatefulWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        inputs = torch.tensor([[0]])
        _, mem = model(inputs)
        self.mem = [torch.zeros_like(m) for m in mem]

    def forward(self, x):
        x, self.mem = self.model(x, self.mem)
        return x