from .sqrll import sqrll_kernel
import torch


class SqrllGate(torch.nn.Module):
    def __init__(self, n_in, n_mem, n_out):
        super().__init__()
        self.wf = torch.nn.Linear(n_in, n_mem)
        self.wi = torch.nn.Linear(n_in, n_mem, bias=False)
        self.wig = torch.nn.Linear(n_in, n_mem)
        self.wog = torch.nn.Linear(n_in, n_mem)
        self.wo = torch.nn.Linear(n_mem, n_out, bias=False)

    def forward(self, x, mem=None):
        y = self.wi(x) * self.wig(x).sigmoid()
        r = self.wf(x).sigmoid()

        y = sqrll_kernel(y, r, mem)
        mem = y[:,-1].detach().clone()
        
        y = torch.nn.functional.softsign(y)
        y = y * self.wog(x).sigmoid()
        y = self.wo(y)

        return y, mem


class SqrllFFN(torch.nn.Module):
    def __init__(self, n_embed, n_ffn, dropout=0.1):
        super().__init__()
        self.norm = torch.nn.LayerNorm(n_embed)
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
    def __init__(self, n_embed, n_mem, dropout=0.1):
        super().__init__()
        self.norm = torch.nn.LayerNorm(n_embed)
        self.sqrll = SqrllGate(n_embed, n_mem, n_embed)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.ffn = SqrllFFN(n_embed, n_mem, dropout=dropout)

    def forward(self, x, mem=None):
        y = self.norm(x)
        y, mem = self.sqrll(y, mem)
        x = x + self.dropout(y)
        x = self.ffn(x)

        return x, mem


class SqrllStack(torch.nn.Module):
    def __init__(
            self, 
            n_embed = 1024,
            n_mem = 1024,
            n_layer = 16,
            dropout = 0.1,
            ):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            SqrllResid(n_embed, n_mem, dropout=dropout)
            for l in range(n_layer)
        ])

    def forward(self, x, mem=None):
        nexmem = []
        for i, block in enumerate(self.blocks):
            m = mem[i] if mem else None
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
            dropout = 0.1,
            ):
        super().__init__()
        self.w_in = torch.nn.Embedding(n_in, n_embed)
        self.sqrll = SqrllStack(n_embed, n_mem, n_layer)
        self.w_out = torch.nn.Linear(n_embed, n_out)

    def forward(self, x, mem=None):
        x = self.w_in(x)
        x, mem = self.sqrll(x, mem)
        return self.w_out(x), mem