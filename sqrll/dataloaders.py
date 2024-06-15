import torch
import multiprocessing
import collections
import numpy as np
import os


def read_raw(files, chunk=2**16):
    for fname in files:
        with open(fname, 'r', encoding='utf8') as f:
            offset = torch.randint(0, chunk, ())
            if x := f.read(offset):
                yield x
            while x := f.read(chunk):
                yield x


def str_tensor(data):
    for x in data:
        # torch.frombuffer does not retain the memory
        x = x if type(x)==bytes else bytes(x, encoding='utf8')
        x = np.frombuffer(x, dtype=np.uint8)
        yield torch.tensor(x)


def tokenize(data, tokenizer):
    for x in data:
        x = x if type(x)==str else str(x, encoding='utf8')
        x = tokenizer.encode(x).ids
        yield torch.tensor(x)


def tensor_to(data, dst):
    for x in data:
        yield x.to(dst)


def map_parallel(func, data, workers=4, lookahead=128, timeout=10):
    with multiprocessing.Pool(workers) as pool:
        q = collections.deque()
        for x in data:
            q.append(pool.apply_async(func, (x,)))
            if len(q) >= lookahead:
                try:
                    if (r := q.popleft().get(timeout=timeout)) is not None:
                        yield r
                except Exception as e:
                    print(e)
        while len(q):
            try:
                if (r := q.popleft().get(timeout=timeout)) is not None:
                    yield r
            except Exception as e:
                print(e)

class Tetris:
    def __init__(self, batch, targetlen):
        self.seqs = [[]] * batch
        self.seq_len = np.zeros(batch, dtype=int)
        self.target_len = targetlen
    
    def push(self, x):
        insert = np.argmin(self.seq_len)
        if self.seq_len[insert] == 0:
            self.seqs[insert] = x
        else:
            self.seqs[insert] = torch.cat((self.seqs[insert], x), dim=0)
        self.seq_len[insert] += len(x)

    def ready(self):
        return np.min(self.seq_len) // self.target_len

    def pop(self):
        outlen = self.target_len
        out = torch.stack([s[:outlen] for s in self.seqs], dim=0)
        self.seqs = [s[outlen:] for s in self.seqs]
        self.seq_len -= outlen
        return out



def tetris(data, batch=16, seqlen=256):
    tet = Tetris(batch, seqlen)
    for d in data:
        if d is None: continue
        tet.push(d)
        while tet.ready():
            yield tet.pop()

        # insert = np.argmin(seq_len)
        # if seq_len[insert] == 0:
        #     seqs[insert] = d
        # else:
        #     seqs[insert] = torch.cat((seqs[insert], d), dim=0)
        # seq_len[insert] += len(d)

        # while np.min(seq_len) >= seqlen:
        #     yield torch.stack([
        #         s[:seqlen] for s in seqs
        #     ], dim=0)
        #     seqs = [s[seqlen:] for s in seqs]
        #     seq_len -= seqlen


def multitetris(data, streams, batch=16, seqlen=256):
    tets = [Tetris(batch, seqlen) for _ in range(streams)]
    for d in data:
        if d is None: continue
        for i, x in enumerate(d):
            tets[i].push(x)
        ready = min([t.ready() for t in tets])
        for _ in range(ready):
            yield tuple(t.pop() for t in tets)

def batch(data, batch=8):
    buf = []
    for x in data:
        buf += [x]
        if len(buf) == batch:
            yield torch.stack(buf)
            buf = []


def take(data, count):
    for i, x in enumerate(data):
        yield x
        if i+1 == count:
            return


def mix(*datas):
    empty = False
    while not empty:
        empty = True
        for data in datas:
            d = next(data, None)
            if d is not None:
                yield d
                empty = False


def shuffle(data, bufsize=1024):
    buffer = [None] * bufsize
    for midi in data:
        i = torch.randint(0, bufsize, ())
        out = buffer[i]
        buffer[i] = midi
        if out is not None:
            yield out
    for out in buffer:
        if out is not None:
            yield out


def dir_iter(rootdir):
    root = os.path.expanduser(rootdir)
    for curdir, _, files in os.walk(root):
        for f in files:
            yield os.path.join(curdir, f)