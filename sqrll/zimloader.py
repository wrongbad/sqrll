from .dataloaders import map_parallel

from libzim.reader import Archive
from bs4 import BeautifulSoup

import random

g_zims = {}

def read_zim_entry(args):
    fname, entry = args
    global g_zims
    if fname not in g_zims:
        g_zims[fname] = Archive(fname)
    
    entry = g_zims[fname]._get_entry_by_id(entry).get_item()
    if entry.mimetype != 'text/html':
        return None
    entry = entry.content.tobytes().decode('utf-8')
    entry = BeautifulSoup(entry, features="lxml").get_text()

    # filter for substantial paragraphs
    # drop lists and random notes
    new_entry = ''
    for line in entry.splitlines():
        line = line.strip()
        if len(line) < 100: continue
        if 'Wikipedia' in line: continue
        if 'Retrieve' in line: continue
        if 'United' in line: continue
        if 'Archive' in line: continue
        new_entry += line + '\n\n'
    entry = new_entry
    if len(entry) < 200:
        return None

    return entry

# read_zim_entry is pretty slow
# but throwing more threads at it seems to fix the problem for me
def read_zims(files, nthreads=1):
    for fname in files:
        count = Archive(fname).all_entry_count
        args = [(fname, i) for i in range(count)]
        random.shuffle(args)
        for x in map_parallel(read_zim_entry, args, workers=nthreads):
            if x is not None:
                yield x