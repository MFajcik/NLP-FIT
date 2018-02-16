# PyVersion: 3.6
# Authors: Martin Fajčík FIT@BUT 2018
import math
import os
import threading
from typing import List
import numpy as np
from tqdm import tqdm


def get_character(f):
    try:
        return str(f.read(1), encoding="utf-8")
    except UnicodeDecodeError:
        return "ERROR"

#Custom (de)serialization
def dump_npdict(r, fname):
    with open(fname, mode="w") as f:
        for key, v in tqdm(zip(r[0].keys(), r[0].values())):
            f.write("{} {}\n".format(key, ' '.join(v.astype(str))))

def load_npdict(fname):
    d = dict()
    with open(fname, mode="r") as f:
        for line in f.readlines():
            tokens = line.split()
            d[tokens[0]] = np.array(tokens[1:]).astype(int)

def find_textfile_split_points(file: str, n: int) -> List[int]:
    """
    Find suitable spots for file to be splitted into n chunks.
    The suitable spots are nearest following whitespaces
    :param file:
    :param n:
    :return:
    """
    filesize = os.path.getsize(file)
    file_chunk_size = int(math.floor(filesize / float(n)))
    offsets = [None] * n
    with open(file, mode="rb") as f:
        offsets[0] = 0
        for idx in range(1, n):
            offset = idx * file_chunk_size
            f.seek(offset)
            character = get_character(f)
            while not (character.isspace() or character == ''):
                character = get_character(f)
            offsets[idx] = f.tell() - 1
    return offsets


class LockedIterator(object):
    """
    Thread-safe iterator wrapper
    """

    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self):
        return self

    def next(self):
        self.lock.acquire()
        try:
            return self.it.next()
        finally:
            self.lock.release()


class DotDict(dict):
    """
    A dictionary with dot notation for key indexing
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val

    # These are needed when python pickle-ing is done
    # for example, when object is passed to another process
    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass
