# for each word in corpus, we would like to find it's global average context

import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from other.logging_config import logger_stub
from other.paralellism import parallel_worker
from preprocessing.tools import DotDict


def dsum(dicts):
    for d in dicts[1:]:
        for k, v in d.items():
            dicts[0][k] += v
    return dict(dicts[0])


def count_context(textchunk, result, idx, opts, logger):
    """

    Context calculation is approximate
    Word context for word near the start/end of chunk are not considered
    :param textchunk:
    :param j:
    :param opts:
    :param logger:
    """
    vocab = opts.vocabulary()
    if result is None:
        result = csr_matrix((3, 4))
    tokens = textchunk.split()
    w2i = {x: i for i, x in enumerate(vocab)}
    window = opts.window
    tokenwindow = tokens[window:-window]
    for i, token in enumerate(tokenwindow):
        for j, neighbor in enumerate(tokens[i - window:i + window]):
            if j == window or neighbor not in vocab:
                continue
            result[token][w2i[neighbor]] += 1
    return result


def find_word_bow_vectors(corpus, vocabulary, window=5, num_of_processes=8, logger=logger_stub()):
    """

    :param corpus:
    :param vocabulary: list of words in vocabulary
    :param window:
    :param num_of_processes:
    :param logger:
    """
    start_time = time.time()
    opts = DotDict()
    opts.window = window
    opts.vocabulary = vocabulary
    logger.info("Starting workers")
    futures = parallel_worker(count_context, corpus, opts, logger, num_of_processes=num_of_processes)
    logger.info("Appending dicts")
    context_bow = dsum(list(map(lambda x: x.result(), futures)))

    duration = time.time() - start_time
    return context_bow, duration


def get_most_coocurent(r,i2w,count=10):
    pass