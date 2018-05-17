# for each word in corpus, we would like to find it's global average context

import time
import pickle
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm

from other.logging_config import logger_stub
from other.paralellism import parallel_worker
from preprocessing.tools import DotDict


class WordContextCalculator():
    """
    TODO: documentation
    """
    def __init__(self, corpus, vocabulary, window=5, num_of_processes=8, logger=logger_stub()):
        self.vocab = vocabulary
        self.window = window
        self.corpus = corpus
        self.num_of_processes = num_of_processes
        self.logger = logger
        self.w2i = {x: i for i, x in enumerate(self.vocab.keys())}
        self.i2w = {i: x for i, x in enumerate(self.vocab.keys())}
        # we wil keep both matrices
        # this can be memory inefficient
        self.bow = None
        self.avgbow = None

    def word_bow(self):
        if self.bow is None:
            self.bow = self.find_word_bow_vectors()
        return self.bow

    def avg_bow(self):
        if self.avgbow is None:
            if self.bow is None:
                self.bow = self.find_word_bow_vectors()
            self.avgbow = self.find_avg_bow_vectors()
        return self.avgbow

    def find_avg_bow_vectors(self):
        # TODO: implement
        pass

    def find_word_bow_vectors(self):

        assert (self.corpus is not None)
        assert (self.window is not None)
        assert (self.vocab is not None)
        assert (self.logger is not None)

        start_time = time.time()
        opts = DotDict()
        opts.window = self.window
        opts.vocabulary = self.vocab.keys
        self.logger.info("Starting workers")
        # futures = parallel_worker(calculate_words_context, self.corpus, opts, self.logger, num_of_processes=self.num_of_processes)
        self.logger.info("Summing matrices")
        # bow_for_each_word = sum(list(map(lambda x: x.result(), futures)))
        duration = time.time() - start_time
        self.logger.info("Finished after %d seconds" % duration)
        return bow_for_each_word

    def find_topn_context(self, idx, topn=20):
        if self.bow is None:
            self.bow = self.find_word_bow_vectors()
        row = self.bow[idx, :].toarray()[0]
        most_coocurent_ind = np.argpartition(row, -topn)[-topn:]
        most_coocurent = list(map(lambda x: self.i2w[x], most_coocurent_ind))
        return most_coocurent

    def save_topn_context(self, file, topn=20, leave_occurence_count=True):
        """
        #TODO: This method is very VERY slow, use on your own danger
        :param file:
        :param topn:
        :param leave_occurence_count:
        """
        if self.bow is None:
            self.bow = self.find_word_bow_vectors()
        with open(file, mode="w") as f:
            for key, word in tqdm(list(enumerate(self.vocab))):
                row = self.bow[key, :].toarray()[0]
                most_coocurent_ind = np.argpartition(row, -topn)[-topn:]
                most_coocurent = list(map(lambda x: self.i2w[x], most_coocurent_ind))
                if not leave_occurence_count:
                    f.write("{} {}\n".format(word, ' '.join(most_coocurent)))
                else:
                    most_coocuren_vals = row[most_coocurent_ind]
                    f.write("{} {}\n".format(word, ' '.join(["{}({})".format(w, c) \
                                                             for w, c in zip(most_coocurent, most_coocuren_vals)])))

    def savebow(self, fn):
        if self.bow is None:
            self.bow = self.find_word_bow_vectors()
        with open(fn, mode="wb") as f:
            pickle.dump(self.bow, f)

    def loadbow(self, fn):
        with open(fn, mode="rb") as f:
            self.bow = pickle.load(f)


def calculate_words_context(textchunk, result, idx, opts, logger):
    """

    Context calculation is approximate
    Word context for word near the start/end of chunk are not considered
    :param textchunk:
    :param j:
    :param opts:
    :param logger:
    """
    vocab = opts.vocabulary()
    vocablen = len(vocab)
    if result is None:
        result = lil_matrix((vocablen, vocablen))
    tokens = textchunk.split()
    w2i = {x: i for i, x in enumerate(vocab)}
    window = opts.window
    tokenwindow = tokens[window:-window]
    for i, token in enumerate(tokenwindow):
        if token in vocab:
            for j, neighbor in enumerate(tokens[i - window:i + window]):
                if j == window or neighbor not in vocab:
                    continue
                result[w2i[token], w2i[neighbor]] += 1
    return result


def get_most_coocurent(r, i2w, count=10):
    pass
