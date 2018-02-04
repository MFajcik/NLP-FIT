# PyVersion: 3.6
# Authors: Martin Fajčík FIT@BUT 2018
# Last edited: 01.02.2018
import threading

from nlpfit.preprocessing.io import read_words


def count_words(input_file, output_file, to_sort=True):
    wordcounter = dict()
    for word in read_words(input_file):
        wordcounter[word] = wordcounter.get(word, 0) + 1
    if to_sort:
        import operator
        wordcounter  = sorted(wordcounter.items(), key=operator.itemgetter(1), reverse=True)

    with open(output_file, 'w') as outf:
        for key, value in wordcounter:
            outf.write("%s %d\n" % (key, value))


class LockedIterator(object):
    """
    Thread-safe iterator wrapper
    """
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def next(self):
        self.lock.acquire()
        try:
            return self.it.next()
        finally:
            self.lock.release()
