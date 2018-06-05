import time

from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
from gensim.models.callbacks import CallbackAny2Vec

QW = "/home/ifajcik/deep_learning/word2vec/evaluation/analogy_questions/questions-words.txt"
corpus = "/home/ifajcik/deep_learning/word2vec/corpus_data/text8/text8"

import logging

logging.basicConfig(level=logging.INFO)


class EvalAnalogyQuestions(CallbackAny2Vec):
    def __init__(self):
        self.benchmarktime = time.time()

    def on_epoch_end(self, model):
        print("Time passed (WO last eval): {:.2f} min".format((time.time()- self.benchmarktime)/60))
        model.wv.accuracy(QW, restrict_vocab=None)
        print("Time passed: {:.2f} min".format((time.time()- self.benchmarktime)/60))

evaluator = EvalAnalogyQuestions()
model = Word2Vec(Text8Corpus(fname=corpus),iter=100,negative=25,size=300, sample=1e-4, window=5, min_count=5, workers=24, sg=1,
                 callbacks=[evaluator])