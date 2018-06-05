#!/usr/bin/python3
# Martin Fajcik
import sys

from nlpfit.explanation.training import WordContextCalculator
from nlpfit.explanation.visualisation.tsne_on_vec import read_vocab

if __name__ == "__main__":
    CWC2011VOCAB = "/home/ifajcik/deep_learning/word2vec/corpus_data/corpus_vocabs/vocab_cwc2011.txt"
    CWC2011CORPUS = "/home/ifajcik/deep_learning/word2vec/corpus_data/cwc_corpus2011/cwc2011"
    CWC1megtest = "/home/ifajcik/deep_learning/word2vec/corpus_data/cwc_corpus2011/cwc1meg"
    EBOOKSVOCAB = "/home/ifajcik/deep_learning/word2vec/corpus_data/corpus_vocabs/vocab_ebooks_nostopwords.txt"
    EBOOKSCORPUS = "/home/ifajcik/deep_learning/word2vec/corpus_data/ebooks_corpus_CZ/e_knihy_preprocessed.txt"
    vocab = read_vocab(EBOOKSVOCAB, min_freq=5)
    bow_vectors = WordContextCalculator(EBOOKSCORPUS, vocab, num_of_processes=int(sys.argv[1]), window=5)
    bow_vectors.word_bow()
    print("Saving BOW matrix")
    # bow_vectors.savebow("bow_CWC2011")
    bow_vectors.savebow("bow_eboooks_w5")
    # avg_word_bow = bow_vectors.avg_bow()

    # bow_vectors = WordContextCalculator(None,vocab)
    # print("Loading model")
    # bow_vectors.loadbow("bow")

    # from other.gensim_demo import _start_shell
    # _START_SHELL(locals())

    # dump_npdict(r,"context.txt")
    # d=read_npdict("context.txt")
