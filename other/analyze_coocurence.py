from explanation.training.find_word_contexts import find_word_bow_vectors
from explanation.visualisation.tsne_on_vec import read_vocab
from other.gensim_demo import _start_shell

vocab = read_vocab("/home/ifajcik/deep_learning/word2vec/corpus_data/corpus_vocabs/t.txt", min_freq=5)
r = find_word_bow_vectors("/home/ifajcik/deep_learning/word2vec/corpus_data/ebooks_corpus_CZ/cwc1meg", vocab.keys, num_of_processes=2)
print("Finished after %d s"%r[1])

i2w = {i: x for i, x in enumerate(vocab.keys())}

get_most_coocurent()
_start_shell(locals())

#dump_npdict(r,"context.txt")
#d=read_npdict("context.txt")