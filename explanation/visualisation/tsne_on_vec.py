import sys

import matplotlib.pyplot as plt
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE

from preprocessing.tools import read_frequency_vocab


def plot_with_labels(low_dim_embs, labels, filename='tsne_.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    fig = plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    pp = PdfPages(filename)
    pp.savefig(fig)
    pp.close()


def run_TSNE(model, freqs, outputs=10):
    # Getting the tokens
    limit = 100
    length = len(model.vocab) if len(model.vocab) < limit else limit

    words = length * [None]
    embeddings = np.ndarray(shape=(length, model.vector_size))

    import operator
    sorted_vocab = sorted(freqs.items(), key=operator.itemgetter(1), reverse=True)
    i = 0
    drawn_graphs = 0
    for word in sorted_vocab:
        if i == limit:
            print("Drawing graph #{}".format(str(drawn_graphs)))
            embeddings = embeddings.reshape(limit, model.vector_size)
            # Creating the tsne plot [Warning: will take time]
            tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)
            low_dim_embedding = tsne.fit_transform(embeddings)
            plot_with_labels(low_dim_embedding, words,
                             filename="ebooks_TSNE_{}-{}".format(freqs[words[0]], freqs[words[i - 1]]))
            i = 0
            drawn_graphs += 1
            if (drawn_graphs == outputs):
                break
        words[i] = word[0]
        try:
            embeddings[i] = model[word[0]]
        except:
            i -= 1
        i += 1

    # pca = PCA(n_components=2)
    # low_dim_embedding = pca.fit_transform(embeddings)
    # plot_with_labels(low_dim_embedding, words)
    # pca = PCA(embeddings,standardize=True)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("Loading model...")
        model = KeyedVectors.load_word2vec_format(sys.argv[1], binary=False)
        print("Done")
        run_TSNE(model)
    elif len(sys.argv) > 2:
        word_frequencies = read_frequency_vocab(sys.argv[2])
        model = KeyedVectors.load_word2vec_format(sys.argv[1], binary=False)
        run_TSNE(model, word_frequencies)
    else:
        sys.stderr.write("Not enough arguments. Expecting embedding file.\n")
