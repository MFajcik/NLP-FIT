import argparse
import os

import sys

import pandas as pd
import re
import numpy as np

from gensim.models import KeyedVectors
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from other.logging_config import init_logging
from preprocessing.preprocessor import lemmatize_list


def get_words(input):
    cnwa = pd.read_csv(input, delimiter="\t", header=None)
    words = set()
    for idx, row in cnwa.iterrows():
        col1_words = re.split(':|\s', row[0][:-1].strip())
        col3_words = re.split(':|\s', row[2].strip())
        words = words.union(set(col1_words + col3_words))
    return words


def draw_TSNE(model, words):
    words = list(words)
    embeddings = np.ndarray(shape=(len(words), model.vector_size))
    for i, word in enumerate(words):
        if word in model.vocab:
            embeddings[i] = model[word]
        else:
            logging.critical("Word %s not in vocabulary!" % word)
            embeddings[i] = None
            words[i] = None
    # filter out Nones #775x30
    words = [x for x in words if x is not None]
    embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]
    tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)
    low_dim_embedding = tsne.fit_transform(embeddings)
    plot_with_labels(low_dim_embedding, words, filename=args.output + ".pdf")


def plot_with_labels(low_dim_embs, labels, filename='tsne_.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    fig = plt.figure(figsize=(18, 18))  # in inches
    for i, label in tqdm(enumerate(labels)):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, s=3)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     size=3)
    pp = PdfPages(filename)
    pp.savefig(fig)
    pp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        required=True,
                        help="Input file in cnwa format.")
    parser.add_argument("-o", "--out_buff",
                        required=True,
                        help="Output file.")
    parser.add_argument("-l", "--logpath",
                        default=os.getcwd(),
                        help="Explicit setting of log folder path")
    parser.add_argument("-m", "--model",
                        required=True,
                        help="Vec model to do tsne on.")
    parser.add_argument("-d", "--dictionary",
                        default="/mnt/minerva1/nlp/projects/semantic_relatedness9" + \
                                "/models/cz_morphodita/czech-morfflex-160310.dict",
                        help="Morphological analyzer.")

    args = parser.parse_args()
    logging = init_logging(os.path.basename(sys.argv[0]).split(".")[0], logpath=args.logpath)
    logging.info("Counting words from cnwa...")
    words = get_words(args.input)
    logging.info("Lemmatizing words...")
    lemmatized_words = lemmatize_list(list(words), args.dictionary)
    model = KeyedVectors.load_word2vec_format(args.model, binary=False)
    logging.info("Drawing TSNE...")
    draw_TSNE(model, words)
