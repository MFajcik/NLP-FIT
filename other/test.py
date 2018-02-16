#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# Author: Viliam Samuel Hostak
# Login: xhosta04
# Email: xhosta04@stud.fit.vutbr.cz
# =============================================================================

import os
import sys
import multiprocessing
import argparse
import datetime
from six import iteritems
from glove import Glove
from gensim import utils
import logging
from loggingConfig import init_logging
from cooccurrence import build_matrix
from cooccurrence import load_matrix


def save(glove, fname):
    """
    Save glove model in format which is compatible with gensim.
    Parameters:
        glove:          Glove model
        fname:          Output filename
    """
    assert (len(glove.inverse_dictionary), glove.no_components) == glove.word_vectors.shape
    with utils.smart_open(fname, 'wb') as file:
        file.write(utils.to_utf8("%s %s\n" % glove.word_vectors.shape))
        for index, word in iteritems(glove.inverse_dictionary):
            row = glove.word_vectors[index]
            file.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))


def train_glove(co_matrix, output_dir, epochs, dimensions):
    """
    Train glove model on input corpus.
    Parameters:
        co_matrix:      Co-occurrence matrix
        output_dir:     Directory path where output glove model will be saved
        epochs:         Number of epochs
        dimensions:     Number of dimensions
    """
    now = datetime.datetime.now()
    output = os.path.join(output_dir,
                          "GloVe" + "_" +
                          str(dimensions) + "_" +
                          str(epochs) + "_" +
                          now.strftime("%Y-%m-%d_%H:%M"))
    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))
    # Train glove model
    logging.info("Glove model training starts.")
    glove = Glove(no_components=dimensions, learning_rate=0.05)
    glove.fit(co_matrix.matrix, epochs=epochs, no_threads=multiprocessing.cpu_count(), verbose=True)
    glove.add_dictionary(co_matrix.dictionary)
    output += ".vec"
    logging.info("storing %sx%s projection weights into %s" % (len(glove.inverse_dictionary),
                                                               glove.no_components,
                                                               output))
    save(glove, output)
    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        required=True,
                        help="Input corpus or co-occurrence matrix.")
    parser.add_argument("-o", "--output_dir",
                        default="./models/glove",
                        help="Directory path where output glove model will be saved.")
    parser.add_argument("-om", "--output_mdir",
                        default="./models/co-occurrence",
                        help="Directory path where output co-occurrence matrix will be saved.")
    parser.add_argument("-c", "--count",
                        default=5,
                        help="Minimal number of word occurrences.")
    parser.add_argument("-e", "--epochs",
                        default=10,
                        help="Number of training epochs.")
    parser.add_argument("-d", "--dimensions",
                        default=300,
                        help="Number of dimensions.")
    parser.add_argument("-m", "--mode",
                        choices=["matrix", "corpus"],
                        default="corpus",
                        help="Select input corpus or co-occurrence matrix.")
    args = parser.parse_args()
    init_logging(os.path.basename(sys.argv[0]).split(".")[0])
    if args.mode == "corpus":
        co_matrix = build_matrix(args.input, args.output_mdir, int(args.count))
    else:
        co_matrix = load_matrix(args.input)
    train_glove(co_matrix,
                args.output_dir,
                int(args.epochs),
                int(args.dimensions))