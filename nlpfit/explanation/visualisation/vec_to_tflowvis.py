import argparse
import os
import re
import sys

import pandas as pd
from gensim.models import KeyedVectors

from nlpfit.other.logging_config import init_logging, logger_stub
from nlpfit.preprocessing import lemmatize_list


def get_words(input):
    cnwa = pd.read_csv(input, delimiter="\t", header=None)
    words = set()
    for idx, row in cnwa.iterrows():
        col1_words = re.split(':|\s', ' '.join(row[0].split()[:-1]).strip())
        col3_words = re.split(':|\s', row[2].strip())
        words = words.union(set(col1_words + col3_words))
    return words


def generate_files_for_tflow_embedding_projector(model, lemmatized_words, o_vec="vecfile.tsv", o_label="labelfile.tsv",
                                                 logging=logger_stub()):
    with open(o_vec, mode="w") as ov, open(o_label, mode="w") as ol:
        already_written = []
        for word in lemmatized_words:
            # We want to show similar words and the important word
            if word in model.vocab:
                candidates = list(map(lambda x: x[0], model.wv.most_similar(positive=word))) + [word]
                for candidate in candidates:
                    if candidate not in already_written:
                        ol.write(candidate + "\n")
                        ov.write('\t'.join(model.wv[candidate].astype(str)) + "\n")
                        already_written.append(candidate)
            else:
                logging.critical("Word {} out of vocabulary.".format(word))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        required=True,
                        help="Input file in cnwa format.")
    parser.add_argument("-o_vec",
                        required=True,
                        help="Output file with vectors.")
    parser.add_argument("-o_labels",
                        required=True,
                        help="Output file with labels.")
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
    logging.info("Generating tensorflow embedding projection outputs")
    generate_files_for_tflow_embedding_projector(model, lemmatized_words, o_vec=args.o_vec, o_label=args.o_labels)
