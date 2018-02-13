# Authors:
# Viliam Samuel Hostak -- basic implementation
# Martin Fajcik --
# frequency file and frequency options
# removed second out_buff column,
# changes related to showing out of vocabulary tests and words
# removed deprecated way of loading vec files with the current one
# added word counting
# changes to MAP calculation -- from now on, out of vocabulary tests have fixed MAP score of 0

import argparse
import logging
import os
import sys

from gensim.models.keyedvectors import KeyedVectors
from six import iteritems
from tqdm import tqdm
from ufal.morphodita import *

from other.logging_config import init_logging
from preprocessing.preprocessor import get_lemma_dict

morpho = None
lemmas = None
FIXED_OOV_SCORE = 0
STRIP_NON_FREQUENT = 10
DEFAULT_DICT = "../contrib/preprocessing/cz_morphodita/models/czech-morfflex-160310.dict"


def read_frequency_vocab(frequency_file):
    vocab_dict = dict()
    with open(frequency_file) as f:
        for line in tqdm(f):
            word_and_count = line.split()
            if len(word_and_count)<2:
                logging.critical("Encouneted line with missing/whitespace word:\n'{}'".format(line))
            elif int(word_and_count[1]) > STRIP_NON_FREQUENT:
                vocab_dict[word_and_count[0]] = int(word_and_count[1])
    return vocab_dict


def cnwa2cnwae(input_file, dictionary, model, sort = True, frequency_file=None):
    """
    Converts file in cnwa format to file in cnwar format
    Parameters:
        input_file:     Input file on cnwa format
        dictionary:     Dictionary compatible with morphodita library
        model:          Word embedding model
        sort:           Sort out_buff
        frequency_file: File containing number of occurences of each word in training corpus
    """

    global morpho, lemmas

    frequency_dict = dict()
    if frequency_file is not None:
        logging.info("Loading frequency file '%s'\n" % frequency_file)
        frequency_dict = read_frequency_vocab(frequency_file)
        logging.info("Done.\n")
    try:
        logging.info("Loading input file '%s'\n" % input_file)
        input_file = open(input_file, encoding='utf-8', mode='r')
    except IOError:
        logging.critical("Cannot open input file '%s'\n" % input_file)
        sys.exit(1)
    model = KeyedVectors.load_word2vec_format(model, binary=False)

    logging.info("Loading dictionary '%s'.\n" % dictionary)
    morpho = Morpho.load(dictionary)
    if not morpho:
        logging.critical("Cannot load dictionary from file '%s'\n" % dictionary)
        sys.exit(1)
    logging.info("Done.\n")

    lemmas = TaggedLemmas()
    # Out of vocabulary counter and buffer
    oov = 0.0
    oov_set = set()
    totalterms = 0.0
    output_buf = []
    logging.info("Processing lines...")
    hint_dict = dict() if frequency_file is not None else None
    for line in tqdm(list(input_file)):
        similarity_dict = {}
        line_list = line.split("\t")

        # Process first column
        # tokens_1col = [token.split() for token in line_list[0].split("/")]
        tokens_1col = line_list[0].split("/")
        tokens_1col[-1] = " ".join(tokens_1col[-1].split()[0:-1])  # Remove number
        for i in range(0, len(tokens_1col)):
            tokens_1col[i] = "_".join(tokens_1col[i].split())
        tokens_1col = list(set(tokens_1col))
        tokens_1col_dict = get_lemma_dict(tokens_1col, morpho, lemmas)
        for token_1col, token_lemmas in tokens_1col_dict.items():
            topn_1col = []
            tokens_1col_dict[token_1col] = [token_lemmas, topn_1col]  # {token : [[lemmas], [topn]]}

        # Process second column
        eval_tokens = []
        tokens_2col = []
        if len(line_list[1]) > 1:
            if line_list[1][0] == "[":
                continue
        groups_2col = line_list[1].split("/")
        for i in range(0, len(groups_2col)):
            if i < len(groups_2col) - 2:  # For future use
                eval_tokens.extend(groups_2col[i].split())
            tokens_2col.extend(groups_2col[i].split())

        # Process third column
        tokens_3col = []
        groups_3col = line_list[2].split("/")
        for i in range(0, len(groups_3col)):
            tokens_3col.extend(groups_3col[i].split())
        tokens_3col_dict = get_lemma_dict(tokens_3col, morpho, lemmas)
        for token_3col, token_lemmas in tokens_3col_dict.items():
            topn_3col = []
            tokens_3col_dict[token_3col] = [token_lemmas, topn_3col]  # {token : [[lemmas], [topn]]}

        # Calculate similarities between hint and agents
        # {token : [[lemmas], [topn]]}
        for hint, hint_lemmas in tokens_1col_dict.items():
            for agent, agent_lemmas in tokens_3col_dict.items():
                for hint_lemma in hint_lemmas[0]:
                    # record hint frequencies
                    if frequency_file is not None:
                        hint_dict[hint] = frequency_dict.get(hint_lemma, 0)
                    if not hint_lemma in model.vocab:
                        logging.error("Hint %s not in vocabulary" % hint_lemma)
                        oov_set.add(hint+"({})".format(hint_lemma))
                        continue
                    for agent_lemma in agent_lemmas[0]:
                        if not agent_lemma in model.vocab:
                            logging.error("Agent %s not in vocabulary" % agent_lemma)
                            oov_set.add(agent+"({})".format(agent_lemma))
                            continue
                        # Calculate similarity
                        similarity = model.similarity(hint_lemma, agent_lemma)
                        if agent in similarity_dict:
                            if similarity > similarity_dict[agent]:
                                similarity_dict[agent] = similarity
                        else:
                            similarity_dict[agent] = similarity
                totalterms += 1
                if agent not in similarity_dict:
                    oov += 1
        if not similarity_dict:
            output_buf.append([line, "Hint is out of vocabulary", FIXED_OOV_SCORE])
        else:
            # Sort cosine similarity between hint and agents
            similarity_list = sorted(iteritems(similarity_dict), key=lambda item: -item[1])
            sim_buf = ""
            if frequency_file is not None:
                for agent, sim in similarity_list:
                    sim_buf = sim_buf + agent + "({0:,d})".format(
                        int(frequency_dict.get(agent.lower(), 0))) + ":{0:.3f} ".format(sim)
            else:
                for agent, sim in similarity_list:
                    sim_buf = sim_buf + agent + ":" + "{0:.3f} ".format(sim)

            # Evaluate results - mean average precision
            avg_precision = 0.0
            j = 1
            for i in range(0, len(similarity_list)):
                if similarity_list[i][0] in tokens_2col:
                    avg_precision += j / (i + 1)
                    j += 1
            if j != 1:
                avg_precision /= (j - 1)
            output_buf.append([line, sim_buf, avg_precision])

    if sort:
        output_buf.sort(key=lambda x: float(x[2]),reverse=True)
    return output_buf, oov, totalterms, list(oov_set), hint_dict


def parse_keywords(raw_keyword):
    tokens_1col = [token.split() for token in raw_keyword.split("/")]
    tokens_1col[-1] = " ".join(tokens_1col[-1].split()[0:-1])  # Remove number
    for i in range(0, len(tokens_1col)):
        tokens_1col[i] = "_".join(tokens_1col[i].split())
    tokens_1col = list(set(tokens_1col))
    return get_lemma_dict(tokens_1col, morpho, lemmas)


def process_line(line, hint_occurence_count):
    # Remove column with unsorted sequence
    splitted = line.split('\t')
    col0 = splitted[0]
    col1 = splitted[1]
    if hint_occurence_count is not None:
        first_col_splitted = col0.split()
        num = first_col_splitted[-1]
        first_col_without_num = ' '.join(first_col_splitted[:-1])
        first_col_words = first_col_without_num.split('/')
        first_col_words_no_empty = list(filter(None, first_col_words))
        first_col_words_no_empty = list(map(str.strip, first_col_words_no_empty))
        col0 = '/'.join([word + "({0:,d})".format(hint_occurence_count['_'.join(word.split())]) \
                         for word in first_col_words_no_empty]) + " {}".format(num)

    return '\t'.join([col0, col1])


def print_to_cnwae(out_buff, oov, totalterms, oov_buf, ofile, hint_occurence_count=None):
    """

    :param out_buff:
    :param oov:
    :param totalterms:
    :param oov_buf:
    :param ofile:
    :param hint_occurence_count:
    """
    try:
        if not ofile.endswith(".cnwae"):
            ofile += ".cnwae"
        ofile = open(ofile, encoding='utf-8', mode='w')
    except IOError:
        logging.critical("Cannot open out_buff file '%s'\n" % out_buff)
        sys.exit(1)

    line_cnt = 0
    avg_precision_sum = 0.0
    for line in out_buff:
        # Uncomment this to NOT count OOV words in total MAP score
        # if line[2] - FIXED_OOV_SCORE < 1e-6:
        line_cnt += 1
        avg_precision_sum += line[2]

        ofile.write(process_line(line[0], hint_occurence_count) + "\t" + line[1] + "\tAverage precision: " + str(
            line[2]) + "\n")

    ofile.write("Total average precision: " + str(avg_precision_sum / line_cnt) + "\n")
    ofile.write("Out of vocabulary[%]: " + str(oov / totalterms * 100) + "\n")
    ofile.write("Out of vocabulary:\n" + '\n'.join(list(oov_buf)) + "\n")

    logging.info("Writing out_buff to %s" % ofile)
    logging.info("Out of vocabulary[%]: " + str(oov / totalterms * 100) + "\n")
    logging.info("Out of vocabulary: " + ', '.join(list(oov_buf)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="Input file in cnwa format.")
    parser.add_argument("-o", "--output",
                        required=True,
                        help="Output file.")
    parser.add_argument("-m", "--model",
                        required=True,
                        help="Trained word embedding model - needs to be compatible with gensim.")
    parser.add_argument("-d", "--dictionary",
                        default=DEFAULT_DICT,
                        help="Morphological analyzer.")
    parser.add_argument("-s", "--sort",
                        action="store_true",
                        default=True,
                        help="Sort out_buff.")
    parser.add_argument("-f", "--frequency",
                        default=None,
                        help="Dictionary of word occurences in corpus.")
    parser.add_argument("-l", "--logpath",
                        default=os.getcwd(),
                        help="Explicit setting of log folder path")

    args = parser.parse_args()
    init_logging(os.path.basename(sys.argv[0]).split(".")[0], logpath=args.logpath)
    out_buff, oov, totalterms, oov_buf, hint_occurence_count = cnwa2cnwae(args.input, args.dictionary,
                                                                          args.model, args.sort, args.frequency)
    print_to_cnwae(out_buff, oov, totalterms, oov_buf, args.output, hint_occurence_count)
