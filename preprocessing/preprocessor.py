# PyVersion: 3.6
# Authors: Martin Fajčík
# Some parts are based on code of Viliam Samuel Hošťák

import itertools
import os
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import List
from ufal_morphodita import Morpho, TaggedLemmas

from tqdm import tqdm
from ufal import morphodita

from other.logging_config import logger_stub
from preprocessing.io import read_word_chunks_with_offset
from preprocessing.tools import DotDict, find_textfile_split_points

string_list = List[str]

def get_lemma_dict(words: string_list, morpho, lemmas):
    """
    :param words: list with word to analyze
    :param morpho: morphodita morphological analyzator
    :param lemmas: tagged lemmas
    :return: dictionary with token lemmas
    """

    lemma_dict = {}
    for word in words:
        tokens_lemma_list = []
        lemmas_list = []
        if word.find("_"):
            token_list = word.split("_")
        else:
            token_list = [word]
        for token in token_list:
            morpho.analyze(token, 0, lemmas)
            lemma_list = []
            for lemma in lemmas:
                lemma_list.append(morpho.rawLemma(lemma.lemma).lower())
            lemma_list = list(set(lemma_list))
            tokens_lemma_list.append(lemma_list)
        for lemma in itertools.product(*tokens_lemma_list):
            lemmas_list.append("_".join(lemma))
        lemma_dict[word] = lemmas_list
    return lemma_dict


def lemmatize_list(words: string_list, dictionary: str) -> string_list:
    """
    :param words: list of words to be lemmatized
    :param dict: word dictionary, used for lemmatization
    :return: list of lemmatized words
    """

    morpho = Morpho.load(dictionary)
    lemmas = TaggedLemmas()
    lemma_dict = get_lemma_dict(words, morpho, lemmas)
    from itertools import chain
    return list(chain.from_iterable(list(lemma_dict.values())))


def merge_files(tmpdir, ofile):
    if not ofile:
        return
    regexp = re.compile(r"^(\d+)_" + os.path.basename(ofile) + r"$")
    tmpdir_files = os.listdir(tmpdir)
    valid_files = list(map(lambda x: bool(regexp.match(x)), tmpdir_files))
    files = list(itertools.compress(tmpdir_files, valid_files))
    to_be_merged_files = sorted(files, key=lambda a: int(re.findall(r"\d+", a)[0]))

    with open(os.path.join(tmpdir, to_be_merged_files[0]), mode="a") as destination:
        for i in tqdm(list(range(1, len(to_be_merged_files)))):
            with open(os.path.join(tmpdir, to_be_merged_files[i]), mode='r') as source:
                shutil.copyfileobj(source, destination)
    shutil.move(os.path.join(tmpdir, to_be_merged_files[0]), ofile)


# 10MB chunks per process
def _worker(idx, s_offset, e_offset, tmpdir, opts, logger, chunk_size=10485760):
    ofilename = "{}_".format(idx) + os.path.basename(opts.ofile) if opts.ofile else None
    wordcounter = Counter() if opts.count_words else None

    preprocessing_tools = DotDict()
    preprocessing_tools.tagger = morphodita.Tagger.load(opts.tagger_file)
    preprocessing_tools.forms = morphodita.Forms()
    preprocessing_tools.lemmas = morphodita.TaggedLemmas()
    preprocessing_tools.tokens = morphodita.TokenRanges()
    preprocessing_tools.tokenizer = preprocessing_tools.tagger.newTokenizer()
    preprocessing_tools.stopwords = open(opts.stopwords_file, encoding="utf-8", mode='r').read().splitlines()
    if opts.ofile:
        with open(os.path.join(tmpdir, ofilename), mode="w") as of:
            for chunk in read_word_chunks_with_offset(opts.ifile, chunk_size, s_offset, e_offset):
                preprocessed, wordcounter = process_text(chunk, preprocessing_tools, opts, logger, wordcounter)
                of.write(preprocessed)
    else:
        for chunk in read_word_chunks_with_offset(opts.ifile, chunk_size, s_offset, e_offset):
            preprocessed, wordcounter = process_text(chunk, preprocessing_tools, opts, logger, wordcounter)
    return wordcounter


def process_text(chunk: str, tools, opts, logger, wordcounter) -> (str, dict):
    tools.tokenizer.setText(chunk)
    processed_chunk = ""
    while tools.tokenizer.nextSentence(tools.forms, tools.tokens):
        tools.tagger.tag(tools.forms, tools.lemmas)
        for i in range(len(tools.lemmas)):
            lemma = tools.lemmas[i]

            # Adds language tags (i.e. 'jet-1_^(pohybovat_se,_ne_však_chůzí)'
            # See http://ufal.mff.cuni.cz/morphodita/users-manual#tagger_output_formats
            if opts.tag_words:
                output = lemma.lemma
            elif opts.lemmatize_words:
                output = tools.tagger.getMorpho().rawLemma(lemma.lemma)
            else:
                output = tools.forms[i]

            # Filter stopwords and punctuation
            if not opts.remove_stop_words or\
                (lemma.tag[0] != "Z" and tools.forms[i] not in tools.stopwords):
                output = output.lower()
                processed_chunk += "%s " % (output)
                if opts.count_words:
                    wordcounter[output] = wordcounter.get(output, 0) + 1
    return processed_chunk, wordcounter

default_tagger_file = "../contrib/preprocessing/cz_morphodita/models/czech-morfflex-pdt-160310.tagger"
default_stopwords_file = "../contrib/preprocessing/cz_stopwords/czechST.txt"


def preprocess_file(ifile: str, ofile, lemmatize_words: bool = True,
                    remove_stop_words : bool =True, tag_words: bool = False, count_words: bool = False, logger=None,
                    num_of_processes: int = 8, tagger_file=default_tagger_file,
                    stopwords_file=default_stopwords_file, tmpdir="tmp") -> (float, dict):
    """
    Universal function for parallel file preprocessing
    :param remove_stop_words:
    :param ifile: input file name
    :param ofile: out_buff file name
    :param lemmatize_words:
    :param tag_words: if option is True, words are tagged with their POS tags
    :param count_words:
    :param logger: custom logger
    :param num_of_processes: number of parallel processes used in preprocessing
    :param tagger_file:
    :param stopwords_file:
    :param tmpdir: name of TMP dir used during preprocessing, do NOT delete this dir during the preprocess stage
    :return time spent by preprocessing, dict of word counts if specified
    """
    opts = DotDict()
    opts.ifile = ifile
    opts.ofile = ofile
    opts.lemmatize_words = lemmatize_words
    opts.tag_words = tag_words
    opts.count_words = count_words
    opts.tagger_file = tagger_file
    opts.stopwords_file = stopwords_file
    opts.tmpdir = tmpdir
    opts.remove_stop_words = remove_stop_words

    if logger is None:
        logger = logger_stub()
    logger.info("Starting preprocessing")
    start_time = time.time()
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    filesize = os.path.getsize(ifile)
    logger.info("File size %d" % filesize)
    offsets = find_textfile_split_points(ifile, num_of_processes)
    # append end of file
    offsets.append(filesize)
    logger.info("Starting %d processes in parallel..." % num_of_processes)
    futures = [None] * num_of_processes
    with ProcessPoolExecutor(max_workers=num_of_processes) as p:
        for idx in tqdm(list(range(num_of_processes))):
            start_offset = offsets[idx]
            end_offset = offsets[idx + 1]
            futures[idx] = p.submit(_worker, idx, start_offset, end_offset, tmpdir, opts, logger)

    wordcounter = sum(list(map(lambda x: x.result(), futures)), Counter()) if opts.count_words else None
    logger.info("Merging files...")
    merge_files(tmpdir, opts.ofile)
    logger.info("Cleaning...")
    shutil.rmtree(tmpdir)
    duration = time.time() - start_time
    logger.info("Preprocessing finished after {} seconds.".format(duration))
    return duration, wordcounter



def count_words(input_file, output_file, to_sort=True, lemmatize=True, remove_stop_words=False, num_of_processes=8):
    """
    The words are lemmatized but stop words are not removed by default during counting
    :param input_file:
    :param output_file:
    :param to_sort:
    """
    duration,wordcounter = preprocess_file(input_file, None, lemmatize_words=lemmatize, remove_stop_words=remove_stop_words, count_words=True, num_of_processes=num_of_processes)
    print("Counting finished in {} seconds.".format(duration))

    if to_sort:
        import operator
        wordcounter = sorted(wordcounter.items(), key=operator.itemgetter(1), reverse=True)

    with open(output_file, 'w') as outf:
        for key, value in wordcounter:
            outf.write("%s %d\n" % (key, value))
