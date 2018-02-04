# PyVersion: 3.6
# Authors: Martin Fajčík, Viliam Samuel Hošťák
# Last edited: 02.02.2018

import itertools
import math
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

from evaluation.logging_config import logger_stub
from nlpfit.preprocessing.io import read_word_chunks_with_offset

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


def getCharacter(f):
    try:
        return str(f.read(1),encoding="utf-8")
    except UnicodeDecodeError:
        return "ERROR"

def find_textfile_split_points(file: str, n: int) -> List[int]:
    """
    Find suitable spots for file to be splitted into n chunks.
    The suitable spots are nearest following whitespaces
    :param file:
    :param n:
    :return:
    """
    filesize = os.path.getsize(file)
    file_chunk_size = int(math.floor(filesize / float(n)))
    offsets = [None]*n
    with open(file,mode="rb") as f:
        offsets[0]=0
        for idx in range(1,n):
            offset = idx * file_chunk_size
            f.seek(offset)
            character = getCharacter(f)
            while not (character.isspace() or character==''):
                character = getCharacter(f)
            offsets[idx]=f.tell()-1
    return offsets

default_tagger_file =    "../contrib/contrib/preprocessing/cz_morphodita/models/czech-morfflex-pdt-160310.tagger"
default_stopwords_file = "../contrib/preprocessing/cz_stopwords/czechST.txt"

# TODO: Test & Doc
def preprocess_file(ifile: str, ofile: str, lemmatize_words: bool = True, stem_words: bool = True,
                    remove_stop_words=True, tag_words: bool = False, count_words: bool = False, logger=None,
                    num_of_processes: int = 8, tagger_file=default_tagger_file,
                    stopwords_file=default_stopwords_file, tmpdir="tmp") -> (float, dict):
    """
    Universal function for parallel file preprocessing
    :param remove_stop_words:
    :param ifile:
    :param ofile:
    :param lemmatize_words:
    :param stem_words:
    :param tag_words:
    :param count_words:
    :param logger:
    :param num_of_processes:
    :param tagger_file:
    :param stopwords_file:
    :param tmpdir:
    :return time spent by lemmatization, dict of word counts if specified
    """
    opts = DotDict()
    opts.ifile = ifile
    opts.ofile = ofile
    opts.lemmatize_words = lemmatize_words
    opts.stem_words = stem_words
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
    offsets = find_textfile_split_points(ifile,num_of_processes)
    #append end of file
    offsets.append(filesize)
    logger.info("Starting %d processes in parallel..." % num_of_processes)
    futures = [None]*num_of_processes
    with ProcessPoolExecutor(max_workers=num_of_processes) as p:
        for idx in tqdm(list(range(num_of_processes))):
            start_offset = offsets[idx]
            end_offset = offsets[idx+1]
            futures[idx]= p.submit(_worker, idx, start_offset, end_offset, tmpdir, opts, logger)

    wordcounter = sum(list(map(lambda x: x.result(),futures)),Counter()) if opts.count_words else None
    logger.info("Merging files...")
    merge_files(tmpdir,opts.ofile)
    logger.info("Cleaning...")
    shutil.rmtree(tmpdir)
    duration = time.time() - start_time
    logger.info("Preprocessing finished after {} seconds.".format(duration))
    return duration, wordcounter

def merge_files(tmpdir, ofile):
    regexp = re.compile(r"^(\d+)_"+ofile+r"$")
    tmpdir_files = os.listdir(tmpdir)
    valid_files = list(map(lambda x: bool(regexp.match(x)),tmpdir_files))
    files = list(itertools.compress(tmpdir_files, valid_files))
    to_be_merged_files = sorted(files,key=lambda a:int(re.findall(r"\d+",a)[0]))

    with open(os.path.join(tmpdir,to_be_merged_files[0]), mode="a") as destination:
        for i in tqdm(list(range (1,len(to_be_merged_files)))):
            with open(os.path.join(tmpdir,to_be_merged_files[i]), mode='r') as source:
                shutil.copyfileobj(source, destination)
    shutil.move(os.path.join(tmpdir,to_be_merged_files[0]), ofile)

class DotDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val
    def __getstate__(self):
        pass
    def __setstate__(self, state):
        pass

#10MB chunks per process
def _worker(idx, s_offset, e_offset, tmpdir, opts, logger,chunk_size = 10485760):
    ofile = "{}_".format(idx) + opts.ofile
    wordcounter = Counter() if opts.count_words else None

    preprocessing_tools = DotDict()
    preprocessing_tools.tagger = morphodita.Tagger.load(opts.tagger_file)
    preprocessing_tools.forms = morphodita.Forms()
    preprocessing_tools.lemmas = morphodita.TaggedLemmas()
    preprocessing_tools.tokens = morphodita.TokenRanges()
    preprocessing_tools.tokenizer = preprocessing_tools.tagger.newTokenizer()
    preprocessing_tools.stopwords = open(opts.stopwords_file, encoding="utf-8", mode='r').read().splitlines()

    with open(os.path.join(tmpdir, ofile), mode="w") as of:
        for chunk in read_word_chunks_with_offset(opts.ifile, chunk_size, s_offset, e_offset):
            preprocessed,wordcounter = process_text(chunk, preprocessing_tools, opts, logger, wordcounter)
            of.write(preprocessed)
    return wordcounter


def process_text(chunk: str, tools, opts, logger, wordcounter) -> (str,dict):
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
                output=tools.forms[i]

            # Filter stopwords and punctuation
            if not(opts.stem_words and opts.remove_stop_words) or\
               (opts.stem_words and lemma.tag[0] != "Z") or\
               (opts.remove_stop_words and tools.forms[i] not in tools.stopwords) or\
               (opts.stem_words and opts.remove_stop_words and
                   (lemma.tag[0] != "Z" or tools.forms[i] not in tools.stopwords)):
                output = output.lower()
                processed_chunk += "%s " % (output)
                if opts.count_words:
                    wordcounter[output] = wordcounter.get(output,0)+1
    return processed_chunk,wordcounter