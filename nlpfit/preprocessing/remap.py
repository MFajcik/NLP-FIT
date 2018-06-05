# Author: Martin Fajcik
import sys
import pickle

import os

from nlpfit.preprocessing import read_word_chunks
from nlpfit.other.paralellism import DEFAULT_CHUNK_SIZE
from tqdm import tqdm


def remap_chunk(f1, f2):
    result = dict()
    Aoffset = Boffset = 0
    ch_size = DEFAULT_CHUNK_SIZE // 100
    chunks = os.path.getsize(f1) // ch_size
    # d_f1 = open(f1, "rb")
    # d_f2 = open(f2, "rb")
    with tqdm(total=chunks + 1) as pbar:
        gen_chunksA = read_word_chunks(f1, ch_size)
        gen_chunksB = read_word_chunks(f2, ch_size)
        chunk_a = next(gen_chunksA)
        chunkB = next(gen_chunksB)
        chunk_a = bytes(chunk_a, 'utf-8')
        chunkB = bytes(chunkB, 'utf-8')
        baseoffsetA = 0
        baseoffsetB = 0
        origA = chunk_a
        origB = chunkB
        # dbg_lastA = 0
        # dbg_lastB = 0
        # i = 0
        # TODO: test with assert uncommentedd
        while chunk_a:
            # We always want to read at least 1 character
            a_nonempty = False
            b_nonempty = False
            for c in chunk_a:
                if str.isspace(chr(c)) and a_nonempty:
                    break
                Aoffset += 1
                a_nonempty = True

            for c in chunkB:
                if str.isspace(chr(c)) and b_nonempty:
                    break
                Boffset += 1
                b_nonempty = True

            result[Aoffset + baseoffsetA] = Boffset + baseoffsetB
            # DEBUG:
            # d_f1.seek(dbg_lastA+baseoffsetA)
            # d_f2.seek(dbg_lastB+baseoffsetB)
            # readedA= d_f1.read(Aoffset - dbg_lastA)
            # readedB = d_f2.read(Boffset-dbg_lastB)
            # print("{}:{} == {}".format(i,str(readedA)[2:-1].strip()+"[{}]".format(readedA.decode().strip()),str(readedB)[2:-1].strip()+"[{}]".format(readedB.decode().strip()) ))
            # i+=1
            # dbg_lastA=Aoffset
            # dbg_lastB=Boffset
            Aoffset += 1
            Boffset += 1
            chunk_a = origA[Aoffset:]
            chunkB = origB[Boffset:]
            if not chunk_a:
                try:
                    chunk_a = next(gen_chunksA)
                    chunk_a = bytes(chunk_a, 'utf-8')
                    origA = chunk_a
                    baseoffsetA += Aoffset
                    Aoffset = 0
                    dbg_lastA = 0
                    pbar.update(1)
                except StopIteration:
                    break
            if not chunkB:
                chunkB = next(gen_chunksB)
                chunkB = bytes(chunkB, 'utf-8')
                origB = chunkB
                baseoffsetB += Boffset
                Boffset = 0
                dbg_lastB = 0
    assert (not chunkB)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 2:
        #     input_corpus = sys.argv[1]
        #     output_corpus = sys.argv[2]
        input_corpus = "../../corpus_data/ebooks_corpus_CZ/e_knihy_preprocessed.txt"
        output_corpus = "../../corpus_data/ebooks_corpus_CZ/e_knihy_preprocessed_nolemma.txt"
        d = remap_chunk(input_corpus, output_corpus)
        with open("mapping_{}_to_{}.pkl".format(os.path.basename(input_corpus), os.path.basename(output_corpus)),
                  mode="wb") as mapfile:
            pickle.dump(d, mapfile, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise BaseException("Not enough arguments! Expecting 2 files to map.")
