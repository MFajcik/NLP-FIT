#!/usr/bin/python3
#Martin Fajcik
import scipy
from gensim.models import KeyedVectors

from preprocessing.tools import ipython_shell


def cdis(u,v):
    return scipy.spatial.distance.cosine(u, v)


if __name__ == "__main__":
    fasttextmodel = "/mnt/minerva1/nlp/projects/semantic_relatedness9/models/fasttext_sg_ns_d300_e10_c10_cwc/sg_ns_300_10_2017-04-30_04:17.vec"
    model = KeyedVectors.load_word2vec_format(fasttextmodel, binary=False)
    ipython_shell(locals())
