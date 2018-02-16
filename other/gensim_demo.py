#!/usr/bin/python3
#Martin Fajcik
import scipy
from gensim.models import KeyedVectors

#Function taken from tensorflow tutorials project
def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)

def cdis(u,v):
    return scipy.spatial.distance.cosine(u, v)

fasttextmodel = "/mnt/minerva1/nlp/projects/semantic_relatedness9/models/fasttext_sg_ns_d300_e10_c10_cwc/sg_ns_300_10_2017-04-30_04:17.vec"
model = KeyedVectors.load_word2vec_format(fasttextmodel, binary=False)
_start_shell(locals())