import os
import pickle

import sys

import numpy as np
import scipy
from gensim.models import KeyedVectors
from sortedcontainers import SortedList
from tqdm import tqdm

from preprocessing.tools import DotDict, ipython_shell

opts = DotDict()
opts.window_size = 5
opts.average_word_bytelen = 100


def format_contexts(contexts, positions, grads=None):
    class color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    r = []
    if grads is not None:
        for c, p, g in zip(contexts, positions, grads):
            posinfo = color.RED + "[{}]".format(p) + color.END
            gradinfo = color.CYAN + " [{0:.3f}]".format(g) + color.END
            prefix = posinfo + gradinfo
            left_nonwindow = ' '.join(c[:len(c) // 4]) + " "
            left_window = color.YELLOW + ' '.join(c[len(c) // 4:len(c) // 2]) + color.END
            middle = color.BOLD + " {} ".format(c[len(c) // 2]) + color.END
            right_window = color.YELLOW + ' '.join(c[len(c) // 2 + 1: len(c) // 2 + len(c) // 4 + 1]) + color.END + " "
            right_nonwindow = ' '.join(c[len(c) // 2 + len(c) // 4 + 1:])
            r.append("{0}: {1}{2}{3}{4}{5}".format(
                prefix,
                left_nonwindow,
                left_window,
                middle,
                right_window,
                right_nonwindow
            ))
    else:
        for c, p in zip(contexts, positions):
            prefix = color.RED + "[{}]".format(p) + color.END
            left_nonwindow = ' '.join(c[:len(c) // 4]) + " "
            left_window = color.YELLOW + ' '.join(c[len(c) // 4:len(c) // 2]) + color.END
            middle = color.BOLD + " {} ".format(c[len(c) // 2]) + color.END
            right_window = color.YELLOW + ' '.join(c[len(c) // 2 + 1: len(c) // 2 + len(c) // 4 + 1]) + color.END + " "
            right_nonwindow = ' '.join(c[len(c) // 2 + len(c) // 4 + 1:])
            r.append("{0}: {1}{2}{3}{4}{5}".format(
                prefix,
                left_nonwindow,
                left_window,
                middle,
                right_window,
                right_nonwindow
            ))

    return r


def word_context_formatted(topgrads, toppositions, word_index, corpus, window_size=opts.window_size, *args, **kwargs):
    contexts, positions, grads = find_word_contexts(toppositions, word_index, corpus, grads=topgrads,
                                                    window_size=window_size * 2, *args, **kwargs)
    return format_contexts(contexts, positions, grads)


def find_word_contexts(toppositions, word_index, corpus, window_size=opts.window_size * 2,
                       average_word_bytelen=opts.average_word_bytelen, grads=None):
    positions = toppositions[word_index]
    grads = grads[word_index] if grads is not None else None
    size = window_size * average_word_bytelen  # 10 for average word
    contexts = []
    with open(corpus, mode="rb") as data:
        for pos in positions:
            data.seek(max(int(pos - size), 0))
            chunk1 = data.read(size).decode("utf-8", errors="ignore").split()
            chunk2 = data.read(size).decode("utf-8", errors="ignore").split()
            contexts.append(chunk1[-(window_size + 1):] + chunk2[:window_size])
    return contexts, positions, grads


def word_context_cross_formatted(top_grad_pos, x, y, corpus, window_size=opts.window_size, *args, **kwargs):
    contexts, positions, grads = find_word_context_cross(top_grad_pos, x, y, corpus,
                                                         window_size=window_size, *args, **kwargs)
    return format_contexts(contexts, positions, grads)


def get_nearby_words(corpus, pos, size=opts.average_word_bytelen, window_size=5, remove_middle_word = True):
    with open(corpus, mode="rb") as data:
        data.seek(max(int(pos - size*2), 0))
        chunk1 = data.read(size*2).decode("utf-8", errors="ignore").split()
        chunk2 = data.read(size*2).decode("utf-8", errors="ignore").split()
        nearby = (chunk1[-(window_size + 1):] + chunk2[:window_size])
        if remove_middle_word:
            del nearby[len(nearby)//2]
        wide_context = (chunk1[-(window_size*2 + 1):] + chunk2[:window_size*2])
    return nearby,wide_context


def getEmbedding(x, model):
    try:
        return model[x]
    except KeyError:
        return np.zeros(shape=(model.vector_size,), dtype=np.float32)


def average_words(words, model):
    embeddings = list(map(lambda x: getEmbedding(x, model), words))
    return np.mean(embeddings, axis=0)


def concat_words(words, model):
    embeddings = list(map(lambda x: getEmbedding(x, model), words))
    return np.concatenate(embeddings)


class DistancePosContext():
    def __init__(self, distance, x_pos, y_pos, x_context, y_context):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_context = x_context
        self.y_context = y_context
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __eq__(self, other):
        return self.distance == other.distance

    def __gt__(self, other):
        return self.distance > other.distance

    def __str__(self):
        return str(self.distance)

def cosine_distance_method(pos_x,pos_y,vec_method = None):
    u, x_context = vec_method(pos_x)
    v, y_context = vec_method(pos_y)
    distance = scipy.spatial.distance.cosine(u, v)
    return distance, x_context, y_context


def find_most_vec_similar(positions_x, positions_y, context_vec_method=None,model=None, distance_method = cosine_distance_method, topn=50):
    ladder = SortedList()
    i = 0
    for pos_x in tqdm(positions_x):
        i+=1
        ladder = SortedList(ladder[:topn])
        # if i > 10:
        #     break
        for pos_y in positions_y:
            distance, x_wide_context, y_wide_context = distance_method(pos_x,pos_y,vec_method=context_vec_method)
            ladder.add(DistancePosContext(distance, pos_x, pos_y, x_wide_context, y_wide_context))
    return ladder

def find_word_context_cross(top_grad_pos, x_ind, y_ind, corpus, method="average", model=None, x_word=None,
                            y_word=None, window_size=opts.window_size,
                            average_word_bytelen=opts.average_word_bytelen, grads=None):
    methods = ["average", "concatenate", "distance_correlation", "wmdistance"]
    contexts = []
    distances = []
    positions = []
    positions_x, positions_y = top_grad_pos[x_ind], top_grad_pos[y_ind]
    #grad_x, grad_y = (grads[x_ind], grads[y_ind]) if grads is not None else (None, None)

    if method not in methods:
        raise ("Only methods {} are supported.".format(methods))

    if method == "average":
        def get_averaged_context_vector(pos):
            nearby_words, context = get_nearby_words(corpus, pos, window_size=window_size, size=average_word_bytelen)
            return average_words(nearby_words, model), context

        similar = find_most_vec_similar(positions_x, positions_y, context_vec_method=get_averaged_context_vector)
        for s in similar:
            contexts.append(s.x_context)
            contexts.append(s.y_context)
            positions.append(s.x_pos)
            positions.append(s.y_pos)
            distances.append(s.distance)
            distances.append(s.distance)

    elif method == "concatenate":
        def get_concatenated_context_vector(pos):
            nearby_words, context = get_nearby_words(corpus, pos)
            return concat_words(nearby_words, model), context

        similar = find_most_vec_similar(positions_x, positions_y, context_vec_method=get_concatenated_context_vector)
        for s in similar:
            contexts.append(s.x_context)
            contexts.append(s.y_context)
            positions.append(s.x_pos)
            positions.append(s.y_pos)
            distances.append(s.distance)
            distances.append(s.distance)

    elif method == "wmdistance":
        assert model is not None

        def WMD_distance_method(pos_x, pos_y,vec_method = None):
            x_context, x_wide_context = get_nearby_words(corpus, pos_x, window_size=window_size, size=average_word_bytelen)
            y_context, y_wide_context = get_nearby_words(corpus, pos_y, window_size=window_size, size=average_word_bytelen)
            distance = model.wv.wmdistance(x_context,y_context)
            return distance, x_wide_context, y_wide_context

        similar = find_most_vec_similar(positions_x, positions_y,distance_method=WMD_distance_method)
        for s in similar:
            contexts.append(s.x_context)
            contexts.append(s.y_context)
            positions.append(s.x_pos)
            positions.append(s.y_pos)
            distances.append(s.distance)
            distances.append(s.distance)
    elif method == "distance_correlation":
        assert model is not None

        def pearson_coeff_emb_method(pos_x, pos_y,vec_method = None):
            x_context, x_wide_context = get_nearby_words(corpus, pos_x, window_size=window_size, size=average_word_bytelen)
            y_context, y_wide_context = get_nearby_words(corpus, pos_y, window_size=window_size, size=average_word_bytelen)

            emb_projection = lambda x: getEmbedding(x, model)
            x_embeddings = list(map(emb_projection, x_context))
            y_embeddings = list(map(emb_projection, y_context))
            distance = scipy.spatial.distance.correlation(np.concatenate(x_embeddings),np.concatenate(y_embeddings))

            return distance, x_wide_context, y_wide_context

        similar = find_most_vec_similar(positions_x, positions_y,distance_method=pearson_coeff_emb_method)
        for s in similar:
            contexts.append(s.x_context)
            contexts.append(s.y_context)
            positions.append(s.x_pos)
            positions.append(s.y_pos)
            distances.append(s.distance)
            distances.append(s.distance)

    return contexts, positions, distances


if __name__ == "__main__":
    ROOTDIR = "../../.."
    gdir = os.path.join(ROOTDIR, "corpus_data/gradient_ladders")
    mdir = os.path.join(ROOTDIR, "trainedmodels")
    models = ["ebooks", "cwcdemo"]

    gradient_positions_ladder_name = \
        gradient_ladder_name = \
        corpus = w2i_fn = \
        model_fn = None

    if len(sys.argv) < 2 or sys.argv[1] not in models:
        print("Available demo models are: " + ' '.join(models))
        raise BaseException("Unsupported demo model.")

    if sys.argv[1] == models[0]:
        gradient_positions_ladder_name = os.path.join(gdir, "ebooks/toppositions_ebooks.pkl")
        gradient_ladder_name = os.path.join(gdir, "ebooks/topgrads_ebooks.pkl")
        corpus = os.path.join(ROOTDIR, "corpus_data/ebooks_corpus_CZ/e_knihy_preprocessed.txt")
        w2i_fn = os.path.join(gdir, "ebooks/w2i_ebooks.pkl")
        model_fn = os.path.join(mdir, "tf_w2vopt_ebooks_gradient_ladder/ebooks_10_gl_model.vec")
    elif sys.argv[1] == models[1]:
        gradient_positions_ladder_name = os.path.join(gdir, "cwc50m/toppositions_CWC50M.pkl")
        gradient_ladder_name = os.path.join(gdir, "cwc50m/topgrads_CWC50M.pkl")
        w2i_fn = os.path.join(gdir, "cwc50m/w2i_ebooks.pkl")
        corpus = "/home/ifajcik/deep_learning/word2vec/corpus_data/cwc_corpus2011/cwc50megs"
        model_fn = os.path.join(mdir, "tf_w2vopt_ebooks_gradient_ladder/ebooks_10_gl_model.vec")

    with open(gradient_positions_ladder_name, 'rb') as gradient_pos_input:
        with open(gradient_ladder_name, 'rb') as gradient_input:
            with open(w2i_fn, 'rb') as w2i_f:
                print("Loading gradient positions...")
                top_grad_pos = pickle.load(gradient_pos_input)
                print("Loading gradients...")
                # top_grads = pickle.load(gradient_input)
                top_grads = None
                print("Loading vocab dict...")
                w2i = pickle.load(w2i_f)


                def show(x, n=50):
                    l = word_context_formatted(top_grad_pos, w2i[x], corpus, grads=top_grads)
                    [print(l[i]) for i in range(n)]
                    return None


                def show_cross(x, y, n=50):
                    print("Loading model...")
                    # model = KeyedVectors.load_word2vec_format(model_fn, binary=False)
                    # with open("kwmodel_ebooks.pkl",mode="wb") as f:
                    #     pickle.dump(model,f,protocol=pickle.HIGHEST_PROTOCOL)
                    # exit()

                    with open("kwmodel_ebooks.pkl", mode="rb") as f:
                        model = pickle.load(f)
                    print("Finding relations...")
                    l = word_context_cross_formatted(top_grad_pos, w2i[x], w2i[y], corpus, model=model, x_word=x,
                                                     grads=top_grads,
                                                     y_word=y, method  = sys.argv[2])
                    [print(l[i]) for i in range(n)]

                    return l


                cross = show_cross("hudba", "démon")
                ipython_shell(locals())
