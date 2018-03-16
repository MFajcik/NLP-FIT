# Taken from
# https://github.com/aubry74/visual-word2vec
import gensim.models as w2v
import sklearn.decomposition as dcmp
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
import re
import nltk
from gensim.models import KeyedVectors

"""
Visualize Word2Vec
Word2vec + PCA + Clustering
"""
__author__ = "Aubry Cholleton"

# model_path = '/mnt/minerva1/nlp/projects/semantic_relatedness10/models/w2v/cbow_ns_300_5_2017-12-20_23:29.vec'
model_path = '/home/ifajcik/word2vec/trainedmodels/tf_w2vopt_ebooks_lemmatized_and_stemmed_32t_30negs_6window/model.vec'


class SemanticMap:
    def __init__(self, model_path):
        print('Loading model ...')
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        print('Ready')

    def __split_words(self, input_string):
        return re.findall(r"[\w']+", input_string)

    def __clean_words(self, words):
        clean_words = []
        for w in words:
            clean_words.append(re.sub(r'\W+', '', w.lower()))
        return clean_words

    def __remove_stop_words(self, words):
        return [w for w in words if not w in nltk.corpus.stopwords.words('english')]

    def __get_non_compositional_entity_vector(self, entity):
        return self.model[entity[0]]

    def __get_compositional_entity_vector(self, entity):
        array = np.array(self.model[entity[0]])
        for ind in range(1, len(entity)):
            array = array + np.array(self.model[entity[ind]])
        return array / len(entity)

    def __get_vector(self, term):
        words = self.__clean_words(self.__split_words(term))

        if len(words) < 1:
            print('All the terms have been filtered.')
        if len(words) == 1:
            try:
                return self.__get_non_compositional_entity_vector(words)
            except:
                print('Out-of-vocabulary entity')
                raise
        elif len(words) < 4:
            try:
                return self.__get_compositional_entity_vector(words)
            except:
                print('Out-of-vocabulary word in compositional entity')
                raise
        else:
            print('Entity is too long.')

    def __reduce_dimensionality(self, word_vectors, dimension=2):
        data = np.array(word_vectors)
        pca = dcmp.PCA(n_components=dimension)
        pca.fit(data)
        return pca.transform(data)

    def cluster_results(self, data, threshold=0.13):
        return hcluster.fclusterdata(data, threshold, criterion="distance")

    def map_words(self, words, sizes):
        final_words = []
        final_sizes = []
        vectors = []

        for word in words:
            try:
                vect = self.__get_vector(word)
                vectors.append(vect)
                if sizes is not None:
                    final_sizes.append(sizes[words.index(word)])
                final_words.append(word)
            except Exception:
                print('not valid ' + word)

        return vectors, final_words, final_sizes

    def plot(self, vectors, lemmas, clusters, sizes=80):
        if sizes == []:
            sizes = 80
        plt.scatter(vectors[:, 0], vectors[:, 1], s=sizes, c=clusters)
        for label, x, y in zip(lemmas, vectors[:, 0], vectors[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.show()

    def map_cluster_plot(self, words, sizes, threshold):
        vectors, words, sizes = self.map_words(words, sizes)
        vectors = self.__reduce_dimensionality(vectors)
        clusters = self.cluster_results(vectors, threshold)
        self.plot(vectors, words, clusters, sizes)

    def print_results(self, words, clusters):
        print(words)
        print(clusters.tolist())


def cli(mapper_cli):
    # while True:
    #     line = input('Enter words or MWEs > ')
    #     if line == 'exit':
    #         break
    line = "pes, mačka, kocour, král, královna"
    mapper_cli.map_cluster_plot(line.split(','), None, 0.2)


if __name__ == "__main__":
    mapper = SemanticMap(model_path)
    cli(mapper)
