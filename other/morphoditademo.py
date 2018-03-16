#!/usr/bin/python3
from ufal import morphodita

if __name__ == "__main__":
    testfilecontent = open("../../corpus_data/ebooks_corpus_CZ/few_sentences.txt").read()
    tf = "../../contrib/preprocessing/cz_morphodita/models/czech-morfflex-pdt-160310.tagger"
    tagger = morphodita.Tagger.load(tf)
    forms = morphodita.Forms()
    lemmas = morphodita.TaggedLemmas()
    tranges = morphodita.TokenRanges()
    tokenizer = tagger.newTokenizer()
    processed_chunk = []
    tokenizer.setText(testfilecontent)
    while tokenizer.nextSentence(forms, tranges):
        tagger.tag(forms, lemmas)
        for i in range(len(lemmas)):
            lemmatized = tagger.getMorpho().rawLemma(lemmas[i].lemma)
            processed_chunk.append(lemmatized)
    print(" ".join(processed_chunk))
