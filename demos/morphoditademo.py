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
    stopwords_file = "../../contrib/preprocessing/cz_stopwords/czechST.txt"
    stopwords = open(stopwords_file, encoding="utf-8", mode='r').read().splitlines()
    tokenizer.setText("pravdě je velmi zvláštní důkaz zrzavých genocida nedotkla jaké teď plány vyptával ho stále týž civilista a mírně něho usmál jedna věc jistá panamerické nevrátím projekt lady diany naprosto nesmyslná záležitost a následek polovina až čtvrtiny světové energie spotřebovány k jedinému účelu k vybudování obřího tunelu sever jih umožnit přístup k bohatství skrytému pod ledem k d.ol ům naftovým zdrojům lesům nalezištím přírodních surovin takového šílenství nezúčastním výpadek energie způsobit polovina lidstva i třetiny vymře nemyslím si stavba neobejde přinejmenším však dojde k velkému zpoždění žádám o politický azyl tím mohu desítkám milionů lidí zachránit život si lady diana najde náhradu")
    while tokenizer.nextSentence(forms, tranges):
        tagger.tag(forms, lemmas)
        for i in range(len(lemmas)):
            lemma = lemmas[i]
            lemmatized = tagger.getMorpho().rawLemma(lemma.lemma)
            if lemma.tag[0] != "Z" and forms[i] not in stopwords:
                processed_chunk.append(lemmatized)
    print(" ".join(processed_chunk))
