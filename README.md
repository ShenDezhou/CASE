# CASE
A bigram of Chinese Case Documents

# Introduction
First, segemnt all the sentences of the case document. Then use scikit-learn's tfidf vertorizor tool to get bigram of the corpus.
Thirdly, use bigram as features to generate tfidf document matrix. Finally output the vocabulary of the tfidf model and sort in revese order of Tfidf order.

# Structure
bigram:  
    folder to store bigrams of dev/small/full case documents.

case_corpus:  
    dev set: 10 lines do case documents with Accuse/Basis/Sentence.
    small set: less than 10000 case documents.
    full set: whole cases.

model:  
    tfidf model of small set.
    tfidf model of full set.



