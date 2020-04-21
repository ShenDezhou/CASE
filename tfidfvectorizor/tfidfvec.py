import gzip
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
import numpy
#from nltk.stem.snowball import SnowballStemmer
from PUB_BiLSTM_BN import PUB_BiLSTM_BN

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='../case_corpus/AccuseBasisSentenceFullClean.utf8.gz', help='input file')
args = parser.parse_known_args()[0]

base = os.path.basename(args.file).split(".")[0]

bilstm = PUB_BiLSTM_BN()
print("load keras model:")
bilstm.loadKeras()
print("keras model loaded.")
segs = bilstm.cut(["我昨天去清华大学。", "他明天去北京大学，再后天去麻省理工大学。"])
print(segs)


MODELMAXLEN=1019
GPUBATCHSIZE=10000


class StemmedTfidfVectorizer(TfidfVectorizer):

    def __init__(self, stemmer, *args, **kwargs):
        super(StemmedTfidfVectorizer, self).__init__(*args, **kwargs)
        self.stemmer = stemmer

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(word) for word in analyzer(doc.replace('\n', ' ')))


with gzip.open(args.file, "rt", encoding='utf-8') as fr:
    with codecs.open("../bigram/"+base+'bigram.utf8', 'w', encoding='utf-8') as fb:
        result = []
        counter = 1
        resl = []
        for _line in fr:
            if len(_line) >= MODELMAXLEN:
                a = _line.strip().split('#')
                a = [line.strip() for line in a if len(line.strip()) > 0]
                for al in a:
                    if len(al) >= MODELMAXLEN:
                        b = al.split("。")
                        b = [line.strip() for line in b if len(line.strip()) > 0]
                        for bl in b:
                            if len(bl) >= MODELMAXLEN:
                                c=[]
                                while len(bl) > MODELMAXLEN:
                                    c.append(bl[:MODELMAXLEN])
                                    bl = bl[MODELMAXLEN:]
                                c.append(bl)
                                resl.extend(c)
                            else:
                                resl.append(bl)
                    else:
                        resl.append(al)
            else:
                resl.append(_line)

            while len(resl)>GPUBATCHSIZE:
                print('.')
                result.extend(bilstm.cut(resl[:GPUBATCHSIZE]))
                resl = resl[GPUBATCHSIZE:]
            counter += 1
        if len(resl)>0:
            result.extend(bilstm.cut(resl))
        print(len(result))

        vectorizer = TfidfVectorizer(
                                    analyzer='word',
                                    lowercase=False,
                                    ngram_range=(2, 2),
                                    max_features =None,)

        X = numpy.array(result)
        vectorizer.fit_transform(X)

        joblib.dump(vectorizer,'../model/tfidf%s.pkl.gz'%base, compress=('gzip', 3))
        dic = vectorizer.vocabulary_
        dic = sorted(dic.items(), key=lambda d: int(d[1]), reverse=True)
        voc = [i[0] for i in dic]
        print(len(voc))

        for w in voc:
            fb.write(w+'\n')

print('FIN')
