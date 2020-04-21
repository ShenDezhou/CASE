import codecs
import logging
import os

import falcon
from falcon_cors import CORS
import json
import waitress


logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
l = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['http://localhost:8081'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )

class BigResource:


    bilstm=None
    uutrtcrf=None
    bigramdic={}

    def __init__(self):
        for root, dirs, files in os.walk('../case_bigram/'):
            for file in files:
                with codecs.open(root+"/"+file,'r', encoding='utf-8') as f:
                    xlines = f.readlines()
                    xlines = [line.strip() for line in xlines]
                    print(len(xlines))
                    dic = dict(zip(xlines, range(len(xlines)+1,1,-1)))
                    for i in dic.items():
                        self.bigramdic[i[0]] = self.bigramdic.get(i[0],0)+int(i[1])
            print(len(self.bigramdic))
            # for k in self.bigramdic:
            #     print(k)

    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.set_header('Access-Control-Allow-Origin', 'http://localhost:8081')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        sentence = req.get_param('q', True)
        print('sentence:', sentence)
        res = [(k,v) for (k,v) in self.bigramdic.items() if sentence in k]
        res = sorted(res, key=lambda d:d[1],reverse=True)
        scoresum = sum([i[1] for i in res])
        counter = len(res)
        print("seg result:", res)
        print("ALL-DONE")
        resp.media = {"words": res,'count':counter,
                      'score': scoresum
                      }


    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', 'http://localhost:8081')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        data = req.stream.read(req.content_length)
        reqdata = json.loads(data, encoding='utf-8')
        print('sentence:', reqdata['sents'])
        sentences = reqdata['sents']
        sentences = [s.strip() for s in sentences if len(s.strip())>0]
        if not isinstance(sentences, list):
            sentences = [sentences]
        if 'model' in reqdata and reqdata['model']=='crf':
            segsents = self.uutrtcrf.cut(sentences)
        else:
            segsents = self.bilstm.cut(sentences)
        print("seg result:", segsents)
        resp.media = {'data':{"seg":segsents}}

if __name__=="__main__":
    api = falcon.API(middleware=[cors_allow_all.middleware])
    api.req_options.auto_parse_form_urlencoded = True
    api.add_route('/bigram', BigResource())
    waitress.serve(api, port=8080, url_scheme='http')
