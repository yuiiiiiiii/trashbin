# -*- coding: utf-8  -*-
import multiprocessing
import json
import codecs
import emoji
import re
import numpy as np
import pickle
from stanfordcorenlp import StanfordCoreNLP
import jieba
from gensim.corpora import WikiCorpus
from opencc import OpenCC
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.svm import LinearSVC
import sys
reload(sys)
sys.setdefaultencoding('utf8')


'''
ignoring unknown word
'''

def preprocessing():
    res = []
    i = 0
    converter = OpenCC('t2s')		#trannsorm into simplified Chinese
    #nlp = StanfordCoreNLP(r'/home/yuyi/stanford-corenlp-full-2018-02-27',lang='zh')

    wiki =WikiCorpus('/home/yuyi/zhwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary=[])#gensim里的维基百科处理类WikiCorpus
    for text in wiki.get_texts():#通过get_texts将维基里的每篇文章转换位1行text文本，并且去掉了标点符号等内容
        cleaned = ''
        text = ''.join(text)
        for char in text:
            char  = converter.convert(char)
            cleaned += char

        if len(cleaned):
            sentence = list(jieba.cut(cleaned))
            res.append(sentence)

        i = i + 1
        if (i % 1000) == 0:
        #if i == 10:
            print "Saved "+str(i)+" articles."
          # break
    
    with open('wiki_zh.pkl','w') as f:
        pickle.dump(res,f)

    print "Finished Saved "+str(i)+" articles."

def getVec(file):
	docv = []
	senv = np.zeros(300)
	model = Word2Vec.load('wiki_zh.model')
	wv = model.wv

        with open('/home/yuyi/taobao/taobao/items_text.pkl','r') as f:
            sentences = pickle.load(f)

        for sentence in sentences:
            cnt = len(sentence)
            for word in sentence:
	        if word in wv.vocab:
		    senv = np.add(model[word],senv)
		else:
		    cnt -= 1
                
            if cnt > 0:
                senv = np.true_divide(senv,cnt)
 	    docv.append(senv)

 	outdir = file + '.pkl'
 	with open(outdir,'wb') as f:
 		pickle.dump(docv,f)
    
def getLabel(file):
    labels = []
    filename = file + '.json'

    for line in codecs.open(filename,'r',encoding='utf8'):
        item = json.loads(line)
        label = item['label']
        labels.append(label)

    outdir = file + '_label.pkl'
    with open(outdir,'w') as f:
        pickle.dump(labels,f)
        

def pretrain():
    with open('wiki_zh.pkl','r') as f:
	sentences = pickle.load(f)

    with open('/home/yuyi/taobao/taobao/test_text.pkl') as f:
        new_sentences = pickle.load(f)

    sentences = sentences + new_sentences

    outdir = 'wiki_zh.model'
   
    model = Word2Vec(sentences, size= 300, window=5, min_count=1,
                     workers=multiprocessing.cpu_count()-1)

    model.save(outdir)





def normalize(filename):
	#nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27',lang='zh')
	labels = []
	cnt = 0
	doc_words = []
	mapping = { u'书包':0, u'T恤':1, u'阔腿裤':2, u'运动鞋':3}

	for line in codecs.open(filename,'rb',encoding='utf8'):
		item = json.loads(line)
		text = item['comment']

		print "tokenizing "+str(cnt) + ' comment ...'
		labels.append(mapping[item['label']])
		word_list = nlp.word_tokenize(text.encode('utf8'))

		doc_words.append(word_list)
                cnt += 1

	return labels,doc_words

def train():
    model = Word2Vec.load('wiki_zh.model')
    print model[u'阔腿裤']
    print model[u'书包']

#        for line in codecs.open('wiki_zh.txt','r',encoding = 'utf8'):
#            line = line.encode('utf8')
#            print str(line)

def test():
    with open('wiki_zh.pkl','r') as f:
        text = pickle.load(f)

    print len(text)


if __name__ == '__main__':
    
    getLabel('items')
