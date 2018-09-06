# -*- coding: UTF-8 -*-
import numpy as np
import json
import emoji
import jieba
import codecs
from stanfordcorenlp  import StanfordCoreNLP
from mxnet.contrib.text.embedding import FastText
from sklearn.svm import LinearSVC
'''
firstly clean and normalize the text
'''
def normalize(filename):
	#nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27',lang='zh')
	labels = []
        cnt = 0
	doc_words = []
	mapping = { u'书包':0, u'T恤':1, u'阔腿裤':2, u'运动鞋':3}

	for line in codecs.open(filename,'rb',encoding='utf8'):
		item = json.loads(line)
		text = item['comment']
                #print str(cnt)+":"+text
                print "tokenizing "+str(cnt) + ' comment ...'
		labels.append(mapping[item['label']])
		#word_list = nlp.word_tokenize(text.encode('utf8'))
                word_list = list(jieba.cut(text))
		doc_words.append(word_list)
                cnt += 1

	return labels,doc_words



if __name__ == '__main__':
        total = 0
	taobao_train = []
        
        labels,doc_words = normalize('/home/yuyi/taobao/taobao/test.json')

        model = FastText('wiki.zh.vec')
        
        for sentence in doc_words:
            print "calculating "+str(total)+" sentence vector..."
            cnt = len(sentence)
            sen_vec = np.zeros((300L,))
            for word in sentence:
                #print  model.get_vecs_by_tokens(word).shape 
                sen_vec = np.add(sen_vec,model.get_vecs_by_tokens(word))
            sen_vec = np.true_divide(sen_vec,cnt)
            taobao_train.append(sen_vec)
            total += 1

        clf = LinearSVC()
        clf.fit(taobao_train, labels)
        
        print "training finished ..."
        test_train = []
        test_labels,test_doc=normalize('/home/yuyi/taobao/taobao/items.json')
        total = 0
        for sentence in test_doc:
            print "calculating "+str(total)+" test sentence vector..."
            cnt = len(sentence)
            sen_vec = np.zeros((300L,))
            for word in sentence:
                #print  model.get_vecs_by_tokens(word).shape 
                sen_vec = np.add(sen_vec,model.get_vecs_by_tokens(word))
            sen_vec = np.true_divide(sen_vec,cnt)
            test_train.append(sen_vec)
            total += 1
        
        test_pred = clf.predict(test_data)
        score = np.mean(test_pred == test_labels)
