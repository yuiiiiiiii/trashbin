# -*- coding: UTF-8 -*-
import numpy as np
import json
import codecs
from stanfordcorenlp import StanfordCoreNlp
from mxnet.contrib.text.embedding import FastText
'''
firstly clean and normalize the text
'''
def normalize(filename):
	nlp = StanfordCoreNlp(r'stanford-corenlp-full-2018-02-27',lang='zh')
	labels = []
	doc_words = []
	mapping = { u'书包':0, u'T恤':1, u'阔腿裤':2, u'运动鞋':3}

	for line in codec.open(filename,'rb',encoding='utf8'):
		item = json.loads(line)
		text = item['comment']
		labels.append(mapping[item['label']])
		word_list = nlp.word_tokenize(text.encode('utf8'))
		doc_words.append(word_list)

	return labels,doc_words



if __name__ == '__main__':
	normalize('/home/yuyi/taobao/taobao/test.json')
