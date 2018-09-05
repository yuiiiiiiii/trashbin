# -*- coding: UTF-8 -*-
import numpy as np
import json
import codecs
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import f1_score

def create_dataset(filename,size):
	cnt = 0
	mapping = { u'书包':0, u'T恤':1, u'阔腿裤':2, u'运动鞋':3}
	examples = []
	target = np.zeros((size,),dtype=np.int64)
	for line in codecs.open(filename,'rb',encoding='utf8'):
		item = json.loads(line)
		examples.append(item['comment'])
		taret[cnt] = mapping[item['label']]
		cnt += 1

	dataset = sklearn.datasets.base.Bunch(data=examples,target=target)
	return dataset


if __name__ == '__main__':
	'''
	create a test dataset using Taobao comments
	0 -- shubao   3036
	1 -- Tshirt   2943
	2 -- kuotuiku 2792
	3 -- sneakers 2329
	total size -- 11100
	'''
	taobao_train = create_dataset('/home/yuyi/taobao/taobao/test.json',11100)

	#initialize BOW
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(taobao_train.data)

	#using tfidf transformer
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	#applying the data to the SVM classifier
	clf = SVC()
	clf.fit(X_train_tfidf, taobao_train.target)

	#testing on new datasets
	'''
	create a dataset using Taobao comments
	0 -- shubao   2814
	1 -- Tshirt   2555
	2 -- kuotuiku 3127
	3 -- sneakers 2320
	total size -- 10872
	'''

	test_train = create_dataset('/home/yuyi/taobao/taobao/items.json',10872)
	X_new_counts = count_vect.transform(test_train)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)

	y_pred = clf.predict(X_new_tfidf)
	score = f1_score(test_train.target, y_pred, average='macro')
	print score