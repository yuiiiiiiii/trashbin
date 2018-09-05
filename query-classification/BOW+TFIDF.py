# -*- coding: UTF-8 -*-
import numpy as np
import json
import codecs
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

'''
create a dataset using Taobao comments
0 -- shubao   2814
1 -- Tshirt   2555
2 -- kuotuiku 3127
3 -- sneakers 2376
'''
cnt = 0
mapping = { u'书包':0, u'T恤':1, u'阔腿裤':2, u'运动鞋':3}
examples = []
target = np.zeros((10872,),dtype=np.int64)
for line in codecs.open('items.json','rb',encoding='utf8'):
	item = json.loads(line)
	examples.append(item['comment'])
	taret[cnt] = mapping[item['label']]
	cnt += 1

taobao_train = sklearn.datasets.base.Bunch(data=examples,target=target)


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
# docs_new = ['God is love', 'OpenGL on the GPU is fast']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicted = clf.predict(X_new_tfidf)
