
# from stanfordcorenlp  import StanfordCoreNLP
# from mxnet.contrib.text.embedding import FastText
# '''
# firstly clean and normalize the text
# '''


# if __name__ == '__main__':
#         total = 0
# 	taobao_train = []
        
#         labels,doc_words = normalize('/home/yuyi/taobao/taobao/test.json')

#         model = FastText('wiki.zh.vec')
        
#         for sentence in doc_words:
#             print "calculating "+str(total)+" sentence vector..."
#             cnt = len(sentence)
#             sen_vec = np.zeros((300L,))
#             for word in sentence:
#                 #print  model.get_vecs_by_tokens(word).shape 
#                 sen_vec = np.add(sen_vec,model.get_vecs_by_tokens(word))
#             sen_vec = np.true_divide(sen_vec,cnt)
#             taobao_train.append(sen_vec)
#             total += 1

#         clf = LinearSVC()
#         clf.fit(taobao_train, labels)
        
#         print "training finished ..."
#         test_train = []
#         test_labels,test_doc=normalize('/home/yuyi/taobao/taobao/items.json')
#         total = 0
#         for sentence in test_doc:
#             print "calculating "+str(total)+" test sentence vector..."
#             cnt = len(sentence)
#             sen_vec = np.zeros((300L,))
#             for word in sentence:
#                 #print  model.get_vecs_by_tokens(word).shape 
#                 sen_vec = np.add(sen_vec,model.get_vecs_by_tokens(word))
#             sen_vec = np.true_divide(sen_vec,cnt)
#             test_train.append(sen_vec)
#             total += 1
        
#         test_pred = clf.predict(test_data)
#         score = np.mean(test_pred == test_labels)




#!/usr/bin/env python
# -*- coding: utf-8  -*-
#将xml的wiki数据转换为text格式
import multiprocessing
import codecs
import emoji
from stanfordcorenlp import StanfordCoreNLP
from gensim.corpora import WikiCorpus
from opencc import openCC
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.svm import LinearSVC

def preprocessing():
    i = 0
    converter = openCC('t2s')		#trannsorm into simplified Chinese
    nlp = StanfordCoreNLP(r'/home/yuyi/stanford-corenlp-full-2018-02-27',lang='zh')

    output = open('wiki_zh.txt', 'w')
    wiki =WikiCorpus('/home/yuyi/zhwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary=[])#gensim里的维基百科处理类WikiCorpus
    for text in wiki.get_texts():#通过get_texts将维基里的每篇文章转换位1行text文本，并且去掉了标点符号等内容
        text = converter.convert(text)
        sentence = nlp.word_tokenize(text)
        output.write(' '.join(sentence) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            print "Saved "+str(i)+" articles."

    output.close()
    print "Finished Saved "+str(i)+" articles."
    

def pretrain():

    indir = 'wiki_zh.txt'
    outdir = 'wiki_zh.model'
   
    model = Word2Vec(LineSentence(inp), size= 300, window=5, min_count=5,
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

if __name__ == '__main__':

	preprocessing()