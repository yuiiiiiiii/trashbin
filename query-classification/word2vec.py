# -*- coding: utf-8  -*-
import multiprocessing
import json
import codecs
import emoji
import re
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


def preprocessing():
    i = 0
    converter = OpenCC('t2s')		#trannsorm into simplified Chinese
    #nlp = StanfordCoreNLP(r'/home/yuyi/stanford-corenlp-full-2018-02-27',lang='zh')

    output = open('retry.txt', 'w')
    wiki =WikiCorpus('/home/yuyi/zhwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary=[])#gensim里的维基百科处理类WikiCorpus
    for text in wiki.get_texts():#通过get_texts将维基里的每篇文章转换位1行text文本，并且去掉了标点符号等内容
        cleaned = ''
        text = ''.join(text)
        for char in text:
            char  = converter.convert(char)
            cleaned += char

        if len(cleaned):
            sentence = jieba.cut(cleaned)
            output.write(' '.join(sentence) + "\n")

        i = i + 1
        #if (i % 1000) == 0:
        #if i == 10:
           # print "Saved "+str(i)+" articles."
           # break
    
    output.close()
    print "Finished Saved "+str(i)+" articles."

def getVec(file):
	docv = []
	senv = np.zeros(300)
	model = Word2Vec.load('wiki_zh.model')
	wv = model.wv

	filename = os.path.join('/home/yuyi/taobao/taobao',file + '.' + 'json')

	for line in codecs.open(filename,'rb',encoding='utf8'):
		item = json.loads(line)
 		text = item['comment']
 		text = list(jieba.cut(text))
		cnt = len(text)
		 
		for word in text:
		 	if word in wv.vocab:
		 		senv = np.add(model[word],senv)
		 	else:
		 		senv = np.add(senv,)
 		senv = np.true_divide(x,cnt)
 		docv.append(senv)

 	outdir = file + '.pkl'
 	with open(outdir,'wb') as f:
 		pickle.dump(docv,f)
    

def pretrain():
	with open('wiki_zh.pkl','r') as f:
		sentences = pickle.load(f)

    outdir = 'wiki_zh.model'
   
    model = Word2Vec(sentences, size= 300, window=5, min_count=1,
                     workers=multiprocessing.cpu_count()-1)

    model.save(outdir)


def retrain():
    stopwords = []
    sentences = []
    filtered_text = []
    
    with open('stopwords.txt','r') as f:
        line = f.read().strip()
        result = re.split(r"[\s\n]",line)

#    print result        

    punc = list("！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~")
    #print punc

    for line in codecs.open('/home/yuyi/taobao/taobao/test.json','rb',encoding='utf8'):
         item = json.loads(line)
         text = item['comment']
         text = list(jieba.cut(text))
         sentences.append(text)

    model = Word2Vec.load('wiki_zh.model')
    model.train(sentences,len(sentences),epochs = model.epochs)
    
    model.save('wiki_zh.model')



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
	#model = Word2Vec.load('wiki_zh.model')
        #print model[u'good']
        #print model[u'shit']

        for line in codecs.open('wiki_zh.txt','r',encoding = 'utf8'):
            line = line.encode('utf8')
            print str(line)

if __name__ == '__main__':
    
    preprocessing()
