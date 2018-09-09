# -*- coding: utf-8  -*-
from __future__ import print_function
from collections import Counter
import jieba
import codecs
import json
import itertools
import numpy as np
import re
import pickle
import mxnet as mx
import sys,os

    
    
def load_data_and_labels(file):
    """
    Loads data from taobao crawler files, use jieba to split the data into words and generates labels.
    Returns split sentences and labels.
    """
    mapping = { u'书包':0, u'T恤':1, u'阔腿裤':2, u'运动鞋':3}
    sentences = []
    labels = []

    filename = file + '.json'

    for line in codecs.open(filename,'rb',encoding='utf8'):
        item = json.loads(line)

        text = item['comment']
        label = mapping[item['label']]

        sentence = list(jieba.cut(text))

        sentences.append(sentence)
        labels.append(label)

    out_label = file + '_label.pkl'
    out_text = file + '_text.pkl'
  
    with open(out_text,'w') as f:
        pickle.dump(sentences,f)
    with open(out_label,'w') as f:
        pickle.dump(labels,f)

def check(file):    
    in_label = file + '_label.pkl'
    in_text = file + '_text.pkl'
  
    f = open(in_text,'r')
    text = pickle.load(f)
    f.close()

    f = open(in_label,'r')
    labels = pickle.load(f)
    f.close()

    for sen in text:
        for wd in sen:
            print(wd)


def pad_sentences(sentences, padding_word=""):
    """
    Pads all sentences to be the length of the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
        
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from token to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    
    return vocabulary, vocabulary_inv


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([
            [vocabulary[word] for word in sentence]
            for sentence in sentences])
    y = np.array(labels)
    
    return x, y


def cnn():
    """
    Loads and preprocesses data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    vocab_size = len(vocabulary)

    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split train/dev set
    # there are a total of 10662 labeled examples to train on
    x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

    sentence_size = x_train.shape[1]

    print('Train/Dev split: %d/%d' % (len(y_train), len(y_dev)))
    print('train shape:', x_train.shape)
    print('dev shape:', x_dev.shape)
    print('vocab_size', vocab_size)
    print('sentence max words', sentence_size)

    '''
    Define batch size and the place holders for network inputs and outputs
    '''

    batch_size = 50
    print('batch size', batch_size)

    input_x = mx.sym.Variable('data') # placeholder for input data
    input_y = mx.sym.Variable('softmax_label') # placeholder for output label


    '''
    Define the first network layer (embedding)
    '''

    # create embedding layer to learn representation of words in a lower dimensional subspace (much like word2vec)
    num_embed = 300 # dimensions to embed words into
    print('embedding dimensions', num_embed)

    embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')

    # reshape embedded data for next layer
    conv_input = mx.sym.Reshape(data=embed_layer, shape=(batch_size, 1, sentence_size, num_embed))

    # create convolution + (max) pooling layer for each filter operation
    filter_list=[3, 4, 5] # the size of filters to use
    print('convolution filters', filter_list)

    num_filter=100
    pooled_outputs = []
    for filter_size in filter_list:
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1, 1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)

    # reshape for next layer
    h_pool = mx.sym.Reshape(data=concat, shape=(batch_size, total_filters))

    # dropout layer
    dropout = 0.5
    print('dropout probability', dropout)

    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

    # fully connected layer
    num_label = 2

    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    # softmax output
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    # set CNN pointer to the "back" of the network
    cnn = sm

if __name__ == '__main__':
    
    load_data_and_labels('items')
