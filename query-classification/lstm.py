from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense
from gensim.models import Word2Vec
import numpy as np
import codecs
import json
import jieba
import pickle
import os

'''
number of sentences  ---  nb_samples
length of sentences ---  timesteps
dimension of word vectors --- input_dim
'''
'''
if the sentences have different length
maybe I can set timesteps to None
batchsize to 1 ???
or padding ???
'''
def LSTM():
	data_dim = 16
	timesteps = 8
	num_classes = 4

	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	model.add(LSTM(32, return_sequences=True,
	               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
	model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
	model.add(LSTM(32))  # return a single vector of dimension 32
	model.add(Dense(4, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	# Generate dummy training data
	x_train = np.random.random((1000, timesteps, data_dim))
	y_train = np.random.random((1000, num_classes))

	# Generate dummy validation data
	x_val = np.random.random((100, timesteps, data_dim))
	y_val = np.random.random((100, num_classes))

	model.fit(x_train, y_train,
	          batch_size=64, epochs=5,
	          validation_data=(x_val, y_val))

def preprocessing(file):
	
	sentences = pad_sequences(sequences)

	return sentences




if __name__ == '__main__':

	getVec('items.json')