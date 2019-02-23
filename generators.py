import pandas as pd
import numpy as np
import os
import numpy as  np
import itertools
# Any results you write to the current directory are saved as output.
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import gensim
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from helpers import *
import keras

import scikitplot.plotters as skplt

import nltk

from nltk.tokenize import word_tokenize
from nltk.util import ngrams


#from xgboost import XGBClassifier1996


import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical   




import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from nltk.corpus import stopwords


class Generator(keras.utils.Sequence):
	
	def __init__(self,data, tCount, tCountNgram, s_word):
		
		self.data = data
		self.tCount = tCount
		self.tCountNgram = tCountNgram
		self.s_word = s_word
		#self.folderList = folderList
#         self.Addk = AddK
#         self.MultiplyK = MultiplyK
#         self.AddNoise = AddNoise
#         self.InvertPixel = InvertPixel
#         self.AddBlur = AddBlur
#         self.Contrast = Contrast
#         self.Convolve = Convolve
#         self.Flip = Flip
#         self.Transform = Transform
#         self.Resize = Resize
		
		
		
#         fileList = []
		
#         for i in range(len(folderList)):
#             files = os.listdir(os.path.join(path,folderList[i]))
#             fileList += [os.path.join(path,folderList[i],x) for x in files]
			
#         self.imgList = [x for x in fileList if "img" in x]  #Keeping only the RGB map images
		
		
	def __len__(self):
		
		return 30
	
	def __getitem__(self,index):
		
		pairs = self.getBatch(batch_size)
		featuresX = self.getFeatures(pairs)
		wordEmb = self.getWordEmbeddings(pairs)
		posEmb = self.getPosEmbeddings(pairs)
		
		return pair, target
	
	def getFeatures(self, datas):
		X = getWordCount(datas[['question_text']])
		X = X.reshape(X.shape[0], 1)
		
		X1 = getLowerCase(datas[['question_text']])
		X1 = X1.reshape(X.shape[0], 1)
		
		X2 = getUpperCase(datas[['question_text']])
		X2 = X2.reshape(X.shape[0], 1)
		
		X3 = numQuestionMarks(datas[['question_text']])
		X3 = X3.reshape(X.shape[0], 1)
		
		X4 = stopWordsCount(datas[['question_text']], self.s_word)
		X4 = X4.reshape(X.shape[0], 1)
		
		X5 = datas[['question_text']].apply(lambda x: countStartWord_anywhere(x['question_text'], self.tCount), axis=1)
		X5 = reshapeArray(X.shape[0], X5)
		
		X6 = datas[['question_text']].apply(lambda x: countStartWord_onlyBegining(x['question_text'], self.tCount), axis=1)
		X6 = reshapeArray(X.shape[0], X6)
		
		X7 = datas[['question_text']].apply(lambda x: countStartWordNGram_anywhere(x['question_text'], self.tCountNgram), axis=1)
		X7 = reshapeArray(X.shape[0], X7)
		
		X8 = datas[['question_text']].apply(lambda x: countStartWordNGram_onlyBegining(x['question_text'], self.tCountNgram), axis=1)
		X8 = reshapeArray(X.shape[0], X8)
		
		featuresConc = np.concatenate((X, X1, X2, X3, X4, X5, X6, X7, X8), axis = 1)
		return featuresConc
		
		
		
	
	def getWordEmbeddings(self, dataQ):
		dfQ = dataQ['question_text'].values
		numOfWords = 2000
		tokenizerW = Tokenizer(num_words = numOfWords)
		tokenizerW.fit_on_texts(dfQ)

		word1 = tokenizerW.texts_to_sequences(dfQ)
		words = pad_sequences(word1, maxlen = 100)
		return words
		

	
	def getPosEmbeddings(self, dataPos):
		dataPos = dataPos['question_text'].values
		X1tPos = WordTokenizer(dataPos)
		numOfWords = 2000
		tokenizerPos = Tokenizer(num_words = numOfWords)
		tokenizerPos.fit_on_texts(X1tPos)


		XPos = tokenizerPos.texts_to_sequences(X1tPos)
		XPadPos = pad_sequences(XPos, maxlen = 100)
		return XPadPos
		
			
			
			

	 
		

	def getBatch(self,batchSize):
			  
		
		
		selections = np.random.choice(len(self.data),batchSize,replace=False)
		dataIndexed = self.data.loc[selections,['question_text', 'target']]
		
		
			
		  

		return dataIndexed
	
	def on_epoch_end(self):
		'Updates to be done after each epoch'
		a = 5
		
		
	def generate(self, batch_size, s="train"):
		"""a generator for batches, so model.fit_generator can be used. """
		while True:
			pairs = self.getBatch(batch_size)
			featuresX = self.getFeatures(pairs)
			wordEmb = self.getWordEmbeddings(pairs)
			posEmb = self.getPosEmbeddings(pairs)
			target = to_categorical(pairs['target'].values, num_classes=2)
			yield ([wordEmb, posEmb, featuresX], target)




class GeneratorPred(keras.utils.Sequence):
	
	
	def __init__(self, data, tCount, tCountNgram, s_word):
		
		self.data = data 
		self.indexData = 0
		self.tCount = tCount
		self.tCountNgram = tCountNgram
		self.s_word = s_word
		#self.folderList = folderList
#         self.Addk = AddK
#         self.MultiplyK = MultiplyK
#         self.AddNoise = AddNoise
#         self.InvertPixel = InvertPixel
#         self.AddBlur = AddBlur
#         self.Contrast = Contrast
#         self.Convolve = Convolve
#         self.Flip = Flip
#         self.Transform = Transform
#         self.Resize = Resize
		
		
		
#         fileList = []
		
#         for i in range(len(folderList)):
#             files = os.listdir(os.path.join(path,folderList[i]))
#             fileList += [os.path.join(path,folderList[i],x) for x in files]
			
#         self.imgList = [x for x in fileList if "img" in x]  #Keeping only the RGB map images
		
		
	def __len__(self):
		
		return 30
	
	def __getitem__(self,index):
		
		pairs = self.getBatch(batch_size)
		featuresX = self.getFeatures(pairs)
		wordEmb = self.getWordEmbeddings(pairs)
		posEmb = self.getPosEmbeddings(pairs)
		
		return pair, target
	
	def getFeatures(self, datas):
		X = getWordCount(datas[['question_text']])
		X = X.reshape(X.shape[0], 1)
		
		X1 = getLowerCase(datas[['question_text']])
		X1 = X1.reshape(X.shape[0], 1)
		
		X2 = getUpperCase(datas[['question_text']])
		X2 = X2.reshape(X.shape[0], 1)
		
		X3 = numQuestionMarks(datas[['question_text']])
		X3 = X3.reshape(X.shape[0], 1)
		
		X4 = stopWordsCount(datas[['question_text']], self.s_word)
		X4 = X4.reshape(X.shape[0], 1)
		
		X5 = datas[['question_text']].apply(lambda x: countStartWord_anywhere(x['question_text'], self.tCount), axis=1)
		X5 = reshapeArray(X.shape[0], X5)
		
		X6 = datas[['question_text']].apply(lambda x: countStartWord_onlyBegining(x['question_text'], self.tCount), axis=1)
		X6 = reshapeArray(X.shape[0], X6)
		
		X7 = datas[['question_text']].apply(lambda x: countStartWordNGram_anywhere(x['question_text'], self.tCountNgram), axis=1)
		X7 = reshapeArray(X.shape[0], X7)
		
		X8 = datas[['question_text']].apply(lambda x: countStartWordNGram_onlyBegining(x['question_text'], self.tCountNgram), axis=1)
		X8 = reshapeArray(X.shape[0], X8)
		
		featuresConc = np.concatenate((X, X1, X2, X3, X4, X5, X6, X7, X8), axis = 1)
		return featuresConc
		
		
		
	
	def getWordEmbeddings(self, dataQ):
		dfQ = dataQ['question_text'].values
		numOfWords = 2000
		tokenizerW = Tokenizer(num_words = numOfWords)
		tokenizerW.fit_on_texts(dfQ)

		word1 = tokenizerW.texts_to_sequences(dfQ)
		words = pad_sequences(word1, maxlen = 100)
		return words
		

	
	def getPosEmbeddings(self, dataPos):
		dataPos = dataPos['question_text'].values
		X1tPos = WordTokenizer(dataPos)
		numOfWords = 2000
		tokenizerPos = Tokenizer(num_words = numOfWords)
		tokenizerPos.fit_on_texts(X1tPos)


		XPos = tokenizerPos.texts_to_sequences(X1tPos)
		XPadPos = pad_sequences(XPos, maxlen = 100)
		return XPadPos
		
			
			
			

	 
		

	def getBatch(self,batchSize):
			  
		
		
		selections = np.random.choice(len(self.data),batchSize,replace=False)
		dataIndexed = self.data.loc[self.indexData:self.indexData+batchSize,['question_text']]
		self.indexData = self.indexData + batchSize
		#print('index', self.indexData)
		
		
			
		  

		return dataIndexed
	
	def on_epoch_end(self):
		'Updates to be done after each epoch'
		a = 5
		
		
	def generate(self, batch_size, s="train"):
		"""a generator for batches, so model.fit_generator can be used. """
		while True:
			pairs = self.getBatch(batch_size)
			featuresX = self.getFeatures(pairs)
			wordEmb = self.getWordEmbeddings(pairs)
			posEmb = self.getPosEmbeddings(pairs)
			#target = to_categorical(pairs['target'].values, num_classes=2)
			yield ([wordEmb, posEmb, featuresX])