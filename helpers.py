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

import keras

import scikitplot.plotters as skplt

import nltk

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from models import lstm_model

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


def getWordCount(dataset):
	wordCount = dataset['question_text'].str.split().str.len()
	return wordCount.values
def getUpperCase(dataset):
	upperCount = dataset['question_text'].str.findall(r'[A-Z]').str.len()
	return upperCount.values
def getLowerCase(dataset):
	lowerCase = dataset['question_text'].str.findall(r'[a-z]').str.len()
	return lowerCase.values
def numQuestionMarks(dataset):
	noQ = dataset['question_text'].str.findall(r'\?').str.len()
	return noQ.values
def stopWordsCount(dataset, stop_word):
	stopWords = dataset['question_text'].apply(lambda x: [item for item in x.split() if item in stop_word])
	stopWordsC = stopWords.str.len()
	return stopWordsC.values


def reshapeArray(lenIndex, arr):
	#print('len', arr)
	zer = arr.values
	#print('zer', zer)
	#print('len0', len(arr[0]))
	zer = np.concatenate(zer, axis = 0).reshape((lenIndex, -1))
	#print('zer2', zer)
	return zer
	
def countStartWord_anywhere(x, True_Count):
	zeros = np.zeros(len(True_Count))
	#print(x, xT)
	
	#zeros = np.zeros(len(TrueCount))
	for item in x.split():
		if item in True_Count:
			zeros[True_Count.index(item)] = 1
				
#     elif xT == 0:
#         zeros = np.zeros(len(FalseCount))
#         for item in x.split():
#             if item in FalseCount:
#                 zeros[FalseCount.index(item)] = 1
#                 #return zeros
# #     zeros = zeros.values
# #     zeros = np.concatenate(zeros, axis = 0).reshape()
	return zeros


def countStartWord_onlyBegining(x, True_Count):
	zeros = np.zeros(len(True_Count))
	#zeros = np.zeros(len(TrueCount))
	if x.split()[0] in True_Count:
		zeros[True_Count.index(x.split()[0])] = 1
		#return zeros
#     elif xT == 0:
#         #zeros = np.zeros(len(FalseCount))
#         if x.split()[0] in FalseCount:
#             zeros[FalseCount.index(x.split()[0])] = 1
#             #return zeros
	return zeros

def countStartWordOccurence(x, True_Count):
	count = 0
	for item in x.split():
		if item in True_Count:
			count = count + 1

	return count
	
	

# def countStartWord_anywhereF(x):
#     zeros = np.zeros(len(FalseCount))
#     for item in x.split():
#         if item in FalseCount:
#             zeros[FalseCount.index(item)] = 1
#             return zeros
#     return zeros


# def countStartWord_onlyBeginingF(x):
#     zeros = np.zeros(len(FalseCount))
#     if x.split()[0] in FalseCount:
#         zeros[FalseCount.index(x.split()[0])] = 1
#         return zeros
#     return zeros

# def countStartWordOccurenceF(x):
#     count = 0
#     for item in x.split():
#         if item in FalseCount:
#             count = count + 1
#     return count

def get_ngrams(text):
	n_grams = ngrams(word_tokenize(text), 2)
	return [ ' '.join(grams) for grams in n_grams]

def countStartWordNGram_anywhere(x, TrueCount_NGram):
	zeros = np.zeros(len(TrueCount_NGram))

	for item in get_ngrams(x):
		if item in TrueCount_NGram:
			zeros[TrueCount_NGram.index(item)] = 1
			#return zeros
#     elif xT == 0:
#         for item in get_ngrams(x):
#             if item in FalseCountNGram:
#                 zeros[FalseCountNGram.index(item)] = 1
		
	return zeros


def countStartWordNGram_onlyBegining(x, TrueCount_NGram):
	zeros = np.zeros(len(TrueCount_NGram))

	if len(get_ngrams(x))>0:
		if get_ngrams(x)[0] in TrueCount_NGram:
			zeros[TrueCount_NGram.index(get_ngrams(x)[0])] = 1
			#return zeros
#     elif xT == 0:
#         if len(get_ngrams(x))>0:
#             if get_ngrams(x)[0] in FalseCountNGram:
#                 zeros[FalseCountNGram.index(get_ngrams(x)[0])] = 1
		
	return zeros

def countStartWordOccurenceNGram(x, TrueCount_NGram):
	count = 0
		
	for item in get_ngrams(x):
		if item in TrueCount_NGram:
				count = count + 1

		
	return count





# def countStartWordNGram_anywhereF(x):
#     zeros = np.zeros(len(FalseCountNGram))
#     for item in get_ngrams(x):
#         if item in FalseCountNGram:
#             zeros[FalseCountNGram.index(item)] = 1
#             return zeros
#     return zeros


# def countStartWordNGram_onlyBeginingF(x):
#     zeros = np.zeros(len(FalseCountNGram))
#     if len(get_ngrams(x))>0:
#         if get_ngrams(x)[0] in FalseCountNGram:
#             zeros[FalseCountNGram.index(get_ngrams(x)[0])] = 1
#             return zeros
#     return zeros

# def countStartWordOccurenceNGramF(x):
#     count = 0
#     for item in get_ngrams(x):
#         if item in FalseCountNGram:
#             count = count + 1
#     return count


def LinearizeWords(column):
	words = []
	for rows in column:
		for sentences in nltk.sent_tokenize(rows):
			words += nltk.word_tokenize(sentences)
			return words
def WordTokenizer(columns):
	#Xt = np.zeros((columns.shape[0],1))
	XtPos = []
	XtBoth = []
	for i,rows in enumerate(columns):
		XttPos = []
		XttWord = []
		#print(i, rows)
		for sent in nltk.sent_tokenize(rows):
			#print(i, sent)
			Xtk = np.array(nltk.pos_tag(nltk.word_tokenize(sent)))
			#print(Xtk)
			XttPos += Xtk[:,1].tolist()
			#XttWord += Xtk[:,0].tolist()
			#XttBoth = [m +'_'+n for m,n in zip(XttWord,XttPos)]
			
			#print('Xtt',Xtt)
			strPos = ' '.join(XttPos)
			#strBoth = ' '.join(XttBoth)
			

			#print(str1)
		XtPos.append(strPos)
		#XtBoth.append(strBoth)
		#print('XT', Xt)
	return np.array(XtPos)#, np.array(XtBoth) 