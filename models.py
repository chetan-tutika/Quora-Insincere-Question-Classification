import pandas as pd
import numpy as np
import os
import itertools

import numpy as  np
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

def lstm_model(pretrained_weights = None, embed_dim = 128, lstm_out = 196, numOfWords = 2000, lossFunc = None, accMetrics = None, Xinput = [100, 100, 445]):
	if lossFunc == None:
		lossFunc = 'categorical_crossentropy'

	if accMetrics == None:
		accMetrics = ['categorical_crossentropy', 'acc']

	inputPos = keras.layers.Input(shape=(Xinput[0],))
	EmbedLayerPos = keras.layers.Embedding(numOfWords, embed_dim, input_length = inputPos.shape[1])(inputPos)
	LSTMPos = keras.layers.LSTM(lstm_out, recurrent_dropout = 0.2, dropout = 0.2)(EmbedLayerPos)


	inputWords = keras.layers.Input(shape=(Xinput[1],))
	EmbedLayerWords = keras.layers.Embedding(numOfWords, embed_dim, input_length = inputWords.shape[1])(inputWords)
	LSTMWords = keras.layers.LSTM(lstm_out, recurrent_dropout = 0.2, dropout = 0.2)(EmbedLayerWords)

	concatWordsPos = keras.layers.concatenate([LSTMWords, LSTMPos], axis=-1)
	#concatWordsPos = keras.layers.concatenate([LSTMWords, LSTMPos], axis= 1)

	LSTM_Concat = keras.layers.Dense(10, activation='relu')(concatWordsPos)


	inputFeatures = keras.layers.Input(shape=(Xinput[2],))

	concatLSTM_Features = keras.layers.concatenate([LSTM_Concat, inputFeatures], axis=-1)

	out1 = keras.layers.Dense(10, activation='relu')(concatLSTM_Features)

	out = keras.layers.Dense(2, activation='sigmoid')(out1)

	model = keras.models.Model(inputs=[inputPos, inputWords, inputFeatures], outputs=out)

	model.compile(optimizer = 'adam', loss = lossFunc, metrics = accMetrics)

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

