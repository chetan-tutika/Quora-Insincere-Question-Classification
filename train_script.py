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


#from xgboost import XGBClassifier1996
from generators import GeneratorPred, Generator
from models import lstm_model

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


numpy.random.seed(7)
df = pd.read_csv('train.csv')
df100 = df.head(1000)
df1 = df.copy()
df100 = df1.head(10000)

dfTrue = df1.loc[df1['target'] == 1]
dfFalse = df1.loc[df1['target'] != 1]
dfQ = df1['question_text'].values
dfT = df1[['target']].values
dfTest = pd.read_csv('test.csv')

indexdf = int(3*df.shape[0]/4) 

dfInd = df1.iloc[:indexdf, :]
dfInd = dfInd.reset_index()
dfV = df1.iloc[indexdf:, :]
dfV = dfV.reset_index()


stop = stopwords.words('english')
firstWord = df1['question_text'].str.split().str[0]

TrueCount = firstWord.value_counts().index.tolist()[:20]

secondWord = df1['question_text'].str.split().str[1]
#secondWordFalse2 = dfFalse['question_text'].str.split().str[1]

TrueCount2 = secondWord.value_counts().index.tolist()[:10]
#FalseCount2 = secondWordFalse2.value_counts().index.tolist()[:10]


TrueCountNGram = []
for x in itertools.product(TrueCount,TrueCount2):
	k = ' '.join(x)
	TrueCountNGram.append(k)

numOfWords = 2000
tokenizerPos = Tokenizer(num_words = numOfWords)

gen = Generator(df1, TrueCount, TrueCountNGram, stop)
#genT = Generator(dfV)
X_input  = gen.generate(6).__next__()
epochs = 5

model = lstm_model()

model.fit_generator(gen.generate(200),epochs=epochs,verbose=1,steps_per_epoch = 500)

model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
	yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
