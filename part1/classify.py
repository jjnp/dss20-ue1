import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split

rawdata = pd.read_csv('spam.csv', encoding='cp1250')
rawdata = rawdata.dropna(axis='columns')
X = rawdata['v2']
y = rawdata['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print('Tokenizing text')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_tokenized = tokenizer.texts_to_sequences(X_train)
X_test_tokenized = tokenizer.texts_to_sequences(X_test)



