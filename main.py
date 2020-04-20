
from numpy.random import seed
# Set seed value used by Keras for reproducible results
seed(42)

import pandas as pd
import numpy as np
import sys
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from numpy import zeros

def token_to_vector(sequences, size=10000):
    results = zeros((len(sequences), size))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def class_label_to_vector(class_labels):
    results = zeros(len(class_labels))
    for i, label in enumerate(class_labels):
        if (label.lower() == 'spam'):
            results[i] = 1
    return results

rawdata = pd.read_csv('input_data.csv', encoding='cp1250')
rawdata = rawdata.dropna(axis='columns')
X = rawdata['v2']
y = rawdata['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print('Tokenizing text')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_tokenized = tokenizer.texts_to_sequences(X_train)
X_test_tokenized = tokenizer.texts_to_sequences(X_test)

print('Vectorizing tokenized text')
x_train = token_to_vector(X_train_tokenized, 10000)
x_test = token_to_vector(X_test_tokenized, 10000)

y_train = class_label_to_vector(y_train)
y_test = class_label_to_vector(y_test)

if len(sys.argv) > 1:
    print('Writing preprocessed data to disk...')
    print('This may take a while depending on the speed of your disk and computer so stay patient!')
    train_x_df = pd.DataFrame(x_train)
    train_x_df.to_csv('results/training_input.csv')
    test_x_df = pd.DataFrame(x_test)
    test_x_df.to_csv('results/test_input.csv')
    train_y_df = pd.DataFrame(y_train)
    train_y_df.to_csv('results/training_labels.csv')
    test_y_df = pd.DataFrame(y_test)
    test_y_df.to_csv('results/test_labels.csv')
    print('Finished writing preprocessed data to disk')

print('Starting model fitting...')
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=20,batch_size=100,validation_split=0.5)


# visualisation
epochs=range(1, 21)
history_dict = history.history

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy Score')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='lower right')
plt.savefig('results/accuracy_epoch.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss function')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['training', 'validation'], loc='upper right')
plt.savefig('results/loss_epoch.png')

results = model.evaluate(x_test, y_test)
outputfile = open('results/results.txt', 'w+')
outputfile.write('Final results after training for 20 Epochs\n')
outputfile.write('Accuracy: %s\n' % (results[1]))
outputfile.write('Loss: %s\n' % (results[0]))
outputfile.flush()
outputfile.close()

print('DONE')

