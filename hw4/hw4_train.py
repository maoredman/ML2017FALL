import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import csv
import sys
import os
import errno
import operator
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model

vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", char_level=False)

train_text = []
with open(sys.argv[2], 'rt') as trainfile:
    for idx, row in enumerate(trainfile):
        train_text.append(row.rstrip())
tokenizer.fit_on_texts(train_text) # around 30 seconds
print('finished setting up tokenizer!')

X_train = []
y_train = []
with open(sys.argv[1], 'rt') as trainfile:
    reader = csv.reader(trainfile, delimiter=' ')
    for idx, row in enumerate(reader):                
        words = ' '.join(row[2:]) 
        X_train.append(words)
        y_train.append(row[0])

X_train = tokenizer.texts_to_sequences(X_train)
max_review_length = 40 # max lengths -- training_label:39    testing_data:39
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
print('finished generating training data!')

# create the model
batch_size = 64
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# save_model = ModelCheckpoint('punct_models/{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

print(model.summary()) # loss 0.1768
model.fit(X_train, y_train, validation_split=0.1, epochs=3, batch_size=batch_size)# , callbacks=[save_model, tensorboard])