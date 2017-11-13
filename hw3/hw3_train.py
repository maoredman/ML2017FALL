import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import csv
import sys
import os
import errno
import numpy as np

x_train = [] # [ [len 2304], [len 2304], [len 2304] ]
y_train = [] # [ y1, y2, y3]
with open(sys.argv[1], 'rt', encoding='big5') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    row1 = next(reader) # skip headings
    for idx, row in enumerate(reader):
        img_data = [int(i) for i in row[1].split(' ')]
        x_train.append(img_data)
        y_train.append(int(row[0]))

x_train = np.array(x_train)
y_train = np.array(y_train)

img_rows, img_cols = 48, 48
num_classes = 7
input_shape = (img_rows, img_cols, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_train /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)

batch_size = 128
epochs = 200

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
             activation='relu',
             input_shape=input_shape))                      # 288 params
model.add(Conv2D(64, (3, 3), activation='relu'))            # 576 params
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))                                    # dropout!

model.add(Conv2D(64, (3, 3), activation='relu'))            # 576 params
model.add(MaxPooling2D(pool_size=(2, 2))) # 64 x 12x12
model.add(Dropout(0.25))                                    # dropout!

model.add(Conv2D(64, (3, 3), activation='relu'))            # 576 params
model.add(MaxPooling2D(pool_size=(2, 2))) # 64 x 6x6
model.add(Dropout(0.25))                                    # dropout!


model.add(Flatten())
model.add(Dense(128, activation='relu'))                    # 294912 params
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))                     # 8192 params
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))                     # 4096 params
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))         # 448 params


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

model.fit(x = x_train,
        y = y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[early_stop])


print('finished training!')