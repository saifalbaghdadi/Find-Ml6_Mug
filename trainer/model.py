import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D


batch_size = 64
epochs_steps = 15

def get_batch_size():
    return batch_size

def get_epochs():
    return epochs_steps

def solution(input_layer):

    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation='softmax'))

    opt = Adam(learning_rate=0.0009)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

