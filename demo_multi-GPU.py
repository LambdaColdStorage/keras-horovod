'''
Single-GPU ..........................................
Epoch 1/3
2048/2048 [==============================] - 20s 10ms/step - loss: 3491.6869 - acc: 0.0015
Epoch 2/3
2048/2048 [==============================] - 17s 8ms/step - loss: 3453.9541 - acc: 9.7656e-04
Epoch 3/3
2048/2048 [==============================] - 17s 8ms/step - loss: 3453.9444 - acc: 9.7656e-04
Multi-GPU ...........................................
Epoch 1/3
2048/2048 [==============================] - 11s 6ms/step - loss: 3688.0599
Epoch 2/3
2048/2048 [==============================] - 6s 3ms/step - loss: 3453.9607
Epoch 3/3
2048/2048 [==============================] - 6s 3ms/step - loss: 3453.9572
'''

import tensorflow as tf
import keras
from keras import models
from keras.utils import multi_gpu_model
import numpy as np

keras.backend.set_image_data_format('channels_first')

NUM_EPOCHS = 3
NUM_SAMPLES = 2048
HEIGHT = 224
WIDTH = 224
CHANNEL = 3
NUM_CLASSES = 1000
BATCH_SIZE = 64
NUM_GPU = 2

x = np.random.random((NUM_SAMPLES, CHANNEL, HEIGHT, WIDTH))
y = np.random.random((NUM_SAMPLES, NUM_CLASSES))


def my_cnn():
  model = models.Sequential()
  model.add(keras.layers.Conv2D(128, 3, activation='relu',
                                input_shape=(CHANNEL, HEIGHT, WIDTH)))
  model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
  model.add(keras.layers.Conv2D(256, 3, activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
  model.add(keras.layers.Conv2D(512, 3, activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
  model.add(keras.layers.Conv2D(64, 3, activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
  model.add(keras.layers.Reshape((-1,)))
  model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
  return model


# Single GPU
model = my_cnn()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('')
print('Single-GPU ..........................................')
model.fit(x, y,
          epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE)


# Multi-GPU
model = my_cnn()

parallel_model = multi_gpu_model(model, NUM_GPU)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')
print('')
print('Multi-GPU ...........................................')
parallel_model.fit(x, y,
                   epochs=NUM_EPOCHS,
                   batch_size=BATCH_SIZE * NUM_GPU)