# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:55:16 2023

@author: Timothy Williams
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets

img_width = 25
img_height = 50 #ideally want to have full image res, but computer not strong enough for that
batch_size = 50

model = models.Sequential()
model.add(layers.Conv2D(50, (3, 3), activation='relu', input_shape=(25, 50, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(100, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(100, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10))

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'ImagesOriginalSize/',
    labels = 'inferred',
    label_mode = "int", #categorical, binary
    class_names = ['NormalFinal','ScolFinal','SpondFinal'],
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = (img_width, img_height), #reshape if not in this size
    shuffle = True,
    seed = 123, #makes split same everytime
    validation_split = 0.1,
    subset='training')

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'ImagesOriginalSize/',
    labels = 'inferred',
    label_mode = "int", #categorical, binary
    class_names = ['NormalFinal','ScolFinal','SpondFinal'],
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = (img_width, img_height), #reshape if not in this size
    shuffle = True,
    seed = 123,
    validation_split = 0.1,
    subset='validation')

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(ds_train, epochs = 20, verbose = 2)

spine_rec_model = 'spine_rec_model.pk1'
pickle.dump(model, open(spine_rec_model, 'wb'))
    