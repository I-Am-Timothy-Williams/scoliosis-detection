# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 18:43:46 2023

@author: Timot
"""
import os
import numpy as np
from numpy import asarray
import tensorflow as tf
from tensorflow import keras
import math

path = 'ImagesOriginalSize/'
classes = [0,1,2]
im_width = 250
im_length = 950
IMAGE_SIZE = (im_width,im_length)

normal_count = 0
normal_path = path + 'NormalFinal/'
for path1 in os.scandir(normal_path):
    if path1.is_file():
        normal_count += 1
        
scoliosis_count = 0
scoliosis_path = path + 'ScolFinal/'
for path1 in os.scandir(scoliosis_path):
    if path1.is_file():
        scoliosis_count += 1
        
spondylosis_count = 0
spondylosis_path = path + 'SpondFinal/'
for path1 in os.scandir(spondylosis_path):
    if path1.is_file(): 
        spondylosis_count += 1

train_split_normal = math.ceil(normal_count * .8)
test_split_normal = normal_count - train_split_normal

x_norm_train=np.empty((train_split_normal,im_width*im_length))
y_norm_train=np.empty((train_split_normal,1))

def get_data_normal(folder,start,end,label):
    file_names = os.listdir(folder)
    for i in range(start, end):
        file_path_n = folder + file_names[i]
        im = tf.io.read_file(file_path_n)
        im = tf.image.decode_png(im, channels=1)
        im = tf.image.resize(im, IMAGE_SIZE, method='lanczos5')
        im_array = asarray(im)
        x_norm_train[i,:]=im_array.reshape(1,-1)
        y_norm_train[i,0]=classes[label]
    return x_norm_train,y_norm_train

get_data_normal(normal_path,0,train_split_normal,0)

x_scol_train=np.empty((train_split_scol,im_width*im_length))
y_scol_train=np.empty((train_split_scol,1))

def get_data_scol(folder,start,end,label):
    file_names = os.listdir(folder)
    for i in range(start, end):
        file_path_n = folder + file_names[i]
        im = tf.io.read_file(file_path_n)
        im = tf.image.decode_png(im, channels=1)
        im = tf.image.resize(im, IMAGE_SIZE, method='lanczos5')
        im_array = asarray(im)
        x_scol_train[i,:]=im_array.reshape(1,-1)
        y_scol_train[i,0]=classes[label]
    return x_scol_train,y_scol_train

get_data_scol(scoliosis_path,0,train_split_scol,1)
        

x_spon_train=np.empty((train_split_spon,im_width*im_length))
y_spon_train=np.empty((train_split_spon,1))

def get_data_spon(folder,start,end,label):
    file_names = os.listdir(folder)
    for i in range(start, end):
        file_path_n = folder + file_names[i]
        im = tf.io.read_file(file_path_n)
        im = tf.image.decode_png(im, channels=1)
        im = tf.image.resize(im, IMAGE_SIZE, method='lanczos5')
        im_array = asarray(im)
        x_spon_train[i,:]=im_array.reshape(1,-1)
        y_spon_train[i,0]=classes[label]
    return x_spon_train,y_spon_train

get_data_spon(spondylosis_path,0,train_split_spon,2)

x_train = np.vstack((x_norm_train,x_scol_train,x_spon_train))
y_train = np.vstack((y_norm_train,y_scol_train,y_spon_train))
y_train=y_train.ravel()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(250, 950, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)