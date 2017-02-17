### Python skript to create and train a model

# Import al nescesary libraries

import pickle
import numpy as np
import math
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import random
from keras.layers import Input, Flatten, Dense, Convolution2D, MaxPooling2D, Activation, Dropout, Cropping2D, Lambda
from keras.models import Sequential, Model
import time
import os
import h5py

print('All modules loaded!')

import cv2
print ("You are a champ! You are running OpenCv3 on python 3.5 #BigTime")

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

print("keras properly loaded :)")

batch_size = 100
epochs = 20
pool_size = (2, 2)
input_vec = (160, 320, 3)

# Create Model
model = Sequential()

# Crop Layer
model.add(Cropping2D(cropping=((40,20),(0,0)),input_shape=input_vec))

# Normalizing Lambda Layer
model.add(Lambda(lambda x: (x/255.0)-0.5))

# Convolutional Layer 1 and Dropout
model.add(Convolution2D(5, 5, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Conv Layer 2
model.add(Convolution2D(5, 5, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 3
model.add(Convolution2D(3, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 4
model.add(Convolution2D(3, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=pool_size))

# Flatten and Dropout
model.add(Flatten())
model.add(Dropout(0.5))

# Fully Connected Layer 1 and Dropout
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# FC 2 and Dropout
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# FC Layer 3
model.add(Dense(128))
model.add(Activation('relu'))

# FC Layer 4
model.add(Dense(32))
model.add(Activation('relu'))

# Final FC Layer - just one output - steering angle
model.add(Dense(1))

print("Model created")

# Select image directory

imDir = "fd3"

# Generator

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Generator for flipped images

def flipedGenerator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_image = np.fliplr(center_image)
                center_angle = float(batch_sample[3])
                center_angle = -center_angle
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Compile and train model with Generator

start = time.time()
#batch_size = 100
epochs = 5
pool_size = (2, 2)

samples = []
with open('./'+imDir+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
#print(samples)

# Train with normal images
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
model.compile(metrics=['mean_squared_error'], optimizer='adam', loss='mean_squared_error')
history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, verbose=2, nb_val_samples=len(validation_samples), nb_epoch=epochs)

# Train with Fliped images
train_generator = flipedGenerator(train_samples, batch_size=32)
validation_generator = flipedGenerator(validation_samples, batch_size=32)
model.compile(metrics=['mean_squared_error'], optimizer='adam', loss='mean_squared_error')
history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, verbose=2, nb_val_samples=len(validation_samples), nb_epoch=epochs)

print("Model trained sucsessfully!")

# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

print("Model trained sucsessfully!")

model.save_weights('model.h5')

# Show summary of model
model.summary()

end = time.time()
print("Model trained in %(1)d seconds"%{"1":end-start})