# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:26:59 2020

@author: Aditya Tiwari
"""

import keras
import tensorflow as tf
from tensorflow import keras
keras.__version__
tf.__version__
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
train_dir='C:/Users/Aditya Tiwari/Downloads/cats_and_dogs/train'
test_dir=r'C:\Users\Aditya Tiwari\Downloads\cats_and_dogs\test'
valid_dir=r'C:\Users\Aditya Tiwari\Downloads\cats_and_dogs\validation'
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
#%%
#Data Preprocessing
# Generating Batches , rescaling images
train_datagen= ImageDataGenerator(resacale=1./255)
test_datagen= ImageDataGenerator(rescale=1./255)

# use of flow_from_Directory method to feed the data into class train_datagen
train_generator= train_datagen.flow_from_directory(
     train_dir,
     target_size=(150,150),
     batch_size=20,
     class_mode='binary')

validation_generator= test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')
#%% Model Creation

from tensorflow.keras import layers
from tensorflow.keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
            input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
    
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
    
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
    
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
#%% Model compilation
from tensorflow.keras import optimizers
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc'])