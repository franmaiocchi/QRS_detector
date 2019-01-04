# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:55:57 2018

@author: Francisco
"""

# Importo TensorFlow como tf
#import tensorflow as tf
## Importo keras
#from tensorflow import keras

# Librerias auxiliares
import numpy as np
import matplotlib.pyplot as plt

#def norm(x):
#    return (x - train_stats['mean']) / train_stats['std']
#normed_train_data = norm(train_dataset)
#normed_test_data = norm(test_dataset)

def testbench():
    
    # Load training ...
    train_path = 'data/training.npy'
    training_set = np.load(train_path)[()]
    
    train_input = training_set.get('signals')
    train_input = (train_input - np.mean(train_input, axis = 1,  keepdims = True)) / np.std(train_input, axis = 1,  keepdims = True)
    
    train_label = training_set.get('labels')
    train_label = (train_label - np.min(train_label, axis = 1, keepdims = True)) / (np.max(train_label, axis = 1, keepdims = True) - np.min(train_label, axis = 1, keepdims = True))
    train_label = train_label/np.sum(train_label, axis = 1, keepdims = True)
    
    # Load validation ...
    validation_path = 'data/validation.npy'
    validation_set = np.load(validation_path)[()]
    
    validation_input = validation_set.get('signals')
    validation_input = (validation_input - np.mean(validation_input, axis = 1,  keepdims = True)) / np.std(validation_input, axis = 1,  keepdims = True)
    
    validation_label = validation_set.get('labels')
    validation_label = (validation_label - np.min(validation_label, axis = 1, keepdims = True)) / (np.max(validation_label, axis = 1, keepdims = True) - np.min(validation_label, axis = 1, keepdims = True))
    validation_label = validation_label/np.sum(validation_label, axis = 1, keepdims = True)
    
    # Load test ...
    test_path = 'data/test.npy'
    test_set = np.load(test_path)[()]
    
    test_input = test_set.get('signals')
    test_input = (test_input - np.mean(test_input, axis = 1,  keepdims = True)) / np.std(test_input, axis = 1,  keepdims = True)

    test_label = test_set.get('labels')
    test_label = (test_label - np.min(test_label, axis = 1, keepdims = True)) / (np.max(test_label, axis = 1, keepdims = True) - np.min(test_label, axis = 1, keepdims = True))
    test_label = test_label/np.sum(test_label, axis = 1, keepdims = True)
    
    print(np.random.randint(np.size(test_input,0), size = 10))
    
#    model = tf.keras.Sequential([
#    # Agrego una capa fully-connected de 64 unidades
#    keras.layers.Dense(200, activation='relu'),
#    # Agrego una capa con activación softmax de 10 neuronas
#    keras.layers.Dense(200, activation='softmax')])
#    
#    # Configuración para regresión con error cuadrático medio 
#    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
#                  loss='mse',       # mean squared error
#                  metrics=['mae'])  # mean absolute error

testbench()