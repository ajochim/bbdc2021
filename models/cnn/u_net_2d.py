#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def conv_bn_relu_block(x, numChannels, padding):
    x = layers.Conv2D(numChannels, kernel_size=3, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def u_net_2d(inputShape, channels, padding = "same", activation="softmax", numClasses = 13, inputLayer = None):
    if padding != "same":
        print("Gibt bisher nur Same-Padding")
        return
    conv_block_function = conv_bn_relu_block
    
    shortcuts = []
    if not(inputLayer is None):
        input_layer = inputLayer
    else:
        input_layer = tf.keras.layers.Input(shape=(inputShape))
    x = input_layer
    x = layers.Reshape((K.int_shape(x)[1], K.int_shape(x)[2], 1))(x)
    #Encoder
    for numLayer in range(len(channels)-1):
        x = conv_block_function(x, channels[numLayer], padding)
        x = conv_block_function(x, channels[numLayer], padding)
        shortcuts.append(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        
    x = conv_block_function(x, channels[numLayer+1], padding)
    x = conv_block_function(x, channels[numLayer+1], padding)
    
    #Decoder start
    for numLayer in reversed(range(len(channels)-1)):
        x = layers.UpSampling2D(size=(2,2))(x)
        x = layers.Concatenate()([x,shortcuts[numLayer]])
        x = conv_block_function(x, channels[numLayer], padding)
        x = conv_block_function(x, channels[numLayer], padding)
    x = layers.MaxPool2D(pool_size=(1,32))(x)
    x = layers.Conv2D(numClasses, kernel_size=1, activation=activation, padding=padding)(x)
    x = layers.Reshape((400,numClasses))(x)

    model = tensorflow.keras.models.Model(inputs=input_layer, outputs=x)
    return model