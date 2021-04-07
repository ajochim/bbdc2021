#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import keras
from keras import layers
from keras import backend as K

def conv_bn_relu_block(x, numChannels, padding):
    x = layers.Conv2D(numChannels, kernel_size=3, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def u_net_2d(inputShape, channels, padding = "same", activation="softmax", numClasses = 13):
    if padding != "same":
        print("Gibt bisher nur Same-Padding")
        return
    input_layer = layers.Input(shape=(inputShape))
    conv_block_function = conv_bn_relu_block
    
    shortcuts = []
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
    x = layers.Reshape((400,13))(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model