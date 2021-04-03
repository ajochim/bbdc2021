#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import keras
#from keras import layers
import tensorflow as tf

def conv_bn_relu_block(x, numChannels, padding, kernel_size=3):
    x = tf.keras.layers.Conv1D(numChannels, kernel_size=kernel_size, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def conv_bn_relu_block_lessParameter(x, numChannels, padding, kernel_size=3):
    x = tf.keras.layers.Conv1D(numChannels, kernel_size=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv1D(numChannels, kernel_size=kernel_size, padding = padding)(x)
    return x


def u_net(inputShape, channels, padding="same", activation="softmax",
          numClasses=13, lessParameter=False, kernel_size=3, first_kernel_size=3, inputLayer = None):
    if padding != "same":
        print("Gibt bisher nur Same-Padding")
        return
    if not(inputLayer is None):
        input_layer = inputLayer
    else:
        input_layer = tf.keras.layers.Input(shape=(inputShape))
    conv_block_function = conv_bn_relu_block
    if lessParameter:
        conv_block_function = conv_bn_relu_block_lessParameter
    
    shortcuts = []
    x = input_layer
    #Encoder
    for numLayer in range(len(channels) - 1):
        if numLayer == 0:
            x = conv_block_function(x, channels[numLayer], padding, kernel_size=first_kernel_size)
            x = conv_block_function(x, channels[numLayer], padding, kernel_size=first_kernel_size)
        else:
            x = conv_block_function(x, channels[numLayer], padding, kernel_size=kernel_size)
            x = conv_block_function(x, channels[numLayer], padding, kernel_size=kernel_size)
        shortcuts.append(x)
        x = tf.keras.layers.MaxPooling1D()(x)
        
    x = conv_block_function(x, channels[numLayer + 1], padding)
    x = conv_block_function(x, channels[numLayer + 1], padding)
    
    #Decoder start
    for numLayer in reversed(range(len(channels) - 1)):
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Concatenate()([x,shortcuts[numLayer]])
        x = conv_block_function(x, channels[numLayer], padding, kernel_size=kernel_size)
        x = conv_block_function(x, channels[numLayer], padding, kernel_size=kernel_size)

    x = tf.keras.layers.Conv1D(numClasses, kernel_size=1, activation=activation,
                               padding=padding)(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model