#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import keras
from keras import layers

def conv_bn_relu_block(x, numChannels, padding):
    x = layers.Conv1D(numChannels, kernel_size=3, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def u_net(inputShape, channels, padding = "same", activation="softmax", numClasses = 13):
    if padding != "same":
        print("Gibt bisher nur Same-Padding")
        return
    input_layer = layers.Input(shape=(inputShape))
    shortcuts = []
    x = input_layer
    #Encoder
    for numLayer in range(len(channels)-1):
        x = conv_bn_relu_block(x, channels[numLayer], padding)
        x = conv_bn_relu_block(x, channels[numLayer], padding)
        shortcuts.append(x)
        x = layers.MaxPooling1D()(x)
        
    x = conv_bn_relu_block(x, channels[numLayer], padding)
    x = conv_bn_relu_block(x, channels[numLayer], padding)
    
    #Decoder start
    for numLayer in reversed(range(len(channels)-1)):
        x = layers.UpSampling1D(2)(x)
        x = layers.Concatenate()([x,shortcuts[numLayer]])
        x = conv_bn_relu_block(x, channels[numLayer], padding)
        x = conv_bn_relu_block(x, channels[numLayer], padding)

    x = layers.Conv1D(numClasses, kernel_size=1, activation=activation, padding=padding)(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model