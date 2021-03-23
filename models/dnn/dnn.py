#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alex
"""

import keras
from keras import layers

def simple_dnn(timepoints, channels):
    """Return keras simple multilayer perceptron model. Flattens array of input channels."""
    model = keras.models.Sequential()
    model.add(layers.Flatten(input_shape=(timepoints, channels)))
    model.add(layers.Dense(400, activation='relu'))
    model.add(layers.Dense(timepoints, activation='relu'))
    return model