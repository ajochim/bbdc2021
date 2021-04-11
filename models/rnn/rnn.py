#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

def rnn(input_shape, layers, n_classes=13, cell_type='gru',
        dilation_rates=(1, 2, 4, 8), conv_filter=20, conv_kernel_size=2):
    """Simple gru network."""
    input_layer = tf.keras.layers.Input(shape=(input_shape))
    x = input_layer
    for rate in dilation_rates * 2:
        x = tf.keras.layers.Conv1D(filters=conv_filter,
                                   kernel_size=conv_kernel_size,
                                   padding="causal", activation="relu",
                                   dilation_rate=rate)(x)
    if cell_type == 'gru':
        for n in layers:
            x = tf.keras.layers.GRU(n, return_sequences=True)(x)
    elif cell_type == 'lstm':
        for n in layers:
            x = tf.keras.layers.LSTM(n, return_sequences=True)(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes, activation='softmax'))(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model
