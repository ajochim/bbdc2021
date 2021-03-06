#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 3 13:21:17 2021

@author: Jannes, Max, Alex

General helper functions for the Bremen Big Data Challenge 2021
"""
import pandas as pd
import numpy as np
import keras.backend as K
import tensorflow as tf

LABEL_DICT =  {'Noise': 0, 'Bark': 1, 'Burping_and_eructation': 2, 'Camera':3, 'Cheering':4, 'Church_bell':5, 'Cough':6, 'Doorbell':7, 'Fireworks':8, 'Meow':9, 'Scratching_(performance_technique)':10, 'Shatter':11, 'Shout':12}


def load_data(fileListName, datasetName, pathToDataDir="./../data/"):
    """Liest Daten inklusive Label ein"""
    ## Label laden und zu One-Hot codieren
    fileList = pd.read_csv(pathToDataDir+fileListName)
    fileList = fileList.replace({"event_label": LABEL_DICT}) #Zuerst zu numerischen Werten per Dictionary konvertieren, um zu vermeiden bei fehlenden Labels falsche Decodings zu erzeugen
    trainLabels = fileList["event_label"].values
    trainLabelsOneHot = np.zeros((trainLabels.size, trainLabels.max()+1))
    trainLabelsOneHot[np.arange(trainLabels.size),trainLabels] = 1
    ## Features aller Files laden
    X = []
    Y = []
    currentFile = ""
    timepoints = np.genfromtxt(pathToDataDir+datasetName+"/dev/00001_mix.csv",delimiter=',')[:,0]
    for index, row in fileList.iterrows():
        if row["filename"]!=currentFile:
            currentFile = row["filename"]
            features = np.genfromtxt(pathToDataDir+datasetName+"/dev/"+row["filename"].replace(".wav", ".csv"),delimiter=',')
            X.append(features)
            y = np.zeros((timepoints.size, len(LABEL_DICT)))
            y[:,0] = 1
            Y.append(y)
        Y[-1][np.where(np.logical_and(timepoints>=row["onset"], timepoints<=row["offset"])),:] = trainLabelsOneHot[index]
    return X, Y , timepoints, fileList   

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(y_true * y_pred, axis=-1)
    denominator = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)+smooth
    return tf.math.divide_no_nan(denominator - (2.*intersection+smooth), denominator)
    # Alternative implementation
    #intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    #return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)