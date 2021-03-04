#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 3 13:21:17 2021

@author: Jannes, Max, Alex

General helper functions for the Bremen Big Data Challenge 2021
"""
import pandas as pd
import numpy as np

def read_train(pathToDataset):
    """Liest Trainingsdaten inklusive Label ein"""
    trainLabels = pd.read_csv(pathToDataset+"/dev-labels.csv") 
    labelList = trainLabels["event_label"].unique()
    trainLabelsOneHot = pd.get_dummies(trainLabels['event_label'])
    trainLabelsOneHot["Noise"] = 0
    labelList = trainLabelsOneHot.columns
    X_train = []
    Y_train = []
    currentFile = ""
    timepoints = np.genfromtxt(pathToDataset+"/dev/00001_mix.csv",delimiter=',')[:,0]
    classNum = 13
    for index, row in trainLabels.iterrows():
        if row["filename"]!=currentFile:
            currentFile = row["filename"]
            features = np.genfromtxt(pathToDataset+"/dev/"+row["filename"].replace(".wav", ".csv"),delimiter=',')
            X_train.append(features[:,1:])
            y = np.zeros((timepoints.size, classNum))
            y[:,-1] = 1
            Y_train.append(y)
        Y_train[-1][np.where(np.logical_and(timepoints>=row["onset"], timepoints<=row["offset"])),:] = trainLabelsOneHot.iloc[index].values
    return X_train, Y_train , labelList, timepoints   

def splitTrain(X_train, Y_train, trainPortion=0.8, valPortion=0.1):
    firstIdx = int(len(X_train)*trainPortion)
    secondIdx = int(len(X_train)*valPortion)+firstIdx
    X_validation = X_train[firstIdx:secondIdx]
    Y_validation = Y_train[firstIdx:secondIdx]
    X_test = X_train[secondIdx:]
    Y_test = Y_train[secondIdx:]
    X_train = X_train[:firstIdx]
    Y_train = Y_train[:firstIdx]
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test


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
    # Alternative implementation
    #return 1-dice_coef(y_true, y_pred)   