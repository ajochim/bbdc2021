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
from itertools import groupby
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import colors 

np.random.seed(1)
tf.random.set_seed(1)

LABEL_DICT =  {'Noise': 0, 'Bark': 1, 'Burping_and_eructation': 2, 'Camera':3, 'Cheering':4, 'Church_bell':5, 'Cough':6, 'Doorbell':7, 'Fireworks':8, 'Meow':9, 'Scratching_(performance_technique)':10, 'Shatter':11, 'Shout':12}
invLabelMap = {v: k for k, v in LABEL_DICT.items()}


def load_data(fileListName, datasetName, pathToDataDir="./../data/"):
    """Liest Daten inklusive Label ein"""
    ## Label laden und zu One-Hot codieren
    labelDf = pd.read_csv(pathToDataDir+fileListName)
    labelDf = labelDf.replace({"event_label": LABEL_DICT}) #Zuerst zu numerischen Werten per Dictionary konvertieren, um zu vermeiden bei fehlenden Labels falsche Decodings zu erzeugen
    trainLabels = labelDf["event_label"].values
    trainLabelsOneHot = np.zeros((trainLabels.size, trainLabels.max()+1))
    trainLabelsOneHot[np.arange(trainLabels.size),trainLabels] = 1
    ## Features aller Files laden
    X = []
    Y = []
    fileList = []
    currentFile = ""
    timepoints = np.genfromtxt(pathToDataDir+datasetName+"/dev/00001_mix.csv",delimiter=',')[:,0]
    for index, row in labelDf.iterrows():
        if row["filename"]!=currentFile:
            currentFile = row["filename"]
            fileList.append(currentFile)
            features = np.genfromtxt(pathToDataDir+datasetName+"/dev/"+row["filename"].replace(".wav", ".csv"),delimiter=',')
            X.append(features[:,1:])
            y = np.zeros((timepoints.size, len(LABEL_DICT)))
            y[:,0] = 1
            Y.append(y)
        Y[-1][np.where(np.logical_and(timepoints>=row["onset"], timepoints<=row["offset"])),:] = trainLabelsOneHot[index]
    return np.array(X), np.array(Y) , timepoints, fileList   

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    # Alternative implementation
    #intersection = K.sum(y_true * y_pred, axis=-1)
    #denominator = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)+smooth
    #return tf.math.divide_no_nan(denominator - (2.*intersection+smooth), denominator)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def diceOfDF(y_true_df, y_pred_df, smooth=1):
    pass

def getPredictionAsSequenceDF(prediction, timepoints, fileList):
    y_predicted = np.argmax(prediction, axis=2)
    detectedEvents = []
    for fileNumber in range(len(y_predicted)):
        pred = y_predicted[fileNumber]
        predAndTime = np.zeros((len(pred),2))
        predAndTime[:,0] = pred
        predAndTime[:,1] = timepoints
        groups = [list(group) for key, group in groupby(predAndTime, itemgetter(0))]
        for group in groups:
            firstTime = group[0][1]
            lastTime = group[-1][1]
            key = int(group[0][0])
            if invLabelMap[key] != "Noise":
                detectedEvents.append([fileList[fileNumber], firstTime, lastTime, invLabelMap[key]]) 
    return pd.DataFrame(detectedEvents, columns=["filename", "onset", "offset", "event_label"])

def plotPredictionAndGT(y_true, y_pred, case): 
    y_trueCase = np.argmax(y_true[case], axis=1)
    y_predCase = np.argmax(y_pred[case], axis=1)
    cmap = colors.ListedColormap(['black', 'red','green', 'orange', 'violet', 'blue', 'darkgreen', 'white', 'darkblue', 'brown', 'darkred', 'gray', 'lightblue'])
    both = np.stack((y_trueCase, y_predCase), axis=0)
    plt.figure(figsize=(30,20))
    plt.imshow(both, cmap)

def plotPredictionAndGTFromDf(y_true_df, y_pred_df, case, timeDelta = 100):
    sequenceLength = 10
    timepoints = np.linspace(start=0, stop=sequenceLength, num=timeDelta)
    cmap = colors.ListedColormap(['black', 'red','green', 'orange', 'violet', 'blue', 'darkgreen', 'white', 'darkblue', 'brown', 'darkred', 'gray', 'lightblue'])
    caseTrueDf = y_true_df.loc[y_true_df["filename"].str.match("0+"+str(case)+"_mix")]
    casePredDf = y_pred_df.loc[y_pred_df["filename"].str.match("0+"+str(case)+"_mix")]
    if caseTrueDf.shape[0]==0:
        print("Case nicht in Datensatz")
        return
    toPlot = np.zeros((2, timeDelta))
    for index, row in caseTrueDf.iterrows():
        label = row["event_label"]
        if isinstance(label, str):
            label = LABEL_DICT(label)
        toPlot[0, np.where(np.logical_and(timepoints>=row["onset"], timepoints<=row["offset"]))] = label
    for index, row in casePredDf.iterrows():
        label = row["event_label"]
        if isinstance(label, str):
            label = LABEL_DICT[label]
        toPlot[1, np.where(np.logical_and(timepoints>=row["onset"], timepoints<=row["offset"]))] = label
    print(toPlot[0])
    plt.figure(figsize=(30,20))
    plt.imshow(toPlot, cmap)