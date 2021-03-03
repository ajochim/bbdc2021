#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 3 13:21:17 2021

@author: Jannes, Max, Alex

General helper functions for the Bremen Big Data Challenge 2021
"""
import pandas as pd
import numpy as np

def read_csv_data(pathToDataset): #TODO Val und Test abspalten
    """Liest Trainingsdaten inklusive Label ein"""
    trainLabels = pd.read_csv(pathToDataset+"/dev-labels.csv") 
    trainLabelsOneHot = pd.get_dummies(trainLabels['event_label'])
    trainLabelsOneHot["Noise"] = 0
    X_train = []
    Y_train = []
    currentFile = ""
    timepoints = np.genfromtxt(pathToDataset+"/dev/00001_mix.csv",delimiter=',')[:,0]
    classNum = 13
    for index, row in trainLabels.iterrows():
        if row["filename"]!=currentFile:
            features = np.genfromtxt(pathToDataset+"/dev/"+row["filename"].replace(".wav", ".csv"),delimiter=',')
            X_train.append(features[:,1:])
            y = np.zeros((timepoints.size, classNum))
            y[:,-1] = 1
            Y_train.append(y)
        Y_train[-1][np.where(np.logical_and(timepoints>=row["onset"], timepoints<=row["offset"])),:] = trainLabelsOneHot.iloc[index].values
    return X_train, Y_train    
