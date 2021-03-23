#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alex
"""

import os
import sys  
sys.path.insert(0, './../../')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import bbdc2021 as bbdc

scaling = 'standard'

raw_data_name = 'dataset_fft_l1024_o256_b32'
pathtodata = "./../../data/"
if scaling == 'standard':
    scaler = StandardScaler()
    out_path = pathtodata + 'std_' + raw_data_name + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
if scaling == 'minmax':
    scaler = MinMaxScaler()
    out_path = pathtodata + 'minmax_' + raw_data_name + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

if __name__ == '__main__':
    # load data
    X_tr, Y_tr, timepoints, _ = bbdc.load_data("train.csv", raw_data_name, pathToDataDir=pathtodata)
    X_val, Y_val, _, _ = bbdc.load_data("validation.csv", raw_data_name, pathToDataDir=pathtodata)
    X_test, Y_test, _, _ = bbdc.load_data("test.csv", raw_data_name, pathToDataDir=pathtodata)
    # use dummy entries for all comlumns, but filename in challenge csv filelist!
    X_ch, _ , _, _ = bbdc.load_data("challenge_filelist_dummy.csv", raw_data_name, pathToDataDir=pathtodata, datasettype='eval')
    # fit, every instance because of timeseries shape
    for instance in X_tr:
        scaler.partial_fit(instance)
    for instance in X_val:
        scaler.partial_fit(instance)
    for instance in X_test:
        scaler.partial_fit(instance)
    for instance in X_ch:
        scaler.partial_fit(instance)
    #print(scaler.mean_, scaler.scale_)
    # transform
    for index, instance in enumerate(X_tr):
        X_tr[index] = scaler.transform(instance)
    for index, instance in enumerate(X_val):
        X_val[index] = scaler.transform(instance)
    for index, instance in enumerate(X_test):
        X_test[index] = scaler.transform(instance)
    for index, instance in enumerate(X_ch):
        X_ch[index] = scaler.transform(instance)
    # save data as txt
    np.save((out_path + 'train'), X_tr)
    np.save((out_path + 'validation'), X_val)
    np.save((out_path + 'test'), X_test)
    np.save((out_path + 'challenge_data'), X_ch)
    np.save((out_path + 'train_labels'), Y_tr)
    np.save((out_path + 'validation_labels'), Y_val)
    np.save((out_path + 'test_labels'), Y_test)
    np.save((out_path + 'timepoints'), timepoints)
