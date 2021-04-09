#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 3 13:21:17 2021

@author: Jannes, Max, Alex

General helper functions for the Bremen Big Data Challenge 2021
"""
import os
import glob
import shutil
from itertools import groupby
from operator import itemgetter
from importlib import reload
import scipy.io.wavfile
from scipy.signal import stft
import soundfile as sf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import colors
#import keras
import tensorflow.keras.backend as K
import tensorflow as tf
import evaluation.evaluate as evaluate
import models.cnn.u_net_1d as unet
reload(unet)
import models.cnn.u_net_2d as unet2d
reload(unet2d)
from tqdm import tqdm
import ast

np.random.seed(1)
tf.random.set_seed(1)

LABEL_DICT = {'Noise': 0, 'Bark': 1, 'Burping_and_eructation': 2, 'Camera': 3,
              'Cheering': 4, 'Church_bell': 5, 'Cough': 6, 'Doorbell': 7,
              'Fireworks': 8, 'Meow': 9,
              'Scratching_(performance_technique)': 10, 'Shatter': 11,
              'Shout': 12}

invLabelMap = {v: k for k, v in LABEL_DICT.items()}

def data_folder_format_string(load_param, csv_stage=True):
    """Returns a format string that can be used for the saving path."""
    format_string = ''
    if csv_stage:
        format_string = 'fft_'
    format_string = format_string + 'l' + str(load_param['window_length'])
    format_string = format_string + 'o' + str(load_param['window_overlap'])
    format_string = format_string + 'b' + str(load_param['band_size'])
    format_string = format_string + 's' + str(load_param['sample_rate'])
    if not csv_stage:
        format_string = format_string + '_'
        format_string = format_string + str(load_param['scaling']) + 'scaling'
    format_string = format_string + '/'
    return format_string

def data_folder_format_string_mel(load_param, csv_stage=True):
    """Returns a format string that can be used for the saving path."""
    format_string = ''
    if csv_stage:
        format_string = 'mel_'
    format_string = format_string + 'size' + str(load_param['frame_size'])
    format_string = format_string + 'stride' + str(load_param['frame_stride'])
    format_string = format_string + 'nfft' + str(load_param['nfft'])
    format_string = format_string + 'mel_filter' + str(load_param['mel_filter'])
    if not csv_stage:
        format_string = format_string + '_'
        format_string = format_string + 'mel_'
        format_string = format_string + str(load_param['scaling']) + 'scaling'
    format_string = format_string + '/'
    return format_string

def load_and_calc_features(files, length=1024, overlap=256, band_size=32,
                           sample_rate=16000, verbose=True):
    """Loads all files in the passed files array and calculates the sftft on
    them. Returns two dictionaries:
    feats: {filename: numpy array fft calculated on windows}
    times: {filename: array containing the times of each window}"""
    max_len = len(files)
    max_freq = length//2
    feats = {}
    times = {}
    for file in files:
        f = file.split('/')[-1]
        if verbose and len(feats) % 1000 == 0:
            print(len(feats), '/', max_len)
        data, _ = sf.read(file)
        _, samp_times, data_spec = stft(data, fs=sample_rate, window='blackman',
                                        nperseg=length, noverlap=overlap)
        data_spec = np.log(np.abs(data_spec) + 0.00000000001)
        data_final = np.zeros((max_freq//band_size, data_spec.shape[1] - 1))
        for i in range(0, max_freq, band_size):
            data_final[int(i//band_size) - 1, :] =\
            np.sqrt(np.sum(np.square(data_spec[i:i+band_size, :-1]), axis=0))
        feats[f] = data_final
        times[f] = samp_times
    return feats, times

def plot_fft(fft, times, name):
    """Plots fft data."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.patch.set_facecolor('white')
    ax.imshow(fft, origin="lower", aspect="auto")
    # use Time instead of window number. Uncomment line to use window number.
    #ax.xaxis.set_major_formatter(lambda val, _:
    #times[int(val)] if int(val) >= 0 and int(val) <= len(times) else '')
    plt.xlabel("Zeit")
    plt.ylabel("Frequenzband")
    plt.title("Frequenzen in {}".format(name))
    plt.tight_layout()

def calc_fft(param):
    """Calculates the fft data fromt the wav files. Uses parameter dictionary
    as an argument."""
    # read paramter from dictionary
    dataset_loc = param['data_folder'] + param['wav_files_folder']
    out_folder = param['data_folder'] + data_folder_format_string(param,
                                                                  csv_stage=True)
    window_length = param['window_length'] #Orginal-Skript: 1024
    window_overlap = param['window_overlap'] #Original-Skript 256
    band_size = param['band_size'] #Original-Skript 32
    sample_rate = param['sample_rate']  #Original-Skript 16000
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    os.makedirs(out_folder + '/dev')
    os.makedirs(out_folder + '/eval')
    shutil.copyfile(dataset_loc+"/dev-labels.csv", out_folder+"/dev-labels.csv")
    print('Processing dev files:')
    # train files
    train_files = sorted([x.split('\\')[-1] \
                          for x in glob.glob(f'{dataset_loc}/dev/*.wav')])
    #max_len = len(train_files)
    # load and save train files (we could pass the full array to the function,
    # but not everyone might have the mem space to do so)
    for i, file_name in tqdm(enumerate(train_files)):
        #if i % 1000 == 0:
        #    print(i, '/', max_len)
        # load
        train_feats, train_times = load_and_calc_features([file_name],
                                                          length=window_length,
                                                          overlap=window_overlap,
                                                          sample_rate=sample_rate,
                                                          verbose=False,
                                                          band_size=band_size)
        name = list(train_feats.keys())[0]
        # merge time and fft
        tmp = np.concatenate([np.expand_dims(train_times[name][:-1], axis=0),
                              train_feats[name]], axis=0).T
        # save to csv
        np.savetxt(out_folder + '/dev/' + name.replace('.wav', '.csv'), tmp,
                   delimiter=',')
    # plot example fft:
    #print('===', 'Plotting example fft for', name, '============')
    #plot_fft(train_feats[name], train_times[name], name)
    #plt.savefig(name.replace('.wav', '.png'))
    print('Processing eval files:')
    # load eval files
    eval_files = sorted([x.split('\\')[-1] \
                         for x in glob.glob(f'{dataset_loc}/eval/*.wav')])
    #max_len = len(eval_files)
    # save eval files
    for i, file_name in tqdm(enumerate(eval_files)):
        #if i % 1000 == 0:
        #    print(i, '/', max_len)
        eval_feats, eval_times = load_and_calc_features([file_name],
                                                        length=window_length,
                                                        overlap=window_overlap,
                                                        band_size=band_size,
                                                        sample_rate=sample_rate,
                                                        verbose=False)
        name = list(eval_feats.keys())[0]
        tmp = np.concatenate([np.expand_dims(eval_times[name][:-1], axis=0),
                              eval_feats[name]], axis=0).T
        np.savetxt(out_folder + '/eval/' + name.replace('.wav', '.csv'), tmp,
                   delimiter=',')

def getFilterBank(sample_rate, nfilt, param):
    nfft = param['nfft']
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return fbank

def getFrames(signal, sample_rate, param):
    frame_size = param['frame_size']
    frame_stride = param['frame_stride']
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal,z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    time = np.linspace(0,int(len(signal)/sample_rate), num_frames)
    return frames, time

def getMelFilteredFreqs(filename, nFilter, param, reduceNoise = True, amplifyHighFreqs = True):
    nfft = param['nfft']
    sample_rate, signal = scipy.io.wavfile.read(filename)  # File assumed to be in the same directory
    if len(signal)==0:
        print(filename)
        return None, None
    if signal.ndim>1: #Check for stereo
        signal = signal[:,0] #Use only mono
    if amplifyHighFreqs:
        pre_emphasis = 0.97
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frames, time = getFrames(signal, sample_rate, param)
    
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))  # Magnitude of the FFT
    pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))  # Power Spectrum

    fbank=getFilterBank(sample_rate, nFilter, param)
    
    melFiltered_signal = np.dot(pow_frames, fbank.T)
    melFiltered_signal = np.where(melFiltered_signal == 0, np.finfo(float).eps, melFiltered_signal)  # Numerical Stability
    melFiltered_signal = 20 * np.log10(melFiltered_signal)  # dB
    if reduceNoise:
        melFiltered_signal -= (np.mean(melFiltered_signal, axis=0) + 1e-8)
    return melFiltered_signal, time

def calc_fft_mel(param):
    """Calculates the fft data fromt the wav files with mel filters. Uses
    parameter dictionary as an argument."""
    # read paramter from dictionary
    dataset_loc = param['data_folder'] + param['wav_files_folder']
    out_folder = param['data_folder'] + data_folder_format_string_mel(param,
                                                                      csv_stage=True)
    mel_filter = param['mel_filter']
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    os.makedirs(out_folder + '/dev')
    os.makedirs(out_folder + '/eval')
    shutil.copyfile(dataset_loc+"/dev-labels.csv", out_folder+"/dev-labels.csv")
    print('Processing dev files:')
    # train files
    train_files = sorted([x.split('\\')[-1] \
                          for x in glob.glob(f'{dataset_loc}/dev/*.wav')])
    #max_len = len(train_files)
    # load and save train files (we could pass the full array to the function,
    # but not everyone might have the mem space to do so)
    for i, file_name in tqdm(enumerate(train_files)):
        #if i % 1000 == 0:
        #    print(i, '/', max_len)
        # load
        train_feats, train_times = getMelFilteredFreqs(file_name, mel_filter,
                                                       param,
                                                       reduceNoise=True,
                                                       amplifyHighFreqs=True)
        if train_feats is None or train_times is None:
            continue
        # merge time and fft
        tmp = np.concatenate([np.expand_dims(train_times, axis=1), train_feats], axis=1)
        # save to csv
        name = file_name.split('/')[-1]
        np.savetxt(out_folder + '/dev/' + name.replace('.wav', '.csv'), tmp, delimiter=',')
    # plot example fft:
    #print('===', 'Plotting example fft for', name, '============')
    #plot_fft(train_feats[name], train_times[name], name)
    #plt.savefig(name.replace('.wav', '.png'))
    print('Processing eval files:')
    # load eval files
    eval_files = sorted([x.split('\\')[-1] for x in glob.glob(f'{dataset_loc}/eval/*.wav')])
    #max_len = len(eval_files)
    # save eval files
    for i, file_name in tqdm(enumerate(eval_files)):
        #if i % 1000 == 0:
        #    print(i, '/', max_len)
        eval_feats, eval_times = getMelFilteredFreqs(file_name, mel_filter, param)
        # merge time and fft
        tmp = np.concatenate([np.expand_dims(eval_times, axis=1), eval_feats], axis=1)
        # save to csv
        name = file_name.split('/')[-1]
        np.savetxt(out_folder + '/eval/' + name.replace('.wav', '.csv'), tmp, delimiter=',')

def load_data(fileListName, datasetName, pathToDataDir="./../data/"):
    """Loads csv data with labels. Challenge dummy csv file can be used to
    also load challenge data."""
    ## Label laden und zu One-Hot codieren
    labelDf = pd.read_csv(pathToDataDir+fileListName)
    # Zuerst zu numerischen Werten per Dictionary konvertieren,
    # um zu vermeiden bei fehlenden Labels falsche Decodings zu erzeugen
    labelDf = labelDf.replace({"event_label": LABEL_DICT})
    trainLabels = labelDf["event_label"].values
    trainLabelsOneHot = np.zeros((trainLabels.size, trainLabels.max()+1))
    trainLabelsOneHot[np.arange(trainLabels.size), trainLabels] = 1
    ## Features aller Files laden
    X = []
    Y = []
    fileList = []
    currentFile = ""
    first_file_string = '/' + labelDf['filename'][0][:-4] + '.csv'
    timepoints = np.genfromtxt(pathToDataDir+datasetName+first_file_string,
                               delimiter=',')[:, 0]
    for index, row in tqdm(labelDf.iterrows()):
        if row["filename"] != currentFile:
            currentFile = row["filename"]
            fileList.append(currentFile)
            features = np.genfromtxt(pathToDataDir + datasetName \
                                     + row["filename"].replace(".wav", ".csv"),
                                     delimiter=',')
            X.append(features[:, 1:])
            y = np.zeros((timepoints.size, len(LABEL_DICT)))
            y[:, 0] = 1
            Y.append(y)
        Y[-1][np.where(np.logical_and(timepoints >= row["onset"],\
          timepoints <= row["offset"])), :] = trainLabelsOneHot[index]
    return np.array(X), np.array(Y), timepoints, fileList

def load_train_test(path, label_file="train.csv", test_size=[], **kwargs):
    ''' Loads train data (and test data if required, suffling possible) '''
    X_next, Y_next, timepoints, trainFileList = load_data(label_file, path)
    X_out = []
    Y_out = []
    
    if not isinstance(test_size, list):
        test_size = [test_size]
        
    for i in range(0, len(test_size)):
        X_save, X_next, Y_save, Y_next = train_test_split(X_next, Y_next, test_size=np.prod(test_size[:i+1]), **kwargs)
        X_out.append(X_save)
        Y_out.append(Y_save)
        
    X_out.append(X_next)
    Y_out.append(Y_next)
    return *X_out, *Y_out

def load_audioset(fileListName, datasetName, pathToDataDir="./../googleData/fft/"):
    """Loads csv data with labels. Challenge dummy csv file can be used to
    also load challenge data."""
    ## Label laden und zu One-Hot codieren
    labelDf = pd.read_csv(pathToDataDir+fileListName)
    # Zuerst zu numerischen Werten per Dictionary konvertieren,
    # um zu vermeiden bei fehlenden Labels falsche Decodings zu erzeugen
    labelDf['event_label'] = labelDf['event_label'].apply(ast.literal_eval)
    ## Features aller Files laden
    X = []
    Y = []
    fileList = []
    for index, row in tqdm(labelDf.iterrows()):
        currentFile = row["# YTID"]
        fileList.append(currentFile)
        features = np.genfromtxt(pathToDataDir + datasetName \
                                 + currentFile+".csv",
                                 delimiter=',')
        X.append(features[:, 1:])
        y = np.zeros(len(LABEL_DICT))
        y[0] = 1
        for label in row["event_label"]:
            y[LABEL_DICT[label]]=1
        Y.append(y)
    return np.array(X), np.array(Y), fileList

def scale(x_dev, x_challenge, scaling='no'):
    """Saves and returns dev and challenge data. Reads data if already
    existent."""
    if scaling == 'no':
        return x_dev, x_challenge
    elif scaling == 'standard':
        scaler = StandardScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        print('Unknown scaling mode.')
        return None
    print(scaling, 'scaling chosen.')
    # fit
    for instance in x_dev:
        scaler.partial_fit(instance)
    for instance in x_challenge:
        scaler.partial_fit(instance)
    #transform
    for index, instance in enumerate(x_dev):
        x_dev[index] = scaler.transform(instance)
    for index, instance in enumerate(x_challenge):
        x_challenge[index] = scaler.transform(instance)
    return x_dev, x_challenge

def loading_block1(param):
    """Loading block for pipeline. Uses parameter in param dictionary to
    calculate csv files from raw wav data, scales it it and create usable
    numpy arrays. Saves both, csv files and numpy arrays in two different
    folders, named by data_folder_format_string function. Checks if
    calculations are already existent and reads file if they are to save time.
    """
    csv_folder = (param['data_folder']
                  + data_folder_format_string(param, csv_stage=True))
    #fft from wav
    if not os.path.exists(csv_folder):
        print('Starting transformation from wav files to csv files.')
        calc_fft(param)
    else:
        print('Csv from wav files already existend. Skipping calc_fft.')
    # load
    npy_folder = (param['data_folder']
                  + data_folder_format_string(param, csv_stage=False))
    dev_csv = param['dev_csv']
    eval_csv = param['eval_csv']
    if os.path.exists(npy_folder):
        print('Scaled numpy files already existend.',
              'Skipping scaling and load_data function.')
        X_dev = np.load(npy_folder + 'X_dev.npy')
        Y_dev = np.load(npy_folder + 'Y_dev.npy')
        timepoints = np.load(npy_folder + 'timepoints.npy')
        filelist_dev = np.load(npy_folder + 'filelist_dev.npy')
        X_challenge = np.load(npy_folder + 'X_challenge.npy')
        filelist_challenge = np.load(npy_folder + 'filelist_challenge.npy')
    else:
        print('Loading dev set:')
        X_dev, Y_dev, timepoints, filelist_dev = load_data(dev_csv,
                                                           csv_folder + 'dev/',
                                                           pathToDataDir=param['data_folder'])
        print('Loading eval set:')
        X_challenge, _, _, filelist_challenge = load_data(eval_csv,
                                                          csv_folder + 'eval/',
                                                          pathToDataDir=param['data_folder'])
        print('Scaling files.')
        X_dev, X_challenge = scale(X_dev, X_challenge, scaling='no')
        print('Saving to numpy arrays.')
        os.makedirs(npy_folder)
        np.save((npy_folder + 'X_dev'), X_dev)
        np.save((npy_folder + 'Y_dev'), Y_dev)
        np.save((npy_folder + 'timepoints'), timepoints)
        np.save((npy_folder + 'filelist_dev'), filelist_dev)
        np.save((npy_folder + 'X_challenge'), X_challenge)
        np.save((npy_folder + 'filelist_challenge'), filelist_challenge)
    return X_dev, Y_dev, timepoints, filelist_dev, X_challenge, filelist_challenge

def loading_block2(param):
    """
    Like loading_block1 but with mel filters.
    
    Loading block for pipeline. Uses parameter in param dictionary to
    calculate csv files from raw wav data, scales it it and create usable
    numpy arrays. Saves both, csv files and numpy arrays in two different
    folders, named by data_folder_format_string function. Checks if
    calculations are already existent and reads file if they are to save time.
    """
    csv_folder = (param['data_folder']
                  + data_folder_format_string_mel(param, csv_stage=True))
    print('Mel filter version loading block.')
    #mel from wav
    if not os.path.exists(csv_folder):
        print('Starting transformation from wav files to csv files (mel).')
        calc_fft_mel(param)
    else:
        print('Csv from wav files already existend. Skipping calc_fft_mel.')
    # load
    npy_folder = (param['data_folder']
                  + data_folder_format_string_mel(param, csv_stage=False))
    dev_csv = param['dev_csv']
    eval_csv = param['eval_csv']
    if os.path.exists(npy_folder):
        print('Scaled numpy files already existend.',
              'Skipping scaling and load_data function.')
        X_dev = np.load(npy_folder + 'X_dev.npy')
        Y_dev = np.load(npy_folder + 'Y_dev.npy')
        timepoints = np.load(npy_folder + 'timepoints.npy')
        filelist_dev = np.load(npy_folder + 'filelist_dev.npy')
        X_challenge = np.load(npy_folder + 'X_challenge.npy')
        filelist_challenge = np.load(npy_folder + 'filelist_challenge.npy')
    else:
        print('Loading dev set:')
        X_dev, Y_dev, timepoints, filelist_dev = load_data(dev_csv,
                                                           csv_folder + 'dev/',
                                                           pathToDataDir=param['data_folder'])
        print('Loading eval set:')
        X_challenge, _, _, filelist_challenge = load_data(eval_csv,
                                                          csv_folder + 'eval/',
                                                          pathToDataDir=param['data_folder'])
        print('Scaling files.')
        X_dev, X_challenge = scale(X_dev, X_challenge, scaling='no')
        print('Saving to numpy arrays.')
        os.makedirs(npy_folder)
        np.save((npy_folder + 'X_dev'), X_dev)
        np.save((npy_folder + 'Y_dev'), Y_dev)
        np.save((npy_folder + 'timepoints'), timepoints)
        np.save((npy_folder + 'filelist_dev'), filelist_dev)
        np.save((npy_folder + 'X_challenge'), X_challenge)
        np.save((npy_folder + 'filelist_challenge'), filelist_challenge)
    return X_dev, Y_dev, timepoints, filelist_dev, X_challenge, filelist_challenge

def shuffle_block1(X_dev, Y_dev, filelist_dev, random_seed='random'):
    """Shuffelt trainings daten mitsamt filelist."""
    if random_seed == 'random':
        X_dev, Y_dev, filelist_dev = shuffle(X_dev, Y_dev, filelist_dev)
    else:
        X_dev, Y_dev, filelist_dev = shuffle(X_dev, Y_dev, filelist_dev,
                                             random_state=random_seed)
    return X_dev, Y_dev, filelist_dev

def split_block1(X_dev, Y_dev, timepoints, filelist_dev, split_param):
    """Splits data set by selecting the last chunk for the test set."""
    i_min, i_max = split_param['test_split_range']
    print('Splitting test set at indices', i_min, 'to', i_max, 'from dev set.')
    X_test, Y_test = X_dev[i_min:i_max], Y_dev[i_min:i_max]
    X_train_val = np.delete(X_dev, list(range(i_min, i_max)), axis=0)
    Y_train_val = np.delete(Y_dev, list(range(i_min, i_max)), axis=0)
    testFileList = filelist_dev[i_min:i_max]
    df = getPredictionAsSequenceDF(Y_test, timepoints, testFileList)
    df.to_csv("test_ref_current.csv", index=False)
    return X_train_val, X_test, Y_train_val, Y_test, testFileList

def plot_history(history, measure='accuracy'):
    """Plots history for a speicic measure (accuracy, mae, loss etc)."""
    plt.plot(history.history[measure])
    plt.plot(history.history['val_' + measure])
    plt.title('model ' + measure)
    plt.ylabel(measure)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def model_block1_unet(X_train_val, Y_train_val, unet_param):
    """Trains unet."""
    if os.path.exists('model.h5'):
        os.remove('model.h5')
        print('Existing model.h5 removed.')
    print('Tensorflow version:', tf.__version__)
    if unet_param['load_path'] is not None:
        print('Loading existing model from path:', unet_param['load_path'])
        model = tf.keras.models.load_model(unet_param['load_path'])
        history = None
        return history, model
    channels = unet_param['channels']
    inputShape = X_train_val[0].shape
    model = unet.u_net(inputShape, channels,
                       lessParameter=unet_param['lessParameter'],
                       kernel_size=unet_param['kernel_size'],
                       first_kernel_size=unet_param['first_kernel_size'])
    val_fraction = unet_param['val_split_fraction']
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val,
                                                      test_size=val_fraction,
                                                      shuffle=False)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', verbose=1,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    mode='auto')
    opt = tf.keras.optimizers.Adam(learning_rate=unet_param['learning_rate'])
    model.compile(optimizer=opt, loss=unet_param['loss'],
                  metrics=['mae', 'accuracy'])
    history = model.fit(X_train, Y_train, batch_size=unet_param['batch_size'],
                        epochs=unet_param['epochs'],
                        validation_data=(X_val, Y_val),
                        shuffle=True, callbacks=[checkpoint])
    model.save(unet_param['model_save_path'] + 'model')
    plot_history(history, 'accuracy')
    plot_history(history, 'mae')
    plot_history(history, 'loss')
    return history, model

def model_block2_unet(X_train_val, Y_train_val, unet_param):
    """Trains unet."""
    if os.path.exists('model.h5'):
        os.remove('model.h5')
        print('Existing model.h5 removed.')
    print('Tensorflow version:', tf.__version__)
    if unet_param['load_path'] is not None:
        print('Loading existing model from path:', unet_param['load_path'])
        model = tf.keras.models.load_model(unet_param['load_path'])
        history = None
        return history, model
    channels = unet_param['channels']
    inputShape = X_train_val[0].shape
    model = unet.u_net(inputShape, channels,
                       lessParameter=unet_param['lessParameter'],
                       kernel_size=unet_param['kernel_size'],
                       first_kernel_size=unet_param['first_kernel_size'])
    i_min, i_max = unet_param['val_split_range']
    print('Splitting val set at indices', i_min, 'to', i_max, 'from train set.')
    X_val, Y_val = X_train_val[i_min:i_max], Y_train_val[i_min:i_max]

    X_train = np.delete(X_train_val, list(range(i_min, i_max)), axis=0)
    Y_train = np.delete(Y_train_val, list(range(i_min, i_max)), axis=0)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', verbose=1,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    mode='auto')
    opt = tf.keras.optimizers.Adam(learning_rate=unet_param['learning_rate'])
    model.compile(optimizer=opt, loss=unet_param['loss'],
                  metrics=['mae', 'accuracy'])
    history = model.fit(X_train, Y_train, batch_size=unet_param['batch_size'],
                        epochs=unet_param['epochs'],
                        validation_data=(X_val, Y_val),
                        shuffle=True, callbacks=[checkpoint])
    model.save(unet_param['model_save_path'] + 'model')
    plot_history(history, 'accuracy')
    plot_history(history, 'mae')
    plot_history(history, 'loss')
    return history, model

def model_block1_unet2d(X_train_val, Y_train_val, unet_param):
    """Trains unet2d."""
    if os.path.exists('model.h5'):
        os.remove('model.h5')
        print('Existing model.h5 removed.')
    print('Tensorflow version:', tf.__version__)
    if unet_param['load_path'] is not None:
        print('Loading existing model from path:', unet_param['load_path'])
        model = tf.keras.models.load_model(unet_param['load_path'])
        history = None
        return history, model
    channels = unet_param['channels']
    inputShape = X_train_val[0].shape
    model = unet2d.u_net_2d(inputShape, channels)
    i_min, i_max = unet_param['val_split_range']
    print('Splitting val set at indices', i_min, 'to', i_max, 'from train set.')
    X_val, Y_val = X_train_val[i_min:i_max], Y_train_val[i_min:i_max]

    X_train = np.delete(X_train_val, list(range(i_min, i_max)), axis=0)
    Y_train = np.delete(Y_train_val, list(range(i_min, i_max)), axis=0)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', verbose=1,
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    mode='auto')
    opt = tf.keras.optimizers.Adam(learning_rate=unet_param['learning_rate'])
    model.compile(optimizer=opt, loss=unet_param['loss'],
                  metrics=['mae', 'accuracy'])
    history = model.fit(X_train, Y_train, batch_size=unet_param['batch_size'],
                        epochs=unet_param['epochs'],
                        validation_data=(X_val, Y_val),
                        shuffle=True, callbacks=[checkpoint])
    model.save(unet_param['model_save_path'] + 'model')
    plot_history(history, 'accuracy')
    plot_history(history, 'mae')
    plot_history(history, 'loss')
    return history, model

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return ((2. * intersection + smooth) / (K.sum(K.square(y_true), -1)
                                            + K.sum(K.square(y_pred), -1)
                                            + smooth))
    # Alternative implementation
    #intersection = K.sum(y_true * y_pred, axis=-1)
    #denominator = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)+smooth
    #return tf.math.divide_no_nan(denominator - (2.*intersection+smooth), denominator)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def diceOfDF(y_true_df, y_pred_df, smooth=1):
    pass

def getPredictionAsSequenceDF(prediction, timepoints, fileList, calculateProbs=False):
    """Creates sequence from one-hot encoded model prediction."""
    detectedEvents = []
    for fileNumber in range(len(prediction)):
        groups = groupSequences(prediction[fileNumber], timepoints)
        for group in groups:
            firstTime = group[0][1]
            lastTime = group[-1][1]
            firstIndex = int(group[0][2])
            lastIndex = int(group[-1][2])
            key = int(group[0][0])
            columns = ["filename", "onset", "offset", "event_label"]
            events = [fileList[fileNumber], firstTime, lastTime, invLabelMap[key]]
            if calculateProbs:
                columns.append("MeanProb")
                columns.append("MedianProb")
                events.append(np.mean(prediction[fileNumber, firstIndex:lastIndex, key]))
                events.append(np.median(prediction[fileNumber, firstIndex:lastIndex, key]))
            if invLabelMap[key] != "Noise":
                detectedEvents.append(events)
    return pd.DataFrame(detectedEvents, columns=columns)

def plot_confusion_matrix(y_true, y_pred):
    """Plots confusion matrix usind sklearn. Transforms one hot encoded predictions."""
    y_true = np.argmax(y_true, axis=2).flatten()
    y_pred = np.argmax(y_pred, axis=2).flatten()
    mtrx = confusion_matrix(y_true, y_pred)
    print(mtrx)
    plt.imshow(mtrx)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.show()
    plt.imshow(mtrx[1:, 1:])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.show()

def evaluation_block1(X_test, Y_test, timepoints, testFileList, model,
                      eval_param):
    """Evaluates current model on test data from dev set. Also calculates
    PSDS score."""
    prediction = model.predict(X_test)
    scores_list = model.evaluate(X_test, Y_test)
    print('')
    print('Evaluation:')
    print('Loss, MAE, Accuracy', scores_list)
    plot_confusion_matrix(Y_test, prediction)
    prediction_df = getPredictionAsSequenceDF(prediction, timepoints, testFileList)
    pred_csv_path = eval_param['prediction_path'] + 'test_pred_model.csv'
    prediction_df.to_csv(pred_csv_path, index=False) 
    #dev_csv_path = eval_param['data_folder'] + eval_param['dev_csv']
    #dev_df = pd.read_csv(dev_csv_path, header=0, usecols=[0, 1, 2, 3])
    #test_df = dev_df[dev_df.filename.isin(testFileList)]
    test_csv_path = eval_param['prediction_path'] + 'test_ref_current.csv'
    psds_info = evaluate.evaluate(pred_csv_path, test_csv_path)
    print('PSDS', psds_info)
    print('')
    return scores_list, psds_info[0]
 
def get_labelcolormap():
    """Gets colormap and norm for plotting labels."""
    cmap = colors.ListedColormap(['black', 'red', 'green', 'orange', 'violet',
                                  'blue', 'darkgreen', 'white', 'darkblue',
                                  'brown', 'darkred', 'gray', 'lightblue'])
    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
                  10.5, 11.5, 12.5]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm

def plotPredictionAndGT(y_true, y_pred, case):
    """Plots a comparision between ground truth and prediction."""
    y_trueCase = np.argmax(y_true[case], axis=1)
    y_predCase = np.argmax(y_pred[case], axis=1)
    cmap, norm = get_labelcolormap()
    both = np.stack((y_trueCase, y_predCase), axis=0)
    plt.figure(figsize=(30, 20))
    plt.imshow(both, cmap=cmap, norm=norm)

def plotPredictionAndGTFromDf(y_true_df, y_pred_df, case, timeDelta=100):
    """Plots a comparision between ground truth and prediction from pandas
    dataframes."""
    sequenceLength = 10
    timepoints = np.linspace(start=0, stop=sequenceLength, num=timeDelta)
    caseTrueDf = y_true_df.loc[y_true_df["filename"].str.match("0+"+str(case)+"_mix")]
    casePredDf = y_pred_df.loc[y_pred_df["filename"].str.match("0+"+str(case)+"_mix")]
    cmap, norm = get_labelcolormap()
    if caseTrueDf.shape[0] == 0:
        print("Case nicht in Datensatz")
        return
    toPlot = np.zeros((2, timeDelta))
    for index, row in caseTrueDf.iterrows():
        label = row["event_label"]
        if isinstance(label, str):
            label = LABEL_DICT(label)
        toPlot[0, np.where(np.logical_and(timepoints >= row["onset"],
                                          timepoints <= row["offset"]))] = label
    for index, row in casePredDf.iterrows():
        label = row["event_label"]
        if isinstance(label, str):
            label = LABEL_DICT[label]
        toPlot[1, np.where(np.logical_and(timepoints >= row["onset"],
                                          timepoints <= row["offset"]))] = label
    print(toPlot[0])
    plt.figure(figsize=(30, 20))
    plt.imshow(toPlot, cmap=cmap, norm=norm)

def postProcess(prediction_one_hot, timepoints, timeThresh=0.5, noiseThresh=0.3,
                base=2):
    """Post process by filling up values between predictions."""
    groups = groupSequences(prediction_one_hot, timepoints)
    classNum = prediction_one_hot.shape[1]
    improvedPrediction = np.zeros((prediction_one_hot.shape[0], classNum+1))
    for group in groups:
        firstTime = group[0][1]
        lastTime = group[-1][1]
        firstIndex = int(group[0][2])
        lastIndex = int(group[-1][2])
        key = int(group[0][0])
        if lastTime-firstTime > timeThresh or\
        (key == 0 and (lastTime - firstTime>noiseThresh or 10 - lastTime < 0.1\
                     or firstTime < 0.1)):
            improvedPrediction[firstIndex:lastIndex + 1, key] = 1
        else:
            improvedPrediction[firstIndex:lastIndex + 1, -1] = 1
    finalPrediction = np.zeros(prediction_one_hot.shape)
    groups = groupSequences(improvedPrediction, timepoints)
    for i in range(len(groups)):
        group = groups[i]
        firstTime = group[0][1]
        lastTime = group[-1][1]
        firstIndex = int(group[0][2])
        lastIndex = int(group[-1][2])
        key = int(group[0][0])
        if key != prediction_one_hot.shape[1]: #Bereich wurde als sicher markiert
            finalPrediction[firstIndex:lastIndex + 1, key] =\
            prediction_one_hot[firstIndex:lastIndex+1, key]
        else: #Bereich unsicher, weil zu kurz
            probabilitiesOfGroup = prediction_one_hot[firstIndex:lastIndex + 1]
            probabilityForClass = np.sum(probabilitiesOfGroup, axis=0) #TODO prod oder sum?
            ownLength = lastIndex - firstIndex + 1
            beforeLength = np.zeros(classNum)
            afterLength = np.zeros(classNum)
            isNoiseSurrounded = True
            if i > 0:
                beforeLength = getSequenceLength(groups[i-1], classNum)
                isNoiseSurrounded = (groups[i - 1][0][0]==0)
            if i + 1 < len(groups):
                afterLength = getSequenceLength(groups[i + 1], classNum)
                isNoiseSurrounded = isNoiseSurrounded\
                and (groups[i + 1][0][0] == 0)
            if isNoiseSurrounded and lastTime-firstTime >= timeThresh:
                newKey = np.argmax(probabilityForClass)
            else:
                overallLength = beforeLength+afterLength + ownLength
                maxLength = np.max(overallLength)
                #lengthFactor = np.exp2(-maxLength/overallLength)
                lengthFactor = base ** (-maxLength/overallLength)
                #print(lengthFactor)
                #print((probabilityForClass*100).astype(int))
                weightedProbabilities = probabilityForClass*lengthFactor
                #print(weightedProbabilities)
                newKey = np.argmax(weightedProbabilities)
            finalPrediction[firstIndex:lastIndex + 1, newKey] =\
            probabilitiesOfGroup[:, newKey]
    return finalPrediction

def post_process_window(prediction_one_hot, class_threshold=0.05):
    """Uses a window with 1s and 2s and fills class that has the most probable
    prediction. Uses at least two classes and maximum.
    class_threshold in percentage of total time. Decides if class will be included."""
    for instance in prediction_one_hot[:]:
        # find 4 most probable classes
        label_encoded = np.argmax(instance, axis=1)
        print(label_encoded)
        values, counts = np.unique(label_encoded, return_counts=True)
        print(values, counts)
        most_counted_4_indexes = counts.argsort()[-5:-1][::-1]
        print(most_counted_4_indexes)
        most_counted_4 = values[most_counted_4_indexes]
        print(most_counted_4)
        # klassen auch oft doppelt! einfach direkt windows abfahren?
        # vlt nur von denen die mind 1 mal predicted wurde um zeit zu sparen

        # window function 1s to 2s?

        # get most probable
        
        # check overlap
        
        # del 1 or 2 depening on threshold

        # create new sequence
    # wahrscheinlich nicht lohnenswert zu implementieren
    pass

def postprocessing_with_evaluation_block1(x_test, y_test, timepoints,
                                          test_file_list, x_challenge,
                                          challenge_file_list, model, param):
    """Post processes test and challenge files. Evaluates new PSDS on test set
    and returns post processed test and prediction data
    """
    prediction = model.predict(x_test)
    challenge_prediction = model.predict(x_challenge)
    if param['post_processing'] == 'no':
        print('No post processing used')
        return prediction, challenge_prediction
    elif param['post_processing'] == 'fill':
        print('Filling post processing used.')
        timethres = param['post_timethres']
        noisethres = param['post_noisethres']
        base = param['post_base']
        post_processed_test = np.array([postProcess(pred, timepoints, timeThresh=timethres, noiseThresh=noisethres,
                base=base) for pred in prediction])
        post_processed_prediction = np.array([postProcess(pred, timepoints, timeThresh=timethres, noiseThresh=noisethres,
                base=base) for pred in challenge_prediction])
        plot_confusion_matrix(y_test, post_processed_test)
        # calculate psds for test set
        df_test_pred = getPredictionAsSequenceDF(post_processed_test, timepoints, test_file_list)
        pred_csv_path = param['prediction_path'] + 'test_pred_postprocessed.csv'
        df_test_pred.to_csv(pred_csv_path, index=False)
        #dev_csv_path = param['data_folder'] + param['dev_csv']
        #dev_df = pd.read_csv(dev_csv_path, header=0, usecols=[0, 1, 2, 3])
        #test_df = dev_df[dev_df.filename.isin(test_file_list)]
        test_csv_path = param['prediction_path'] + 'test_ref_current.csv'
        post_processed_psds_info = evaluate.evaluate(pred_csv_path, test_csv_path)
        print('PSDS', post_processed_psds_info)    
    return post_processed_test, post_processed_prediction, post_processed_psds_info[0]

def getSequenceLength(group, classNum):
    groupLength = np.zeros(classNum)
    firstIndex = int(group[0][2])
    lastIndex = int(group[-1][2])
    key = int(group[0][0])
    groupLength[key] = lastIndex-firstIndex
    return groupLength

def groupSequences(prediction_one_hot, timepoints):
    y_predicted = np.argmax(prediction_one_hot, axis=-1)
    predAndTime = np.zeros((len(y_predicted), 3))
    predAndTime[:, 0] = y_predicted
    predAndTime[:, 1] = timepoints
    predAndTime[:, 2] = np.arange(len(y_predicted))
    return [list(group) for key, group in groupby(predAndTime, itemgetter(0))]

def challenge_prediction_block1(challenge_prediction, timepoints,
                                filelist_challenge, param):
    """Takes X_challenge data and creates submission csv file."""
    df = getPredictionAsSequenceDF(challenge_prediction, timepoints,
                                   filelist_challenge, calculateProbs=True)
    df.to_csv(param['submission_file_path'], index=False)
    print('Submission file saved at', param['submission_file_path'])
