#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alex
"""
import sys
sys.path.insert(0, './../')
import bbdc2021 as bbdc

def pipeline_u_net_1(pipe_param, model_param):
    """Current u_net pipeline from loading data to prediction.
    Takes paramater dictionary as argument"""
    # load
    x_dev, y_dev, timep, filelist_dev, x_ch, filelist_ch = bbdc.loading_block1(pipe_param)
    # split
    x_tv, x_test, y_tv, y_test, filelist_test = bbdc.split_block1(x_dev, y_dev,
                                                                  filelist_dev,
                                                                  pipe_param)
    # model fit
    history, model = bbdc.model_block1_unet(x_tv, x_test, y_tv, y_test, model_param)
    # evaluate
    bbdc.evaluation_block1(x_test, y_test, timep, filelist_test, model, pipe_param)
    # challenge prediction
    bbdc.challenge_prediction_block1(x_ch, timep, filelist_ch, model, pipe_param)

def main():
    """Main function for different hyperparmeter searches."""
    pipe_param = {'data_folder': './../data/',
                  'wav_files_folder': 'final_pre_dataset',
                  'window_length': 1024, 'window_overlap': 523,
                  'band_size': 4, 'sample_rate': 16000,
                  'dev_csv': 'dev-labels_mini.csv',
                  'eval_csv': 'challenge_filelist_dummy_mini.csv',
                  'scaling': 'no',
                  'test_split_fraction': 0.1,
                  'prediction_path': './test_pred.csv',
                  'submission_file_path': './test_challenge_submission.csv'}
    unet_param = {'channels': [32, 64, 90],
                  'lessParameter': True,
                  'val_split_fraction': 1/9,
                  'loss': 'categorical_crossentropy',
                  'learning_rate': 0.001,
                  'batch_size': 15,
                  'epochs': 100}
    pipeline_u_net_1(pipe_param, unet_param)

if __name__ == '__main__':
    main()
