#!/usr/bin/env python

import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from utils import compute_mape_error, compute_mae_error, print_model_error

import torch
from torch.utils.data import Dataset, DataLoader


def load_data_from_csv(input_filepath):
    def extract_last_float(line):
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        # Get the last number and convert it to float
        last_number = float(numbers[-1])
        return last_number

    data = []
    with open(input_filepath, 'r') as file:
        for line in file:
            last_float = float(extract_last_float(line))
            data.append(last_float)

    return np.array(data)


def get_sample_indices(data_length, x_window_size=120, y_window_size=24):
    sample_length = data_length - x_window_size - y_window_size

    # create feature indices
    feature_indices = torch.arange(x_window_size).view(1, -1) + torch.arange(sample_length).view(-1, 1)

    # create label indices
    label_indices = torch.arange(y_window_size).view(1, -1) + torch.arange(sample_length).view(-1, 1)
    label_indices = torch.add(label_indices, x_window_size)

    return feature_indices, label_indices


def import_scaled_data():
    data = load_data_from_csv("75e9_2023_09_07_11_02_to_2023_09_13_11_02.csv")
    # print(data)

    # scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))

    data = scaler.fit_transform(np.array(data[:, np.newaxis]))
    return np.array(data[:, 0])


def unify_data_base_models(X, Y, bm_1, bm_2):
    # bm_1 = np.load('bm_1.npy')
    # bm_2 = np.load('bm_2.npy')
    # X = np.load('X.npy')
    # Y = np.load('Y.npy')

    # the predictions are shorter because the feature length is subtracted???
    length = len(bm_1)
    feature_samples = X[-length:]
    label_samples = Y[-length:]

    bm_1 = np.expand_dims(bm_1, axis=1)
    bm_2 = np.expand_dims(bm_2, axis=1)

    base_model_samples = np.concatenate([bm_1, bm_2], axis=1)

    # L = len(feature_samples) // 8
    L = len(feature_samples) - 2993

    test_X = feature_samples[-L:]
    valid_X = feature_samples[-(2 * L):-L]
    train_X = feature_samples[:-(2 * L)]

    test_y = label_samples[-L:]
    valid_y = label_samples[-(2 * L):-L]
    train_y = label_samples[:-(2 * L)]

    test_preds = base_model_samples[-L:]
    valid_preds = base_model_samples[-(2 * L):-L]
    train_preds = base_model_samples[:-(2 * L)]

    # test_X = feature_samples[:L]
    # valid_X = feature_samples[L:(2 * L)]
    # train_X = feature_samples[(2 * L):]
    #
    # test_y = label_samples[:L]
    # valid_y = label_samples[L:(2 * L)]
    # train_y = label_samples[(2 * L):]
    #
    # test_preds = base_model_samples[:L]
    # valid_preds = base_model_samples[L:(2 * L)]
    # train_preds = base_model_samples[(2 * L):]

    test_error_df  = compute_mape_error(test_y , test_preds)
    valid_error_df = compute_mape_error(valid_y, valid_preds)
    train_error_df = compute_mape_error(train_y, train_preds)

    print_model_error(1, test_y, test_preds[:, 0])
    print_model_error(2, test_y, test_preds[:, 1])

    np.save('dataset/bm_test_preds.npy', test_preds)
    np.save('dataset/bm_valid_preds.npy', valid_preds)
    np.save('dataset/bm_train_preds.npy', train_preds)

    np.savez('dataset/input.npz',
             test_X=test_X,
             valid_X=valid_X,
             train_X=train_X,
             test_y=test_y,
             valid_y=valid_y,
             train_y=train_y,
             test_error=test_error_df,
             valid_error=valid_error_df,
             train_error=train_error_df
            )


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]