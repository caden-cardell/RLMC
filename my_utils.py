#!/usr/bin/env python

import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

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


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]