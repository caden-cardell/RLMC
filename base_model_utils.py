#!/usr/bin/env python

import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from utils import compute_mape_error, print_model_error
import torch


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


def generate_sequence_sampling_indices(total_sequence_length, input_sequence_length, output_sequence_length):
    num_possible_samples = total_sequence_length - input_sequence_length - output_sequence_length

    # create feature indices
    input_indices = (torch.arange(input_sequence_length).view(1, -1) +
                     torch.arange(num_possible_samples).view(-1, 1))

    # create label indices
    output_indices = (torch.arange(output_sequence_length).view(1, -1) +
                      torch.arange(num_possible_samples).view(-1, 1))
    output_indices = torch.add(output_indices, input_sequence_length)

    return input_indices, output_indices


def import_scaled_data():
    data = load_data_from_csv("75e9_2023_09_07_11_02_to_2023_09_13_11_02.csv")

    # add axis
    data = data[:, np.newaxis]

    # scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    return np.array(data[:, 0])


def save_door_open_data():
    # manually selected ranges
    door_open_ranges = [(160, 240), (700, 750), (820, 870), (1300, 1350), (1900, 1950), (2520, 2570), (3150, 3200),
                        (3850, 3920)]

    # get data
    data = import_scaled_data()

    # collect door open data from corresponding ranges
    door_open_data = [data[start:end] for start, end in door_open_ranges]

    np.save('door_open_ranges.npy', door_open_data)


def unify_data_base_models(X, Y, bms):
    # the predictions are shorter because the feature length is subtracted???
    lengths = []
    for bm in bms:
        lengths.append(len(bm))  # TODO use dim

    length = min(lengths)
    print(length)

    trunc_bms = []
    for bm in bms:
        trunc_bms.append(bm[-length:])
    bms = trunc_bms

    feature_samples = X[-length:]
    label_samples = Y[-length:]

    expanded_bms = []
    for bm in bms:
        expanded_bm = np.expand_dims(bm, axis=1)
        expanded_bms.append(expanded_bm)

    base_model_samples = np.concatenate(expanded_bms, axis=1)

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

    for base_model in range(len(bms)):  # TODO use dim
        print_model_error(base_model, test_y, test_preds[:, base_model])

    test_error_df = compute_mape_error(test_y, test_preds)
    valid_error_df = compute_mape_error(valid_y, valid_preds)
    train_error_df = compute_mape_error(train_y, train_preds)

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


if __name__ == '__main__':
    pass
