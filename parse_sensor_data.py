import torch
import numpy as np
import re
import math


def load_data_from_csv(input_filepath, output_filepath):
    x_window_size = 120
    y_window_size = 24

    def extract_last_float(line):
        # Extract all float numbers from the line
        floats = re.findall(r'[-+]?\d*\.\d+|\d+', line)
        # Return the last float if any, else return None
        return floats[-1] if floats else None

    data = []
    with open(input_filepath, 'r') as file:
        for line in file:
            last_float = float(extract_last_float(line))
            data.append(last_float)
    data = np.array(data)

    data_length = len(data) - x_window_size - y_window_size

    # Create indices for extracting windows
    indices = torch.arange(x_window_size).view(1, -1) + torch.arange(data_length).view(-1, 1)

    # Use advanced indexing to create train_x
    all_x = data[indices]

    # Create indices y
    indices = torch.arange(y_window_size).view(1, -1) + torch.arange(data_length).view(-1, 1)
    indices = torch.add(indices, x_window_size)

    # Use advanced indexing to create train_x
    all_y = data[indices]

    np.savez(output_filepath,
             all_x=all_x[:, :, np.newaxis],  # add dimension because this model works with multi-feature data
             all_y=all_y)


def synthesize_predictions(input_filepath='real_data/synthesized_data.npz'):
    """
        IMPORT REAL DATA
    """
    input_data = np.load(input_filepath)

    # separate data
    all_x = input_data['all_x']
    all_y = input_data['all_y']

    # get important dimensions of data
    x_window_size = all_x.shape[1]
    y_window_size = all_y.shape[1]
    number_of_points = len(all_y) + x_window_size + y_window_size

    # create y indices
    indices = torch.arange(y_window_size).view(1, -1) + torch.arange(number_of_points).view(-1, 1)
    indices = torch.add(indices, x_window_size)

    """
        SYNTHESIS PREDICTION DATA
    """

    # generate prediction error
    start_mean, end_mean = 0, 5
    start_std, end_std = 1, 3

    # Create tensors for linearly interpolated mean and standard deviation
    means = torch.linspace(start_mean, end_mean, number_of_points)
    stds = torch.linspace(start_std, end_std, number_of_points)

    # Sample from the normal distribution for each index
    dist_1_prediction_deltas = torch.stack([torch.normal(mean, std) for mean, std in zip(means, stds)])

    # Define parameters for the sine wave
    amplitude = 3  # Amplitude of the sine wave
    frequency = 2 * math.pi / number_of_points  # Frequency - one full cycle over the number of points

    # Create a tensor for the x values
    x_values = torch.linspace(0, number_of_points - 1, number_of_points)

    # Calculate the sine wave for the means
    means = amplitude * torch.sin(frequency * x_values)

    # Define standard deviation range
    start_std, end_std = 4, 1

    # Create tensors for linearly interpolated standard deviation
    stds = torch.linspace(start_std, end_std, number_of_points)

    # Sample from the normal distribution for each index
    dist_2_prediction_deltas = torch.stack([torch.normal(mean, std) for mean, std in zip(means, stds)])

    dist_1_pred = dist_1_prediction_deltas[indices] + all_y
    dist_2_pred = dist_2_prediction_deltas[indices] + all_y




if __name__ == '__main__':
    load_data_from_csv(input_filepath='data.csv', output_filepath='real_data/all_data.npz')
    synthesize_predictions(input_filepath='real_data/all_data.npz')
