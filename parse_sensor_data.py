import torch
import numpy as np
import re
import matplotlib.pyplot as plt


def load_from_csv(filename='data.csv'):
    x_window_size = 120
    y_window_size = 24

    def extract_last_float(line):
        # Extract all float numbers from the line
        floats = re.findall(r'[-+]?\d*\.\d+|\d+', line)
        # Return the last float if any, else return None
        return floats[-1] if floats else None

    data = []
    with open(filename, 'r') as file:
        for line in file:
            last_float = float(extract_last_float(line))
            data.append(last_float)
    data = np.array(data)

    data_length = len(data) - x_window_size - y_window_size

    # Create indices for extracting windows
    indices = torch.arange(x_window_size).view(1, -1) + torch.arange(data_length).view(-1, 1)    # (445, 50)

    # Use advanced indexing to create train_x
    train_x = data[indices]  # (445, 50)

    # Create indices y
    indices = torch.arange(y_window_size).view(1, -1) + torch.arange(data_length).view(-1, 1)    # (445, 5)
    indices = torch.add(indices, x_window_size)

    # Use advanced indexing to create train_x
    train_y = data[indices]  # (445, 50)

    np.savez('dataset/synthesized_data.npz',
             train_x=train_x,
             train_y=train_y)


if __name__ == '__main__':
    load_from_csv()