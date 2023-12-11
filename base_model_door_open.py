import numpy as np
from lstm import LSTM
from base_model_utils import import_scaled_data, generate_sequence_sampling_indices
from base_models.trainer import Trainer
from base_models.time_series_dataset import TimeSeriesDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
FEATURE_LENGTH = 7
LEARNING_RATE = 0.001
NUMBER_OF_EPOCHS = 30
TRAIN_TO_TEST_RATIO = 0.7
BATCH_SIZE = 16
LSTM_HIDDEN_SIZE = 5


def import_scaled_data_door_open():

    data = import_scaled_data()

    door_open_points = [(160, 240), (700, 750), (820, 870), (1300, 1350), (1900, 1950), (2520, 2570), (3150, 3200),
                        (3850, 3920)]

    # init empty arrays
    feature_framed_data = np.empty((0, FEATURE_LENGTH))
    label_framed_data = np.empty((0, 1))

    for point in door_open_points:
        # get sub_data
        sub_data = data[point[0]:point[1]]

        # get frames of data
        feature_indices, label_indices = generate_sequence_sampling_indices(total_sequence_length=len(sub_data),
                                                                            input_sequence_length=FEATURE_LENGTH,
                                                                            output_sequence_length=1)

        feature_framed_sub_data = np.array(sub_data)[feature_indices]
        label_framed_sub_data = np.array(sub_data)[label_indices]

        # concat all data
        feature_framed_data = np.append(feature_framed_data, feature_framed_sub_data, axis=0)
        label_framed_data = np.append(label_framed_data, label_framed_sub_data, axis=0)

    return feature_framed_data, label_framed_data


def train_and_predict_model_with_door_open(seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    ################################
    # Train model on training data
    ################################

    # import data
    X, y = import_scaled_data_door_open()

    # split train and test data and add dimension to X
    split_index = int(len(X) * TRAIN_TO_TEST_RATIO)
    X_train = X[:split_index, :, np.newaxis]
    X_test = X[split_index:, :, np.newaxis]
    y_train = y[:split_index]
    y_test = y[split_index:]

    # convert to tensors
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    # create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # initialize model and move to device
    model = LSTM(1, LSTM_HIDDEN_SIZE, 1, device=device)
    model.to(device)

    # train model
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model_trainer = Trainer(model, train_loader, test_loader, optimizer, loss_function, NUMBER_OF_EPOCHS)
    model_trainer.train()

    ################################
    # Use model on entire data set
    ################################
    data = import_scaled_data()

    feature_indices, _ = generate_sequence_sampling_indices(total_sequence_length=len(data),
                                                            input_sequence_length=FEATURE_LENGTH,
                                                            output_sequence_length=1)

    X = data[feature_indices]
    X = X[:, :, np.newaxis]  # add dimension

    # convert to sensor
    X = torch.tensor(X).float()

    with torch.no_grad():
        predicted = model(X.to(device)).to('cpu').numpy()

    return predicted


if __name__ == '__main__':
    train_and_predict_model_with_door_open(seed=3)
