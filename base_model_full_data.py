#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler
from lstm import LSTM
from my_utils import import_scaled_data, get_sample_indices, TimeSeriesDataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# hyper parameters
FEATURE_LENGTH = 7
LABEL_LENGTH = 1
LEARNING_RATE = 0.001
NUMBER_OF_EPOCHS = 30
TRAIN_TO_TEST_RATIO = 0.7
batch_size = 16


def import_scaled_data_door_open():

    data = import_scaled_data()

    door_open = [(160, 240), (700, 750), (820, 870), (1300, 1350), (1900, 1950), (2520, 2570), (3150, 3200),
                 (3850, 3920)]

    # # plotting
    # plt.plot(data, label="data", color='red')
    #
    # for idx in range(len(door_open)):
    #     each = door_open[len(door_open) - idx - 1]
    #     plt.plot(data[0: each[1]], color='blue')
    #     plt.plot(data[0: each[0]], color='red')
    #
    # # labeling
    # plt.xlabel('120 second time steps')
    # plt.ylabel('Degrees F')
    # plt.title('Data')
    # plt.legend()
    #
    # # Display the plot
    # plt.show()

    # init empty arrays
    feature_framed_data = np.empty((0, FEATURE_LENGTH))
    label_framed_data = np.empty((0, LABEL_LENGTH))

    for points in door_open:
        # get sub_data
        sub_data = data[points[0]:points[1]]

        # get frames of data
        sub_data_length = len(sub_data)
        feature_indices, label_indices = get_sample_indices(sub_data_length, x_window_size=FEATURE_LENGTH,
                                                            y_window_size=LABEL_LENGTH)
        feature_framed_sub_data = np.array(sub_data)[feature_indices]
        label_framed_sub_data = np.array(sub_data)[label_indices]

        # concat all data
        feature_framed_data = np.append(feature_framed_data, feature_framed_sub_data, axis=0)
        label_framed_data = np.append(label_framed_data, label_framed_sub_data, axis=0)

    return feature_framed_data, label_framed_data


def main():
    X, y = import_scaled_data_door_open()

    # split train and test data
    split_index = int(len(X) * TRAIN_TO_TEST_RATIO)
    X_train = X[:split_index]
    X_test = X[split_index:]

    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    y_train = y[:split_index]
    y_test = y[split_index:]

    # convert to tensors
    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    # create data sets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # for _, batch in enumerate(train_loader):
    #     x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    #     print(x_batch.shape, y_batch.shape)
    #     break

    model = LSTM(1, 5, 1, device=device)
    model.to(device)

    def train_one_epoch():
        model.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.5f}'.format(batch_index + 1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print()

    def validate_one_epoch():
        model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

        print('Val Loss: {0:.5f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUMBER_OF_EPOCHS):
        train_one_epoch()
        validate_one_epoch()

    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()

    # plt.plot(Y_train[:, 0], label='Actual Temperature')
    # plt.plot(predicted[:, 0], label='Predicted Temperature')
    # plt.xlabel('120 Second Time Step')
    # plt.ylabel('Degrees F')
    # plt.legend()
    # plt.show()

    # do whole thing
    # get frames of data
    data = import_scaled_data()

    data_length = len(data)
    feature_indices, label_indices = get_sample_indices(data_length, x_window_size=FEATURE_LENGTH, y_window_size=LABEL_LENGTH)

    feature_framed_data = data[feature_indices]
    label_framed_data = data[label_indices]

    X = feature_framed_data
    Y = label_framed_data

    X = X[:, :, np.newaxis]

    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()

    with torch.no_grad():
        predicted = model(X.to(device)).to('cpu').numpy()

    plt.plot(Y[:, 0], label='Actual Temperature')
    plt.plot(predicted[:, 0], label='Predicted Temperature')
    plt.xlabel('120 Second Time Step')
    plt.ylabel('Degrees F')
    plt.legend()
    plt.show()

    np.save('predicted.npy', predicted)


if __name__ == '__main__':
    main()