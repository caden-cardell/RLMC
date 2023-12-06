#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # !/usr/bin/env python
    # coding: utf-8

    # In[1]:

    torch.rand(1)  # to test that numbers are random

    torch.manual_seed(0)

    # In[2]:

    # hyper parameters
    FEATURE_LENGTH = 7
    LABEL_LENGTH = 1
    LEARNING_RATE = 0.001
    NUMBER_OF_EPOCHS = 20
    TRAIN_TO_TEST_RATIO = 0.7


    # In[3]:

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


    data = load_data_from_csv("75e9_2023_09_07_11_02_to_2023_09_13_11_02.csv")

    # In[4]:

    # plotting
    plt.plot(data, label="data", color='red')

    # labeling
    plt.xlabel('120 second time steps')
    plt.ylabel('Degrees F')
    plt.title('Data')
    plt.legend()

    # Display the plot
    plt.show()


    # In[5]:

    def get_sample_indices(data_length, x_window_size=120, y_window_size=24):

        sample_length = data_length - x_window_size - y_window_size

        # create feature indices
        feature_indices = torch.arange(x_window_size).view(1, -1) + torch.arange(sample_length).view(-1, 1)

        # create label indices
        label_indices = torch.arange(y_window_size).view(1, -1) + torch.arange(sample_length).view(-1, 1)
        label_indices = torch.add(label_indices, x_window_size)

        return feature_indices, label_indices


    # get frames of data
    data_length = len(data)
    feature_indices, label_indices = get_sample_indices(data_length, x_window_size=FEATURE_LENGTH,
                                                        y_window_size=LABEL_LENGTH)  # y window doesn' tmatter because it's not used

    feature_framed_data = np.array(data)[feature_indices]  # [:, :, np.newaxis]
    label_framed_data = np.array(data)[label_indices]  # [:, :, np.newaxis]

    feature_framed_data.shape, label_framed_data.shape

    # In[6]:

    # scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))

    feature_framed_data_fit = scaler.fit_transform(feature_framed_data)
    label_framed_data_fit = scaler.fit_transform(label_framed_data)

    feature_framed_data_fit.shape, label_framed_data_fit.shape

    # In[7]:

    X = feature_framed_data_fit
    Y = label_framed_data_fit

    X.shape, Y.shape

    # In[8]:

    # split train and test data
    split_index = int(len(X) * TRAIN_TO_TEST_RATIO)

    split_index

    # In[9]:

    X_train = X[:split_index]
    X_test = X[split_index:]

    Y_train = Y[:split_index]
    Y_test = Y[split_index:]

    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # In[10]:

    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    Y_train = Y_train
    Y_test = Y_test

    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # In[25]:

    plt.plot(X[:, 0], label='Test')
    plt.plot(X_train[:, 0], label='Train')
    plt.xlabel('120 Second Time Step')
    plt.ylabel('Degrees F')
    plt.legend()
    plt.show()

    # In[12]:

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    Y_train = torch.tensor(Y_train).float()
    Y_test = torch.tensor(Y_test).float()

    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    # In[13]:

    from torch.utils.data import Dataset


    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]


    train_dataset = TimeSeriesDataset(X_train, Y_train)
    test_dataset = TimeSeriesDataset(X_test, Y_test)

    # In[14]:

    from torch.utils.data import DataLoader

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # In[15]:

    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break


    # In[16]:

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers, output_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_stacked_layers = num_stacked_layers

            self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                                batch_first=True)

            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.randn(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out


    model = LSTM(1, 2, 1, 1)
    model.to(device)
    model


    # In[17]:

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


    # In[18]:

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


    # In[19]:

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUMBER_OF_EPOCHS):
        train_one_epoch()
        validate_one_epoch()

    # In[20]:

    X = X[:, :, np.newaxis]

    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()

    # In[21]:

    with torch.no_grad():
        predicted = model(X.to(device)).to('cpu').numpy()

    other_predicted = np.load('predicted.npy')

    plt.plot(Y[:, 0], label='Actual Temperature')
    plt.plot(other_predicted[:, 0], label='Other Predicted Temperature')
    plt.plot(predicted[:, 0], label='Predicted Temperature')
    plt.xlabel('120 Second Time Step')
    plt.ylabel('Degrees F')
    plt.legend()
    plt.show()

    # In[22]:

    # save to file
    print(predicted.shape)
    np.save('bm_3_preds.npy', predicted)

    # In[23]:

    np.save('bm_1.npy', other_predicted)
    np.save('bm_2.npy', predicted)
    np.save('X.npy', X)
    np.save('Y.npy', Y)

