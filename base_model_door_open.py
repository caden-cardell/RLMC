#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from lstm import LSTM
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.rand(1)  # to test that numbers are random

# In[2]:


# hyper parameters
FEATURE_LENGTH = 7
LABEL_LENGTH = 1
LEARNING_RATE = 0.001
NUMBER_OF_EPOCHS = 30
TRAIN_TO_TEST_RATIO = 0.7

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


# scale data
scaler = MinMaxScaler(feature_range=(-1, 1))

data = scaler.fit_transform(np.array(data[:, np.newaxis]))
data = data[:, 0]

door_open = [(160, 240), (700, 750), (820, 870), (1300, 1350), (1900, 1950), (2520, 2570), (3150, 3200), (3850, 3920)]

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


def get_sample_indices(data_length, x_window_size=120, y_window_size=24):
    sample_length = data_length - x_window_size - y_window_size

    # create feature indices
    feature_indices = torch.arange(x_window_size).view(1, -1) + torch.arange(sample_length).view(-1, 1)

    # create label indices
    label_indices = torch.arange(y_window_size).view(1, -1) + torch.arange(sample_length).view(-1, 1)
    label_indices = torch.add(label_indices, x_window_size)

    return feature_indices, label_indices


# init empty arrays
feature_framed_data = np.empty((0, FEATURE_LENGTH))
label_framed_data = np.empty((0, LABEL_LENGTH))

for points in door_open:
    # get sub_data
    sub_data = data[points[0]:points[1]]

    # get frames of data
    sub_data_length = len(sub_data)
    feature_indices, label_indices = get_sample_indices(sub_data_length, x_window_size=FEATURE_LENGTH,
                                                        y_window_size=LABEL_LENGTH)  # y window doesn' tmatter because it's not used
    feature_framed_sub_data = np.array(sub_data)[feature_indices]
    label_framed_sub_data = np.array(sub_data)[label_indices]

    # concat all data
    feature_framed_data = np.append(feature_framed_data, feature_framed_sub_data, axis=0)
    label_framed_data = np.append(label_framed_data, label_framed_sub_data, axis=0)

feature_framed_data.shape, label_framed_data.shape

# In[7]:


X = feature_framed_data
Y = label_framed_data

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

# In[11]:


X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
Y_train = torch.tensor(Y_train).float()
Y_test = torch.tensor(Y_test).float()

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# In[12]:


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

# In[13]:


from torch.utils.data import DataLoader

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# In[14]:


for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

model = LSTM(1, 5, 1, device=device)
model.to(device)
model

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


# In[17]:


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


# In[18]:


loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUMBER_OF_EPOCHS):
    train_one_epoch()
    validate_one_epoch()

# In[19]:


with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

plt.plot(Y_train[:, 0], label='Actual Temperature')
plt.plot(predicted[:, 0], label='Predicted Temperature')
plt.xlabel('120 Second Time Step')
plt.ylabel('Degrees F')
plt.legend()
plt.show()

# In[20]:


# do whole thing
# get frames of data
data_length = len(data)
feature_indices, label_indices = get_sample_indices(data_length, x_window_size=FEATURE_LENGTH,
                                                    y_window_size=LABEL_LENGTH)  # y window doesn' tmatter because it's not used

feature_framed_data = np.array(data)[feature_indices]  # [:, :, np.newaxis]
label_framed_data = np.array(data)[label_indices]  # [:, :, np.newaxis]

feature_framed_data.shape, label_framed_data.shape

# In[21]:


# scale data
scaler = MinMaxScaler(feature_range=(-1, 1))

feature_framed_data_fit = scaler.fit_transform(feature_framed_data)
label_framed_data_fit = scaler.fit_transform(label_framed_data)

feature_framed_data_fit.shape, label_framed_data_fit.shape

# In[22]:


X = feature_framed_data_fit
Y = label_framed_data_fit

X.shape, Y.shape

# In[23]:


X = X[:, :, np.newaxis]

X.shape, Y.shape

# In[24]:


X = torch.tensor(X).float()
Y = torch.tensor(Y).float()

X.shape, Y.shape

# In[25]:


with torch.no_grad():
    predicted = model(X.to(device)).to('cpu').numpy()

plt.plot(Y[:, 0], label='Actual Temperature')
plt.plot(predicted[:, 0], label='Predicted Temperature')
plt.xlabel('120 Second Time Step')
plt.ylabel('Degrees F')
plt.legend()
plt.show()

# In[26]:


np.save('predicted.npy', predicted)

