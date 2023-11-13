import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import re

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, forecast, actual):
        numerator = torch.abs(forecast - actual)
        denominator = (torch.abs(actual) + torch.abs(forecast))
        smape = 200 * torch.mean(numerator / denominator)
        return smape


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, forecast, actual):
        numerator = torch.abs(actual - forecast)
        denominator = torch.abs(actual)
        mape = 100 * torch.mean(numerator / denominator)
        return mape


def load_from_csv():
    x_window_size = 120
    y_window_size = 24

    def extract_last_float(line):
        # Extract all float numbers from the line
        floats = re.findall(r'[-+]?\d*\.\d+|\d+', line)
        # Return the last float if any, else return None
        return floats[-1] if floats else None

    data = []
    with open('data.csv', 'r') as file:
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


def main():
    load_from_csv()

    # Constants
    BATCH_SIZE = 32    # Batch size for training

    # separate data
    all_data = np.load('dataset/synthesized_data.npz')
    input_data = all_data['train_x']
    output_data  = all_data['train_y']

    # Convert to PyTorch tensors and move to device
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    output_tensor = torch.tensor(output_data, dtype=torch.float32).to(device)

    # Create dataset and dataloader
    dataset = TensorDataset(input_tensor, output_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # RNN Model
    class RNNModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNNModel, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x, _ = self.rnn(x)
            x = self.fc(x[:, -1, :])  # Using the last output
            return x

    # Model parameters
    input_size = 120
    hidden_size = 100  # Example hidden size
    output_size = 24
    num_layers = 1

    # Initialize the model and move to device
    model = RNNModel(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = MAPELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 100  # Number of epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Reshape inputs to (batch_size, seq_len, input_size)
            inputs = inputs.view(-1, 1, input_size)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')


if __name__ == "__main__":
    main()
