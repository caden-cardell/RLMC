import torch
from tqdm import trange
import sys
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    """
    Trainer class
    """
    def __init__(self, model, train_loader, test_loader, optimizer, loss_function, len_epoch):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.len_epoch = len_epoch

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train(True)
        running_loss = 0.0
        avg_loss_across_batches = 0.0
        for batch_index, batch in enumerate(self.train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = self.model(x_batch)
            loss = self.loss_function(output, y_batch)
            running_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                running_loss = 0.0

        return avg_loss_across_batches

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(self.test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = self.model(x_batch)
                loss = self.loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(self.test_loader)
        return avg_loss_across_batches

    def train(self):
        """
        Full training logic
        """
        for epoch in trange(self.len_epoch, desc='Epoch'):
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
