import torch

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
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0

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
                print('Batch {0}, Loss: {1:.5f}'.format(batch_index + 1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print()

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

        print('Val Loss: {0:.5f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.len_epoch):
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
            self._progress(epoch)
