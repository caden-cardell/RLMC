#!/usr/bin/env python

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.randn(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out