from torch import nn
from einops import rearrange, repeat
import torch
import torch.nn.functional as F


class MixHistorical(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=2, num_layers=4, batch_first=True)
        self.fc1 = nn.Linear(8, 1)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, historical, pre_pred):
        _, historical = self.rnn(historical)
        historical = rearrange(historical, "l b h -> b l h")
        historical = torch.flatten(historical, 1)
        historical = self.fc1(historical)
        historical = torch.tanh(historical)
        output = historical + pre_pred
        return output

