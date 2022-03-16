import torch
from torch import nn

epochs = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
batch_size = 64
time_step = 28  # rnn 时间步数 / 图片高度
input_size = 28  # rnn 每步输入值 / 图片每行像素
hidden_size = 64
num_layers = 1
num_classes = 10
lr = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class simpleLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])

        return out


model = simpleLSTM(input_size, hidden_size, num_layers, num_classes)
