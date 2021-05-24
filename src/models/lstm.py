import torch
import torch.nn as nn
from torch.autograd import Variable


class CustomLSTM(nn.Module):

    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 seq_len: int):
        """ Creates class instance.

        :param num_classes:
        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param seq_len:
        """

        super(CustomLSTM, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """ Forward pass through the model.

        :param x: Input tensor
        :return: Output tensor
        """

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n = h_n.view(-1, self.hidden_size)

        out = self.relu(h_n)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc(out)

        return out
