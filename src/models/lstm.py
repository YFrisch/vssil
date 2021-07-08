import torch
import torch.nn as nn
from torch.autograd import Variable, Function

from .layers import FullyConnected


class CustomLSTM(nn.Module):

    """ LSTM followed by a variable number of heads
        consisting of two fully connected layers.
    """

    def __init__(self,
                 input_size: tuple,
                 num_out_classes: list,
                 lstm_hidden_size: int,
                 num_lstm_layers: int,
                 num_fc_layers: list,
                 num_heads: int,
                 fc_activation: nn.Module,
                 out_activations: list,
                 seq_len: int):
        """ Creates class instance.

        :param input_size: Shape of the input tensors
        :param num_out_classes: List of number of output nodes / classes (per header)
        :param lstm_hidden_size: Number of nodes in (hidden) lstm layers
        :param num_lstm_layers: Number of hidden lstm layers
        :param num_fc_layers: Number of fc layers per head
        :param num_heads: Number of heads (A 2 fc layers)
        :param fc_activation: Activation function of fc layers
        :param out_activations: List of activation functions of final layer (per header)
        :param seq_len: Number of concat. frames per sample
        """

        super(CustomLSTM, self).__init__()

        self.num_out_classes = num_out_classes
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers,
                            batch_first=True)

        assert len(out_activations) == num_heads
        assert len(num_fc_layers) == num_heads

        self.heads = [
            nn.Sequential(
                *[FullyConnected(
                    in_features=lstm_hidden_size,
                    out_features=128,
                    activation=fc_activation
                ) for layer_id in range(0, num_fc_layers[head_id])]
            ) for head_id in range(0, num_heads)
        ]

    def forward(self, x: torch.Tensor):
        """ Forward pass through the model.

        :param x: Input tensor
        :return: Output tensor
        """

        h_0 = Variable(torch.zeros((self.num_lstm_layers, x.shape[0], self.lstm_hidden_size)))  # hidden state
        c_0 = Variable(torch.zeros((self.num_lstm_layers, x.shape[0], self.lstm_hidden_size)))  # internal state

        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n = h_n.view(-1, self.lstm_hidden_size)

        out = self.relu(h_n)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc(out)

        return out
