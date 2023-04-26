import torch
import torch.nn as nn
import torch.nn.functional as F



class LSTMNet(nn.Module):

    def __init__(self, input_size):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])
        # print(out.shape)
        return F.relu(out)


class Transformer(nn.Module):
    def __init__(self,input_size, nhead=1, num_encoder_layers=3, num_decoder_layers=1, dim_feedforward=512):
        super(Transformer, self).__init__()
        self.trans = nn.Transformer(input_size, nhead, num_encoder_layers,
                                    num_decoder_layers, dim_feedforward)
        self.out = nn.Sequential(
            nn.Linear(4, 1),
        )

    def forward(self, x):
        r_out = self.trans(x, x)  # None 表示 hidden state 会用全0的 state
        # print(r_out.shape)
        out = self.out(r_out[:, -1, :])

        # print(out.shape)
        return F.relu(out)


class NWP(nn.Module):
    def __init__(self, input_size, n_hidden=100, n_hidden_2=50, n_output=1):
        super(NWP, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden_2),
        )
        self.predict = torch.nn.Linear(n_hidden_2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


class SDM(nn.Module):
    def __init__(self, input_size, n_hidden=256, n_hidden_2=128, n_hidden_3=56, n_output=1):
        super(SDM, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden_2),
            nn.Sigmoid(),
            nn.Linear(n_hidden_2, n_hidden_3),
        )
        self.predict = torch.nn.Linear(n_hidden_3, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return F.relu(x)