# attention network
import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder
class LstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, outlayer_size, num_emotions):
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_size, num_emotions)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        hiddens, _ = self.lstm(x)
        # print("lstm hidden shape {}".format(hiddens.shape))
        hiddens = hiddens.squeeze(1)
        return hiddens


# Attention 
class Attention(nn.Module):
    def __init__(self, num_emotions, dan_hidden_size, attention_hidden_size):
        super(Attention, self).__init__()
        N = dan_hidden_size
        N2 = attention_hidden_size

        self.W = nn.Linear(N, N2)  # input size N, output size N2
        self.W_m = nn.Linear(N, N2)
        self.W_h = nn.Linear(N2, 1)  # should be scalar

        self.fc = nn.Linear(N, num_emotions)

    def forward(self, hyp, dan_hidden_size, attention_hidden_size, attention_type):
        N = dan_hidden_size
        N2 = attention_hidden_size
        if attention_type == 'attention':
            m = hyp.mean(0).unsqueeze(0)
            m = m.permute(1, 0, 2)
            hyp = hyp.permute(1, 0, 2)
            mx = m.repeat(1, hyp.size(1), 1)
            h = torch.tanh(self.W(hyp)) * torch.tanh(self.W_m(mx))
            a = F.softmax(self.W_h(h), dim=1)
            c = (a.repeat(1, 1, N) * hyp).sum(1)
        elif attention_type == 'additive':
            hyp = hyp.permute(1, 0, 2)
            h = torch.tanh(self.W(hyp))
            score = self.W_h(h)
            a = F.softmax(score, dim=1)
            c = (a.repeat(1, 1, N) * hyp).sum(1)
        return c


# memory to emotion decoder
class Predictor(nn.Module):
    def __init__(self, num_emotions, hidden_size, output_scale_factor=1, output_shift=0):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(hidden_size, num_emotions)

    def forward(self, x):
        x = self.fc(x)
        return x
