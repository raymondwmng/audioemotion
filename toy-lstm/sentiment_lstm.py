import torch
import torch.nn as nn

class SentimentLstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,outlayer_size,num_emotions):
        super(SentimentLstm, self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)
        self.fc = nn.Linear(outlayer_size, num_emotions)
 
    def forward(self,x):
       x = torch.transpose(x,0,1)      # to swap the batch dimension and position dimension
       hiddens,_ = self.lstm(x)
       last_hiddens=hiddens[-1]
       y = self.fc(last_hiddens)
       return y
