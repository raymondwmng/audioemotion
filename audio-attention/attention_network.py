# attention network
import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder
class LstmNet(nn.Module):
	"""
	input:  features (B * N * D), B=batch_size, N=length_of_utt, D=dimension
	hidden: layers (num_layers, 2) of size (B * N * H), H=hidden_size(512)
	output: lstm hidden states (B * N * O), O=outlayer_size(1024)
	"""
	def __init__(self,input_size,hidden_size,num_layers,outlayer_size,num_emotions):
		super(LstmNet, self).__init__()
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)
		self.linear = nn.Linear(hidden_size, num_emotions)

	def forward(self,x):
		x = torch.transpose(x,0,1)      # to swap the batch dimension and position dimension
#		x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)
		hiddens,_ = self.lstm(x)
#		hiddens, _ = torch.nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)	
#		print(x.shape, hiddens.shape)
#		sys.exit()	
		return hiddens


# Attention 
class Attention(nn.Module):
	"""
	input:  lstm hidden states (B * N * O), O=outlayer_size(1024)
	att:    dual attention network layer (B * N * AH), AH=attention_hidden_size(128)
	output: context vector (B * DH), DH=dan_hidden_size(1024)
	"""
	def __init__(self, num_emotions, dan_hidden_size, attention_hidden_size):
		super(Attention, self).__init__()
		N = dan_hidden_size
		N2 = attention_hidden_size

		self.W = nn.Linear(N,N2)		# input size N, output size N2
		self.W_m = nn.Linear(N,N2)
		self.W_h = nn.Linear(N2,1)	#should be scalar
#		self.fc = nn.Linear(N, num_emotions)


	def forward(self, hyp, dan_hidden_size, attention_hidden_size, BATCHSIZE):
		N = dan_hidden_size
		N2 = attention_hidden_size

		m = hyp.mean(0).unsqueeze(0)
		m = m.permute(1,0,2)
		hyp = hyp.permute(1,0,2)
		mx = m.repeat(1, hyp.size(1),1)
		h = torch.tanh(self.W(hyp))*torch.tanh(self.W_m(mx))
		a = F.softmax(self.W_h(h),dim=1)
		c = (a.repeat(1,1,N)*hyp).sum(1)
		return c


# final layer for classifying emotion
class Predictor(nn.Module):
	"""
	input:  context vector (B * DH), DH=dan_hidden_size(1024)
	output: prediction (B * NE), NE=num_emotions(6) 
	"""
	def __init__(self,num_emotions,hidden_size,output_scale_factor = 1, output_shift = 0):
		super(Predictor, self).__init__()
		self.fc = nn.Linear(hidden_size, num_emotions)

	def forward(self,x):
		x = self.fc(x)
#		# use sigmoid/tanh after comparing outputs with clamped outputs?
#		x = torch.sigmoid(x)
		return x


