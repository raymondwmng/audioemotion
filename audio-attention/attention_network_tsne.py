# attention network
import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder
class LstmNet(nn.Module):
	"""
	input:  features (B * L * D), B=batch_size, L=length_of_utt, D=dimension
	hidden: layers (num_layers, 2) of size (B * L * H), H=hidden_size(512)
	output: lstm hidden states (B * L * O), O=outlayer_size(1024)
	"""
	def __init__(self,input_size,hidden_size,num_layers,outlayer_size,num_emotions):
		super(LstmNet, self).__init__()
#		self.outlayer_size = outlayer_size
#		self.num_emotions = num_emotions
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)
#		if ATTENTION:
		self.linear = nn.Linear(hidden_size, num_emotions)
#		self.linear = nn.Linear(outlayer_size, num_emotions)

	def forward(self, x, ATTENTION):
		x = torch.transpose(x,0,1)      # to swap the batch dimension and position dimension
		hiddens, (ht,ct) = self.lstm(x)
		# https://github.com/claravania/lstm-pytorch/blob/master/model.py
		# ht is the last hidden state of the sequences
		# ht = (1 x batch_size x hidden_dim)
		# ht[-1] = (batch_size x hidden_dim)
		if not ATTENTION:
			return ht[-1]
		else:
			return hiddens, ht[-1]


# Attention 
class Attention(nn.Module):
	"""
	input:  lstm hidden states (L * B * O), O=outlayer_size(1024)
	att:    dual attention network layer (L * B * AH), AH=attention_hidden_size(128)
	output: context vector (B * DH), DH=dan_hidden_size(1024)
	"""
	def __init__(self, num_emotions, dan_hidden_size, attention_hidden_size):
		super(Attention, self).__init__()
		N = dan_hidden_size		#N
		N2 = attention_hidden_size	#AH

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
		return [a, c]


# final layer for classifying emotion
class Predictor(nn.Module):
	"""
	input:  context vector (B * DH), DH=dan_hidden_size(1024)
	output: prediction (B * NE), NE=num_emotions(6) 
	"""
	def __init__(self,num_emotions,hidden_size):#,output_scale_factor = 1, output_shift = 0):
		super(Predictor, self).__init__()
		self.fc = nn.Linear(hidden_size, num_emotions)

	def forward(self,x):
		x = self.fc(x)
#		# use sigmoid/tanh after comparing outputs with clamped outputs? Raymond
#		x = torch.sigmoid(x)
		return x


