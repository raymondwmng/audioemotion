# attention network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
#		self.linear = nn.Linear(outlayer_size, num_emotions)

	def forward(self,x, ATTENTION):
		x = torch.transpose(x,0,1)      # to swap the batch dimension and position dimension
		hiddens,_ = self.lstm(x)
#		if not ATTENTION:
#			hiddens = self.linear(hiddens)
		return hiddens


# Attention 
class Attention(nn.Module):
	"""
	input:  lstm hidden states (B * N * O), O=outlayer_size(1024)
	att:    dual attention network layer (B * N * N2), N2=attention_hidden_size(128)
	output: context vector (B * N), N=dan_hidden_size(1024)
	"""
	def __init__(self, num_emotions, dan_hidden_size, attention_hidden_size, multihead_size):
		super(Attention, self).__init__()
		H = multihead_size
		N = dan_hidden_size
		N2 = attention_hidden_size
		N_H = int(dan_hidden_size/H)
		N2_H = int(attention_hidden_size/H)

		# projecting the utterance hidden vector into H parts of size N x N_H
		multihead_linears = []
		for i in range(H):
			multihead_linears.append(nn.Linear(N, N_H).cuda())
		self.multihead_linears = multihead_linears
#		print(multihead_linears)

		# attention
		self.W = nn.Linear(N_H, N2_H)		# input size N_H, output size N2_H
		self.W_m = nn.Linear(N_H, N2_H)
		self.W_h = nn.Linear(N2_H, 1)	#should be scalar


	# https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XMcZvEPTVhE
	# H=8 attention layers (aka “heads”): that represent linear projection (for the purpose of dimension reduction) of key K and query Q into dk-dimension and value V into dv- dimension
	def forward(self, hyp, dan_hidden_size, attention_hidden_size, multihead_size):
		H = multihead_size
		N = dan_hidden_size
		N_H = int(dan_hidden_size/H)
		N2_H = int(attention_hidden_size/H)
		C = torch.zeros([0,N_H])
		C = Variable(C.float()).cuda()
		# split input into H parts, attention separately on each, concat output	
		for i in range(H):
			# split
#			print(hyp.shape, N, N_H)
#			MH = self.multihead_linears[i]
#			print(MH)
			x = self.multihead_linears[i](hyp)
#			x = self.multihead_linears[i](hyp)
#			print(i, hyp.shape, H, N_H, N2_H, C.shape, (i*N_H), ((i+1)*N_H), x.shape)
			# attention
			m = x.mean(0).unsqueeze(0)
			m = m.permute(1,0,2)
			x = x.permute(1,0,2)
			mx = m.repeat(1, x.size(1),1)
			h = torch.tanh(self.W(x))*torch.tanh(self.W_m(mx))
			a = F.softmax(self.W_h(h),dim=1)
			c = (a.repeat(1,1,N_H)*x).sum(1)
			# concat
			C = torch.cat((C, c), 1)
#			print("\t", C.shape, c.shape)
		return C


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
#		# use sigmoid/tanh after comparing outputs with clamped outputs?
#		x = torch.sigmoid(x)
		return x


