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
	att:    dual attention network layer (B * N * AH), AH=attention_hidden_size(128)
	output: context vector (B * DH), DH=dan_hidden_size(1024)
	"""
	def __init__(self, num_emotions, dan_hidden_size, attention_hidden_size, multihead_size):
		super(Attention, self).__init__()
		H = multihead_size
		N = int(dan_hidden_size/H)
		N2 = int(attention_hidden_size/H)

		# attention
		self.W = nn.Linear(N, N2)		# input size N, output size N2
		self.W_m = nn.Linear(N, N2)
		self.W_h = nn.Linear(N2, 1)	#should be scalar
#		self.fc = nn.Linear(N, num_emotions)


	# https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XMcZvEPTVhE
	# H=8 attention layers (aka “heads”): that represent linear projection (for the purpose of dimension reduction) of key K and query Q into dk-dimension and value V into dv- dimension
	def forward(self, hyp, dan_hidden_size, attention_hidden_size, multihead_size):
		H = multihead_size
		N = int(dan_hidden_size/H)
		N2 = int(attention_hidden_size/H)
		C = torch.zeros([0,N])
		C = Variable(C.float()).cuda()
		# split input into H parts, attention separately on each, concat output	
		for i in range(H):
			# split
			x = hyp[:,:,(i*N):((i+1)*N)]
#			print(i, hyp.shape, H, N, N2, C.shape, (i*N), ((i+1)*N), x.shape)
			# attention
			m = x.mean(0).unsqueeze(0)
			m = m.permute(1,0,2)
			x = x.permute(1,0,2)
			mx = m.repeat(1, x.size(1),1)
			h = torch.tanh(self.W(x))*torch.tanh(self.W_m(mx))
			a = F.softmax(self.W_h(h),dim=1)
			c = (a.repeat(1,1,N)*x).sum(1)
#			print("\t", C.shape, c.shape)
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


