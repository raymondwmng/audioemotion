import sys
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
import h5py
from PIL import Image
from sklearn.externals import joblib
import shutil
import os
import random
import pickle
import time
import gc
import re
from tensorboardX import SummaryWriter
import time
import math
from torchvision import datasets, models, transforms
import matplotlib.cm as cm
import cv2
import pandas as pd 
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from mosei_dataloader import mosei
from torch.nn.parameter import Parameter
from cmu_score import ComputePerformance

torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)


preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'---------------------------------------------------LSTM VocalNet-------------------------------------------------------'

class VocalNet(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers):
		super(VocalNet, self).__init__()
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)
		self.linear = nn.Linear(hidden_size, 6)


	def forward(self,x):
		x = torch.transpose(x,0,1)
		hiddens,_ = self.lstm(x)
		hiddens = hiddens.squeeze(1)
		return hiddens
'---------------------------------------------------LSTM TextNet-------------------------------------------------------'

class WordvecNet(nn.Module):
	# def __init__(self,input_size,hidden_size,num_layers,out_size,dropout=0.2,bidirectional=False):
	def __init__(self,input_size,hidden_size,num_layers,dropout=0.2,bidirectional=True):
		super(WordvecNet, self).__init__()
		# self.rnn = nn.LSTM(input_size,hidden_size,num_layers,dropout,bidirectional,batch_first=True)
		self.rnn = nn.LSTM(input_size,hidden_size,num_layers,dropout,bidirectional=bidirectional)

	def forward(self,x):
		"""
		param x: tensor of shape (batch_size, seq_len, in_size)
		"""
		# output,final_hiddens = self.rnn(x)
		# return output,final_hiddens
		x = torch.transpose(x,0,1)
		hiddens,_ = self.rnn(x)
		hiddens = hiddens.squeeze(1)
		return hiddens




'---------------------------------------------------LSTM VisualNet-------------------------------------------------------'

class VisionNet(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers):
		super(VisionNet, self).__init__()
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=True)


	def forward(self,x):
		x = torch.transpose(x,0,1)
		hiddens,_ = self.lstm(x)
		hiddens = hiddens.squeeze(1)
		return hiddens


'---------------------------------------------------Gated Attention----------------------------------------------------'

class GatedAttention(nn.Module):
	def __init__(self,att_input_size,att_hidden_size,att_num_layers,no_of_emotions):
		super(GatedAttention, self).__init__()
		self.lstm = nn.LSTM(att_input_size,att_hidden_size,att_num_layers)
		self.linear = nn.Linear(att_hidden_size, no_of_emotions)


	def forward(self,vocal,vision):

		vocal = vocal.repeat(45,1)
		vision = vision.squeeze(2)
		vision = vision.squeeze(2)
		fusion = vocal*vision
		fusion = fusion.unsqueeze(0)
		fusion = fusion.transpose(0,1)
		hiddens,_ = self.lstm(fusion)
		outputs = self.linear(hiddens[-1])
		return outputs	
'---------------------------------------------------Triple Attention----------------------------------------------------'

class TripleAttention(nn.Module):
	def __init__(self,no_of_emotions,dan_hidden_size,attention_hidden_size):
		super(TripleAttention, self).__init__()
		N = dan_hidden_size
		N2 = attention_hidden_size
		''' K= 1 ''' 
		self.Wvision_1 = nn.Linear(N,N2)
		self.Wvision_m1 = nn.Linear(N,N2)
		self.Wvision_h1 = nn.Linear(N2,1)
		self.Wvocal_1 = nn.Linear(N,N2)
		self.Wvocal_m1 = nn.Linear(N,N2)
		self.Wvocal_h1 = nn.Linear(N2,1)
		self.Wemb_1 = nn.Linear(N,N2)
		self.Wemb_m1 = nn.Linear(N,N2)
		self.Wemb_h1 = nn.Linear(N2,1)

		''' K = 2 '''
		self.Wvision_2 = nn.Linear(N,N2)
		self.Wvision_m2 = nn.Linear(N,N2)
		self.Wvision_h2 = nn.Linear(N2,1)
		self.Wvocal_2 = nn.Linear(N,N2)
		self.Wvocal_m2 = nn.Linear(N,N2)
		self.Wvocal_h2 = nn.Linear(N2,1)
		self.Wemb_2 = nn.Linear(N,N2)
		self.Wemb_m2 = nn.Linear(N,N2)
		self.Wemb_h2 = nn.Linear(N2,1)

		self.fc = nn.Linear(N, no_of_emotions)


	def forward(self,vocal,vision,emb):
		N = dan_hidden_size
		N2 = attention_hidden_size
		# Sorting out vision
		# print(resnet_output.size())
		# resnet_output = resnet_output.mean(0)
		# resnet_output = resnet_output.view(512,49)
		# vision = resnet_output.transpose(0,1)

		'-------------------------------------------------Initializing Memory--------------------------------------'
		vision_zero = vision.mean(0).unsqueeze(0)
		vocal_zero = vocal.mean(0).unsqueeze(0)
		emb_zero = emb.mean(0).unsqueeze(0)
		
		m_zero = vision_zero * vocal_zero * emb_zero
		m_zero_vision = m_zero.repeat(vision.size(0),1)
		m_zero_vocal = m_zero.repeat(vocal.size(0),1)
		m_zero_emb = m_zero.repeat(emb.size(0),1)
		'--------------------------------------------------K = 1 ---------------------------------------------------'
		# Visual Attention
		h_one_vision = F.tanh(self.Wvision_1(vision))*F.tanh(self.Wvision_m1(m_zero_vision))
		a_one_vision = F.softmax(self.Wvision_h1(h_one_vision),dim=0)  ## along dimsions
		# vision_one = (a_one_vision.repeat(1,N)*vision).mean(0).unsqueeze(0)
		vision_one = (a_one_vision.repeat(1,N)*vision).sum(0)
		# avision_one = (a_one_vision.repeat(1,N)*vision).mean(0).unsqueeze(0)
		# vision_one = F.tanh(self.Pvision_1(avision_one))

		# Vocal Attention
		h_one_vocal = F.tanh(self.Wvocal_1(vocal))*F.tanh(self.Wvocal_m1(m_zero_vocal))
		a_one_vocal = F.softmax(self.Wvocal_h1(h_one_vocal),dim=0)
		# vocal_one = (a_one_vocal.repeat(1,N)*vocal).mean(0).unsqueeze(0)
		vocal_one = (a_one_vocal.repeat(1,N)*vocal).sum(0)
		# avocal_one = (a_one_vocal.repeat(1,N)*vocal).mean(0).unsqueeze(0)
		# vocal_one = F.tanh(self.Pvocal_1(avocal_one))

		# Emb Attention
		h_one_emb = F.tanh(self.Wemb_1(emb))*F.tanh(self.Wemb_m1(m_zero_emb))
		a_one_emb = F.softmax(self.Wemb_h1(h_one_emb),dim=0)
		# emb_one = (a_one_emb.repeat(1,N)*emb).mean(0).unsqueeze(0)
		emb_one = (a_one_emb.repeat(1,N)*emb).sum(0)
		# aemb_one = (a_one_emb.repeat(1,N)*emb).mean(0).unsqueeze(0)
		# emb_one = F.tanh(self.Pemb_1(aemb_one))

		# Memory Update
		m_one = m_zero + ( vision_one * vocal_one * emb_one )
		m_one_vision = m_one.repeat(vision.size(0),1)
		m_one_vocal = m_one.repeat(vocal.size(0),1)
		m_one_emb = m_one.repeat(emb.size(0),1)

		'--------------------------------------------------K = 2  ---------------------------------------------------'

		# Visual Attention
		h_two_vision = F.tanh(self.Wvision_2(vision))*F.tanh(self.Wvision_m2(m_one_vision))
		a_two_vision = F.softmax(self.Wvision_h2(h_two_vision),dim=0)
		# vision_two = (a_two_vision.repeat(1,N)*vision).mean(0).unsqueeze(0)
		vision_two = (a_two_vision.repeat(1,N)*vision).sum(0)
		# avision_two = (a_two_vision.repeat(1,N)*vision).mean(0).unsqueeze(0)
		# vision_two = F.tanh(self.Pvision_2(avision_two))	

		# Vocal Attention
		h_two_vocal = F.tanh(self.Wvocal_2(vocal))*F.tanh(self.Wvocal_m2(m_one_vocal))
		a_two_vocal = F.softmax(self.Wvocal_h2(h_two_vocal),dim=0)
		# vocal_two = (a_two_vocal.repeat(1,N)*vocal).mean(0).unsqueeze(0)
		vocal_two = (a_two_vocal.repeat(1,N)*vocal).sum(0)
		# avocal_two = (a_two_vocal.repeat(1,N)*vocal).mean(0).unsqueeze(0)
		# vocal_two = F.tanh(self.Pvocal_2(avocal_two))

		# Emb Attention
		h_two_emb = F.tanh(self.Wemb_2(emb))*F.tanh(self.Wemb_m2(m_one_emb))
		a_two_emb = F.softmax(self.Wemb_h2(h_two_emb),dim=0)
		# emb_two = (a_two_emb.repeat(1,N)*emb).mean(0).unsqueeze(0)
		emb_two = (a_two_emb.repeat(1,N)*emb).sum(0)
		# aemb_two = (a_two_emb.repeat(1,N)*emb).mean(0).unsqueeze(0)
		# emb_two = F.tanh(self.Pemb_2(aemb_two))

		# Memory Update
		m_two = m_one + ( vision_two * vocal_two * emb_two )
		return m_two
		'-------------------------------------------------Prediction--------------------------------------------------'
		# return m_two
		# outputs = self.fc(m_two)
		# # print(outputs)
		# return outputs	
'---------------------------------------------------Memory to Emotion Decoder------------------------------------------'
class predictor(nn.Module):
	def __init__(self,no_of_emotions,hidden_size,output_scale_factor = 1, output_shift = 0):
		super(predictor, self).__init__()
		self.fc = nn.Linear(hidden_size, no_of_emotions)
		# self.output_scale_factor = Parameter(torch.FloatTensor([output_scale_factor]), requires_grad=False)
		# self.output_shift = Parameter(torch.FloatTensor([output_shift]), requires_grad=False)

	def forward(self,x):
		x = self.fc(x)
		# x = F.sigmoid(x)
		# x = x*self.output_scale_factor + self.output_shift

		return x
'------------------------------------------------------Hyperparameters-------------------------------------------------'
batch_size = 1
mega_batch_size = 1
no_of_emotions = 6
use_CUDA = True
use_pretrained = True
num_workers = 20

test_mode = True
val_mode = False
train_mode = False

no_of_epochs = 1000
vocal_input_size = 74 # Dont Change
vision_input_size = 35 # Dont Change
wordvec_input_size = 300
vocal_num_layers = 2
vision_num_layers = 2
wordvec_num_layers = 2
vocal_hidden_size = 512
vision_hidden_size = 512
wordvec_hidden_size = 512
dan_hidden_size = 1024
attention_hidden_size = 128
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = VocalNet(vocal_input_size, vocal_hidden_size, vocal_num_layers)
Vision_encoder = VisionNet(vision_input_size, vision_hidden_size, vision_num_layers)
Wordvec_encoder = WordvecNet(wordvec_input_size, wordvec_hidden_size, wordvec_num_layers)
Attention = TripleAttention(no_of_emotions,dan_hidden_size,attention_hidden_size)
Predictor = predictor(no_of_emotions,dan_hidden_size)
if train_mode:
	train_dataset = mosei(mode= "train")
	data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,num_workers = num_workers)
elif val_mode:
	val_dataset = mosei(mode = "val")
	data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=1,
                                        shuffle=False,num_workers = num_workers)
	no_of_epochs = 1
else:
	test_dataset = mosei(mode = "test")
	data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=1,
                                        shuffle=False,num_workers = num_workers)
	no_of_epochs = 1
curr_epoch = 0
total = 0
'----------------------------------------------------------------------------------------------------------------------'
Vocal_encoder = Vocal_encoder.cuda()
Attention = Attention.cuda()
Vision_encoder = Vision_encoder.cuda()
Wordvec_encoder = Wordvec_encoder.cuda()
Predictor = Predictor.cuda()
'----------------------------------------------------------------------------------------------------------------------'
criterion = nn.MSELoss(size_average = False)
# params =  list(Vocal_encoder.parameters())+ list(Attention.parameters()) + list(Wordvec_encoder.parameters()) + list(Vision_encoder.parameters()) + list(Predictor.parameters())[2:]
params =  list(Vocal_encoder.parameters())+ list(Attention.parameters()) + list(Wordvec_encoder.parameters()) + list(Vision_encoder.parameters()) + list(Predictor.parameters())
print('Parameters in the model = ' + str(len(params)))
optimizer = torch.optim.Adam(params, lr = 0.0001)
# optimizer = torch.optim.SGD(params, lr =0.001,momentum = 0.9 )

'------------------------------------------Saving Intermediate Models--------------------------------------------------'


def save_checkpoint(state, is_final, filename='attention_net'):
	filename = filename +'_'+str(state['epoch'])+'.pth.tar'
	os.system("mkdir -p TAN_1024_scalarTime_oldattention") 
	torch.save(state, './TAN_1024_scalarTime_oldattention/'+filename)
	if is_final:
		shutil.copyfile(filename, 'model_final.pth.tar')


'-------------------------------------------Setting into train mode----------------------------------------------------'

if not train_mode:
	Vision_encoder.train(False)
	Vocal_encoder.train(False)
	Wordvec_encoder.train(False)
	Attention.train(False)
	Predictor.train(False)
else:
	Vision_encoder.train(True)
	Vocal_encoder.train(True)
	Wordvec_encoder.train(True)
	Attention.train(True)
	Predictor.train(True)

'----------------------------------------------------------------------------------------------------------------------'
epoch = 0
y_true = []
y_pred = []
while epoch<no_of_epochs:
	j_start = 0
	running_loss = 0
	running_corrects = 0
	if use_pretrained:
		pretrained_file = './TAN_1024_scalarTime_seeded777_sumtime/triple_attention_net__4.pth.tar'
		# pretrained_file = './TAN/triple_attention_net__8.pth.tar'

		checkpoint = torch.load(pretrained_file)
		Vocal_encoder.load_state_dict(checkpoint['Vocal_encoder'])
		Vision_encoder.load_state_dict(checkpoint['Vision_encoder'])
		Wordvec_encoder.load_state_dict(checkpoint['Wordvec_encoder'])
		Attention.load_state_dict(checkpoint['Attention'])
		Predictor.load_state_dict(checkpoint['Predictor'])
		use_pretrained = False
		if train_mode:
			epoch = checkpoint['epoch']+1
			optimizer.load_state_dict(checkpoint['optimizer'])

	K = 0
	# initialise reference and hypothesis ######
	overall_hyp = np.zeros((0,no_of_emotions))
	overall_ref = np.zeros((0,no_of_emotions))
	############################################
	for i,(vision,vocal,emb,gt) in enumerate(data_loader):
		if use_CUDA:
			# if i==0 or i==1:
			# 	print('To load data into CUDA')
			# 	print(vision.size())
			# 	print(vocal.size())
			# 	print(emb.size())
			vision = Variable(vision.float()).cuda()
			vocal = Variable(vocal.float()).cuda()
			emb = Variable(emb.float()).cuda()
			gt = Variable(gt.float()).cuda()

		vision_output = Vision_encoder(vision)
		vocal_output = Vocal_encoder(vocal)
		emb_output = Wordvec_encoder(emb)
		# output = Attention(vocal_output,vision_output)
		output = Attention(vocal_output,vision_output,emb_output)
		outputs = Predictor(output)
		outputs = torch.clamp(outputs,0,3)
		loss = criterion(outputs, gt)
 
		# concatenate reference and hypothesis ######
		overall_hyp = np.concatenate((overall_hyp,outputs.data.cpu().numpy()),axis=0)
		overall_ref = np.concatenate((overall_ref,gt.data.cpu().numpy()),axis=0)
		#############################################

		if train_mode and K%mega_batch_size==0:
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			Vocal_encoder.zero_grad()
			Vision_encoder.zero_grad()
			Wordvec_encoder.zero_grad()
			Attention.zero_grad()
			Predictor.zero_grad()

		# Compute performance metrics at the end of each epoch ------
		if K%100 == 0 and K > 0:
			score=ComputePerformance(overall_ref,overall_hyp);
			print('Scoring -- Epoch [%d], Sample [%d], Binary accuracy %.4f' % (epoch+1, K, score['binaryaccuracy']))
			print('Scoring -- Epoch [%d], Sample [%d], MSE %.4f' % (epoch+1, K, score['MSE']))
			print('Scoring -- Epoch [%d], Sample [%d], MSE_class %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['MSE_class'][0], score['MSE_class'][1], score['MSE_class'][2], score['MSE_class'][3], score['MSE_class'][4], score['MSE_class'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], MAE %.4f' % (epoch+1, K, score['MAE']))
			print('Scoring -- Epoch [%d], Sample [%d], MAE_class %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['MAE_class'][0], score['MAE_class'][1], score['MAE_class'][2], score['MAE_class'][3], score['MAE_class'][4], score['MAE_class'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], TP %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['TP'][0], score['TP'][1], score['TP'][2], score['TP'][3], score['TP'][4], score['TP'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], TN %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['TN'][0], score['TN'][1], score['TN'][2], score['TN'][3], score['TN'][4], score['TN'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], FP %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['FP'][0], score['FP'][1], score['FP'][2], score['FP'][3], score['FP'][4], score['FP'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], FN %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['FN'][0], score['FN'][1], score['FN'][2], score['FN'][3], score['FN'][4], score['FN'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], P %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['P'][0], score['P'][1], score['P'][2], score['P'][3], score['P'][4], score['P'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], N %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['N'][0], score['N'][1], score['N'][2], score['N'][3], score['N'][4], score['N'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], WA %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['WA'][0], score['WA'][1], score['WA'][2], score['WA'][3], score['WA'][4], score['WA'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], F1customised %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['F1customised'][0], score['F1customised'][1], score['F1customised'][2], score['F1customised'][3], score['F1customised'][4], score['F1customised'][5]))
			print('Scoring -- Epoch [%d], Sample [%d], F1 %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['F1'][0], score['F1'][1], score['F1'][2], score['F1'][3], score['F1'][4], score['F1'][5]))
		# -------------------------------------------------
		# outputs_ = Variable(torch.FloatTensor([ 0.1565 ,0.1233,  0.0401,  0.4836 , 0.1596,  0.04842])).cuda()
		# loss = criterion(outputs_, gt)

		running_loss += loss.data[0]
		K+=1
		average_loss = running_loss/K
		if train_mode and K%mega_batch_size==0:			
			print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f'
			% (epoch+1, K, average_loss))
		elif val_mode:
			print('Validating -- Epoch [%d], Sample [%d], Average Loss: %.4f'
			% (epoch+1, K, average_loss))
		elif test_mode:
			print('Testing -- Epoch [%d], Sample [%d], Average Loss: %.4f'
			 % (epoch+1, K, average_loss))
	'-------------------------------------------------Saving model after every epoch-----------------------------------'
	# Compute performance metrics at the end of each epoch ------
	score=ComputePerformance(overall_ref,overall_hyp);
	print('Scoring -- Epoch [%d], Sample [%d], Binary accuracy %.4f' % (epoch+1, K, score['binaryaccuracy']))
	print('Scoring -- Epoch [%d], Sample [%d], MSE %.4f' % (epoch+1, K, score['MSE']))
	print('Scoring -- Epoch [%d], Sample [%d], MSE_class %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['MSE_class'][0], score['MSE_class'][1], score['MSE_class'][2], score['MSE_class'][3], score['MSE_class'][4], score['MSE_class'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], MAE %.4f' % (epoch+1, K, score['MAE']))
	print('Scoring -- Epoch [%d], Sample [%d], MAE_class %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['MAE_class'][0], score['MAE_class'][1], score['MAE_class'][2], score['MAE_class'][3], score['MAE_class'][4], score['MAE_class'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], TP %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['TP'][0], score['TP'][1], score['TP'][2], score['TP'][3], score['TP'][4], score['TP'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], TN %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['TN'][0], score['TN'][1], score['TN'][2], score['TN'][3], score['TN'][4], score['TN'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], FP %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['FP'][0], score['FP'][1], score['FP'][2], score['FP'][3], score['FP'][4], score['FP'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], FN %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['FN'][0], score['FN'][1], score['FN'][2], score['FN'][3], score['FN'][4], score['FN'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], P %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['P'][0], score['P'][1], score['P'][2], score['P'][3], score['P'][4], score['P'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], N %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['N'][0], score['N'][1], score['N'][2], score['N'][3], score['N'][4], score['N'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], WA %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['WA'][0], score['WA'][1], score['WA'][2], score['WA'][3], score['WA'][4], score['WA'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], F1customised %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['F1customised'][0], score['F1customised'][1], score['F1customised'][2], score['F1customised'][3], score['F1customised'][4], score['F1customised'][5]))
	print('Scoring -- Epoch [%d], Sample [%d], F1 %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch+1, K, score['F1'][0], score['F1'][1], score['F1'][2], score['F1'][3], score['F1'][4], score['F1'][5]))

	'-------------------------------------------------Saving model after every epoch-----------------------------------'
	if train_mode:
		save_checkpoint({
			'epoch': epoch,
			'loss' : running_loss,
			'correct' : running_corrects,
			'j_start' : 0,
			'Vocal_encoder': Vocal_encoder.state_dict(),
			'Vision_encoder' : 	Vision_encoder.state_dict(),
			'Wordvec_encoder' : Wordvec_encoder.state_dict(),
			'Attention' : Attention.state_dict(),
			'Predictor' : Predictor.state_dict(),
			'optimizer': optimizer.state_dict(),
		}, False,'triple_attention_net_')
	epoch+= 1 
'------------------------------------------------------Saving model after training completion--------------------------'
if train_mode:
	save_checkpoint({
		'epoch': epoch,
		'loss' : running_loss,
		'j_start' : 0,
		'Vocal_encoder': Vocal_encoder.state_dict(),
		'Vision_encoder' : 	Vision_encoder.state_dict(),
		'Wordvec_encoder' : Wordvec_encoder.state_dict(),
		'Attention' : Attention.state_dict(),
		'Predictor' : Predictor.state_dict(),
		'optimizer': optimizer.state_dict(),
	}, False)

# print('Accuracy:', accuracy_score(y_true, y_pred))
# print('F1 score:', f1_score(y_true, y_pred,average = 'weighted'))
# print('Recall:', recall_score(y_true, y_pred,average ='weighted'))
# print('Precision:', precision_score(y_true, y_pred,average = 'weighted'))
