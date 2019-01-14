# learning model
from attention_network import LstmNet
from attention_network import Attention
from attention_network import Predictor
import torch
import torch.nn as nn
from torch.autograd import Variable
from fea_data import fea_data_npy
from fea_data import fea_test_data_npy
import sys
import os
import shutil
import numpy as np
from cmu_score_v2 import ComputePerformance
from cmu_score_v2 import PrintScore
from cmu_score_v2 import PrintScoreWiki
from datasets import database


### ----------------------------------------- pytorch setup
use_CUDA = True
# set seed to be able to reproduce output
torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)


### ----------------------------------------- training variables
MAX_ITER=100
LEARNING_RATE=0.0001
BATCHSIZE = 1
PADDING = False

### ----------------------------------------- script input
if len(sys.argv) >= 3:
        datalbl = sys.argv[1]
        ext = sys.argv[2]
else:
        print("Error: Missing datalbl and feature extension")
        sys.exit()
if "MOSEI" in datalbl:
        MAX_ITER = 10


### ----------------------------------------- test model
def test_model(dataitems):
	# given validation or test set, test the model
	overall_hyp = np.zeros((0,num_emotions))
	overall_ref = np.zeros((0,num_emotions))
	for i,(fea,ref) in enumerate(dataitems):
		if use_CUDA:
			fea = Variable(fea.float()).cuda()
			ref = Variable(ref.float()).cuda()
		else:
			fea = Variable(fea.float())
			ref = Variable(ref.float())
		hyp = encoder(fea)
		output = attention(hyp, dan_hidden_size, att_hidden_size, BATCHSIZE=1)
		outputs = predictor(output)
		outputs = torch.clamp(outputs,0,3)
		overall_hyp = np.concatenate((overall_hyp, outputs.unsqueeze(0).data.cpu().numpy()),axis=0)
		overall_ref = np.concatenate((overall_ref, ref.data.cpu().numpy()),axis=0)
	score = ComputePerformance(overall_ref, overall_hyp)
	return score


### ----------------------------------------- save model
def save_model(state, is_final):
	# save intermediate models
	savedir = './models/%s/torch.%d.%d-%d.%d.%d.%d/' % (datalbl, input_size, hidden_size, num_layers, outlayer_size, att_hidden_size, num_emotions)
	filename = "%s/model-iter%d-loss%.4f.pth.tar" % (savedir, state['epoch'], state['loss'])
	os.system("mkdir -p %s" % savedir)
	torch.save(state, filename)
	if is_final:
		shutil.copyfile(filename, '%s/final_model-iter%d.pth.tar'% (savedir, state['epoch']))


### ----------------------------------------- load data
# load npy data
trainset = fea_data_npy(database[datalbl][ext]['train']['fea'], database[datalbl][ext]['train']['ref'], datalbl, BATCHSIZE, PADDING)
validset = fea_test_data_npy(database[datalbl][ext]['valid']['fea'], database[datalbl][ext]['valid']['ref'], datalbl)
testset = fea_test_data_npy(database[datalbl][ext]['test']['fea'], database[datalbl][ext]['test']['ref'], datalbl)
# pytorch data loader
train_dataitems=torch.utils.data.DataLoader(dataset=trainset,batch_size=BATCHSIZE,shuffle=True,num_workers=2)
valid_dataitems=torch.utils.data.DataLoader(dataset=validset,batch_size=1,shuffle=False,num_workers=2)
test_dataitems=torch.utils.data.DataLoader(dataset=testset,batch_size=1,shuffle=False,num_workers=2)
# shuffle - reshuffles data at every epoch
# num_workers - how many subprocesses to use for data loading


### ----------------------------------------- quickrun for debugging
if 'debug' in sys.argv:
	debug_mode = True
	l = 15
	trainset.fea = trainset.fea[:l]
	trainset.ref = trainset.ref[:l]
	validset.fea = validset.fea[:l]
	validset.ref = validset.ref[:l]
	testset.fea = testset.fea[:l]
	testset.ref = testset.ref[:l]
	MAX_ITER = 1
else:
	debug_mode = False


### ----------------------------------------- model setup
input_size = trainset.fea[0].shape[1]	# get from data
hidden_size = 512
num_layers = 2
outlayer_size = 1024
num_emotions = trainset.ref[0].shape[0]
dan_hidden_size = 1024 # dan = dual attention network
att_hidden_size = 128
modelname = "lstm_%d.%dx%d.%d-att_%d.%d-pred_%d" % (input_size, hidden_size, num_layers, outlayer_size, att_hidden_size, dan_hidden_size, num_emotions)
# summarise details
print("Batchsize = %s" % BATCHSIZE)
print("Max epochs = %s" % MAX_ITER)
print("Learning rate = %s" % LEARNING_RATE)
print("Dataset = %s" % datalbl)
print("Model = %s" % modelname)
print("Emotion classes = %d" % num_emotions)


### ----------------------------------------- model initialisation
encoder = LstmNet(input_size, hidden_size, num_layers, outlayer_size, num_emotions)
attention = Attention(num_emotions, dan_hidden_size, att_hidden_size)
predictor = Predictor(num_emotions, dan_hidden_size)
if use_CUDA:
	encoder = encoder.cuda()
	attention = attention.cuda()
	predictor = predictor.cuda()


### ----------------------------------------- loss function 
# computes a value that estimates how far away the output is from the target
#if "MOSEI" in datalbl:
criterion = nn.MSELoss()
print("Criterion = MSELoss")
#else:
#	criterion = nn.CrossEntropyLoss()
#	print("Criterion = CrossEntropyLoss")
params = list(encoder.parameters()) + list(attention.parameters()) + list(predictor.parameters())
# different update rules - Adam: A Method for Stochastic Optimization
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
print('Parameters in the model = ' + str(len(params)))
print("Optimiser = Adam")


### ----------------------------------------- train network
epoch = 1
accumulated_loss = 0
while epoch <= MAX_ITER:
	accumulated_loss = 0
	overall_hyp = np.zeros((0,num_emotions))
	overall_ref = np.zeros((0,num_emotions))
	for i,(fea,ref) in enumerate(train_dataitems):
		if use_CUDA:
			fea = Variable(fea.float()).cuda()
			ref = Variable(ref.float()).cuda()
		else:
			fea = Variable(fea.float())
			ref = Variable(ref.float())
		hyp = encoder(fea)
		output = attention(hyp, dan_hidden_size, att_hidden_size, BATCHSIZE)
		outputs = predictor(output) # produces tensor.shape[1,6]
		# clamp/clip/send to 0 values below 0 and above 3
		outputs = torch.clamp(outputs,0,3)		
		if debug_mode == True:
			print("i:", i)
			print("ref:", ref, ref.shape)
			print("fea:", fea, fea.shape)
			print("Variable(fea):", fea, fea.shape)
			print("hyp=encoder(Var(fea)):", hyp, hyp.shape)
			print("output=attention(hyp):", output, output.shape)
			print("outputs=predictor(output):", outputs, outputs.shape)
			print("torch.clamp(outputs,0,3):", outputs, outputs.shape)
		# computes loss using mean-squared error between th einput and the target
		if BATCHSIZE > 1:
			loss = criterion(outputs, ref) # related to shape of outputs
		else:
			loss = criterion(outputs, ref[0])
		# the whole graph is differentiated w.r.t. the loss, and all Variables in the graph will have their .grad Variable accumulated with the gradient
		# backprop
		loss.backward()
		# update weights
		optimizer.step()
		# zero the gradient buffers of all params with random gradient
		optimizer.zero_grad()
		encoder.zero_grad()
		attention.zero_grad()
		predictor.zero_grad()
		accumulated_loss += loss.item()
		# concatenate reference and hypothesis
		if BATCHSIZE > 1:
			overall_hyp = np.concatenate((overall_hyp, outputs.data.cpu().numpy()),axis=0)
		else:
			overall_hyp = np.concatenate((overall_hyp, outputs.unsqueeze(0).data.cpu().numpy()),axis=0)
		overall_ref = np.concatenate((overall_ref, ref.data.cpu().numpy()),axis=0)
	# compute score at end of every epoch 
	train_score = ComputePerformance(overall_ref, overall_hyp)
	valid_score = test_model(valid_dataitems)
	test_score = test_model(test_dataitems)
	# print scores
	print('Scoring Overall -- Epoch [%d], Samples [%d], Loss: train %.4f, valid %.4f, test %.4f' % (epoch, i+1, train_score['MSE'], valid_score['MSE'], test_score['MSE']))
	# ready for next epoch
	accumulated_loss /= (i+1)
	accumulated_loss = float(accumulated_loss)
	# save intermediate models
	save_model({
		'epoch': epoch,
		'loss' : accumulated_loss,
		'encoder' : encoder.state_dict(),
		'attention' : attention.state_dict(),
		'predictor' : predictor.state_dict(),
		'optimizer' : optimizer.state_dict(),
	}, False)
	epoch += 1
else:
	if epoch == MAX_ITER:
		print("TRAINING STOPPED as Epoch [%d] == MAX_ITER [%d]" % (epoch, MAX_ITER))
	else:
		print("TRAINING STOPPED")
	PrintScore(test_score, epoch-1, i+1, 'test')


### -----------------------------------------print test scores in wiki format
PrintScoreWiki(test_score, epoch-1)


### -----------------------------------------save the final trained model
save_model({
	'epoch': epoch-1,
	'loss': accumulated_loss,
	'encoder': encoder.state_dict(),
	'attention': attention.state_dict(),
	'predictor': predictor.state_dict(),
	'optimizer': optimizer.state_dict()
}, True)



