# learning model
from attention_network import LstmNet
from attention_network import Attention
from attention_network import Predictor
import torch
import torch.nn as nn
from torch.autograd import Variable
from fea_data import fea_data
import sys
import os
import shutil
import numpy as np
from cmu_score_v2 import ComputePerformance
from cmu_score_v2 import PrintScore
from cmu_score_v2 import PrintScoreEpochs
from datasets import database

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
use_CUDA = True
use_pretrained = False
debug_mode = False
SHUFF=False

# set seed to be able to reproduce output
torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)

# training variables
MAX_ITER=5
LEARNING_RATE=0.0001


# given validation or test set, test the model
def testModel(dataitems):
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
		output = attention(hyp, dan_hidden_size, att_hidden_size, attention_type)
		outputs = predictor(output)
		outputs = torch.clamp(outputs,0,3)
		overall_hyp = np.concatenate((overall_hyp, outputs.unsqueeze(0).data.cpu().numpy()),axis=0)
		overall_ref = np.concatenate((overall_ref, ref.data.cpu().numpy()),axis=0)
	score = ComputePerformance(overall_ref, overall_hyp)
	return score

# save intermediate models
def save_checkpoint(state, is_final):
	savedir = './models/%s/torch.%d.%d-%d.%d.%d-att_%s/' % (datalbl, input_size, hidden_size, num_layers, outlayer_size, att_hidden_size, attention_type)
	filename = "%s/model-iter%d-loss%.4f.pth.tar" % (savedir, state['epoch'], state['loss'])
	os.system("mkdir -p %s" % savedir)
	torch.save(state, filename)
	if is_final:
		shutil.copyfile(filename, '%s/final_model-iter%d.pth.tar'% (savedir, state['epoch']))


# set attention type
if "-a" in sys.argv:
	attention_type = sys.argv[sys.argv.index("-a")+1]
else:
	attention_type = 'attention'


# load data
datalbl = 'MOSEI_acl2018'#'MOSEI_edinacl2018' 'misc' 'ent05p2'
#file_fea = database[datalbl]['fea_covarep']
file_fea = database[datalbl]['scp_fbk']
file_ref = database[datalbl]['ref_etm']
trainset = fea_data(file_fea, file_ref, dataset_name=datalbl, dataset_split='train')
validset = fea_data(file_fea, file_ref, dataset_name=datalbl, dataset_split='valid')
testset = fea_data(file_fea, file_ref, dataset_name=datalbl, dataset_split='test')
train_dataitems=torch.utils.data.DataLoader(dataset=trainset,batch_size=1,shuffle=SHUFF,num_workers=2)
valid_dataitems=torch.utils.data.DataLoader(dataset=validset,batch_size=1,shuffle=SHUFF,num_workers=2)
test_dataitems=torch.utils.data.DataLoader(dataset=testset,batch_size=1,shuffle=SHUFF,num_workers=2)
# shuffle - reshuffles data at every epoch
# num_workers - how many subprocesses to use for data loading

# quickrun for debugging
if debug_mode == True:
	trainset.fea = trainset.fea[:50]
	trainset.ref = trainset.ref[:50]
	validset.fea = validset.fea[:50]
	validset.ref = validset.ref[:50]
	testset.fea = testset.fea[:50]
	testset.ref = testset.ref[:50]


# about model 
input_size = trainset.fea[0].shape[1]	# get from data
hidden_size = 512
num_layers = 2
outlayer_size = 1024
num_emotions = trainset.ref[0].shape[0]
dan_hidden_size = 1024 # ???
att_hidden_size = 128
modelname = "%dl.%d.%d.%d.%d" % (num_layers, input_size, hidden_size, outlayer_size, att_hidden_size)

# summarise details
print("Max epochs = %s" % MAX_ITER)
print("Learning rate = %s" % LEARNING_RATE)
print("Dataset = %s" % datalbl)
print("Features = %s" % file_fea)
print("Reference = %s" % file_ref)
print("Model = %s" % modelname)
print("Attention = %s" % attention_type) 
print("Emotion classes = %d" % num_emotions)


# model initialisation
encoder = LstmNet(input_size, hidden_size, num_layers, outlayer_size, num_emotions)
attention = Attention(num_emotions, dan_hidden_size, att_hidden_size)
predictor = Predictor(num_emotions, dan_hidden_size)
if use_CUDA:
	encoder = encoder.cuda()
	attention = attention.cuda()
	predictor = predictor.cuda()

# loss function 
# computes a value that estimates how far away the output is from the target
criterion = nn.MSELoss()
params = list(encoder.parameters()) + list(attention.parameters()) + list(predictor.parameters())
print('Parameters in the model = ' + str(len(params)))
# different update rules - Adam: A Method for Stochastic Optimization
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
print("Criterion = MSE")
print("Optimiser = Adam")


# train network
epoch = 1
scores = []
accumulated_loss, prev_loss = 0, 1
loss_diff = 0.01
while epoch <= MAX_ITER: #and np.abs(prev_loss-accumulated_loss) > loss_diff:
	prev_loss = accumulated_loss
	accumulated_loss = 0
	overall_hyp = np.zeros((0,num_emotions))
	overall_ref = np.zeros((0,num_emotions))
	for i,(fea,ref) in enumerate(train_dataitems):
		if debug_mode == True:
			print("i:", i)
			print("ref:", ref)
			print("fea:", fea)
		if use_CUDA:
			fea = Variable(fea.float()).cuda()
			ref = Variable(ref.float()).cuda()
		else:
			fea = Variable(fea.float())
			ref = Variable(ref.float())
		hyp = encoder(fea)
		output = attention(hyp, dan_hidden_size, att_hidden_size, attention_type)
		outputs = predictor(output) # produces tensor.shape[1,7]
		# clamp/clip/send to 0 values below 0 and above 3
		if debug_mode == True:
			print("Variable(fea):", fea)
			print("hyp=encoder(Var(fea)):", hyp)
			print("output=attention(hyp):", output)
			print("outputs=predictor(output):", outputs)
		#outputs = torch.clamp(outputs[1:],0,3)
		outputs = torch.cat((outputs[0].unsqueeze(0),(torch.clamp(outputs[1:],0,3))))
		if debug_mode == True:
			print("torch.clamp(outputs,0,3):", outputs)
		# computes loss using mean-squared error between th einput and the target
		loss = criterion(outputs, ref[0]) # related to shape of outputs
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
		accumulated_loss += loss.data[0]
		# concatenate reference and hypothesis
		overall_hyp = np.concatenate((overall_hyp, outputs.unsqueeze(0).data.cpu().numpy()),axis=0)
		overall_ref = np.concatenate((overall_ref, ref.data.cpu().numpy()),axis=0)
		# compute loss and score
		if (i+1)%500==0:
			print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f' % (epoch, i+1, accumulated_loss/(i+1)))
	### compute score at end of every epoch # trainset score
	train_score = ComputePerformance(overall_ref, overall_hyp)
	PrintScore(train_score, epoch, i+1, 'train')
	### validset score
	valid_score = testModel(valid_dataitems)
	PrintScore(valid_score, epoch, i+1, 'valid')
	### testset score
	test_score = testModel(test_dataitems)
	PrintScore(test_score, epoch, i+1, 'test')
	scores.append(test_score)
	# print scores
#	print('Training -- Epoch [%d], Sample [%d], Average Loss: %.4f' % (epoch+1, i+1, accumulated_loss/(i+1)))
	print('Scoring Overall -- Epoch [%d], Sample [%d], Train MSE: %.4f' % (epoch, i+1, train_score['MSE']))
	print('Scoring Overall -- Epoch [%d], Sample [%d], Valid MSE: %.4f' % (epoch, i+1, valid_score['MSE']))
	print('Scoring Overall -- Epoch [%d], Sample [%d], Test  MSE: %.4f' % (epoch, i+1, test_score['MSE']))
	# ready for next epoch
	accumulated_loss /= (i+1)
	accumulated_loss = float(accumulated_loss)
	# save intermediate models
	save_checkpoint({
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
		print("TRAINING STOPPED as |prev_loss [%.4f] - accum_loss [%.4f]| < %f" % (prev_loss, accumulated_loss, loss_diff))



# print all test scores in wiki format
PrintScoreEpochs(scores)


# save the final trained model
save_checkpoint({
	'epoch': epoch-1,
	'loss': accumulated_loss,
	'encoder': encoder.state_dict(),
	'attention': attention.state_dict(),
	'predictor': predictor.state_dict(),
	'optimizer': optimizer.state_dict()
}, True)



