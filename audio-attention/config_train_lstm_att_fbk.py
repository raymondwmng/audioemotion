#!/usr/bin/python
import torch
import numpy as np

### ----------------------------------------- pytorch setup
USE_CUDA = True
seed = 777	# set seed to be able to reproduce output
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

print("# -- PYTORCH SETUP -- #")
print("USE_CUDA = ", USE_CUDA)
print("seed = ", seed)

### ----------------------------------------- training variables
MAX_ITER=100
LEARNING_RATE=0.0001
BATCHSIZE = 1
PADDING = False # padding keeps whole segments in batchmode, else segments chopped
VALIDATION = False
SAVE_MODEL = True
USE_PRETRAINED = True

print("# -- TRAINING VARIABLES -- #")
print("MAX_ITER = ", MAX_ITER)
print("LEARNING_RATE = ", LEARNING_RATE)
print("VALIDATION = ", VALIDATION)
print("SAVE_MODEL = ", SAVE_MODEL)
print("USE_PRETRAINED = ", USE_PRETRAINED)

### ----------------------------------------- model setup
EXT = "fbk"
input_size = 23	# feature dependent
hidden_size = 512
num_layers = 2
outlayer_size = 1024
num_emotions = 6
dan_hidden_size = 1024 # dan = dual attention network
att_hidden_size = 128
model_name = "%s-lstm%d.%dx%d.%d-att%d.%d-out%d" % (EXT, input_size, hidden_size, num_layers, outlayer_size, att_hidden_size, dan_hidden_size, num_emotions)

print("# -- MODEL SETUP -- #")
#print("Train dataset = ", traindatalbl)
#print("Test dataset = ", testdatalbl)
print("Model = %s" % model_name)
print("Features = %s" % EXT)
print("Emotion classes = %d" % num_emotions)


