#!/usr/bin/python
import torch
import numpy as np
import os

### ----------------------------------------- enivorment
DEBUG_MODE=False #False #False #True #False
WDIR="/share/spandh.ami1/emo/dev/6class/vlog/mosei/tools/audioemotion/audio-attention/"

### ----------------------------------------- pytorch setup
USE_CUDA = True
seed = 777	# set seed to be able to reproduce output
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

### ----------------------------------------- training variables
OPTIM = "Adam"
MAX_ITER = 100
LEARNING_RATE = 0.0001
BATCHSIZE = 1
PADDING = False # padding keeps whole segments in batchmode, else segments chopped ... yet to implement
SAVE_MODEL = True #True
USE_PRETRAINED = True #True #False
VALIDATION = False

### ----------------------------------------- model setup
EXT = "fbk"
input_size = 23
hidden_size = 512
num_layers = 2
outlayer_size = 1024
num_emotions = 6
dan_hidden_size = 1024 # dan = dual attention network
att_hidden_size = 128
model_name = "%s-lstm%d.%dx%d.%d-att%d.%d-out%d" % (EXT, input_size, hidden_size, num_layers, outlayer_size, att_hidden_size, dan_hidden_size, num_emotions)

### ---------------------------------------- unchanging
if DEBUG_MODE:
    VALIDATION = False

### ----------------------------------------
def printConfig(EXT, traindatalbl, TRAIN_MODE):
    SAVEDIR=WDIR+"/models/%s/%s/" % ("+".join(traindatalbl), model_name)
    if not os.path.isdir(SAVEDIR):
        os.makedirs(SAVEDIR)
    if EXT == "fbk":
#        global input_size
        input_size = 23
    elif EXT == "covarep":
#        global input_size
        input_size = 74
    elif EXT == "mfcc":
#        global input_size
        input_size = 13
    elif EXT == "plp":
#        global input_size
        input_size = 14
    print("# -- ENVIRONMENT -- #")
    print("WDIR = ", WDIR)
    print("SAVEDIR = ", SAVEDIR)
    print("DEBUG_MODE = ", DEBUG_MODE)
    print("# -- PYTORCH SETUP -- #")
    print("USE_CUDA = ", USE_CUDA)
    print("seed = ", seed)
    print("# -- MODEL SETUP -- #")
    print("Features = %s" % EXT)
    print("Model = %s" % model_name)
    print("Emotion classes = %d" % num_emotions)
    if TRAIN_MODE:
        print("# -- TRAINING VARIABLES -- #")
        print("OPTIMISER = ", OPTIM)
        print("MAX_ITER = ", MAX_ITER)
        print("LEARNING_RATE = ", LEARNING_RATE)
        print("VALIDATION = ", VALIDATION)
        print("SAVE_MODEL = ", SAVE_MODEL)
        print("USE_PRETRAINED = ", USE_PRETRAINED)
    return SAVEDIR

