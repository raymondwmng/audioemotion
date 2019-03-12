#!/usr/bin/python
import torch
import numpy as np
import os

### ----------------------------------------- enivorment
DEBUG_MODE=False # # #False #False #True #False
WDIR="/share/spandh.ami1/emo/dev/6class/vlog/mosei/tools/audioemotion/audio-attention/"
path="train-MOSEI_acl2018-fbk-ReduceLROnPlateau1h0.5"#"train-MOSEI_acl2018-fbk-ReduceLROnPlateau1h0.5"#"train-MOSEI_acl2018-fbk-ReduceLROnPlateau1h0.5"

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
MAX_ITER = 100 #100 #100 #100 #200 #200 #200 #100
LEARNING_RATE = 0.0001
LR_schedule="ReduceLROnPlateau"#"ReduceLROnPlateau"#"ReduceLROnPlateau"
LR_size=1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#10#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#1#2#5#5#15#20#1
LR_factor=0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5#0.5
BATCHSIZE = 1
PADDING = False # padding keeps whole segments in batchmode, else segments chopped ... yet to implement
SAVE_MODEL = True #True
USE_PRETRAINED = True #True #False
VALIDATION = False
MULTITASK=False

### ----------------------------------------- model setup
EXT = "fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk80" #"fbk60" #"fbk40" #"fbk80" #"fbk60" #"fbk40" #"fbk80" #"fbk60" #"fbk40" #"fbk80" #"fbk60" #"fbk40" #"covarep" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk" #"fbk"
input_size = 23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #80 #60 #40 #80 #60 #40 #80 #60 #40 # # # #74 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23 #23
hidden_size = 512
num_layers = 2
outlayer_size = 1024
num_emotions = 6
dan_hidden_size = 1024 # dan = dual attention network
att_hidden_size = 128
model_name = "%s-lstm%d.%dx%d.%d-att%d.%d-out%d.lr%f" % (EXT, input_size, hidden_size, num_layers, outlayer_size, att_hidden_size, dan_hidden_size, num_emotions, LEARNING_RATE)

### ---------------------------------------- unchanging
if DEBUG_MODE:
    VALIDATION = False

### ----------------------------------------
def printConfig(EXT, traindatalbl, TRAIN_MODE):
    SAVEDIR=WDIR+"/logs/"+path+"/models/%s/" % (model_name)
    if not os.path.isdir(SAVEDIR):
        os.makedirs(SAVEDIR)
    print("# -- ENVIRONMENT -- #")
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
        print("MAX_ITER = 100 #100 #100 #100 #200 #200 #200 #", MAX_ITER)
        print("LEARNING_RATE = ", LEARNING_RATE)
        print("LR_schedule = ", LR_schedule)
        print("VALIDATION = ", VALIDATION)
        print("SAVE_MODEL = ", SAVE_MODEL)
        print("USE_PRETRAINED = ", USE_PRETRAINED)
    return SAVEDIR
