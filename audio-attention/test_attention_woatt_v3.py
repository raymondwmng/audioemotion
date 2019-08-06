#!/usr/bin/python
import sys
import os
import glob
import numpy as np
import torch
from cmu_score_v3 import ComputePerformance
from cmu_score_v3 import PrintScore
from cmu_score_v3 import PrintScoreEmo
from cmu_score_v3 import PrintScoreDom
from attention_model_woatt import calculate_priors
from attention_model_woatt import load_model
from attention_model_woatt import load_data
from attention_model_woatt import read_cfg as read_config
from attention_model_woatt import model_init
from attention_model_woatt import define_loss
from attention_model_woatt import train_model
import configparser

### ----------------------------------------- variables
TRAIN_MODE = False


### ----------------------------------------- seed
seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)



### ----------------------------------------- config.txt
def read_cfg(config):
    cfg = configparser.ConfigParser()
    cfg.read(config)
    for var in cfg['DEFAULT']:
        print("%s = %s" % (var, cfg['DEFAULT'][var]))
    # cuda
    global USE_CUDA
    USE_CUDA = cfg['DEFAULT'].getboolean('USE_CUDA')
    # training
    global OPTIM
    OPTIM = cfg['DEFAULT']['OPTIM']
    global MAX_ITER
    MAX_ITER = cfg['DEFAULT'].getint('MAX_ITER')
    global LEARNING_RATE
    LEARNING_RATE = cfg['DEFAULT'].getfloat('LEARNING_RATE')
    global LR_schedule
    LR_schedule = cfg['DEFAULT']['LR_schedule']
    global LR_size
    LR_size = cfg['DEFAULT'].getint('LR_SIZE')
    global LR_factor
    LR_factor = cfg['DEFAULT'].getfloat('LR_factor')
    global BATCHSIZE
    BATCHSIZE = cfg['DEFAULT'].getint('BATCHSIZE')
    global PADDING
    PADDING = cfg['DEFAULT'].getboolean('PADDING')
    global SAVE_MODEL
    SAVE_MODEL = cfg['DEFAULT'].getboolean('SAVE_MODEL')
    global SAVE_ITER
    SAVE_ITER = cfg['DEFAULT'].getint('SAVE_ITER')
    global SELECT_BEST_MODEL
    SELECT_BEST_MODEL = cfg['DEFAULT'].getboolean('SELECT_BEST_MODEL')
    global USE_PRETRAINED
    USE_PRETRAINED = cfg['DEFAULT'].getboolean('USE_PRETRAINED')
    global VALIDATION
    VALIDATION = cfg['DEFAULT'].getboolean('VALIDATION')
    global MULTITASK
    MULTITASK = cfg['DEFAULT'].getboolean('MULTITASK')
    global PRIORS
    PRIORS = cfg['DEFAULT'].getboolean('PRIORS')
    # model
    global EXT
    EXT = cfg['DEFAULT']['EXT']
    global input_size
    input_size = cfg['DEFAULT'].getint('input_size')
    global hidden_size
    hidden_size = cfg['DEFAULT'].getint('hidden_size')
    global num_layers
    num_layers = cfg['DEFAULT'].getint('num_layers')
    global outlayer_size
    outlayer_size = cfg['DEFAULT'].getint('outlayer_size')
    global num_emotions
    num_emotions = cfg['DEFAULT'].getint('num_emotions')
    global dan_hidden_size
    dan_hidden_size = cfg['DEFAULT'].getint('dan_hidden_size')
    global att_hidden_size
    att_hidden_size = cfg['DEFAULT'].getint('att_hidden_size')
    global model_name
    model_name = "lstm%d.%dx%d.%d-att%d.%d-out%d" % (input_size, hidden_size, num_layers, outlayer_size, att_hidden_size, dan_hidden_size, num_emotions)
    # environment
    global WDIR
    WDIR = cfg['DEFAULT']['WDIR']
    global exp
    if 'EXP' in cfg['DEFAULT']:
        exp = cfg['DEFAULT']['exp']
    else:
        exp = "logs"
        print("exp not defined:", exp)
    global path
    path = cfg['DEFAULT']['path']
    global SAVEDIR
    SAVEDIR = "/%s/%s/%s/models/%s/" % (WDIR, exp, path, model_name)
    if not os.path.isdir(SAVEDIR):
        os.makedirs(SAVEDIR)
    global DEBUG_MODE
    DEBUG_MODE = cfg['DEFAULT'].getboolean('DEBUG_MODE')
    if DEBUG_MODE:
        VALIDATION = False
    return WDIR, path

### ----------------------------------------- main
def main():
    # load config
    if "-c" in sys.argv:
        config = sys.argv[sys.argv.index("-c")+1]
    else:
        config = "config.ini"
    print("CONFIG: %s" % config)
    read_cfg(config)
    read_config(config)



    # read in variables
    if "--train" in sys.argv:
        traindatalbl = sys.argv[sys.argv.index("--train")+1].split("+")
    if "--test" in sys.argv:
        testdatalbl = sys.argv[sys.argv.index("--test")+1].split("+")
    if "--model" in sys.argv:
        modelfile = sys.argv[sys.argv.index("--model")+1].split("+")
    else:
        modelfile = False
    if "-e" in sys.argv:
        EXT = sys.argv[sys.argv.index("-e")+1]
    if "--epochs" in sys.argv:
        epochs = [int(sys.argv[sys.argv.index("--epochs")+1]), int(sys.argv[sys.argv.index("--epochs")+2])]
    else:
        epochs = [1,MAX_ITER]
    if "--no-cuda" in sys.argv:
        USE_CUDA=False
        print("CHANGED: USE_CUDA=False")


    # setup/init data and model
    train_dataitems, valid_dataitems, multi_test_dataitems = load_data(traindatalbl, testdatalbl, EXT, TRAIN_MODE, DEBUG_MODE)
    network = model_init(OPTIM, TRAIN_MODE)
    criterions = define_loss()
    priors = calculate_priors(train_dataitems)

    # find models
    if modelfile:
        models = modelfile
    else:
#        models = sorted(glob.glob('/%s/*/%s/models/%s/epoch*tar' % (WDIR, path, model_name)))
        print("SAVEDIR=", SAVEDIR)
        models = sorted(glob.glob('%s/epoch*tar' % SAVEDIR)) 
        models2 = []
        for m in models:
            e = int(m.split("epoch")[1].split("-samples")[0])
            if epochs[0] <= e <= epochs[1]:
                models2.append(m)
        models = models2
    print("Models[%d,%d] = " % (epochs[0], epochs[1]), models, len(models))

    # test one model
    if DEBUG_MODE:
        models = [models[-1]]

    # loop over trained models
    for pretrained_model in models:

        # check for pretrained model
        network, epoch = load_model(pretrained_model, network, TRAIN_MODE)
        # test
        for [datalbl, test_dataitems] in multi_test_dataitems:
            print("Testing: ", datalbl)
            [loss, ref, hyp] = train_model(datalbl, test_dataitems, network, criterions, priors, False, DEBUG_MODE)
            print("---\nSCORING TEST-- Epoch[%d]: [%d] %.4f" % (epoch, len(test_dataitems.dataset), loss))
            PrintScore(ComputePerformance(ref, hyp, datalbl, "EMO"), epoch, len(test_dataitems.dataset), [pretrained_model,datalbl], "EMO")

            # test one database
            if DEBUG_MODE:
                continue


if __name__ == "__main__": main()
