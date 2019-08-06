#!/usr/bin/python
from attention_network_multihead import LstmNet
from attention_network_multihead import Attention
from attention_network_multihead import Predictor
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from fea_data import fea_data_npy
from fea_data import fea_test_data_npy
import sys
import os
import shutil
import glob
import numpy as np
from cmu_score_v3 import ComputePerformance
from cmu_score_v3 import PrintScore
from datasets import database
import configparser

### ----------------------------------------- seed
seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


### ----------------------------------------- config.ini
def read_cfg(config):
    cfg = configparser.ConfigParser()   
    cfg.read(config)
    for var in cfg['DEFAULT']:
        print("%s = %s" % (var, cfg['DEFAULT'][var]))
    # cuda
    global USE_CUDA
    USE_CUDA = cfg['DEFAULT'].getboolean('USE_CUDA')
    # training
    global ATTENTION
    ATTENTION = cfg['DEFAULT'].getboolean('ATTENTION')
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
    global multihead_size
    multihead_size = cfg['DEFAULT'].getint('multihead_size')
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


### ----------------------------------------- Convert to numpy
def to_npy(x):
    # convert tensor to numpy format
    return x.data.cpu().numpy()


### ----------------------------------------- Return current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


### ----------------------------------------- load data
def load_data(traindatalbl, testdatalbl, EXT, TRAIN_MODE, DEBUG_MODE):
    if TRAIN_MODE:
        # load train and valid sets and combine
        train_fea, valid_fea, train_ref, valid_ref = [], [], [], []
        for datalbl in traindatalbl:
            train_fea += database[datalbl][EXT]['train']['fea']
            train_ref += database[datalbl][EXT]['train']['ref']
            valid_fea += database[datalbl][EXT]['valid']['fea']
            valid_ref += database[datalbl][EXT]['valid']['ref']
        if not VALIDATION:
            train_fea += valid_fea
            train_ref += valid_ref
            valid_dataitems = []
        else:
            validset = fea_data_npy(valid_fea, valid_ref,  BATCHSIZE, traindatalbl)
            valid_dataitems=torch.utils.data.DataLoader(dataset=trainset,batch_size=1,shuffle=False,num_workers=2)
        trainset = fea_data_npy(train_fea, train_ref, BATCHSIZE, traindatalbl)
        train_dataitems=torch.utils.data.DataLoader(dataset=trainset,batch_size=BATCHSIZE,shuffle=True,num_workers=2)
    else:
        train_dataitems, valid_dataitems = [], []
    # load (multiple) test sets separately
    multi_test_dataitems = []
    if testdatalbl:
        for datalbl in testdatalbl:
            print(datalbl, testdatalbl)
            test_fea = database[datalbl][EXT]['test']['fea']
            test_ref = database[datalbl][EXT]['test']['ref']
            testset = fea_test_data_npy(test_fea, test_ref, datalbl, traindatalbl)
            test_dataitems = torch.utils.data.DataLoader(dataset=testset,batch_size=1,shuffle=False,num_workers=2)
            multi_test_dataitems.append([datalbl, test_dataitems])
    # reduce datasets if debugging code
    if DEBUG_MODE:
        l = 10
        if TRAIN_MODE:
            trainset.fea, trainset.ref = trainset.fea[:l], trainset.ref[:l]
#            if VALIDATION:
#                validset.fea, validset.ref = validset.fea[:l], validset.ref[:l]
        if testdatalbl:
            testset.fea, testset.ref = testset.fea[:l], testset.ref[:l]
    return train_dataitems, valid_dataitems, multi_test_dataitems


### ----------------------------------------- save model
def save_model(state, is_final):
    # save intermediate models
    filename = "%s/epoch%03d-samples%d-loss%.10f-LR%.10f.pth.tar" % (SAVEDIR, state['epoch'], state['samples'], state['loss'], state['LEARNING_RATE'])
    print("Saving model: %s" % filename)
    torch.save(state, filename)
#    if is_final:
#        shutil.copyfile(filename, '%s/final_epoch%d-loss%.4f.pth.tar'% (SAVEDIR, state['epoch'], state['loss']))


### ----------------------------------------- find model
def find_model(epoch):
    # check if model at MAX_ITER already exists
    pretrained_model = "%s/epoch%03d*.pth.tar" % (SAVEDIR, MAX_ITER)
    if glob.glob(pretrained_model) != []:
        print("Model at MAX_ITER=%d already trained (%s)" % (MAX_ITER, glob.glob(pretrained_model)[0]))
        sys.exit()
    # find model at starting epoch
    start_pretrained_model = "%s/epoch%03d*.pth.tar" % (SAVEDIR, epoch)
    if glob.glob(start_pretrained_model) != []:
#        print("Model at epoch=%d will be loaded (%s)" % (epoch, start_pretrained_model))
        start_pretrained_model = glob.glob(pretrained_model)[0]
    else:
        start_pretrained_model = False
    # find highest train model
    if os.path.isdir(SAVEDIR):
        # if savedir exists
        models = os.listdir(SAVEDIR)
        if models != []:
            # if models exist in savedir, find highest and start epoch model
            highest_epoch = int(sorted(models)[-1].split("-")[1].strip("epoch"))
            highest_pretrained_model = "%s/%s" % (SAVEDIR,  sorted(models)[-1])
            start_pretrained_model = "%s/epoch%03d*.pth.tar" % (SAVEDIR, epoch)
            if glob.glob(start_pretrained_model) != []:
                # if start exists
                start_pretrained_model = glob.glob(pretrained_model)[0]
                if highest_epoch > epoch:
                    pretrained_model = highest_pretrained_model
                    print("Model at highest trained epoch=%d will be loaded (%s)" % (highest_epoch, pretrained_model))
                else:
                    pretrained_model = highest_pretrained_model
            else:
                # starting model does not exist, go from highest
                pretrained_model = highest_pretrained_model
                print("Model at highest trained epoch=%d will be loaded (%s)" % (highest_epoch, pretrained_model))
        else:
            # no model exist, train from scratch
            pretrained_model = False
            print("No models exist in folder (%s), training from scratch" % SAVEDIR)
    else:
        # model folder does not exist, train from scratch
        # should always exist as created in config
        pretrained_model = False
        print("Model folder (%s) doe snot exist, creating and training from scratch" % SAVEDIR)
        os.mkdirs(SAVEDIR)
    return pretrained_model


### ----------------------------------------- save model
def save_model(traindatalbl, samples, epoch, network, train_loss):
    if ATTENTION:
        [encoder, attention, predictor, optimizer] = network
    else:
        [encoder, predictor, optimizer] = network
    # save intermediate models
    if ATTENTION:
        state = {
        'data' : "+".join(traindatalbl),
        'epoch': epoch,
        'samples' : samples,
        'loss' : train_loss,
        'LEARNING_RATE' : get_lr(network[-1]),
        'encoder' : encoder.state_dict(),
        'attention' : attention.state_dict(),
        'predictor' : predictor.state_dict(),
        'optimizer' : optimizer.state_dict(),
        }
    else:
        state = {
        'data' : "+".join(traindatalbl),
        'epoch': epoch,
        'samples' : samples,
        'loss' : train_loss,
        'LEARNING_RATE' : get_lr(network[-1]),
        'encoder' : encoder.state_dict(),
        'predictor' : predictor.state_dict(),
        'optimizer' : optimizer.state_dict(),
        }

    filename = "%s/epoch%03d-samples%d-loss%.10f-LR%.10f.pth.tar" % (SAVEDIR, state['epoch'], state['samples'], state['loss'], state['LEARNING_RATE'])
    print("Saving model: %s" % filename)
    torch.save(state, filename)

### ----------------------------------------- load model
def load_model(pretrained_model, network, TRAIN_MODE):
    if ATTENTION:
        [encoder, attention, predictor, optimizer] = network
    else:
        [encoder, predictor, optimizer] = network
#    checkpoint = torch.load(pretrained_model, map_location=lambda storage, location: storage)
    checkpoint = torch.load(pretrained_model)
    encoder.load_state_dict(checkpoint['encoder'])
    if ATTENTION:
        attention.load_state_dict(checkpoint['attention'])
    predictor.load_state_dict(checkpoint['predictor'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    accumulated_loss = checkpoint['loss']
    LEARNING_RATE = checkpoint['LEARNING_RATE']
    data = checkpoint['data']
    samples = checkpoint['samples']
    print("Loaded model (%s[%d]) at epoch (%d) with loss (%.4f) and LEARNING_RATE (%f)" % (pretrained_model, samples, epoch, accumulated_loss, LEARNING_RATE))
    if TRAIN_MODE:
        encoder.train()
        if ATTENTION:
            attention.train()
        predictor.train()
    else:
        encoder.eval()
        if ATTENTION:
            attention.eval()
        predictor.eval()
    if ATTENTION:
        return [encoder, attention, predictor, optimizer], epoch
    else:
        return [encoder, predictor, optimizer], epoch


### ----------------------------------------- model initialisation
def model_init(optim, TRAIN_MODE):
    encoder = LstmNet(input_size, hidden_size, num_layers, outlayer_size, num_emotions)
    if ATTENTION:
        attention = Attention(num_emotions, dan_hidden_size, att_hidden_size, multihead_size)
        predictor = Predictor(num_emotions, dan_hidden_size)
    else:
        predictor = Predictor(num_emotions, hidden_size)
    if USE_CUDA:
        encoder = encoder.cuda()
        if ATTENTION:
            attention = attention.cuda()
        predictor = predictor.cuda()
    if TRAIN_MODE: 
        # sets the mode (useful for batchnorm, dropout)
        encoder.train()
        if ATTENTION:
            attention.train()
        predictor.train()
    else:
        encoder.eval()
        if ATTENTION:
            attention.eval()
        predictor.eval()
    params = list(encoder.parameters())
    print('Parameters:encoder = %d' % len(list(encoder.parameters())))
    if ATTENTION:
        params += list(attention.parameters())
        print('Parameters:attention = %d' % len(list(attention.parameters())))
    params += list(predictor.parameters()) 
    print('Parameters:predictor = %d' % len(list(predictor.parameters())))
    print('Parameters:total = %d' % len(params))
    if optim == "Adam":
        # different update rules - Adam: A Method for Stochastic Optimization
        optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    if ATTENTION:
        return [encoder, attention, predictor, optimizer]
    else:
        return [encoder, predictor, optimizer]


### ----------------------------------------- loss function 
# computes a value that estimates how far away the output is from the target
def define_loss():
    criterion_c = nn.CrossEntropyLoss()
    criterion_r = nn.MSELoss()
    return [criterion_c, criterion_r]


### ----------------------------------------- train model
def train_model(datalbl, dataitems, network, criterions, TRAIN_MODE, DEBUG_MODE):
    # train the model or test if TRAIN_MODE == False
    if ATTENTION:
        [encoder, attention, predictor, optimizer] = network
    else:
        [encoder, predictor, optimizer] = network
    [criterion_c, criterion_r] = criterions
    accumulated_loss = 0
    overall_hyp = np.zeros((0,num_emotions))
    overall_ref = np.zeros((0,num_emotions))

    if datalbl == "iemocap_t1234t5_haex1sa1an1ne1":
        overall_ref = np.zeros((0,4))

    for i,(fea,ref,tsk) in enumerate(dataitems):
        # send to cuda
        if USE_CUDA:
            fea = Variable(fea.float()).cuda()
            ref = Variable(ref.float()).cuda()
        else:
            fea = Variable(fea.float())
            ref = Variable(ref.float())

        # train
#        print("feature", fea.shape)
        hyp = encoder(fea, ATTENTION)
#        print("encoder", hyp.shape)
        if ATTENTION:
            output = attention(hyp, dan_hidden_size, att_hidden_size, multihead_size)
        else:
            output = hyp
#        print("(if att)", output.shape)
        outputs = predictor(output)
#        print("predict", outputs.shape)

        # loss
        loss = criterion_c(outputs, torch.max(ref, 1)[1])
        accumulated_loss += loss.item()
        overall_hyp = np.concatenate((overall_hyp, to_npy(outputs)),axis=0)
        overall_ref = np.concatenate((overall_ref, to_npy(ref)),axis=0)

        if DEBUG_MODE and TRAIN_MODE:
            print("%4d ref:" % i, ref, ref.shape, torch.max(ref, 1)[1])
            print("%4d fea:" % i, fea, fea.shape)
            print("%4d hyp=encoder(fea):" % i, hyp, hyp.shape)
            if ATTENTION:
                print("%4d output=attention(hyp):" % i, output, output.shape)
            print("%4d outputs=predictor(output):" % i, outputs, outputs.shape)
            print("%4d loss:" % i, loss, accumulated_loss, accumulated_loss/(i+1))

        if TRAIN_MODE:
            # backprop
            loss.backward()
            # update weights    
            optimizer.step()
            # zero the gradient
            encoder.zero_grad()
            if ATTENTION:
                attention.zero_grad()
            predictor.zero_grad()
            optimizer.zero_grad()

    # overall loss
    accumulated_loss /= (i+1)
    if TRAIN_MODE:
        return [accumulated_loss, overall_ref, overall_hyp, network]
    else:
        return [accumulated_loss, overall_ref, overall_hyp]


### ----------------------------------------- main
def main():
    # load config
    if "-c" in sys.argv:
        config = sys.argv[sys.argv.index("-c")+1]
    else:
        config = "config.ini"
    print("CONFIG: %s" % config)
    read_cfg(config)



    # read in variables
    if "--train" in sys.argv:
        traindatalbl = sys.argv[sys.argv.index("--train")+1].split("+")
    if "--test" in sys.argv:
        testdatalbl = sys.argv[sys.argv.index("--test")+1].split("+")
    if "--train-mode" in testdatalbl:
	    testdatalbl = False 
    else:
        testdatalbl = False
    if "--train-mode" in sys.argv:
        TRAIN_MODE = True
    else:
        TRAIN_MODE = False
    # starting epoch
    epoch = 1
    if "--epochs" in sys.argv:
        epoch = int(sys.argv[sys.argv.index("--epochs")+1])
        MAX_ITER = int(sys.argv[sys.argv.index("--epochs")+2])
#        global USE_PRETRAINED
        USE_PRETRAINED = True
#    global LEARNING_RATE
    # model specified?
    oodmodel = ""
    if "-m" in sys.argv:
        oodmodel = sys.argv[sys.argv.index("-m")+1]
    elif exp == "ood-adapt":
        print("ood-adapt experiment but no pretrained model specified!")
        sys.exit()


    # load data and setup model
    train_dataitems, valid_dataitems, multi_test_dataitems = load_data(traindatalbl, testdatalbl, EXT, TRAIN_MODE, DEBUG_MODE)
    network = model_init(OPTIM, TRAIN_MODE)
    criterions = define_loss()


    # learning rate decay
    if LR_schedule == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(network[-1], step_size=LR_size, gamma=LR_factor)     # optimizer
    elif LR_schedule == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(network[-1], 'min', patience=LR_size, factor=LR_factor)


    # check for previous models
    if USE_PRETRAINED:
        # check if model at MAX_ITER already exists
        print("Check if MAX_ITER model exists...")
        max_pretrained_model = "%s/epoch%03d*.pth.tar" % (SAVEDIR, MAX_ITER)
        if glob.glob(max_pretrained_model) != []:
            print("Model at MAX_ITER=%d already trained (%s)" % (MAX_ITER, glob.glob(max_pretrained_model)[0]))
            sys.exit()
        # check if any models exist
        print("Check if any models exist already...")
        pretrained_model = "%s/epoch*.tar" % (SAVEDIR)
        models = sorted(glob.glob(pretrained_model))
        print(pretrained_model)
        print(models)
        if models == []:
            # no models exist, train from  scratch
            print("No models exist, train from epoch=1")
            epoch = 1
        else:
            print("Models exist...")
            network, epoch = load_model(models[-1], network, TRAIN_MODE)
            epoch += 1
        USE_PRETRAINED = False
 
    if epoch == 1 and exp == "ood-adapt":
        print("OOD pretrained model specified and no adapt models already saved...")
        network, pretrained_epoch = load_model(oodmodel, network, TRAIN_MODE)
        USE_PRETRAINED = False


    # train
    halving = 0
#    prev_pretrained_model = False
    prev_loss = 9999
    running_train_loss = []
    while epoch <= MAX_ITER:
        print("Epoch %d/%d" % (epoch, MAX_ITER))
        if LR_schedule == "StepLR":
            scheduler.step()
            print("LR scheduler: %s, LR=%f" % (LR_schedule,get_lr(network[-1])))


        # training
        [train_loss, ref, hyp, network] = train_model(traindatalbl, train_dataitems, network, criterions, TRAIN_MODE, DEBUG_MODE)
        running_train_loss.append(train_loss)
        print("---\nSCORING TRAIN-- Epoch[%d]: [%d] %.10f" % (epoch, len(train_dataitems.dataset), train_loss))
        PrintScore(ComputePerformance(ref, hyp, traindatalbl, "EMO"), epoch, len(train_dataitems.dataset), traindatalbl, "EMO")

        # valid and test
        if VALIDATION:
            [loss, ref, hyp] = train_model(traindatalbl, valid_dataitems, network, criterions, False, DEBUG_MODE)
            print("---\nSCORING VALID-- Epoch[%d]: [%d] %.10f" % (epoch, len(valid_dataitems.dataset), loss))
            PrintScore(ComputePerformance(ref, hyp, traindatalbl, "EMO"), epoch, len(valid_dataitems.dataset), traindatalbl, "EMO")
        if testdatalbl:
            for [datalbl, test_dataitems] in multi_test_dataitems:
                [loss, ref, hyp] = train_model(datalbl, test_dataitems, network, criterions, False, DEBUG_MODE)
                print("---\nSCORING TEST-- Epoch[%d]: [%d] %.10f" % (epoch, len(test_dataitems.dataset), loss))
                PrintScore(ComputePerformance(ref, hyp, datalbl, "EMO"), epoch, len(test_dataitems.dataset), datalbl, "EMO")

        # save intermediate models - must save before learning rate changed
        if SAVE_MODEL and epoch%SAVE_ITER == 0:
            save_model(traindatalbl, len(train_dataitems.dataset), epoch, network, train_loss)
        epoch += 1


        if LR_schedule == "ReduceLROnPlateau":
            curr_lr = get_lr(network[-1])
            print("LR scheduler: %s, LR=%.10f" % (LR_schedule,curr_lr))
            scheduler.step(train_loss)
            new_lr = get_lr(network[-1])
            if curr_lr != new_lr:
                print("LR has been updated: LR=%.10f" % (new_lr))
                if SELECT_BEST_MODEL:
                    print("Selecting best model at epoch...")
                    for p in range(1, LR_size+1):	# patience
                        print("e[%d] = %.10f" % (epoch-p, running_train_loss[-p]))
                    print(running_train_loss, running_train_loss[-LR_size:], min(running_train_loss[-LR_size:]), running_train_loss.index(min(running_train_loss[-LR_size:])), running_train_loss.index(min(running_train_loss[-LR_size:]))+1)
                    # learning rate has changed so choose previous best model to load
                    # consider only models in last LR_size/patience/stepsize epochs (??)
                    e = running_train_loss.index(min(running_train_loss[-LR_size:]))+1
                    print("Selecting best model at epoch: ", e)
                    network, epoch_loaded = load_model(glob.glob("%s/epoch%03d*.pth.tar" % (SAVEDIR, e))[0], network, TRAIN_MODE=True)

    else:
        if epoch >= MAX_ITER:
            print("---\nTRAINING STOPPED: Epoch [%d] >= MAX_ITER [%d]" % (epoch, MAX_ITER))
        else:
            print("---\nTRAINING STOPPED")



if __name__ == "__main__": main()
