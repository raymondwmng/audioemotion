#!/usr/bin/python
from attention_network_dat import LstmNet
from attention_network_dat import Attention
from attention_network_dat import Predictor
from attention_network_dat import DomainClassifier
from attention_network_dat import GradReverse
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from fea_data_dat import fea_data_npy
from fea_data_dat import fea_test_data_npy
import sys
import os
import shutil
import glob
import numpy as np
from cmu_score_v2 import ComputePerformance
from cmu_score_v2 import PrintScoreEmo
from cmu_score_v2 import PrintScoreDom
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
    # dat
    global DAT
    DAT = cfg['DEFAULT'].getboolean('DAT')
    global c
    c = cfg['DEFAULT'].getfloat('c')
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
    global num_domains
    num_domains = cfg['DEFAULT'].getint('num_domains')
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
    exp = cfg['DEFAULT']['exp']
    global path
    path = cfg['DEFAULT']['path']
    global SAVEDIR
    SAVEDIR = WDIR+"/"+exp+"/"+path+"/models/%s/" % (model_name)
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
    # load train and valid sets and combine
    if TRAIN_MODE:
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
            validset = fea_data_npy(valid_fea, valid_ref,  BATCHSIZE, traindatalbl, MULTITASK)
            valid_dataitems=torch.utils.data.DataLoader(dataset=trainset,batch_size=1,shuffle=False,num_workers=2)
        trainset = fea_data_npy(train_fea, train_ref, BATCHSIZE, traindatalbl, MULTITASK)
        train_dataitems=torch.utils.data.DataLoader(dataset=trainset,batch_size=BATCHSIZE,shuffle=True,num_workers=2)
    else:
        train_dataitems, valid_dataitems = [], []

    # load (multiple) test sets separately
    multi_test_dataitems = []
    if testdatalbl:
        for datalbl in testdatalbl:
            test_fea = database[datalbl][EXT]['test']['fea']
            test_ref = database[datalbl][EXT]['test']['ref']
            testset = fea_test_data_npy(test_fea, test_ref, datalbl, traindatalbl, MULTITASK)
            test_dataitems = torch.utils.data.DataLoader(dataset=testset,batch_size=1,shuffle=False,num_workers=2)
            multi_test_dataitems.append([datalbl, test_dataitems])

    # reduce datasets if debugging code
    if DEBUG_MODE:
        l = 10
        if TRAIN_MODE:
            trainset.fea, trainset.ref = trainset.fea[:l], trainset.ref[:l]
        if testdatalbl:
            testset.fea, testset.ref = testset.fea[:l], testset.ref[:l]

    return train_dataitems, valid_dataitems, multi_test_dataitems


### ----------------------------------------- save model
def save_model(traindatalbl, samples, epoch, network, train_loss):
    [encoder, attention, predictor, domainclassifier, optimizer] = network
    # save intermediate models
    state = {
        'data' : "+".join(traindatalbl),
        'epoch': epoch,
        'samples' : samples,
        'loss' : train_loss,
        'LEARNING_RATE' : get_lr(network[-1]),
        'encoder' : encoder.state_dict(),
        'attention' : attention.state_dict(),
        'predictor' : predictor.state_dict(),
        'domainclassifier': domainclassifier.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }

    filename = "%s/epoch%03d-samples%d-loss%.10f-LR%.10f.pth.tar" % (SAVEDIR, state['epoch'], state['samples'], state['loss'], state['LEARNING_RATE'])
    print("Saving model: %s" % filename)
    torch.save(state, filename)


### ----------------------------------------- load model
def load_model(pretrained_model, network, TRAIN_MODE):
    [encoder, attention, predictor, domainclassifier, optimizer] = network

#    checkpoint = torch.load(pretrained_model, map_location=lambda storage, location: storage)
    checkpoint = torch.load(pretrained_model)
    data = checkpoint['data']
    epoch = checkpoint['epoch']
    samples = checkpoint['samples']
    accumulated_loss = checkpoint['loss']
    LEARNING_RATE = checkpoint['LEARNING_RATE']
    encoder.load_state_dict(checkpoint['encoder'])
    attention.load_state_dict(checkpoint['attention'])
    predictor.load_state_dict(checkpoint['predictor'])
    domainclassifier.load_state_dict(checkpoint['domainclassifier'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded model (%s[%d]) at epoch (%d) with loss (%.4f) and LEARNING_RATE (%f)" % (pretrained_model, samples, epoch, accumulated_loss, LEARNING_RATE))

    if TRAIN_MODE:
        encoder.train()
        attention.train()
        predictor.train()
        domainclassifier.train()
    else:
        encoder.eval()
        attention.eval()
        predictor.eval()
        domainclassifier.eval()
#        for var_name in encoder.state_dict():
#            print(var_name, "\t", encoder.state_dict()[var_name])
#        for var_name in attention.state_dict():
#            print(var_name, "\t", attention.state_dict()[var_name])
#        for var_name in predictor.state_dict():
#            print(var_name, "\t", predictor.state_dict()[var_name])
#        for var_name in optimizer.state_dict():
#            print(var_name, "\t", optimizer.state_dict()[var_name])

    return [encoder, attention, predictor, domainclassifier, optimizer], epoch


### ----------------------------------------- model initialisation
def model_init(optim, TRAIN_MODE, c):
    # model
    encoder = LstmNet(input_size, hidden_size, num_layers, outlayer_size, num_emotions)
    attention = Attention(num_emotions, dan_hidden_size, att_hidden_size)
    predictor = Predictor(num_emotions, dan_hidden_size)
    domainclassifier = DomainClassifier(num_domains, dan_hidden_size, c)

    # use cuda
    if USE_CUDA:
        encoder = encoder.cuda()
        attention = attention.cuda()
        predictor = predictor.cuda()
        domainclassifier = domainclassifier.cuda()

    # train or test mode
    if TRAIN_MODE: 
        # (useful for batchnorm, dropout)
        encoder.train()
        attention.train()
        predictor.train()
        domainclassifier.train()
    else:
        encoder.eval()
        attention.eval()
        predictor.eval()
        domainclassifier.eval()

    params = list(encoder.parameters()) + list(attention.parameters()) + list(predictor.parameters()) + list(domainclassifier.parameters())
    print('Parameters:encoder = %d' % len(list(encoder.parameters())))
    print('Parameters:attention = %d' % len(list(attention.parameters())))
    print('Parameters:predictor = %d' % len(list(predictor.parameters())))
    print('Parameters:domainclassifier = %d' % len(list(domainclassifier.parameters())))
    print('Parameters:total = %d' % len(params))

    # optimizer
    if optim == "Adam":
        # different update rules - Adam: A Method for Stochastic Optimization
        optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    return [encoder, attention, predictor, domainclassifier, optimizer]


### ----------------------------------------- loss function 
# computes a value that estimates how far away the output is from the target
def define_loss():
    criterion_c = nn.CrossEntropyLoss()
    criterion_r = nn.MSELoss()
    return [criterion_c, criterion_r]


### ----------------------------------------- train model
def train_model(dataitems, network, criterions, TRAIN_MODE, DEBUG_MODE):
    # train the model or test if TRAIN_MODE == False
    [encoder, attention, predictor, domainclassifier, optimizer] = network
    [criterion_c, criterion_r] = criterions # leftover from multitask
    accumulated_loss = 0
    overall_hyp = np.zeros((0,num_emotions))
    overall_ref = np.zeros((0,num_emotions))
    overall_domain_hyp = np.zeros((0,num_domains))
    overall_domain_ref = np.zeros((0,num_domains))
    for i,(fea,ref,tsk,domain_ref) in enumerate(dataitems):
        # send to cuda
        if USE_CUDA:
            fea = Variable(fea.float()).cuda()
            ref = Variable(ref.float()).cuda()
#            print(fea)
#            print(ref)
#            print(domain_ref)
            domain_ref = Variable(domain_ref.int()).cuda()
        else:
            fea = Variable(fea.float())
            ref = Variable(ref.float())
            domain_ref = Variable(domain_ref.int())

        # train
        hyp = encoder(fea)
        output = attention(hyp, dan_hidden_size, att_hidden_size, BATCHSIZE=1)

        # emotion
        outputs = predictor(output)
        emotion_loss = criterion_c(outputs, torch.max(ref, 1)[1])
        overall_hyp = np.concatenate((overall_hyp, to_npy(outputs)),axis=0)
        overall_ref = np.concatenate((overall_ref, to_npy(ref)),axis=0)

        # domain
        domain_outputs = domainclassifier(output)
        domain_loss =  criterion_c(domain_outputs, torch.max(domain_ref, 1)[1])
        overall_domain_hyp = np.concatenate((overall_domain_hyp, to_npy(domain_outputs)),axis=0)
        overall_domain_ref = np.concatenate((overall_domain_ref, to_npy(domain_ref)),axis=0)

        # combine losses
        loss = emotion_loss + domain_loss
        accumulated_loss += loss.item()

        if DEBUG_MODE and TRAIN_MODE:
            print("%4d ref:" % i, ref, ref.shape, torch.max(ref, 1)[1])
            print("%4d domain_ref:" % i, domain_ref, domain_ref.shape, torch.max(domain_ref, 1)[1])
            print("%4d fea:" % i, fea, fea.shape)
            print("%4d hyp=encoder(fea):" % i, hyp, hyp.shape)
            print("%4d output=attention(hyp):" % i, output, output.shape)
            print("%4d emotion_outputs=predictor(output):" % i, outputs, outputs.shape)
            print("%4d emotion_loss:" % i, emotion_loss, accumulated_loss, accumulated_loss/(i+1))
            print("%4d domain_outputs=domain_classifier(output):" % i, domain_outputs, domain_outputs.shape)
            print("%4d domain_loss:" % i, domain_loss, accumulated_loss, accumulated_loss/(i+1))

        if TRAIN_MODE:
            # backprop
            loss.backward()
            # update weights    
            optimizer.step()
            # zero the gradient
            encoder.zero_grad()
            attention.zero_grad()
            predictor.zero_grad()
            optimizer.zero_grad()
            domainclassifier.zero_grad()

    # overall loss
    accumulated_loss /= (i+1)

    if TRAIN_MODE:
        return [accumulated_loss, overall_ref, overall_domain_ref, overall_hyp, overall_domain_hyp, network]
    else:
        return [accumulated_loss, overall_ref, overall_domain_ref, overall_hyp, overall_domain_hyp]


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


#    # num_domains defined by input datasets
#    num_domains = len(traindatalbl)


    # load data and setup model
    train_dataitems, valid_dataitems, multi_test_dataitems = load_data(traindatalbl, testdatalbl, EXT, TRAIN_MODE, DEBUG_MODE)
    network = model_init(OPTIM, TRAIN_MODE, c)
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


    # train
    running_train_loss = []
    while epoch <= MAX_ITER:
        print("Epoch %d/%d" % (epoch, MAX_ITER))
        if LR_schedule == "StepLR":
            scheduler.step()
            print("LR scheduler: %s, LR=%f" % (LR_schedule,get_lr(network[-1])))

        # training
        [train_loss, ref, domain_ref, hyp, domain_hyp, network] = train_model(train_dataitems, network, criterions, TRAIN_MODE, DEBUG_MODE)
        running_train_loss.append(train_loss)
        print("---\nSCORING TRAIN-- Epoch[%d]: [%d] %.10f" % (epoch, len(train_dataitems.dataset), train_loss))
        PrintScoreEmo(ComputePerformance(ref, hyp), epoch, len(train_dataitems.dataset), traindatalbl)
        PrintScoreDom(ComputePerformance(domain_ref, domain_hyp), epoch, len(train_dataitems.dataset), traindatalbl)

        # valid and test
        if VALIDATION:
            [loss, ref, domain_ref, hyp, domain_hyp] = train_model(valid_dataitems, network, criterions, False, DEBUG_MODE)
            print("---\nSCORING VALID-- Epoch[%d]: [%d] %.10f" % (epoch, len(valid_dataitems.dataset), loss))
            PrintScoreEmo(ComputePerformance(ref, hyp), epoch, len(valid_dataitems.dataset), traindatalbl)
            PrintScoreDom(ComputePerformance(domain_ref, domain_hyp), epoch, len(valid_dataitems.dataset), traindatalbl)
        if testdatalbl:
            for [datalbl, test_dataitems, domain_hyp] in multi_test_dataitems:
                [loss, ref, domain_ref, hyp] = train_model(test_dataitems, network, criterions, False, DEBUG_MODE)
                print("---\nSCORING TEST-- Epoch[%d]: [%d] %.10f" % (epoch, len(test_dataitems.dataset), loss))
                PrintScoreEmo(ComputePerformance(ref, hyp), epoch, len(test_dataitems.dataset), datalbl)
                PrintScoreDom(ComputePerformance(domain_ref, domain_hyp), epoch, len(test_dataitems.dataset), datalbl)


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
                if SELECT_BEST_MODEL:	## untested
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
