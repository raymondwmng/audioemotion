#!/usr/bin/python
from attention_network import LstmNet
from attention_network import Attention
from attention_network import Predictor
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
from cmu_score_v2 import ComputePerformance
from cmu_score_v2 import ComputeAccuracy
from cmu_score_v2 import PrintScore
from cmu_score_v2 import PrintScoreWiki
from datasets import database
from config import *


### ----------------------------------------- Convert to numpy
def to_npy(x):
    # convert tensor to numpy format
    return x.data.cpu().numpy()


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
            validset = fea_data_npy(valid_fea, valid_ref,  BATCHSIZE, PADDING)
            valid_dataitems=torch.utils.data.DataLoader(dataset=trainset,batch_size=1,shuffle=False,num_workers=2)
        trainset = fea_data_npy(train_fea, train_ref, BATCHSIZE, PADDING)
        train_dataitems=torch.utils.data.DataLoader(dataset=trainset,batch_size=BATCHSIZE,shuffle=True,num_workers=2)
    else:
        train_dataitems, valid_dataitems = [], []
    # load (multiple) test sets separately
    multi_test_dataitems = []
    for datalbl in testdatalbl:
        test_fea = database[datalbl][EXT]['test']['fea']
        test_ref = database[datalbl][EXT]['test']['ref']
        testset = fea_test_data_npy(test_fea, test_ref, datalbl)
        test_dataitems = torch.utils.data.DataLoader(dataset=testset,batch_size=1,shuffle=False,num_workers=2)
        multi_test_dataitems.append([datalbl, test_dataitems])
    # reduce datasets if debugging code
    if DEBUG_MODE:
        l = 10
        if TRAIN_MODE:
            trainset.fea, trainset.ref = trainset.fea[:l], trainset.ref[:l]
#            if VALIDATION:
#                validset.fea, validset.ref = validset.fea[:l], validset.ref[:l]
        testset.fea, testset.ref = testset.fea[:l], testset.ref[:l]
    return train_dataitems, valid_dataitems, multi_test_dataitems


### ----------------------------------------- save model
def save_model(state, is_final):
    # save intermediate models
    filename = "%s/epoch%03d-samples%d-loss%.4f.pth.tar" % (SAVEDIR, state['epoch'], state['samples'], state['loss'])
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


### ----------------------------------------- load model
def load_model(pretrained_model, network, TRAIN_MODE):
    [encoder, attention, predictor, optimizer] = network
#    checkpoint = torch.load(pretrained_model, map_location=lambda storage, location: storage)
    checkpoint = torch.load(pretrained_model)
    encoder.load_state_dict(checkpoint['encoder'])
    attention.load_state_dict(checkpoint['attention'])
    predictor.load_state_dict(checkpoint['predictor'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    accumulated_loss = checkpoint['loss']
    data = checkpoint['data']
    samples = checkpoint['samples']
    print("Loaded model (%s[%d]) at epoch (%d) with loss (%.4f)" % (pretrained_model, samples, epoch, accumulated_loss))
    if TRAIN_MODE:
        encoder.train()
        attention.train()
        predictor.train()
    else:
#        encoder.train(False) # == encoder.eval() set for testing mode
#        attention.train(False)
#        predictor.train(False)
        encoder.eval()
        attention.eval()
        predictor.eval()
#        for var_name in encoder.state_dict():
#            print(var_name, "\t", encoder.state_dict()[var_name])
#        for var_name in attention.state_dict():
#            print(var_name, "\t", attention.state_dict()[var_name])
#        for var_name in predictor.state_dict():
#            print(var_name, "\t", predictor.state_dict()[var_name])
#        for var_name in optimizer.state_dict():
#            print(var_name, "\t", optimizer.state_dict()[var_name])
    return [encoder, attention, predictor, optimizer], epoch


### ----------------------------------------- model initialisation
def model_init(optim, TRAIN_MODE):
    encoder = LstmNet(input_size, hidden_size, num_layers, outlayer_size, num_emotions)
    attention = Attention(num_emotions, dan_hidden_size, att_hidden_size)
    predictor = Predictor(num_emotions, dan_hidden_size)
    if USE_CUDA:
        encoder = encoder.cuda()
        attention = attention.cuda()
        predictor = predictor.cuda()
    if TRAIN_MODE: 
        # sets the mode (useful for batchnorm, dropout)
        encoder.train()
        attention.train()
        predictor.train()
    else:
#        encoder.train(False) # == encoder.eval() set for testing mode
#        attention.train(False)
#        predictor.train(False)
        encoder.eval()
        attention.eval()
        predictor.eval()
    params = list(encoder.parameters()) + list(attention.parameters()) + list(predictor.parameters())
    print('Parameters:encoder,attention,predictor = %d,%d,%d = %d' % (len(list(encoder.parameters())), len(list(attention.parameters())), len(list(predictor.parameters())), len(params)))
    if optim == "Adam":
        # different update rules - Adam: A Method for Stochastic Optimization
        optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    return [encoder, attention, predictor, optimizer]


### ----------------------------------------- loss function 
# computes a value that estimates how far away the output is from the target
def define_loss():
    criterion_c = nn.CrossEntropyLoss()
    criterion_r = nn.MSELoss()
    return [criterion_c, criterion_r]


### ----------------------------------------- train model
def train_model(dataitems, network, criterions, TRAIN_MODE, DEBUG_MODE):
    # train the model or test if TRAIN_MODE == False
    [encoder, attention, predictor, optimizer] = network
    [criterion_c, criterion_r] = criterions
    accumulated_loss = 0
    overall_hyp = np.zeros((0,num_emotions))
    overall_ref = np.zeros((0,num_emotions))
    for i,(fea,ref,tsk) in enumerate(dataitems):
        # send to cuda
        if USE_CUDA:
            fea = Variable(fea.float()).cuda()
            ref = Variable(ref.float()).cuda()
        else:
            fea = Variable(fea.float())
            ref = Variable(ref.float())
        # train
        hyp = encoder(fea)
        output = attention(hyp, dan_hidden_size, att_hidden_size, BATCHSIZE=1)
        outputs = predictor(output)
        #outputs = torch.clamp(outputs,0,3)
        # loss
        if tsk:
            loss = criterion_r(outputs, ref)
        elif not tsk:
            loss = criterion_c(outputs, torch.max(ref, 1)[1])
        accumulated_loss += loss.item()
        overall_hyp = np.concatenate((overall_hyp, to_npy(outputs)),axis=0)
        overall_ref = np.concatenate((overall_ref, to_npy(ref)),axis=0)
        if DEBUG_MODE and TRAIN_MODE:
            print("%4d ref:" % i, ref, ref.shape)
            print("%4d fea:" % i, fea, fea.shape)
            print("%4d hyp=encoder(fea):" % i, hyp, hyp.shape)
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
    # read in variables
    if "--train" in sys.argv:
        traindatalbl = sys.argv[sys.argv.index("--train")+1].split("+")
    if "--test" in sys.argv:
        testdatalbl = sys.argv[sys.argv.index("--test")+1].split("+")
    if "-e" in sys.argv:
        global EXT
        EXT = sys.argv[sys.argv.index("-e")+1]
    if "--train-mode" in sys.argv:
        TRAIN_MODE = True
    else:
        TRAIN_MODE = False
    if "-p" in sys.argv:
        PADDING = True
        BATCHSIZE = int(sys.argv[sys.argv.index("-p")+1])
        if BATCHSIZE == 1:
            PADDING = False
    # starting epoch
    epoch = 1
    if "--epochs" in sys.argv:
        epoch = int(sys.argv[sys.argv.index("--epochs")+1])
        MAX_ITER = int(sys.argv[sys.argv.index("--epochs")+2])
        global USE_PRETRAINED
        USE_PRETRAINED = True


    # setup/init data and model
    global SAVEDIR
    SAVEDIR = printConfig(EXT, traindatalbl, TRAIN_MODE)
    train_dataitems, valid_dataitems, multi_test_dataitems = load_data(traindatalbl, testdatalbl, EXT, TRAIN_MODE, DEBUG_MODE)
    network = model_init(OPTIM, TRAIN_MODE)
    criterions = define_loss()

    # train
    prev_pretrained_model = False
    while epoch <= MAX_ITER:
        print("Epoch %d/%d" % (epoch, MAX_ITER))

        # check for pretrained model
#        if USE_PRETRAINED:
#            # check if model at MAX_ITER already exists
#            max_pretrained_model = "%s/epoch%03d*.pth.tar" % (SAVEDIR, MAX_ITER)
#            if glob.glob(max_pretrained_model) != []:
#                print("Model at MAX_ITER=%d already trained (%s)" % (MAX_ITER, glob.glob(max_pretrained_model)[0]))
#                sys.exit()
#            # find start epoch trained
#            pretrained_model = "%s/epoch%03d*.pth.tar" % (SAVEDIR, epoch)
#            if glob.glob(pretrained_model) != []:
#                # model at current epoch exists
#                prev_pretrained_model = glob.glob(pretrained_model)[0]
#                print("Found model (%s) and checking next epoch" % prev_pretrained_model)
#                epoch += 1
#                continue
#            else:
#                # current epoch model does not exist, so load previous and train from there
#                if prev_pretrained_model:
#                    network, epoch = load_model(prev_pretrained_model, network, TRAIN_MODE)
#                    USE_PRETRAINED = False
#                else:
#                    print("Model at epoch=%d does not exist to load" % epoch)
#                    sys.exit()
#        else:
#            print("Not loading a model")                

        # training
        [train_loss, ref, hyp, network] = train_model(train_dataitems, network, criterions, TRAIN_MODE, DEBUG_MODE)
        print("---\nSCORING TRAIN-- Epoch[%d]: [%d] %.4f" % (epoch, len(train_dataitems.dataset), train_loss))
        PrintScore(ComputePerformance(ref, hyp), epoch, len(train_dataitems.dataset), traindatalbl)

        # valid and test
        if VALIDATION:
            [loss, ref, hyp] = train_model(valid_dataitems, network, criterions, False, DEBUG_MODE)
            print("---\nSCORING VALID-- Epoch[%d]: [%d] %.4f" % (epoch, len(valid_dataitems.dataset), loss))
            PrintScore(ComputePerformance(ref, hyp), epoch, len(valid_dataitems.dataset), traindatalbl)
        for [datalbl, test_dataitems] in multi_test_dataitems:
            [loss, ref, hyp] = train_model(test_dataitems, network, criterions, False, DEBUG_MODE)
            print("---\nSCORING TEST-- Epoch[%d]: [%d] %.4f" % (epoch, len(test_dataitems.dataset), loss))
            PrintScore(ComputePerformance(ref, hyp), epoch, len(test_dataitems.dataset), datalbl)

        # save intermediate models
        [encoder, attention, predictor, optimizer] = network
        if SAVE_MODEL: # and epoch%10 == 0:
            save_model({
                'data' : "+".join(traindatalbl),
                'epoch': epoch,
                'samples' : len(train_dataitems.dataset),
                'loss' : train_loss,
                'encoder' : encoder.state_dict(),
                'attention' : attention.state_dict(),
                'predictor' : predictor.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, False)

        epoch += 1

    else:
        if epoch >= MAX_ITER:
            print("---\nTRAINING STOPPED as Epoch [%d] >= MAX_ITER [%d]" % (epoch, MAX_ITER))
        else:
            print("---\nTRAINING STOPPED")



if __name__ == "__main__": main()
