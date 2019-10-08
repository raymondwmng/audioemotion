#!/usr/bin/python
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from htkmfc_python3 import HTKFeat_read
from data_loader import data_loader_npy
from attention_network_dat import LstmNet
from attention_network_dat import Attention
from attention_network_dat import Predictor
from attention_network_dat import DomainClassifier
from attention_network_dat import GradReverse


### ----------------------------------------- seed
seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


### ----------------------------------------- config
### ENV
WDIR = "./"
hcopy = "HCopy"
hcopy_cfg = "./config.hcopy.wav.wb.crbe"
### CUDA
USE_CUDA = True
### MODEL TRAINING
LEARNING_RATE = 0.0001
### ENCODER
input_size = 23
hidden_size = 512
num_layers = 2
outlayer_size = 1024
### ATTENTION
att_hidden_size = 128
dan_hidden_size = 1024
### PREDICTOR
num_emotions = 6
num_domains = 4
### DAT
lmbda = 0.007

### ----------------------------------------- Convert Tensor to numpy
def to_npy(x):
    # convert tensor to numpy format
    return x.data.cpu().numpy()

### ----------------------------------------- extract features
def extract_htk(audiofile):
    # create saving directory
    outfolder = "%s/out_%s/" % (WDIR, audiofile.split("/")[-1].replace(".wav",""))
    featurefile = "%s%s" % (outfolder, audiofile.split("/")[-1].replace(".wav",".fbk"))
    print("Saving outputs to: %s" % outfolder)
    out = os.system("mkdir -p %s" % outfolder)
    # run hcopy
    print("Extracting Log-Mel Filterbanks (23 dims) to: %s" % (featurefile))
    log = "%shcopy.log" % outfolder
    print("Saving HCopy log: %s" % log)
    cmd = "%s -A -D -V -T 7 -C %s %s %s > %s" % (hcopy, hcopy_cfg, audiofile, featurefile, log)
    out = os.system(cmd)
    return featurefile


### ----------------------------------------- convert htk features to npy
def htkfeat_npy(featurefile):
    print("Reading htk features...")
    htk = HTKFeat_read()    
    features = htk.getall(featurefile)
    featurefile_npy = "%s.npy" % featurefile
    print("...and saving to npy format: %s" % featurefile_npy)
    np.save(featurefile_npy, features)    
    return featurefile_npy


### ----------------------------------------- load data
def load_data(featurefile_npy):
    testdata = data_loader_npy(featurefile_npy)    ### do this here...?
    dataitems = torch.utils.data.DataLoader(dataset=testdata,batch_size=1,shuffle=False,num_workers=2)
    return dataitems


### ----------------------------------------- model initialisation
def model_init():
    print("Initialising empty model...")
    # model
    encoder = LstmNet(input_size, hidden_size, num_layers, outlayer_size, num_emotions)
    attention = Attention(num_emotions, dan_hidden_size, att_hidden_size)
    predictor = Predictor(num_emotions, dan_hidden_size)
    domainclassifier = DomainClassifier(num_domains, dan_hidden_size, lmbda)

    # use cuda
    if USE_CUDA:
        encoder = encoder.cuda()
        attention = attention.cuda()
        predictor = predictor.cuda()
        domainclassifier = domainclassifier.cuda()

    # number of parameters
    params = list(encoder.parameters()) + list(attention.parameters()) + list(predictor.parameters()) + list(domainclassifier.parameters())

    # different update rules - Adam: A Method for Stochastic Optimization
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    return [encoder, attention, predictor, domainclassifier, optimizer]


### ----------------------------------------- load model
def load_model(pretrained_model, network):
    print("Loading pretrained model...")
    [encoder, attention, predictor, domainclassifier, optimizer] = network

    if USE_CUDA:
        checkpoint = torch.load(pretrained_model)
    else:
        checkpoint = torch.load(pretrained_model, map_location=lambda storage, location: storage)
#    data = checkpoint['data']
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

    # set to testing mode
    encoder.eval()
    attention.eval()
    predictor.eval()
    domainclassifier.eval()

    return [encoder, attention, predictor, domainclassifier, optimizer], epoch


### ----------------------------------------- test model
def test_model(dataitems, network):
    print("Running test...")
    # network structure
    [encoder, attention, predictor, domainclassifier, optimizer] = network
    # define loss
    criterion = nn.CrossEntropyLoss()

    print(len(dataitems))
    for i, fea in enumerate(dataitems):
        # send to cuda
        if USE_CUDA:
            fea = Variable(fea.float()).cuda()
        else:
            fea = Variable(fea.float())

        # encoder
        hyp = encoder(fea)
        # attention
        output = attention(hyp, dan_hidden_size, att_hidden_size, BATCHSIZE=1)
        # emotion
        emotion_outputs = predictor(output)
        # domain
        domain_outputs = domainclassifier(output)

    return list(to_npy(emotion_outputs.squeeze(0)))


### ----------------------------------------- print score
def print_score(emotion_outputs):
    classes = ["happiness","sad","anger","surprise","disgust","fear"]
    ind_max = emotion_outputs.index(max(emotion_outputs))
    print("Utterance has emotion: %s" % classes[ind_max])


### ----------------------------------------- main
def main():
    # read in audio file
    if "-a" in sys.argv:
        audiofile = sys.argv[sys.argv.index("-a")+1]
        print("Audio file: ", audiofile)
    else:
        print("Error: audio file required (-a)")
        sys.exit()

    # read in pretrained model
    if "-m" in sys.argv:
        modelfile = sys.argv[sys.argv.index("-m")+1]
        print("Model: %s" % modelfile)
    else:
        print("Error: pretrained model required (-m)")
        sys.exit()

    # extract features (wav to LMFB23)
    featurefile = extract_htk(audiofile)

    # convert htk to npy
    featurefile_npy = htkfeat_npy(featurefile)

    # load features
    dataitems = load_data(featurefile_npy)

    # init model and criterion
    network = model_init()

    # load pretrained model
    network, epoch = load_model(modelfile, network)

    # test audio
    outputs = test_model(dataitems, network)

    # print prediction
    print_score(outputs)


if __name__ == "__main__": main()
