#!/usr/bin/python
import sys
import os
from cmu_score_v2 import ComputePerformance
from cmu_score_v2 import PrintScore
from config_train_lstm_att_fbk import *
from attention_model import load_model
from attention_model import load_data
from attention_model import model_setup ### think
from attention_model import model_init
from attention_model import define_loss
from attention_model import train_model

### ----------------------------------------- variables
TRAIN_MODE = False


### ----------------------------------------- main
def main():
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
    if "--debug-mode" in sys.argv:
        DEBUG_MODE = True
    else:
        DEBUG_MODE = False

    # setup/init data and model
#    model_setup(EXT)
    train_dataitems, valid_dataitems, multi_test_dataitems = load_data(traindatalbl, testdatalbl, EXT, TRAIN_MODE, DEBUG_MODE)
    network = model_init("Adam", TRAIN_MODE)
    criterions = define_loss()

    # find models
    if modelfile:
        models = modelfile
    else:
        savedir = './models/%s/%s/' % ("+".join(traindatalbl), model_name)
        models = [savedir+m for m in os.listdir(savedir)]
    print("Models = ", models)

    # loop over trained models
    for pretrained_model in models:

        # check for pretrained model
        network, epoch = load_model(pretrained_model, network)

        # test
        for [datalbl, test_dataitems] in multi_test_dataitems:
            [loss, ref, hyp] = train_model(test_dataitems, network, criterions, False, DEBUG_MODE)
            print("---\nSCORING TEST-- Epoch[%d]: [%d] %.4f" % (epoch, len(test_dataitems.dataset), loss))
            PrintScore(ComputePerformance(ref, hyp), epoch, len(test_dataitems.dataset), [pretrained_model,datalbl])


if __name__ == "__main__": main()
