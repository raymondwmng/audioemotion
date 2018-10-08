from sentiment_lstm import SentimentLstm
import torch
import numpy as np
from torch.autograd import Variable
from gv_data import gv_test_data

use_CUDA = False

# -- about data ---
test_gv = './test_glove_vectors.npy'

# -- about model ---
lstm_input_size=300
lstm_hidden_size=512
lstm_num_layers=2
lstm_outlayer_size=1024
num_emotions=7

# -- load data
testset = gv_test_data(test_gv) 
dataitems = torch.utils.data.DataLoader(dataset=testset,batch_size=1,shuffle=False,num_workers=2)

# -- model initialisation ---
sentiment_lstm = SentimentLstm(lstm_input_size,lstm_hidden_size,lstm_num_layers,lstm_outlayer_size,num_emotions)
if use_CUDA:
    sentiment_lstm = sentiment_lstm.cuda()


# load the trained model 
checkpoint=torch.load('./sentiment_lstm.pth.tar')
sentiment_lstm.load_state_dict(checkpoint['sentiment_lstm'])

allhyp=[]
for i,gv in enumerate(dataitems):
    if use_CUDA:
        gv = Variable(gv.float()).cuda()
    else:
        gv = Variable(gv.float())
    hyp = sentiment_lstm(gv)
    allhyp.append(hyp)

np.save('./predicted_sentiments.npy',allhyp)
