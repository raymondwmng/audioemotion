from sentiment_lstm import SentimentLstm
import torch
import torch.nn as nn
from torch.autograd import Variable
from gv_data import gv_data


use_CUDA = True


# -- about data ---
train_gv = 'CMU-MOSEI/train_glove_vectors.npy'
train_ref = 'CMU-MOSEI/train_average_emotion_labels.npy'

# -- about model ---
lstm_input_size=300
lstm_hidden_size=512
lstm_num_layers=2
lstm_outlayer_size=1024
num_emotions=7

# -- load data
trainset = gv_data(train_gv,train_ref) 
dataitems = torch.utils.data.DataLoader(dataset=trainset,batch_size=1,shuffle=True,num_workers=2)

# -- model initialisation ---
sentiment_lstm = SentimentLstm(lstm_input_size,lstm_hidden_size,lstm_num_layers,lstm_outlayer_size,num_emotions)
if use_CUDA:
    sentiment_lstm = sentiment_lstm.cuda()

criterion=nn.MSELoss()
params=list(sentiment_lstm.parameters())
optimizer=torch.optim.Adam(params,lr=0.0001)

epoch=0
while epoch<3:
    accumulated_loss=0
    for i,(gv,ref) in enumerate(dataitems):
        if use_CUDA:
            gv = Variable(gv.float()).cuda()
            ref = Variable(ref.float()).cuda()
        else:
            gv = Variable(gv.float())
            ref = Variable(ref.float())
        hyp = sentiment_lstm(gv)
        loss = criterion(hyp, ref)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        sentiment_lstm.zero_grad()

        accumulated_loss = accumulated_loss + loss.data[0]
        if (i+1)%500==0:
            print('Epoch [%d], Sample [%d], Average Loss: %.4f' % (epoch+1, i+1, accumulated_loss/(i+1)))
    epoch=epoch+1

# save the trained model after 4 epochs
torch.save({
    'epoch': epoch,
    'loss': accumulated_loss,
    'sentiment_lstm': sentiment_lstm.state_dict(),
    'optimizer': optimizer.state_dict()
},'./sentiment_lstm.pth.tar')
