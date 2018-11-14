import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

eps=1e-12

def ComputePerformance(ref,hyp):
    # ref_local=ref.data.cpu().numpy()
    # hyp_local=hyp.data.cpu().numpy()
    ref_local=ref
    hyp_local=hyp
    no_of_examples=np.shape(ref_local)[0]
    no_of_classes=np.shape(ref_local)[1]

    # print(ref_local)
    # print(hyp_local)

    ref_binary=np.zeros(np.shape(ref_local))
    # ref_binary[ref_local >= 0.5]=1
    ref_binary[ref_local >= 0.1]=1
    hyp_binary=np.zeros(np.shape(hyp_local))
    # hyp_binary[hyp_local >= 0.5]=1
    hyp_binary[hyp_local >= 0.1]=1

    ref_class_binary=np.zeros((no_of_classes,no_of_examples))
    hyp_class_binary=np.zeros((no_of_classes,no_of_examples))
    score = dict()
    score['WA']=[[] for i in range(0,no_of_classes)]
    score['F1customised']=[[] for i in range(0,no_of_classes)]
    score['F1']=[[] for i in range(0,no_of_classes)]
    score['TP']=[[] for i in range(0,no_of_classes)]
    score['TN']=[[] for i in range(0,no_of_classes)]
    score['FN']=[[] for i in range(0,no_of_classes)]
    score['FP']=[[] for i in range(0,no_of_classes)]
    score['P']=[[] for i in range(0,no_of_classes)]
    score['N']=[[] for i in range(0,no_of_classes)]
    
    for i in range(0,no_of_classes):
      # ref_class_binary[i][ref_local[:,i] >= 0.5]=1
      # hyp_class_binary[i][hyp_local[:,i] >= 0.5]=1
      ref_class_binary[i][ref_local[:,i] >= 0.1]=1
      hyp_class_binary[i][hyp_local[:,i] >= 0.1]=1
      TP=np.sum(np.logical_and(ref_class_binary[i]==1,hyp_class_binary[i]==1))
      TN=np.sum(np.logical_and(ref_class_binary[i]==0,hyp_class_binary[i]==0))
      FP=np.sum(np.logical_and(ref_class_binary[i]==0,hyp_class_binary[i]==1))
      FN=np.sum(np.logical_and(ref_class_binary[i]==1,hyp_class_binary[i]==0))
      P=TP+FN
      N=TN+FP
      score['TP'][i] = TP
      score['TN'][i] = TN
      score['FP'][i] = FP
      score['FN'][i] = FN
      score['P'][i] = P
      score['N'][i] = N
      score['WA'][i] = (TP*N/max(P,eps)+TN)/(2*max(N,eps))
      score['F1customised'][i] =(2*TP)/max(2*TP+FP+FN,eps)
      score['F1'][i] = f1_score(ref_class_binary[i],hyp_class_binary[i])
      # print('customised F1')
      # print(score['F1customised'][i])
      # print('default F1')
      # print(score['F1'][i])
      # print('WA')
      # print(score['WA'][i])
    # print('overall F1')
    score['overallF1'] = f1_score(np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0],np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0])
   
 

    # ref_flat=np.reshape(ref_local,(1,np.prod(np.shape(ref_local))))
    # hyp_flat=np.reshape(hyp_local,(1,np.prod(np.shape(hyp_local))))
   
    # print(ref_binary)
    # print(hyp_binary) 
    # print(accuracy_score(np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0],np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0]))
    score['binaryaccuracy'] = accuracy_score(np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0],np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0])
   
    # print(ref_flat)
    # print(hyp_flat)
    # score['MSE'] = ((ref_flat - hyp_flat) ** 2).mean(axis=0)
    # score['MAE'] = (np.abs(ref_flat - hyp_flat)).mean(axis=0)
    score['MSE_class'] = ((ref_local - hyp_local) ** 2).mean(axis=0)
    score['MAE_class'] = (np.abs(ref_local - hyp_local)).mean(axis=0)
    score['MSE'] = score['MSE_class'].sum(axis=0)/len(score['MSE_class'])
    score['MAE'] = score['MAE_class'].sum(axis=0)/len(score['MAE_class'])


    # print('Accuracy:', accuracy_score(y_true, y_pred))
    # print('F1 score:', f1_score(y_true, y_pred,average = 'weighted'))
    # print('Recall:', recall_score(y_true, y_pred,average ='weighted'))
    # print('Precision:', precision_score(y_true, y_pred,average = 'weighted'))
    return score

def PrintScore(score, epoch, K, lbl):
    print('DATASET -- %s' % lbl)
    print('Scoring -- Epoch [%d], Sample [%d], Binary accuracy %.4f' % (epoch, K, score['binaryaccuracy']))
    print('Scoring -- Epoch [%d], Sample [%d], MSE %.4f' % (epoch, K, score['MSE']))
    print('Scoring -- Epoch [%d], Sample [%d], MSE_class %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch, K, score['MSE_class'][0], score['MSE_class'][1], score['MSE_class'][2], score['MSE_class'][3], score['MSE_class'][4], score['MSE_class'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], MAE %.4f' % (epoch, K, score['MAE']))
    print('Scoring -- Epoch [%d], Sample [%d], MAE_class %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch, K, score['MAE_class'][0], score['MAE_class'][1], score['MAE_class'][2], score['MAE_class'][3], score['MAE_class'][4], score['MAE_class'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], TP %d %d %d %d %d %d' % (epoch, K, score['TP'][0], score['TP'][1], score['TP'][2], score['TP'][3], score['TP'][4], score['TP'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], TN %d %d %d %d %d %d' % (epoch, K, score['TN'][0], score['TN'][1], score['TN'][2], score['TN'][3], score['TN'][4], score['TN'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], FP %d %d %d %d %d %d' % (epoch, K, score['FP'][0], score['FP'][1], score['FP'][2], score['FP'][3], score['FP'][4], score['FP'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], FN %d %d %d %d %d %d' % (epoch, K, score['FN'][0], score['FN'][1], score['FN'][2], score['FN'][3], score['FN'][4], score['FN'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], P %d %d %d %d %d %d' % (epoch, K, score['P'][0], score['P'][1], score['P'][2], score['P'][3], score['P'][4], score['P'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], N %d %d %d %d %d %d' % (epoch, K, score['N'][0], score['N'][1], score['N'][2], score['N'][3], score['N'][4], score['N'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], WA %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch, K, score['WA'][0], score['WA'][1], score['WA'][2], score['WA'][3], score['WA'][4], score['WA'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], F1customised %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch, K, score['F1customised'][0], score['F1customised'][1], score['F1customised'][2], score['F1customised'][3], score['F1customised'][4], score['F1customised'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], F1 %.4f %.4f %.4f %.4f %.4f %.4f' % (epoch, K, score['F1'][0], score['F1'][1], score['F1'][2], score['F1'][3], score['F1'][4], score['F1'][5]))
    print('Scoring -- Epoch [%d], Sample [%d], Overall F1 %.4f' % (epoch, K, score['overallF1']))

def PrintScoreEpochs(scores):
   print("||<|2> '''Epoch''' ||<|2> '''Metric''' ||<|2> '''Overall''' ||<-7> '''Classes''' ||")
   print("|| '''happiness''' || '''sadness''' || '''anger''' || '''surprise'' || '''disgust''' || '''fear''' ||")
   for epoch in range(len(scores)):
      score = scores[epoch]
      print("||<|12> [%d] || Binary Accuracy || %.4f||" % (epoch+1, score['binaryaccuracy']))
      print("|| MSE || %.4f || %.4f || %.4f || %.4f || %.4f || %.4f || %.4f ||" % (score['MSE'], score['MSE_class'][0], score['MSE_class'][1], score['MSE_class'][2], score['MSE_class'][3], score['MSE_class'][4], score['MSE_class'][5]))
      print("|| MAE || %.4f || %.4f || %.4f || %.4f || %.4f || %.4f || %.4f ||" % (score['MAE'], score['MAE_class'][0], score['MAE_class'][1], score['MAE_class'][2], score['MAE_class'][3], score['MAE_class'][4], score['MAE_class'][5]))
      print("|| TP || || %d || %d || %d || %d || %d || %d ||" % (score['TP'][0], score['TP'][1], score['TP'][2], score['TP'][3], score['TP'][4], score['TP'][5]))
      print("|| TN || || %d || %d || %d || %d || %d || %d ||" % (score['TN'][0], score['TN'][1], score['TN'][2], score['TN'][3], score['TN'][4], score['TN'][5]))
      print("|| FP || || %d || %d || %d || %d || %d || %d ||" % (score['FP'][0], score['FP'][1], score['FP'][2], score['FP'][3], score['FP'][4], score['FP'][5]))
      print("|| FN || || %d || %d || %d || %d || %d || %d ||" % (score['FN'][0], score['FN'][1], score['FN'][2], score['FN'][3], score['FN'][4], score['FN'][5]))
      print("|| P || || %d || %d || %d || %d || %d || %d ||" % (score['P'][0], score['P'][1], score['P'][2], score['P'][3], score['P'][4], score['P'][5]))
      print("|| N || || %d || %d || %d || %d || %d || %d ||" % (score['N'][0], score['N'][1], score['N'][2], score['N'][3], score['N'][4], score['N'][5]))
      print("|| WA || || %.4f || %.4f || %.4f || %.4f || %.4f || %.4f ||" % (score['WA'][0], score['WA'][1], score['WA'][2], score['WA'][3], score['WA'][4], score['WA'][5]))
      print("|| F1customised || || %.4f || %.4f || %.4f || %.4f || %.4f || %.4f ||" % (score['F1customised'][0], score['F1customised'][1], score['F1customised'][2], score['F1customised'][3], score['F1customised'][4], score['F1customised'][5]))
      print("|| F1 || %.4f || %.4f || %.4f || %.4f || %.4f || %.4f || %.4f ||" % (score['overallF1'], score['F1'][0], score['F1'][1], score['F1'][2], score['F1'][3], score['F1'][4], score['F1'][5]))
      print("|| ||")
