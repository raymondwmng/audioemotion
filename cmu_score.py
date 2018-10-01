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
    score['MSE'] = score['MSE_class'].sum(axis=0)
    score['MAE'] = score['MAE_class'].sum(axis=0)


    # print('Accuracy:', accuracy_score(y_true, y_pred))
    # print('F1 score:', f1_score(y_true, y_pred,average = 'weighted'))
    # print('Recall:', recall_score(y_true, y_pred,average ='weighted'))
    # print('Precision:', precision_score(y_true, y_pred,average = 'weighted'))
    return score
