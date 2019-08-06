import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score

eps=1e-12	# to prevent dividing by zero
LIMIT=0.1	# any value above this is set to 1

classes = ['happiness','sadness','anger','surprise','disgust','fear']

datalbls = {}
datalbls["MOSEI_acl2018_neNA"] = 2009
datalbls["ent05p2_t34v5t5_shoutclipped"] = 150
datalbls["ravdess_t17v2t5_all1neNAcaNA"] = 240
datalbls["iemocap_t1234t5_neNAfrNAexNAotNA"] = 586
datalbls["iemocap_t1234t5_haex1sa1an1ne0"] = 1241
datalbls["iemocap_t1234t5_haex1sa1an1ne1"] = 1241


def ComputePerformance(ref, hyp, datalbl, TASK):
    # ref_local=ref.data.cpu().numpy()
    # hyp_local=hyp.data.cpu().numpy()
    ref_local=ref
    hyp_local=hyp
    no_of_examples=np.shape(ref_local)[0]
    no_of_classes=np.shape(ref_local)[1]


    ref_binary=np.zeros(np.shape(ref_local))
    hyp_binary=np.zeros(np.shape(hyp_local))

    # if "+" in datalbl: # as in testing all data together ... need to know if training or testing score
    # ... max/limit for correct data
    # ...

    print("DATALBL: %s\tTASK: %s" % (datalbl, TASK))
    if TASK == "DOM": #or datalbl in ["ent05p2_t34v5t5_shoutclipped","ravdess_t17v2t5_all1neNAcaNA","iemocap_t1234t5_neNAfrNAexNAotNA"]:
        print("-take the max, only one class allowed in ref and hyp")
        # take the max value(s) as 1, not thresholding
        ref_binary[np.arange(len(ref_local)), ref_local.argmax(1)] = 1
        hyp_binary[np.arange(len(hyp_local)), hyp_local.argmax(1)] = 1
    else: # TASK='EMO'
#        if datalbl in ["iemocap_t1234t5_haex1sa1an1ne0"]:
#            # single max value above 0 allowed (to account for neutral emotion)
#            print("-take the max, only one class allowed in ref and hyp, unless max below zero (for detecting NA)")
#            # set max value to 1
#            ref_binary[np.arange(len(ref_local)), ref_local.argmax(1)] = 1
#            hyp_binary[np.arange(len(hyp_local)), hyp_local.argmax(1)] = 1
#            # set all values below zero (inc max) to 0
#            ref_binary[ref_local <= 0]=0
#            hyp_binary[hyp_local <= 0]=0
        if datalbl in ["iemocap_t1234t5_haex1sa1an1ne1"]:
            hyp_local = hyp_local[:,:4]
            hyp_binary=np.zeros(np.shape(hyp_local))
            # single max value above 0 allowed (to account for neutral emotion)
            print("-take the max, only one class allowed in ref and hyp, unless max below zero (for detecting NA)")
            # set max value to 1
#            print(ref_local.shape, ref_binary.shape, hyp_local.shape, hyp_binary.shape)
            ref_binary[np.arange(len(ref_local)), ref_local.argmax(1)] = 1
            hyp_binary[np.arange(len(hyp_local)), hyp_local.argmax(1)] = 1
            # set all values below zero (inc max) to 0
            ref_binary[ref_local <= 0]=0
            hyp_binary[hyp_local <= 0]=0
            # if all values 0 then neutral, set last to 1
#            print(ref_local.shape, ref_binary.shape, hyp_local.shape, hyp_binary.shape)
            hyp_binary = hyp_binary[:,:4]
            for h in range(len(hyp_binary)):
                if sum(hyp_binary[h]) == 0:
#                    print(h, h.shape)
                    hyp_binary[h][-1] = 1
            print(ref_binary[:10], hyp_binary[:10])
        elif datalbl in ["MOSEI_acl2018_neNA"]:
            # multiple emotions allowed
            print("-set values about limit (0.1) to 1 in ref and hyp")
            ref_binary[ref_local >= LIMIT]=1
            hyp_binary[hyp_local >= LIMIT]=1
        else:
            print("-take the max, only one class allowed in ref and hyp")
            # take the max value(s) as 1, not thresholding
            ref_binary[np.arange(len(ref_local)), ref_local.argmax(1)] = 1
            hyp_binary[np.arange(len(hyp_local)), hyp_local.argmax(1)] = 1


    print("REFSUM=", sum(ref_local), sum(ref_binary), sum(sum(ref_binary)))
    print("HYPSUM=", sum(hyp_local), sum(hyp_binary), sum(sum(hyp_binary)))


    ref_class_binary=np.zeros((no_of_classes,no_of_examples))
    hyp_class_binary=np.zeros((no_of_classes,no_of_examples))
    score = dict()
    score['UA']=[[] for i in range(0,no_of_classes)]	# unweighted accuracy
    score['WA']=[[] for i in range(0,no_of_classes)]	# weighted accuracy
    score['F1customised']=[[] for i in range(0,no_of_classes)]
    score['Recall']=[[] for i in range(0,no_of_classes)]
    score['Precision']=[[] for i in range(0,no_of_classes)]
    score['F1']=[[] for i in range(0,no_of_classes)]
    score['TP']=[[] for i in range(0,no_of_classes)]
    score['TN']=[[] for i in range(0,no_of_classes)]
    score['FN']=[[] for i in range(0,no_of_classes)]
    score['FP']=[[] for i in range(0,no_of_classes)]
    score['P']=[[] for i in range(0,no_of_classes)]
    score['N']=[[] for i in range(0,no_of_classes)]
    


    #sys.exit()

    for i in range(0,no_of_classes):
      ref_class_binary[i][ref_binary[:,i] == 1]=1
      hyp_class_binary[i][hyp_binary[:,i] == 1]=1
#      print("Class: %d" % i)
#      print(ref_class_binary[i], sum(ref_class_binary[i]), hyp_class_binary[i], sum(hyp_class_binary[i]))
      TP=np.sum(np.logical_and(ref_class_binary[i]==1,hyp_class_binary[i]==1))
      TN=np.sum(np.logical_and(ref_class_binary[i]==0,hyp_class_binary[i]==0))
      FP=np.sum(np.logical_and(ref_class_binary[i]==0,hyp_class_binary[i]==1))
      FN=np.sum(np.logical_and(ref_class_binary[i]==1,hyp_class_binary[i]==0))
      P=TP+FN
      N=TN+FP
#      print(TP, TN, FP, FN, P, N)
      score['TP'][i] = TP
      score['TN'][i] = TN
      score['FP'][i] = FP
      score['FN'][i] = FN
      score['P'][i] = P
      score['N'][i] = N
      score['UA'][i] = (TP+TN)/max(TP+TN+FP+FN,eps)
      score['WA'][i] = 0.5*((TP/max(P,eps)) + (TN/max(N,eps)))
      score['Recall'][i] = TP / max(TP+FN,eps)
      score['Precision'][i] = TP / max(TP+FP,eps)
      score['F1customised'][i] =(2*TP)/max(2*TP+FP+FN,eps)
      score['F1'][i] = f1_score(ref_class_binary[i],hyp_class_binary[i])
      # print('customised F1')
      # print(score['F1customised'][i])
      # print('default F1')
      # print(score['F1'][i])
      # print('WA')
      # print(score['WA'][i])
    # print('overall F1')
    score['overallUA'] = sum(score['UA'])/no_of_classes
    score['overallWA'] = sum(score['WA'])/no_of_classes
    score['overallPrecision'] = precision_score(np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0],np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0])
    score['overallRecall'] = recall_score(np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0],np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0])
    score['overallF1'] = f1_score(np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0],np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0])
    score['overallF1customised'] = (2*score['overallPrecision']*score['overallRecall'])/max(score['overallRecall']+score['overallPrecision'],eps)   
 

    # ref_flat=np.reshape(ref_local,(1,np.prod(np.shape(ref_local))))
    # hyp_flat=np.reshape(hyp_local,(1,np.prod(np.shape(hyp_local))))
   
    # print(ref_binary)
    # print(hyp_binary) 
    # print(accuracy_score(np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0],np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0]))
    score['binaryaccuracy'] = accuracy_score(np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0],np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0])
    score['balancedaccuracy'] = balanced_accuracy_score(np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0],np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0])

#    print( ref_binary)
#    print( (1,np.prod(np.shape(ref_binary))))
#    print( np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary)))))
#    print( np.reshape(ref_binary, (1,np.prod(np.shape(ref_binary))))[0])
#    print( hyp_binary)
#    print( (1,np.prod(np.shape(hyp_binary))))
#    print( np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary)))))
#    print( np.reshape(hyp_binary, (1,np.prod(np.shape(hyp_binary))))[0])
   
    # print(ref_flat)
    # print(hyp_flat)
    # score['MSE'] = ((ref_flat - hyp_flat) ** 2).mean(axis=0)
    # score['MAE'] = (np.abs(ref_flat - hyp_flat)).mean(axis=0)
    score['MSE_class'] = ((ref_local - hyp_local) ** 2).mean(axis=0)
    score['MAE_class'] = (np.abs(ref_local - hyp_local)).mean(axis=0)

    score['SumMAE'] = score['MAE_class'].sum(axis=0)#/len(score['MAE_class'])
    score['SumMSE'] = score['MSE_class'].sum(axis=0)#/len(score['MSE_class'])
    score['AvgMAE'] = score['MAE_class'].sum(axis=0)/len(score['MAE_class'])
    score['AvgMSE'] = score['MSE_class'].sum(axis=0)/len(score['MSE_class'])

    # print('Accuracy:', accuracy_score(y_true, y_pred))
    # print('F1 score:', f1_score(y_true, y_pred,average = 'weighted'))
    # print('Recall:', recall_score(y_true, y_pred,average ='weighted'))
    # print('Precision:', precision_score(y_true, y_pred,average = 'weighted'))
    return score



def PrintScore(score, epoch, K, lbl, TASK="EMO"):
    mseclass, maeclass, tp, tn, fp, fn, p, n, ua, wa, precis, recall, f1cust, f1 = "","","","","", "","","","","", "","","",""
    for i in range(len(score['MSE_class'])):
         mseclass += "%.4f " % score['MSE_class'][i]
         maeclass += "%.4f " % score['MAE_class'][i]
         tp += "%d " % score['TP'][i]
         tn += "%d " % score['TN'][i]
         fp += "%d " % score['FP'][i]
         fn += "%d " % score['FN'][i]
         p += "%d " % score['P'][i]
         n += "%d " % score['N'][i]
         ua += "%.4f " % score['UA'][i]
         wa += "%.4f " % score['WA'][i]
         precis += "%.4f " % score['Precision'][i]
         recall += "%.4f " % score['Recall'][i]
         f1cust += "%.4f " % score['F1customised'][i]
         f1 += "%.4f " % score['F1'][i]
    classification = TASK
    print('DATASET -- %s' % lbl)
#    print('Scoring -- Epoch [%d], Sample [%d], Binary accuracy %.4f' % (epoch, K, score['binaryaccuracy']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Sum MSE %.4f' % (classification, epoch, K, score['SumMSE']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Avg MSE %.4f' % (classification, epoch, K, score['AvgMSE']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], MSE_class %s' % (classification, epoch, K, mseclass))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Sum MAE %.4f' % (classification, epoch, K, score['SumMAE']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Avg MAE %.4f' % (classification, epoch, K, score['AvgMAE']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], MAE_class %s' % (classification, epoch, K, maeclass))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], TP %s' % (classification, epoch, K, tp))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], TN %s' % (classification, epoch, K, tn))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], FP %s' % (classification, epoch, K, fp))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], FN %s' % (classification, epoch, K, fn))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], P  %s' % (classification, epoch, K, p))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], N  %s' % (classification, epoch, K, n))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], UA %s' % (classification, epoch, K, ua))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], WA %s' % (classification, epoch, K, wa))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Overall UA     %.4f (binaryaccur %.4f)' % (classification, epoch, K, score['overallUA'], score['binaryaccuracy']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Overall WA     %.4f (balancedacc %.4f)' % (classification, epoch, K, score['overallWA'], score['balancedaccuracy']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Precis %s' % (classification, epoch, K, precis))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Recall %s' % (classification, epoch, K, recall))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], F1cust %s' % (classification, epoch, K, f1cust))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], F1     %s' % (classification, epoch, K, f1))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Overall Precis %.4f' % (classification, epoch, K, score['overallPrecision']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Overall Recall %.4f' % (classification, epoch, K, score['overallRecall']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Overall F1cust %.4f' % (classification, epoch, K, score['overallF1customised']))
    print('%sScoringv3 -- Epoch [%d], Sample [%d], Overall F1     %.4f' % (classification, epoch, K, score['overallF1']))


# remove these from attention_model*.py
def PrintScoreEmo(score, epoch, K, lbl):
    PrintScore(score, epoch, K, lbl, TASK="EMO")

def PrintScoreDom(score, epoch, K, lbl):
    PrintScore(score, epoch, K, lbl, TASK="DOM")

