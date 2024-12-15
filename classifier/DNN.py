
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale,StandardScaler

from sklearn.metrics import average_precision_score, f1_score, zero_one_loss,hamming_loss,accuracy_score,multilabel_confusion_matrix,jaccard_score,recall_score,label_ranking_loss,coverage_error
from keras import layers
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Input,Dropout
from keras.layers import Flatten

data = pd.read_csv('po.csv')   #
label = pd.read_csv('polabel.csv')

data = pd.DataFrame(data)
label = pd.DataFrame(label)

num_class=4
X = np.array(data)
y = np.array(label)

n=np.shape(X)[1]
label_y=[]
a=0
c=0
for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j]==1:
                label_y.append(j+1)
                break

def build_model():

    model=models.Sequential()

    model.add(layers.Dense(64, activation='relu', input_shape=(n,)))
    model.add(layers.Dense(14,activation='relu'))
    model.add(layers.Dense(num_class, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def to_class(pred_y):
    for i in range(len(pred_y)):
        pred_y[i][pred_y[i]>=0.5]=1
        pred_y[i][pred_y[i]<0.5]=0
    return pred_y

num_iter=0

yscore=np.ones((1,num_class))*0.5
yclass=np.ones((1,num_class))*0.5
ytest=np.ones((1,num_class))*0.5

def evaluate(y_gt, y_pred, threshold_value=0.5):
    print("thresh = {:.6f}".format(threshold_value))
    y_pred_bin = y_pred >= threshold_value

    OAA = accuracy_score(y_gt, y_pred_bin)
    print("accuracy_score = {:.6f}".format(OAA))
    mAP = average_precision_score(y_gt, y_pred)
    print("mAP = {:.2f}%".format(mAP * 100))
    score_f1_macro = f1_score(y_gt, y_pred_bin, average="macro")
    print("Macro_f1_socre = {:.6f}".format(score_f1_macro))
    score_f1_micro = f1_score(y_gt, y_pred_bin, average="micro")
    print("Micro_f1_socre = {:.6f}".format(score_f1_micro))
    score_F1_weighted = f1_score(y_gt, y_pred_bin, average="weighted")
    print("score_F1_weighted = {:.6f}".format(score_F1_weighted))
    Mconfusion_matri = multilabel_confusion_matrix(y_gt, y_pred_bin)
    print("混淆矩阵", Mconfusion_matri)
    h_loss = hamming_loss(y_gt, y_pred_bin)
    print("Hamming_Loss = {:.6f}".format(h_loss))
    R_loss = label_ranking_loss(y_gt, y_pred_bin)
    print("ranking_loss= {:.6f}".format(R_loss))
    z_o_loss = zero_one_loss(y_gt, y_pred_bin)
    print("zero_one_loss = {:.6f}".format(z_o_loss))
    CV = coverage_error(y_gt, y_pred_bin)
    print("coverage_error= {:.6f}".format(CV))

    recall = recall_score(y_gt, y_pred_bin, average='samples')
    print("Recall:", recall)

evaluate (ytest[1:,:],yscore[1:,:], threshold_value=0.5 )
