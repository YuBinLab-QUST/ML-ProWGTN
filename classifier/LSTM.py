
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys,os
import re
from scipy import interp
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Input,Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from sklearn.preprocessing import scale,StandardScaler
from keras.layers import Dense, concatenate,Input,Dropout
from keras.models import Model
from sklearn.metrics import average_precision_score, f1_score, zero_one_loss, hamming_loss, accuracy_score, \
    multilabel_confusion_matrix,coverage_error,jaccard_score,recall_score
from sklearn.metrics import label_ranking_loss

def to_class(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y
####################

data = pd.read_csv('po.csv')
label = pd.read_csv('polabel.csv')

print(data.shape)
print(label.shape)

num_class = 4
data = np.array(data)
y = np.array(label)

X=scale(data)


def get_RNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(LSTM(int(input_dim/2), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(int(input_dim/4), return_sequences=True))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(int(input_dim/4), activation = 'relu'))
    model.add(Dense(int(input_dim/8), activation = 'relu'))
    model.add(Dense(out_dim, activation = 'softmax',name="Dense_2"))
    # model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

[sample_num,input_dim]=np.shape(X)
out_dim=2
yscore = np.ones((1, num_class)) * 0.5
yclass = np.ones((1, num_class)) * 0.5
ytest = np.ones((1, num_class)) * 0.5

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
    CV=coverage_error(y_gt, y_pred_bin)
    print("coverage_error= {:.6f}".format(CV))
    recall = recall_score(y_gt, y_pred_bin, average='samples')
    print("Recall:", recall)

evaluate(ytest[1:, :], yscore[1:, :], threshold_value=0.5)
