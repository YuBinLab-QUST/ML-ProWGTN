import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, zero_one_loss, hamming_loss, accuracy_score, \
    multilabel_confusion_matrix,jaccard_score,recall_score
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error
from tensorflow.keras.layers import Conv1D, Dense, AveragePooling1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Input, Dropout

# 导入数据

data = pd.read_csv('data.csv')   #
label = pd.read_csv('label.csv')

num_class = 13
X = np.array(data)
y = np.array(label)
n = np.shape(X)[1]
label_y = []
for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        if y[i, j] == 1:
            label_y.append(j + 1)
            break

def to_class(pred_y):
    '''
#     输入：
#         pred_y(ndarray):预测值(各分量为概率)
#     作用：
#         本函数将输入的pred_y转换成class
#     输出:
#         pred_y(ndarray):预测值(各分量为0，1)
#     '''
    for i in range(len(pred_y)):
        pred_y[i][pred_y[i] >= 0.5] = 1
        pred_y[i][pred_y[i] < 0.5] = 0
    return pred_y

def build_model(input_dim, num_class):
    model = models.Sequential()
    model.add(Bidirectional(LSTM(int(input_dim / 2), return_sequences=True), input_shape=(1, n)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(int(input_dim / 4), return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_class, activation='softmax', name="Dense_2"))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

[sample_num, input_dim] = np.shape(X)

num_iter = 0

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
    import torch
    print(ytest[1:, :])

    evaluate(ytest[1:, :], yscore[1:, :], threshold_value=0.5)