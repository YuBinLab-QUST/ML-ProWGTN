import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import math
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import average_precision_score, f1_score, zero_one_loss,hamming_loss,accuracy_score,multilabel_confusion_matrix,jaccard_score,recall_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error


def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y

data = pd.read_csv('po.csv')   #
label = pd.read_csv('polabel_csv')
label=np.array(label)
data=np.array(data)

shu=scale(data)
y= label
label_y = []
for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        if y[i, j] == 1:
            label_y.append(j + 1)
            break
[sample_num,input_dim]=np.shape(shu)
X = shu
sepscores = []


def get_shuffle(dataset, label):
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset, label
X, y = get_shuffle(X, y)


num_class=4
num_iter = 0
##进行交叉验证,并训练
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
    CV = coverage_error(y_gt, y_pred_bin)
    print("coverage_error= {:.6f}".format(CV))
    recall = recall_score(y_gt, y_pred_bin, average='samples')
    print("Recall:", recall)

evaluate(ytest[1:, :], yscore[1:, :], threshold_value=0.5)
