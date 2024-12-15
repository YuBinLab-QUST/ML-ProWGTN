import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import scale,StandardScaler 
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn import linear_model
from sklearn.metrics import average_precision_score, f1_score, zero_one_loss,hamming_loss,accuracy_score,multilabel_confusion_matrix,jaccard_score,recall_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error

def get_shuffle(dataset,label):    
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label
def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y

data = pd.read_csv('po.csv',header=None)   #
label = pd.read_csv('polabel',header=None)

data1=np.array(data)
data=data1[:,1:]
label=np.array(label)

shu=scale(data)

X,y_=get_shuffle(shu,label)
y=y_

sepscores = []

cv_clf = xgb.XGBClassifier(max_depth=200, learning_rate=0.01,
                 n_estimators=100, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=5, nthread=5, gamma=1, min_child_weight=1,
                 max_delta_step=2, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=1, reg_lambda=2, scale_pos_weight=1,
                 base_score=0.5)

num_class=4
yscore=np.ones((1,num_class))*0.5
yclass=np.ones((1,num_class))*0.5
ytest=np.ones((1,num_class))*0.5

def evaluate(y_gt, y_pred, threshold_value=0.5):
    print("thresh = {:.6f}".format(threshold_value))
    y_pred_bin = y_pred >= threshold_value

    OAA=accuracy_score(y_gt, y_pred_bin)
    print("accuracy_score = {:.6f}".format(OAA))
    mAP = average_precision_score(y_gt, y_pred)
    print("mAP = {:.2f}%".format(mAP * 100))
    score_f1_macro = f1_score(y_gt, y_pred_bin, average="macro")
    print("Macro_f1_socre = {:.6f}".format(score_f1_macro))
    score_f1_micro = f1_score(y_gt, y_pred_bin, average="micro")
    print("Micro_f1_socre = {:.6f}".format(score_f1_micro))
    score_F1_weighted = f1_score(y_gt, y_pred_bin, average="weighted")
    print("score_F1_weighted = {:.6f}".format(score_F1_weighted))
    Mconfusion_matri=multilabel_confusion_matrix(y_gt, y_pred_bin)
    print("混淆矩阵",Mconfusion_matri)
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

evaluate (ytest[1:,:],yscore[1:,:], threshold_value=0.5)

