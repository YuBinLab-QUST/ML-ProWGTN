from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
from module import GAT
import torch.nn.functional as F
import pandas as pd
import dgl
import dgl.nn as dglnn
import random
from sklearn.model_selection import KFold
import math
import time
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy import interp
import os
import matplotlib.pyplot as plt
from pylab import *
import utils.tools as utils

from sklearn.metrics import average_precision_score, f1_score, zero_one_loss, hamming_loss, accuracy_score, \
    multilabel_confusion_matrix,jaccard_score,recall_score,coverage_error
from sklearn.metrics import label_ranking_loss

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, self.activation))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

def get_shuffle(dataset,label):
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label

sepscores = []
sepscores_ = []
num_class=12

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
    CV=coverage_error(y_gt, y_pred_bin)
    print("coverage_error= {:.6f}".format(CV))

    recall = recall_score(y_gt, y_pred_bin, average='samples')
    print("Recall:", recall)

    sepscore_rnn=[]
    sepscore_rnn.append([OAA,mAP, score_f1_macro,score_f1_micro,score_F1_weighted, h_loss, R_loss, z_o_loss,CV])
    result=sepscore_rnn
    data_csv = pd.DataFrame(data=result)
    data_csv.to_csv('1-new.csv')
    data=pd.read_csv(r'1-new.csv', header=None, names=['OAA', 'AP', 'f1_macro', 'f1_micro', 'F1', 'h_loss', 'R_loss', 'z_o_loss', 'CV'])
    data.to_csv('1-new.csv',index=False)

evaluate(ytest[1:, :], yscore[1:, :], threshold_value=0.5)
