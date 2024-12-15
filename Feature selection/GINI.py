import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

def gini_index(X, y):
    n_samples, n_features = X.shape
    # initialize gini_index for all features to be 0.5
    gini = np.ones(n_features) * 0.5
    # For i-th feature we define fi = x[:,i] ,v include all unique values in fi
    for i in range(n_features):
        v = np.unique(X[:, i])
        for j in range(len(v)):
            left_y = y[X[:, i] <= v[j]]
            right_y = y[X[:, i] > v[j]]
            gini_left = 0
            gini_right = 0
            for k in range(np.min(y), np.max(y)+1):
                if len(left_y) != 0:
                    t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                    t2_left = np.power(t1_left, 2)
                    gini_left += t2_left
                if len(right_y) != 0:
                    t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                    t2_right = np.power(t1_right, 2)
                    gini_right += t2_right
            gini_left = 1 - gini_left
            gini_right = 1 - gini_right
            t1_gini = (len(left_y) * gini_left + len(right_y) * gini_right)
            value = np.true_divide(t1_gini, len(y))
            if value < gini[i]:
                gini[i] = value
    return gini

def feature_ranking(W):
    idx = np.argsort(W)
    return idx


def feature_ranking2(score, **kwargs):
    if 'style' not in kwargs:
        kwargs['style'] = 0
    style = kwargs['style']
    if style == -1 or style == 0:
        idx = np.argsort(score, 0)
        return idx[::-1]
    elif style != -1 and style != 0:
        idx = np.argsort(score, 0)
        return idx

data_train=pd.read_csv('po.csv', header=None)
data=np.array(data_train)
label=pd.read_csv('polabel.csv', header=None)
label=np.array(label)

shu=scale(data)
feature=pd.DataFrame(shu)
X=np.array(feature)
y=label.astype('int64')
score = gini_index(X, y)

idx =feature_ranking(score)
giniresult = feature[feature.columns[idx[0:60]]]
giniresult.to_csv("po.csv",header=None,index=None)


