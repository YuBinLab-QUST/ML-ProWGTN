
import pdb

import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


if __name__ == '__main__':

    data = pd.read_csv('po.csv', header=None)  #
    label = pd.read_csv('polabel.csv', header=None)
    print(data.shape)
    print(label.shape)
    X = np.array(data)
    y = np.array(label)

    model = XGBClassifier(n_estimators=60)

    model.fit(X, y)

    importance = [[idx,score]  for idx,score in enumerate(model.feature_importances_)]
    importance=sorted(importance,key=lambda x:x[1],reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv("po.csv", index=False)

import pdb
import  pandas as pd
importance=pd.read_csv("po.csv")["feature"]
select_fea=list(importance)[0:60]
select_fea=[str(i+1) for i in select_fea]
print(select_fea)
print(len(select_fea))

DF=[]

data = pd.read_csv('po.csv', header=None)

X = np.array(data)
for i in select_fea:
    df = X[:, int(i)-1]
    DF.append(df)
df1 = np.array(DF)
