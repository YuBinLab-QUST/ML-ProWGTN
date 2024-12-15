import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

if __name__ == '__main__':
    data = pd.read_csv('po.csv', header=None)
    label = pd.read_csv('polabel.csv', header=None)
    print(data.shape)
    print(label.shape)
    X = np.array(data)
    y = np.array(label)


    model = RandomForestClassifier(n_estimators=60, random_state=42)

    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    select_fea = indices[:61]
    select_fea = [str(i + 1) for i in select_fea]
    print(select_fea)
    print(len(select_fea))

    DF = []
    for i in select_fea:
        df = X[:, int(i) - 1]
        DF.append(df)
    df1 = np.array(DF)

