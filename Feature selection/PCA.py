import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

if __name__ == '__main__':
    data = pd.read_csv('po.csv', header=None)
    label = pd.read_csv('polabel.csv', header=None)
    print(data.shape)
    print(label.shape)
    X = np.array(data)
    y = np.array(label)

    pca = PCA(n_components=60)
    X_pca = pca.fit_transform(X)

    print(X_pca.shape)

    DF1 = pd.DataFrame(X_pca)
    DF1.to_csv("po.csv", index=False, header=None)
