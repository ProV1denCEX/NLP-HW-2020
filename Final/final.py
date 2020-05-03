import pandas as pd
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

aaa = pd.read_csv("Q7.csv", header=None)

pca1 = PCA(n_components=2)
pca1.fit(aaa)

X_new = pca1.transform(aaa)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
plt.show()



print(1)