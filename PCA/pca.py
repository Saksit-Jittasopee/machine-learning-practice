from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# 100 row - 10 col
x, y = make_blobs(n_samples=100, n_features=10)

# print("Before = ",x.shape) # (100,10)
pca = PCA(n_components=4)
pca.fit_transform(x)
# print("After = ",new_x.shape) # (100,4)

df = pd.DataFrame({'var':pca.explained_variance_ratio_,'pc':['PC1','PC2','PC3','PC4']})
sb.barplot(x='pc', y='var', data=df, color='c')
plt.show()
