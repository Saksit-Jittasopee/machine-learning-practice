from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.decomposition import PCA

iris = sb.load_dataset('iris')
x = iris.drop('species', axis=1)
y = iris['species']

# PCA
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x)

# add before and after
x['PCA1'] = x_pca[:,0]
x['PCA2'] = x_pca[:,1]
x['PCA3'] = x_pca[:,2]

x_train, x_test, y_train, y_test = train_test_split(x,y)

# complete data
x_train = x_train.loc[:,['PCA1','PCA2','PCA3']]
x_test = x_test.loc[:,['PCA1','PCA2','PCA3']]

# model
model = GaussianNB()
# train
model.fit(x_train, y_train)
# predict
y_predict = model.predict(x_test)
# accuracy
print("Accuracy = ",accuracy_score(y_test, y_predict))
