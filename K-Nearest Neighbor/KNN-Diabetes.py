from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes.csv")

# drop column outcomes and show only numbers in 2d array
x = df.drop("Outcome",axis=1).values
y = df["Outcome"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

knn_model = KNeighborsClassifier(n_neighbors=8)

knn_model.fit(x_train, y_train)

y_predict = knn_model.predict(x_test)

# print(classification_report(y_test, y_predict))

# print(confusion_matrix(y_test, y_predict))

print(pd.crosstab(y_test, y_predict, rownames=['True'], colnames=['Prediction'], margins=True))

# find k to model (1,2,3,4,5,6,7,8)
# k_neighbors = np.arange(1, 9)

# Training Score
# train_score = np.empty(len(k_neighbors)) 

# Test Score
# test_score = np.empty(len(k_neighbors))

# for i,k in enumerate(k_neighbors):
#     # Model
#     knn_model = KNeighborsClassifier(n_neighbors=k)
#     # Training
#     knn_model.fit(x_train, y_train)
#     # วัดประสิทธิภาพ
#     train_score[i] = knn_model.score(x_train, y_train)
#     test_score[i] = knn_model.score(x_test, y_test)

# plt.title("Compare k value in model")
# plt.plot(k_neighbors, train_score, label="Train Score")
# plt.plot(k_neighbors, test_score, label="Test Score")
# plt.legend()
# plt.xlabel("K Number")
# plt.ylabel("Score")
# plt.show()
