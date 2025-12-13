from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.4, random_state=0)

# Model
knn_model = KNeighborsClassifier(n_neighbors=1)

# Training
knn_model.fit(x_train, y_train)

# Predict
# predict = knn_model.predict([x_test[1]])
y_predict = knn_model.predict(x_test)

# Show
# print("Prediction: ", predict)
# print("Belong in group: ", iris_dataset['target_names'][predict])
print(classification_report(y_test, y_predict,target_names=iris_dataset['target_names']))
print("Accuracy Score: ", accuracy_score(y_test, y_predict)*100)
