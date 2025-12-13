from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# def displayImage(x):
#     plt.imshow(x.reshape(28,28),cmap=plt.cm.binary,interpolation="nearest")
#     plt.show()

# def displayPredict(clf,actually_y,x):
#     print("Actually = ",actually_y) # ข้อมูลที่ทดสอบจริงๆ
#     print("Prediction = ",clf.predict([x])[0]) # ข้อมูลที่ Train

mnist_raw = loadmat("../mnist-original.mat")
mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}

x,y = mnist["data"], mnist["target"]

# Training & Test
# 1-60000 - Training, 60001-70000 - Test
x_train, x_test, y_train, y_test = x[:60000],x[60000:],y[:60000],y[60000:]

# แบ่ง class กลุ่ม 0 / กลุ่มที่ไม่ใช่ 0
# y_train = [0,0,0 ... , 9]
predict_number = 100
y_train_0 = (y_train==0) # [true, true, ... , false] บอกว่าเป็น 0 หรือไม่ - true / false
y_test_0 = (y_test==0)

# Model
sgd_model = SGDClassifier()
sgd_model.fit(x_train, y_train_0)

# displayPredict(sgd_model, y_test_0[predict_number], x_test[predict_number])
# displayImage(x_test[predict_number])

# cv ทดสอบ 3 ครั้ง scoring ให้แสดงความแม่นยำ (Cross Validation Score)
# score = cross_val_score(sgd_model,x_train, y_train_0, cv=3, scoring="accuracy")
# print(score)

# y_train_predict = cross_val_predict(sgd_model, x_train, y_train_0, cv=3)
# เปรียบเทียบค่าที่ทดสอบ กับค่าทำนาย
# cm = confusion_matrix(y_train_0, y_train_predict)

y_test_predict = sgd_model.predict(x_test)

classes = ['Other Number', 'Number 0']
# เทียบค่าที่ได้จากการพยากรณ์ค่า
# print(classification_report(y_test_0, y_test_predict, target_names=classes))
print("Accuracy percent is = ", accuracy_score(y_test_0, y_test_predict)*100)