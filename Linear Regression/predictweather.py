import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('Weather.csv')

# แสดงค่าเชิงสถิติ สูงสุด ต่ำสุด เฉลี่ยแต่ละตัว
# print(dataset.describe())

# Train & Test Set to 2D Array
x = dataset["MinTemp"].values.reshape(-1, 1)
y = dataset["MaxTemp"].values.reshape(-1, 1)

# 80% - 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training
model = LinearRegression()
model.fit(x_train, y_train)

# Testing
y_predict = model.predict(x_test)

# Compare True Data & Predict Data
df = pd.DataFrame({'Actually':y_test.flatten(), 'Predicted':y_predict.flatten()}) #flatten แปลงจาก 2D Array เป็น 1D Array เพราะโยนเข้าไปใน DataFrame

# การเปรียบเทียบค่า Error ของ Test กับ Predict เข้าใกล้ 0 แสดงว่ามี Error น้อย
print("MAE = ", metrics.mean_absolute_error(y_test, y_predict))
print("MSE = ", metrics.mean_squared_error(y_test, y_predict))
print("RMSE = ", np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

# หาค่า R Score ค่าผันแปรของตัวแปร y พยากรณ์หาค่า y ระหว่าง 0-100% (ค่าความแม่นยำ)
print("Score = ", metrics.r2_score(y_test, y_predict))

# ดึงข้อมูลจาก DataFrame มา 20 ตัว และมาเทียบค่า y_test ที่เป็นข้อมูลจริงที่ใช้ทดสอบ กับ y_predict ที่เป็นการคาดเดาจาก model
# df2=df.head(20)
# df2.plot(kind='bar', figsize=(16, 10))
# plt.show()

# Visualize
# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_predict, color='red', linewidth=2) # สร้างเส้น Linear
# plt.show()

# dataset.plot(x='MinTemp', y='MaxTemp')
# plt.title('Min & Max Temp')
# plt.xlabel('Mintemp')
# plt.ylabel('Maxtemp')
# plt.show()