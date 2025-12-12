import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# สุ่มตัวเลขกลุ่มข้อมูล
rng = np.random
# x = สุ่มตัวเลข 50 ตัวแรก ไม่เอา 0.
x = rng.rand(50)*10

# randn มีค่าติดลบได้
y = 2*x+rng.randn(50)

# Linear Regression Model
model = LinearRegression()
# เปลี่ยนเป็น 2D Array
x_new = x.reshape(-1, 1)

# Train ต้องเป็น 2D Array
model.fit(x_new,y)

# Coefficient คือ ค่าสัมประสิทธิ์แสดงการตัดสินใจ
# Intercept คือ ค่าที่บ่งบอกจุดตัดแกน
# R-Square คือ ค่าผันแปรของตัวแปร y พยากรณ์หาค่า y ระหว่าง 0-100% (ค่าความแม่นยำ)
# print(model.score(x_new, y))

# Test Model
# สร้างกลุ่มข้อมูล -1 จนถึง 11
xfit = np.linspace(-1, 11)
xfit_new = xfit.reshape(-1, 1)

yfit = model.predict(xfit_new)

# Analysis Model
plt.scatter(x,y)
plt.plot(xfit, yfit)
plt.show()

# กระจายแบบ Scatterplot
# plt.scatter(x,y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# Linear Regression (y=ax+b)