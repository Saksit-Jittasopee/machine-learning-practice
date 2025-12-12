# Numpy คือ Array
import numpy as np
import matplotlib.pyplot as plt

# สร้างกลุ่มข้อมูล -5 จนถึง 5 จำนวน 10 ชุด
x = np.linspace(-5, 5, 10)

# Linear Regression (y=ax+b) (a=2, b=1)
y = 2*x+1

# plot graph -r คือ สีแดง
plt.plot(x,y, '-r', label='y=2x+1')
# ข้อความแกน x
plt.xlabel('x')
# ข้อความแกน y
plt.ylabel('y')
# บอกรายละเอียดซ้ายบน
plt.legend(loc='upper left')
# บอกหัวข้อของกราฟ
plt.title('Linear Regression (y=2x+1)')
# สร้างกล่องสีเหลี่ยมตารางด้านใน
plt.grid()
# แสดง
plt.show()

#print(x)