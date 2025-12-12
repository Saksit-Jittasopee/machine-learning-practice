# โหลดไฟล์จาก matlab เข้ามาใช้งาน แทนการใช้ scikit-learn
from scipy.io import loadmat
import matplotlib.pyplot as plt

mnist_raw = loadmat("mnist-original.mat")

# สร้าง Dictionary เก็บข้อมูลเฉพาะ
mnist={
    "data":mnist_raw["data"].T, #data เก็บเฉพาะส่วน data / .T (Transpose) คือสลับ column ไปหน้า row ไปหลัง
    "target":mnist_raw["label"][0] #target เก็บเฉพาะส่วน label index 0
}

# print(mnist_raw)

x = mnist["data"]
y = mnist["target"]

# เอาข้อมูล x ในช่วง 15000
number = x[15000]
# แปลงกลับให้เป็น Array 2 มิติ ขนาด 28x28
number_image = number.reshape(28,28)

# interpolation="nearest" - กำหนดพิกเซลเพื่อนบ้าน "ที่ใกล้ที่สุด" และกำหนดค่าความเข้มของพิกเซลนั้น
plt.imshow(number_image,cmap=plt.cm.binary, interpolation="nearest")
plt.show()

# print(x.shape)