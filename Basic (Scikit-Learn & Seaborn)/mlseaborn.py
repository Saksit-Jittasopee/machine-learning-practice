# seaborn คล้ายๆ Matplotlib แต่จะแสดงข้อมูลสถิติที่ลึก และซับซ้อนกว่า
import seaborn as sb
import matplotlib.pyplot as plt
iris_dataset = sb.load_dataset('iris')

# แสดงข้อมูล 5 แถวแรก
# print(iris_dataset.head(5))

sb.set()
# แสดงผลเป็น pairplot จากข้อมูล iris_dataset โดยสีจะแยกจาก species ขนาด = 2
sb.pairplot(iris_dataset, hue='species', size=2)
plt.show()