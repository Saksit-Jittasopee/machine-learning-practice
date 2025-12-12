#import pylab
import matplotlib.pyplot as plt
from sklearn import datasets
#Import iris as Dictionary {Key:Value} (iris - ข้อมูลดอกไม้)
#iris_dataset = datasets.load_iris()

#Import digits as Dictionary {Key:Value} (digits - ข้อมูลตัวเลข)
digit_dataset = datasets.load_digits()

#แสดงผลรูปภาพเลข 1 cmap คือใส่สี color map ให้เป็น gray (ใช้ pylab)
# print(digit_dataset.target[1])
# pylab.imshow(digit_dataset.images[1],cmap=pylab.cm.gray_r)
# pylab.show()

#แสดงผลรูปภาพเลข 1 cmap คือใส่สี color map ให้เป็น gray (ใช้ matplotlib)
print(digit_dataset.target[1])
plt.imshow(digit_dataset.images[1],cmap=plt.get_cmap('gray'))
plt.show()

# print(iris_dataset.keys())
# print(digit_dataset.keys())

#จะแสดงเลข 0-1 เพราะ Target จะเก็บข้อมูลแบบ Binary
# print(iris_dataset['target']) 

#จะแสดงเป็นชื่อสายพันธ์ุแทน
# print(iris_dataset['target_names']) 
# print(digit_dataset.target_names) 

#จะแสดงเป็นข้อมูลสายพันธ์ุแทน
# print(iris_dataset['feature_names'])

#จะแสดงแบบรายละเอียดสถิติ
# print(iris_dataset['DESCR'])

#จะแสดงเป็นข้อมูล 10 แถวแรก (index)
#print(iris_dataset['data'][0:10])

#แสดงข้อมูลรูปภาพแถวแรก
# print(digit_dataset.images[0])

#แสดงขนาดรูปภาพแถวแรก (pixel x pixel)
# print(digit_dataset.images[0].shape)