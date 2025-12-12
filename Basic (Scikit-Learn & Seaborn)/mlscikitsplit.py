from sklearn.datasets import load_iris
# แบ่งข้อมูล train / test set
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

# แบ่งข้อมูลตามจำนวนที่ระบุ (75%, 25%) = default / x-data, y-target / test_size = 0.2 (20%) train_size ก็จะกลายเป็น 80%
x_train, x_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], test_size=0.2, random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(iris_dataset.data.shape)