import numpy 
from sklearn import preprocessing
input_data = numpy.array([[5.1, 2.9, 3.3], [2.2, 7.8, 3.1], [3.9, 5.4, 4.1], [7.3, 9.9, 4.5]])
# 二値化の情報処理
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("data:\n", data_binarized)
# 平均化の情報処理

# スケーリングの情報処理

# 正規化の情報処理