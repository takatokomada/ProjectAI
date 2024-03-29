import pandas
import numpy 
from sklearn.datasets import load_digits
data_url = "http://lib.stat.cmu.edu/datasets/boston"
data_url1 = "http://lib.stat.cmu.edu/datasets/cloud"
data_url2 = "http://lib.stat.cmu.edu/datasets/boston_corrected.txt"
# data_url = "http://lib.stat.cmu.edu/datasets/bolts"
raw_df = pandas.read_csv(data_url, sep=",",skiprows=22, header=None, encoding='shift-jis')
raw_df1 = pandas.read_csv(data_url1, sep=",",skiprows=15,  header=None, encoding='shift-jis')
raw_df2 = pandas.read_csv(data_url2, sep=",", skiprows=22, header=None, encoding='utf-8')
data = numpy.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
data1 = numpy.hstack([raw_df1.values[::2, :], raw_df1.values[1::2, :2]])
data2 = numpy.hstack([raw_df2.values[::2, :], raw_df2.values[1::2, :2]])
digits = load_digits() 
print(data)
print(data1)
print(data2)
print(digits.images[4])
