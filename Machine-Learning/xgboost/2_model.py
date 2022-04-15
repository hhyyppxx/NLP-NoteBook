import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

train_data = pd.read_csv('output/train.csv')
test_data = pd.read_csv('output/test.csv')

print(train_data)
print(test_data)

# 划分训练集和验证集
train_data = train_data.iloc[:, 1:] # 去掉id字段
train_data = train_data.values
train_data, dev_data = train_test_split(train_data, train_size=0.8, random_state=1)

x_train, y_train = train_data[:, 1:], train_data[:, 0]
x_dev, y_dev = dev_data[:, 1:], dev_data[:, 0]
print(x_train.shape)
print(y_train.shape)

