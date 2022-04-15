import pandas as pd
import os

# 显示全部列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

print('原始训练数据：')
print(train_data)

# 将所需数据转换成数值格式
print('原始训练数据信息：')
train_data.info()

print('原始测试数据信息：')
test_data.info()

def set_Cabin_type(data):
    data.loc[data['Cabin'].notnull(), 'Cabin'] = 'YES'
    data.loc[data['Cabin'].isnull(), 'Cabin'] = 'NO'
    return data

def add_dummy(data):
    # 需要转化的列
    object2num = ['Cabin', 'Embarked', 'Sex', 'Pclass']

    dummies_data = pd.DataFrame()
    for col_name in object2num:
        dummies_data = pd.concat([dummies_data, pd.get_dummies(data[col_name], prefix=col_name)], axis=1)
    
    data = pd.concat([data, dummies_data], axis=1)
    return data

train_data = set_Cabin_type(train_data)
train_data = add_dummy(train_data)
train_data.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print('转换为数值之后的训练数据：')
print(train_data.head())

# 对于Age字段，对没有确切数值的地方进行丢弃（后续会进行填补值的测试）
train_data = train_data.loc[train_data['Age'].notnull(), :]

# 因为Age和Fare两个字段范围不一致，要先进行归一化
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
train_data['Age_scaled'] = scaler.fit_transform(train_data[['Age']])
train_data['Fare_scaled'] = scaler.fit_transform(train_data[['Fare']])
train_data.drop(['Age', 'Fare'], axis=1, inplace=True)
print('将年龄和票价进行归一化后的结果：')
print(train_data)

# 对测试数据做同样的操作
test_data = set_Cabin_type(test_data)
test_data = add_dummy(test_data)
test_data.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# 测试集不要删年龄
# test_data = test_data.loc[test_data['Age'].notnull(), :]
test_data['Age_scaled'] = scaler.fit_transform(test_data[['Age']])
test_data['Fare_scaled'] = scaler.fit_transform(test_data[['Fare']])
test_data.drop(['Age', 'Fare'], axis=1, inplace=True)


# 保存数据文件
def save_file(path, data:pd.DataFrame, type):
    if not os.path.exists(path):
        os.mkdir(path)
    save_path = os.path.join(path, type + '.csv')
    data.to_csv(save_path)

save_file('output', train_data, 'train')
save_file('output', test_data, 'test')

