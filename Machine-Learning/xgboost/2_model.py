import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
# from xgboost.sklearn import XGBClassifier

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

param = {
    'booster': 'gbtree',
    'objective': 'rank:pairwise',
    'eval_metric':'auc',
    'seed':2022,
    'eta': 0.01,
    'gamma':0.1,
    'min_child_weight':1.1,
    'max_depth':5,
    'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'colsample_bylevel':0.7,
    'tree_method':'exact'
}

train_ = xgb.DMatrix(data=x_train, label=y_train)
eval_ = xgb.DMatrix(data=x_dev, label=y_dev)
eval_ = [(eval_, 'val')]
xgb_model = xgb.train(params=param, dtrain=train_, evals=eval_)

cvresult = xgb.cv(param, train_, num_boost_round=100, nfold=5, metrics='auc', seed=2022,
                    callbacks=[xgb.callback.print_evaluation(show_stdv=False), xgb.callback.early_stop(50)]
)

# fig = xgb.to_graphviz(xgb_model, num_trees=1, leaf_node_params={'shape': 'plaintext'})
# print(fig)
xgb.plot_importance(xgb_model)
# print(fig_2)
