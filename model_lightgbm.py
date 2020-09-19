import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # 最大最小归一化
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from dataProcess import *


# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression', # 目标函数
    'metric': {'l2', 'rmse'},  # 评估函数
    'num_leaves': 4,   # 叶子节点数
    'learning_rate': 0.005,  # 学习速率
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

# def makeModel():
#     model =
#     return model

def train_and_test(id, time, X, Y, Y_s):
    X_np = np.array(X)
    Y_np = np.array(Y)
    Y_s_np = np.array(Y_s)

    id_train,id_test,time_train,time_test,X_train,X_test,Y_train,Y_test,Y_s_train,Y_s_test = train_test_split(id,time,X_np,Y_np,Y_s_np)
    print(Y)
    # print(X_train)

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X_train, Y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
    lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)  # 创建验证数据

    gbm  = lgb.train(params,lgb_train,num_boost_round=500,valid_sets=lgb_eval,early_stopping_rounds=5) # 训练数据需要参数列表和数据集
    print("training...")
    gbm.save_model('tree100.model')
    print('training is ok')
    Y_predict = gbm.predict(X_np)
    # print(gbm.predict(X_np))
    print (Y_predict)
    print(Y_np)
    lightgbm_result = []
    for i in range(len(Y_predict)):
        lightgbm_result.append([Y_predict[i], Y_np[i]])
    pd.DataFrame(lightgbm_result, columns=['预测值', '真实值']).to_excel('lightgbm_result.xlsx', encoding='utf8', index=False)

   #  print(model.feature_importances_)
   #  index = []
   #  index_values = []
   #  for i in range(len(gbm.feature_importances_)):
   #      if gbm.feature_importances_[i] > 0:
   #          index.append(i)
   #          index_values.append(gbm.feature_importances_[i])
   #  plt.bar(index, index_values)
   #  plt.show()
   #  print(index)
   #  print(index_values)

if __name__ == '__main__':
    str_path = "../处理数据.xlsx"
    propertys_list, list_values = load_New_data(str_path)
    propertys_list, list_values = load_New_data(str_path)
    propertys_list.remove('产品硫含量,μg/g')
    propertys_list.remove('产品辛烷值RON')
    propertys_list.remove(r'产品RON损失')
    # print(len(propertys_list))
    propertys_token2idx = {tag: i for i, tag in enumerate(propertys_list[2:len(propertys_list)])}
    # print(propertys_token2idx)
    propertys_idx2token = {i: w for w, i in propertys_token2idx.items()}

    # str_path = "降维后的数据30维.xlsx"
    # propertys_list,list_values = load_New_data(str_path)
    id, time, X, Y, Y_s = make_dataset(list_values)
    # train_and_test(id, time, X, Y, Y_s)

    str_path1 = '降维后的数据.xlsx'
    data_df = pd.read_excel(str_path1)
    data_numpy = np.array(data_df.values)
    train_and_test(id, time, data_numpy, Y, Y_s)








