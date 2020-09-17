import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn import metrics
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # 最大最小归一化
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from dataProcess import *

def makeModel():
    model = XGBRegressor(max_depth=30, learning_rate=0.01, n_estimators=5,
             silent=True, booster='gbtree', n_jobs=50,
             nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
             colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
             scale_pos_weight=1, base_score=0.5, random_state=0, seed=None,
             missing=None, importance_type='gain')

    return model

def train_and_test(id, time, X, Y, Y_s):
    X_np = np.array(X)
    Y_np = np.array(Y)
    Y_s_np = np.array(Y_s)

    id_train,id_test,time_train,time_test,X_train,X_test,Y_train,Y_test,Y_s_train,Y_s_test = train_test_split(id,time,X_np,Y_np,Y_s_np)
    print(Y)
    # print(X_train)
    model = makeModel()
    print("training...")
    model.fit(X_train,Y_train)
    model.save_model('tree100.model')
    print('training is ok')
    #  = model.predict(X_test)
    # print(fit_pred)
    # 显示重要特征
    # plot_importance(model)
    # plt.show()
   #  print(model.feature_importances_)
    index = []
    index_values = []
    for i in range(len(model.feature_importances_)):
        if model.feature_importances_[i] > 0:
            index.append(i)
            index_values.append(model.feature_importances_[i])
    plt.bar(index, index_values)
    plt.show()
    print(index)
    print(index_values)
if __name__ == '__main__':
    str_path = "../附件一：325个样本数据.xlsx"
    propertys_list,list_values = load_data(str_path)
    id, time, X, Y, Y_s = make_dataset(list_values)
    train_and_test(id, time, X, Y, Y_s)








