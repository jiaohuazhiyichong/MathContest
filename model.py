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
from xgboost import plot_importance

def makeModel():
    model1 = XGBRegressor(max_depth=30, learning_rate=0.01, n_estimators=5,
             silent=True, booster='gbtree', n_jobs=50,
            objective='reg:squarederror', base_score=0.5, eval_metric=['rmse'], seed=1,
             nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
             colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
             scale_pos_weight=1,  random_state=0,
             missing=None, importance_type='gain')
    model2 = XGBRegressor(booster='gbtree', # general parameters
                        objective='reg:squarederror', base_score=0.5, eval_metric=['rmse'], seed=1,
                        # learning task parameters
                        n_estimators=3000, learning_rate=0.01, # booster parameters: 学习步长
                        subsample=1, colsample_bytree=1, # booster parameters: 训练集的行和列抽样
                        max_depth=3, min_child_weight=1, gamma=0, # booster parameters：树生长相关参数
                        reg_lambda=1, reg_alpha=0, # booster parameters：L1和L2正则化
                        scale_pos_weight=1, max_delta_step=0)
    return model2

def train_and_test_feature(id, time, X, Y, Y_s,propertys_idx2token):
    X_np = np.array(X)
    Y_np = np.array(Y)
    Y_s_np = np.array(Y_s)

    id_train,id_test,time_train,time_test,X_train,X_test,Y_train,Y_test,Y_s_train,Y_s_test = train_test_split(id,time,X_np,Y_np,Y_s_np)
    print(Y)
    # print(X_train)
    model = makeModel()
    print("training...")
    model.fit(X_train,Y_train,verbose=True,eval_set=[(X_test,Y_test)])
    model.save_model('tree100.model')
    print('training is ok')
    #  = model.predict(X_test)
    # print(fit_pred)
    # 显示重要特征
    # plot_importance(model)
    # plt.show()
   #  print(model.feature_importances_)
    # fea_imp_sorted = sorted(model.get_booster().get_score().items(), key=itemgetter(1), reverse=True)
    # print(model.get_booster().get_score().items())
    feature_im = sorted(model.get_booster().get_score().items(), key=lambda a: a[1], reverse=True)
    # print(feature_im)
    feature_im_list = []
    for i in feature_im:
        feature_im_list.append([propertys_idx2token[int(i[0][1:len(i[0])])],i[1]])
    print(feature_im_list)
    pd.DataFrame(feature_im_list, columns=['操作位点', 'xgb特征重要度']).to_excel('相关系数统计/xgb特征重要度.xlsx', encoding='utf8', index=False)
    plot_importance(model)
    plt.show()

def modelTraining(id, time, X, Y, Y_s,propertys_idx2token):
    X_np = np.array(X)
    Y_np = np.array(Y)
    Y_s_np = np.array(Y_s)

    id_train, id_test, time_train, time_test, X_train, X_test, Y_train, Y_test, Y_s_train, Y_s_test = train_test_split(
        id, time, X_np, Y_np, Y_s_np)
    model = makeModel()
    model_s = makeModel()
    print("training...")
    model.fit(X_train, Y_train, verbose=True, eval_set=[(X_test, Y_test)])
    model.save_model('tree1000.model')
    # model_s.fit(X_train, Y_s_train, verbose=True, eval_set=[(X_test, Y_s_test)])
    # model_s.save_model('tree_s_1000.model')
    print('training is ok')

    print(model.predict(X_np))
    print(Y_np)
    result = []

if __name__ == '__main__':
    str_path = "../处理数据.xlsx"
    propertys_list,list_values = load_New_data(str_path)
    propertys_list.remove('产品硫含量,μg/g')
    propertys_list.remove('产品辛烷值RON')
    propertys_list.remove(r'产品RON损失')
    # print(len(propertys_list))
    propertys_token2idx = {tag: i for i, tag in enumerate(propertys_list[2:len(propertys_list)])}
    # print(propertys_token2idx)
    propertys_idx2token = {i: w for w, i in propertys_token2idx.items()}
    # print(propertys_idx2token)
    id, time, X, Y, Y_s = make_dataset(list_values)
    # train_and_test_feature(id, time, X, Y, Y_s,propertys_idx2token)
    # print(propertys_token2idx)
    # print(len(X[0]))
    str_path1 = '../降维后的数据.xlsx'
    data_df = pd.read_excel(str_path1)
    data_numpy = np.array(data_df.values)
    modelTraining(id,time,data_numpy,Y,Y_s,propertys_idx2token)









