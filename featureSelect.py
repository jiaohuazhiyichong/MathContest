import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn import metrics
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  # 最大最小归一化
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from dataProcess import *
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression
from minepy import MINE
# 过滤式特征选择, 筛选出方差较低的特征,由于对回归模型不太适用，因此此方案最终被放弃


# 单变量特征选择过滤式的计算，计算皮尔森相关系数，f_regression, mutual_info_regression, 距离系数
def Univariatefeatureselection(id, time, X, Y, Y_s,propertys_idx2token,propertys_token2idx):
    # 计算皮尔森相关系数，输出前100个特征变量
    pearsonr_values = []
    row_nums = np.shape(X)[0]
    col_nums = np.shape(X)[1]
    for j in range(col_nums):
        p_value = pearsonr(X[:,j],Y)[1]
        f_value_abs = abs(pearsonr(X[:,j],Y)[0])
        pearsonr_values.append([propertys_idx2token[j],f_value_abs])
    # 输出相关特征排名，主要参考p_value
    pearsonr_values = sorted(pearsonr_values,key = (lambda x:x[1]),reverse=True)
    # print(pearsonr_values[0:100])

    # 计算f_regression, 输出前100个特征变量
    f_regression_fvalues, f_regression_pvalues = f_regression(X,Y)
    # print(f_regression_fvalues)
    # print(f_regression_fvalues.tolist())
    f_regression_fvalues = f_regression_fvalues.tolist()
    f_regression_f = []
    for i in range(len(f_regression_fvalues)):
        f_regression_f.append([propertys_idx2token[i], f_regression_fvalues[i]])
    f_regression_f = sorted(f_regression_f, key = (lambda x:x[1]),reverse=True)
    # print(f_regression_f)


    # 计算mutual_info_regression， 最大信息系数，输出特征变量排名
    m = MINE()
    Mic_scores = []
    for j in range(col_nums):
        m.compute_score(X[:,j],Y)
        Mic_scores.append([propertys_idx2token[j], m.mic()])
    Mic_scores = sorted(Mic_scores, key = (lambda x:x[1]),reverse=True)
    print(Mic_scores)

    # 距离系数的计算
    disTance_scores = []
    for j in range(col_nums):
        disTance_scores.append([propertys_idx2token[j], distcorr(X[:,j],Y)])
    disTance_scores = sorted(disTance_scores,key = (lambda x:x[1]),reverse=True)
    print(disTance_scores)

    pd.DataFrame(pearsonr_values,columns=['操作位点','皮尔森相关系数']).to_excel('相关系数统计/皮尔森相关系数.xlsx',encoding='utf8',index=False)
    pd.DataFrame(f_regression_f, columns=['操作位点', 'f_regression']).to_excel('相关系数统计/f_regression.xlsx', encoding='utf8', index=False)
    pd.DataFrame(Mic_scores, columns=['操作位点', '最大信息系数']).to_excel('相关系数统计/最大信息系数.xlsx', encoding='utf8', index=False)
    pd.DataFrame(disTance_scores, columns=['操作位点', '距离系数']).to_excel('相关系数统计/距离系数.xlsx', encoding='utf8', index=False)

# 采用随机森林进行模型的特征排序
def RFfeatureselection(id, time, X, Y, Y_s,propertys_idx2token,propertys_token2idx):
    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    rf_scores = []
    row_nums = np.shape(X)[0]
    col_nums = np.shape(X)[1]
    for i in range(col_nums):
        print(i)
        score = cross_val_score(rf, X[:,i:i+1], Y,scoring="r2",  # 注意X[:, i]和X[:, i:i+1]的区别
                            cv=ShuffleSplit(len(X), 3, .3))
        print(np.mean(score))
        rf_scores.append([propertys_idx2token[i],np.mean(score)])
    rf_scores =sorted(rf_scores,key = (lambda x:x[1]),reverse=True)
    print(rf_scores)
    pd.DataFrame(rf_scores, columns=['操作位点', 'RF排名']).to_excel('相关系数统计/随机森林RF排名.xlsx', encoding='utf8', index=False)

# 利用xgboost

if __name__ == '__main__':
    str_path = "../处理数据.xlsx"
    propertys_list,list_values = load_New_data(str_path)
    # print(propertys_list)
    propertys_list.remove('产品硫含量,μg/g')
    propertys_list.remove('产品辛烷值RON')
    propertys_list.remove(r'产品RON损失')
    # print(len(propertys_list))
    propertys_token2idx = {tag: i for i, tag in enumerate(propertys_list[2:len(propertys_list)])}
    # print(propertys_token2idx)
    propertys_idx2token = {i: w for w, i in propertys_token2idx.items()}
    # print(propertys_idx2token)
    id, time, X, Y, Y_s = make_dataset(list_values)
    # print(propertys_token2idx)
    # print(len(X[0]))
    X_np = np.array(X)
    Y_np = np.array(Y)
    Y_s_np = np.array(Y_s)
    # Univariatefeatureselection(id,time,X_np,Y_np,Y_s_np,propertys_idx2token,propertys_token2idx)
    # print(propertys_idx2token)
    RFfeatureselection(id,time,X_np,Y_np,Y_s_np,propertys_idx2token,propertys_token2idx)