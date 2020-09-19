import numpy as np
import pandas as pd
import xlrd
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler  # 最大最小归一化
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


from scipy.spatial.distance import pdist, squareform

def distcorr(X, Y):
    ''' Compute the distance correlation function
    a = [1,2,3,4,5]
    b = np.array([1,2,9,4,4])
    distcorr(a, b)
    0.762676242417
    '''
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def load_data(str_path):
    data = xlrd.open_workbook(str_path)
    table = data.sheets()[0]
    list_values = []
    # print(table.nrows)
    propertys_list = []
    propertys_list_explain = []
    for x in range(1,table.nrows):
        values = []
        row = table.row_values(x)
        if x == 1:
            for i in row:
                propertys_list.append(i)
        elif x == 2:
            for i in row:
                propertys_list_explain.append(i)
        else:
            for i in row:
                values.append(i)
            list_values.append(values)
    # print(propertys_list)
    # print(list_values[len(list_values)-1])
    propertys_list[0] = '样本编号'
    propertys_list[1] = '时间'
    propertys_list_explain[0] = '样本编号'
    propertys_list_explain[1] = '时间'
    return propertys_list,list_values,propertys_list_explain

def load_dataRange(str_path):
    dict_max = {}
    dict_min = {}
    data = xlrd.open_workbook(str_path)
    table = data.sheets()[0]
    for x in range(1,table.nrows):
        row = table.row_values(x)
        max_num = row

def load_New_data(str_path):
    data = xlrd.open_workbook(str_path)
    table = data.sheets()[0]
    list_values = []
    propertys_list = []
    for x in range(0,table.nrows):
        values = []
        row = table.row_values(x)
        if x == 0:
            for i in row:
                propertys_list.append(i)
        else:
            for i in row:
                values.append(i)
            list_values.append(values)
    return propertys_list,list_values
def make_dataset(list_values):
    Y = []
    Y_s = []
    X = []
    id = []
    time = []

    for i in range(len(list_values)):
        values = []
        for j in range(len(list_values[i])):
            if j == 0:
                id.append(list_values[i][j])
            elif j == 1:
                time.append(list_values[i][j])
            elif j == 11:
                Y.append(list_values[i][j])
            elif j == 9:
                Y_s.append(list_values[i][j])
            elif j == 10:
                continue
            else:
                values.append(list_values[i][j])
        X.append(values)
    return id,time,X,Y,Y_s

def dataCleanout(df,propertys_list):
    print("hello")
    # 统计每列数据的残缺位点,获取缺失值的数目
    varies_0 = []
    varies_0_names = []
    print(df.head())
    for i in range(len(propertys_list)):
        if i >= 2:
            mask_0 = df[propertys_list[i]].isin([0.0])
            # print(df[propertys_list[i]][mask_0].count())
            if df[propertys_list[i]][mask_0].count() > 0:
                varies_0.append(df[propertys_list[i]][mask_0].count())
                varies_0_names.append(propertys_list[i])

    drop_list = ['S-ZORB.FT_1501.PV','S-ZORB.FT_1002.PV','S-ZORB.FC_1202.PV',
                 'S-ZORB.FT_1501.TOTAL','S-ZORB.FT_5102.PV','S-ZORB.FT_2901.DACA','S-ZORB.FC_1104.DACA',
                 'S-ZORB.FT_2803.DACA','S-ZORB.FT_1502.DACA','S-ZORB.TEX_3103A.DACA','S-ZORB.FT_5102.DACA.PV',
                 'S-ZORB.FC_2301.PV','S-ZORB.FT_5104.PV','S-ZORB.FT_9101.PV','S-ZORB.FC_3103.PV','S-ZORB.FT_1002.TOTAL']
   #  print(df)
    df = df.drop(columns=drop_list,axis=1)
    print(df.head())
    propertys_list = [i for i in propertys_list if i not in drop_list]
    print(propertys_list)

    print(len(propertys_list))
    for j in range(len(propertys_list)):
        if j >= 2:
            df[propertys_list[j]].replace(0.0,df[propertys_list[j]].mean(),inplace=True)
    print(df.head())
    return df,propertys_list


if __name__ == '__main__':
    str_path = "../附件一：325个样本数据.xlsx"
    propertys_list,list_values,propertys_list_explain = load_data(str_path)

    df = pd.DataFrame(list_values,columns=propertys_list)
    df,propertys_list = dataCleanout(df,propertys_list)
    df.to_excel('../处理数据.xlsx',encoding='utf8',index=False)
    load_dataRange('../附件四：354个操作变量信息.xlsx')
    propertys_list,list_values = load_New_data('../处理数据.xlsx')
    print(propertys_list)
    print(list_values[0])
    print(list_values[1])
    id, time, X, Y, Y_s = make_dataset(list_values)

