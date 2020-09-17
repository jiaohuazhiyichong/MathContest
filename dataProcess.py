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

def load_data(str_path):
    data = xlrd.open_workbook(str_path)
    table = data.sheets()[0]
    list_values = []
    print(table.nrows)
    propertys_list = []
    for x in range(2,table.nrows):
        values = []
        row = table.row_values(x)
        if x == 2:
            for i in row:
                propertys_list.append(i)
        else:
            for i in row:
                values.append(i)
            list_values.append(values)
    # print(propertys_list)
    # print(list_values[len(list_values)-1])
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

if __name__ == '__main__':
    str_path = "../附件一：325个样本数据.xlsx"
    propertys_list,list_values = load_data(str_path)
    id, time, X, Y, Y_s = make_dataset(list_values)
    print(Y)
    print(X[0])

