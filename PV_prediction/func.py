# coding=gbk
import json
import sys

import pandas as pd
import numpy as np

import pickle
import os

import xlrd
import datetime

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, RepeatVector
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import warnings
import method
from model import *
from pylab import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

pv = ["1A", "1B", "30", "35", "22", "19", "5", "8", "11", "14"]


def date(para):
    # print(str(para))
    delta = pd.Timedelta(str(para) + 'days')
    # print(delta)
    time = pd.to_datetime('1899-12-30') + delta
    return time.round('min')


def get_length(pv_name):
    file_path = 'xls/xls/{}-2017.xls'.format(pv_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    # print(table.row_values(0)[0])
    sum_row = 0
    while table.row_values(sum_row + 1)[0] != 0:
        # print(sum_row)
        sum_row += 1
        # print(table.row_values(sum_row + 1)[0])

    return sum_row


def get_history(pv_name):
    file_path = 'xls/xls/{}-2017.xls'.format(pv_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    # print(table.row_values(0)[0])
    length = get_length(pv_name)
    time_list = []
    for i in range(1, length + 1):
        value = table.row_values(i)[0]
        # print(value)
        str_p = str(date(value))
        # print(str_p)
        dateTime_p = datetime.datetime.strptime(str_p, '%Y-%m-%d %H:%M:%S')
        # #print(str(dateTime_p))
        time_list.append(str(dateTime_p)[:-3])

    pv_power = []

    for i in range(1, length + 1):
        value = table.row_values(i)[3]
        if isinstance(value, str):
            pv_power.append(0)
        else:
            pv_power.append(value)
    history = []
    for j in range(len(time_list)):
        history.append((time_list[j], str(pv_power[j])))
    return json.dumps(history)


def convert(lsit_):
    temp = list(lsit_)
    pred_list = []
    for _ in range(len(temp)):
        pred_list.extend(list(temp[_]))
    return pred_list


def datatest(pv_name):
    file_path = 'data/{}-2017.csv'.format(pv_name)
    data = pd.read_csv(file_path, header=0, low_memory=False, index_col=0)
    data = data.rename(columns={
        # u'1A Trina - Active Energy Delivered-Received (kWh)': 'AE_Power',
        # u'1A Trina - Current Phase Average (A)': 'Current',  # 电流
        # u'1A Trina - Wind Speed (m/s)': 'Wind_speed',  # 风速
        u'timestamp': 'time',
        u'Active_Power': 'Power',  # 功率
        u'Weather_Relative_Humidity': 'Humidity',  # 湿度
        u'Weather_Temperature_Celsius': 'Temp',  # 气温
        u'Global_Horizontal_Radiation': 'GHI',  # 全球水平辐照度
        u'Diffuse_Horizontal_Radiation': 'DHI',  # 扩散水平辐照度
        # u'1A Trina - Wind Direction (Degrees)': 'Wind_dir',  # 风向
        # u'1A Trina - Weather Daily Rainfall (mm)': 'Rainfall'  # 降雨
    })

    # feature = ['Current', 'Wind_speed', 'Power', 'Humidity', 'Temp', 'GHI', 'DHI', 'Wind_dir']
    # 设定输入特征
    input_feature = ['Temp', 'Power', 'Humidity', 'GHI']
    # input_feature = ["Power"]
    feature = input_feature
    input_feature_num = 4
    # 设定目标特征
    target_feature = ['Power']

    # 删除功率为空的数据组
    # data = data.dropna(subset=['Power'])

    # NAN值赋0
    data = data.fillna(0)
    data[data < 0] = 0

    # 设定样本数目
    data = data[23905:]  # Random

    # 找零值点
    # zero_poi = data.isin[0]

    # 归一化
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    data[input_feature] = x_scaler.fit_transform(data[input_feature].to_numpy())
    data[target_feature] = y_scaler.fit_transform(data[target_feature].to_numpy())

    # 数据集分配
    test_x, test_y = method.create_dataset(data, target_feature, input_feature)
    return test_x, test_y, y_scaler


def prediction(model, series_x, series_y, name, pv_name):
    model = model.eval()
    pred = model(series_x)
    pred[pred < 0] = 0
    length = len(series_y)
    for i in range(length):
        if series_y[i] == 0:
            pred[i] = 0
    pred = pred.view(-1).data.numpy()
    test_x, test_y, y_scaler = datatest(pv_name)
    pred = y_scaler.inverse_transform(pred.reshape(-1, 1))

    series_y = y_scaler.inverse_transform(series_y.reshape(-1, 1))

    pred_list = convert(pred)
    #print(pred_list)
    # print(' MAPE: {:.5f}%'.format(MAPE))
    return pred, pred_list


def get_pred(pv_name):
    file_path = 'xls/xls/{}-2017.xls'.format(pv_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    length = get_length(pv_name)

    model = torch.load('modelpth/model_{}.pth'.format(pv_name))
    test_x, test_y, y_scaler = datatest(pv_name)
    temp, y_pred_list = prediction(model, test_x, test_y, 'LSTM', pv_name)
    test_y = y_scaler.inverse_transform(test_y.reshape(-1, 1))
    test_y_list = convert(test_y)
    time_list = []
    for i in range(1, length + 1):
        value = table.row_values(i)[0]
        str_p = str(date(value))
        dateTime_p = datetime.datetime.strptime(str_p, '%Y-%m-%d %H:%M:%S')
        time_list.append(str(dateTime_p)[:-3])
    prediction_tuple = []
    for j in range(len(y_pred_list)):
        prediction_tuple.append((time_list[23904 + j], str(y_pred_list[j])))
    return json.dumps(prediction_tuple)


pred_lstm = []
pred_lstm_list = []
real = []
real_list = []

for i in range(len(pv)):
    # length = get_length(pv[i])
    #history = get_history(pv[i])
    prediction = get_pred(pv[i])
    print(prediction)
    sys.exit()
