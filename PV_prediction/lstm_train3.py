import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import method
# from model import Transformer
from model import LSTMNet
from pylab import *
import pickle
import torch.nn.functional as F

pv = ["1A","1B","30","35","22","19","5","8","11","14"]
def convert(num):
    file_path = 'data/{}.csv'.format(pv[num])
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

    data[324567:460567].to_csv("data2/{}-2017.csv".format(pv[num]))


def lstm_train(num):
    file_path = 'data2/{}-2017.csv'.format(pv[num])
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
    #data[].to_csv("{}-2017".format(pv[num]))
    #feature = ['Current', 'Wind_speed', 'Power', 'Humidity', 'Temp', 'GHI', 'DHI', 'Wind_dir']
    # 设定输入特征
    #input_feature = ['Temp', 'Power', 'Humidity', 'GHI']
    input_feature = ["Power"]
    feature = input_feature
    input_feature_num = 1
    # 设定目标特征
    target_feature = ['Power']

    # 删除功率为空的数据组
    data = data.dropna(subset=['Power'])

    # NAN值赋0
    data = data.fillna(0)
    data[data < 0] = 0

    # 设定样本数目
    data = data[:104040]
    # data = data[:4352]
    # 归一化
    scaler = MinMaxScaler()
    data[feature] = scaler.fit_transform(data[feature].to_numpy())

    # 数据集分配
    # train_x, train_y = method.create_dataset(data, target_feature, input_feature)
    train_x, train_y = method.create_dataset(data, target_feature, input_feature)
    lstm = LSTMNet(input_size=input_feature_num)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.006)
    loss_func = nn.MSELoss()
    epochs = 1000
    print(lstm)
    print('Start training...')
    print(train_x.shape)

    for e in range(epochs):
        # 前向传播
        y_pred = lstm(train_x)
        y_pred = torch.squeeze(y_pred)
        loss = loss_func(y_pred, train_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % 20 == 0:
            print('Epoch:{}, Loss:{:.5f}'.format(e, loss.item()))

    plt.plot(y_pred.detach().numpy(), 'r', label='y_pred')
    plt.plot(train_y.detach().numpy(), 'b', label='y_train')
    plt.title('Photovoltaic History Curve')
    plt.legend()
    plt.show()

    print('Model saving...')

    MODEL_PATH = 'modelpth/model_{}.pth'.format(pv[num])

    torch.save(lstm, MODEL_PATH)

    print('Model saved')


lstm_train(0)
# convert(1)