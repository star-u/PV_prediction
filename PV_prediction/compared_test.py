import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import method
from model import *
from pylab import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

mpl.rcParams['font.sans-serif'] = ['SimHei']
pv = ["1A", "1B", "30", "35", "22", "19", "5", "8", "11", "14"]


def testXY(num):
    file_path = 'data/{}-2017.csv'.format(pv[num])
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


lstm_model = []
# 导入模型
for _ in range(len(pv)):
    lstm_model.append(torch.load('modelpth/model_{}.pth'.format(pv[_])))


# LSTM2 = torch.load('C:/keyan\project\PV_prediction/modelpth\model_lstm_vgg16.pth')

def convert(lsit_):
    temp = list(lsit_)
    pred_list = []
    for _ in range(len(temp)):
        pred_list.extend(list(temp[_]))
    return pred_list


def prediction(model, series_x, series_y, name, num):
    model = model.eval()
    pred = model(series_x)
    pred[pred < 0] = 0
    length = len(series_y)
    for i in range(length):
        if series_y[i] == 0:
            pred[i] = 0
    pred = pred.view(-1).data.numpy()
    pred = y_scaler.inverse_transform(pred.reshape(-1, 1))

    series_y = y_scaler.inverse_transform(series_y.reshape(-1, 1))
    MSE = mean_squared_error(series_y, pred)
    RMSE = sqrt(MSE)
    R2 = r2_score(series_y, pred)
    MAE = mean_absolute_error(series_y, pred)
    # MAPE = method.MAPE_value(series_y, pred)
    print("{}-".format(pv[num]) + name, ' :')
    print(' {}-MSE: {:.3f}'.format(pv[num], MSE))
    print(' {}-RMSE: {:.3f}'.format(pv[num], RMSE))
    print(' {}-MAE: {:.3f}'.format(pv[num], MAE))
    print(' {}-R2: {:.3f}'.format(pv[num], R2))

    pred_list = convert(pred)
    print(pred_list)
    # print(' MAPE: {:.5f}%'.format(MAPE))
    return pred, pred_list


pred_lstm = []
pred_lstm_list = []
real = []
real_list = []
for _ in range(len(pv)):
    test_x, test_y, y_scaler = testXY(_)
    temp, temp2 = prediction(lstm_model[_], test_x, test_y, 'LSTM', _)
    pred_lstm.append(temp)
    pred_lstm_list.append(temp2)
    test_y = y_scaler.inverse_transform(test_y.reshape(-1, 1))
    real.append(test_y)
    test_y_list = convert(test_y)
    real_list.append(test_y_list)
# pred_lstm2 = prediction(LSTM2, test_x, test_y, 'LSTM2')
# print(pred_lstm_list)


print('Drawing...')

x = np.linspace(0, 168, 2015)

plt.plot(x, pred_lstm[0], 'aqua', label='LSTM预测值')
# plt.plot(x, pred_lstm2, 'blue', label='VGG16+LSTM预测值')

plt.plot(x, real[0], 'r', label='实际值')

plt.title('7天预测曲线对比图')
plt.xlabel('时间(单位：小时)')
plt.ylabel('功率(单位：kW)')
plt.xlim(0, 169)
plt.legend(loc='upper right')
plt.show()
print('Done')
