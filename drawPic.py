#  __author__ = 'czx'
# coding=utf-8
import numpy as np
from numpy import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def leakyrelu(x):
    y = x.copy()
    for i in range(y.shape[0]):
        if y[i] < 0:
            y[i] = 0.2 * y[i]
    return y


def mish(x):
    return x * tanh(np.log(1 + np.exp(x)))


def relu(x):
    y = x.copy()
    y[y < 0] = 0
    return y


def elu(x, a):
    y = x.copy()
    for i in range(y.shape[0]):
        if y[i] < 0:
            y[i] = a * (exp(y[i]) - 1)
    return y

def sigmoid(x):
    a = np.exp(x)
    ans = a / ( a + 1)
    return ans

def swish(x,B):
    return x * sigmoid(B * x)

if __name__ == '__main__':
    x = arange(-4.0, 4.0, 0.01)
    y_leakyrelu = leakyrelu(x)
    y_mish = mish(x)
    y_relu = relu(x)
    y_elu = elu(x, 0.25)
    y_swish = swish(x, 1)

    plt.plot(x, y_leakyrelu, 'r', linewidth=1.5, label=u'leakyrelu')
    plt.plot(x, y_mish, 'g', linewidth=1.5, label=u'mish')
    plt.plot(x, y_relu, 'b', linewidth=1.5, label=u'relu')
    plt.plot(x, y_elu, 'k', linewidth=1.5, label=u'elu')
    plt.plot(x, y_swish, 'y', linewidth=1.5, label=u'swish')
    plt.ylim([-2, 2])
    plt.xlim([-4, 4])
    plt.legend()
    plt.grid(color='b', linewidth='0.3', linestyle='--')
    plt.savefig("mish.png", dpi=200)
    plt.show()

import os
import numpy as np
import pandas as pd
#
# def openreadtxt(file_name):
#     data = []
#     file = open(file_name, 'r')  # 打开文件
#     file_data = file.readlines()  # 读取所有行
#     for row in file_data:
#         tmp_list = row.split(' ')  # 按‘，’切分每行的数据
#         tmp_list[-1] = tmp_list[-1].replace('\n', '') #去掉换行符
#         data.append(tmp_list[:][3:])  # 将每行数据插入data中
#     return data
#
#
# if __name__ == "__main__":
#     path = 'data/VOC2007/labels-ssd'
#     files = os.listdir(path)
#     data = []
#     for file in files:
#         position = path + '/' + file
#         file = open(position, 'r')  # 打开文件
#         file_data = file.readlines()  # 读取所有行
#         for row in file_data:
#             tmp_list = row.split(' ')  # 按‘，’切分每行的数据
#             tmp_list[-1] = tmp_list[-1].replace('\n', '')  # 去掉换行符
#             data.append(tmp_list[:][3:])  # 将每行数据插入data中
#
#     data1 = pd.DataFrame(data)



import csv
import matplotlib.pyplot as plt

# df = pd.read_csv('wh.csv')  # 打开csv文件
# plt.scatter(df['x'], df['y'], s=0.6)
# plt.scatter(0.15006525, 0.45527289, s=15, c='r')
# plt.scatter(0.86430525, 0.89536966, s=15, c='r')
# plt.scatter(0.2951833, 0.82939571, s=15, c='r')
# plt.scatter(0.06667491, 0.13657743, s=15, c='r')
# plt.scatter(0.24602455, 0.62782629, s=15, c='r')
# plt.scatter(0.11597167, 0.27853844, s=15, c='r')
# plt.scatter(0.54624204, 0.61158999, s=15, c='r')
# plt.scatter(0.53822006, 0.88753414, s=15, c='r')
# plt.scatter(0.31305935, 0.40262698, s=15, c='r')
# plt.xlabel("w")
# plt.ylabel("h")
#
# plt.show()

# 一张图片中画四个子图

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# ax1 = plt.subplot(2, 2, 1)
# ax2 = plt.subplot(2, 2, 2)
# ax3 = plt.subplot(2, 2, 3)
# ax4 = plt.subplot(2, 2, 4)
#
# ax1.bar("YOLOv7-tiny", 83.36)
# ax1.bar("Our", 83.51)
# ax1.bar("YOLO4-tiny", 76.54)
# ax1.bar("SSD", 69.07)
# ax1.legend()
# ax1.set_xlabel("Model")
# ax1.set_ylabel("Precision(%)")
# ax1.set_ylim(65, 85)
# plt.tight_layout()
#
# ax2.bar("YOLOv7-tiny", 66.81)
# ax2.bar("Our", 73.45)
# ax2.bar("YOLO4-tiny", 62.86)
# ax2.bar("SSD", 71.20)
# ax2.legend()
# ax2.set_xlabel("Model")
# ax2.set_ylabel("Recall(%)")
# ax2.set_ylim(60, 75)
# plt.tight_layout()
#
# ax3.bar("YOLOv7-tiny", 77.95)
# ax3.bar("Our", 79.42)
# ax3.bar("YOLO4-tiny", 74.73)
# ax3.bar("SSD", 72.21)
# ax3.legend()
# ax3.set_xlabel("Model")
# ax3.set_ylabel("mAP(%)")
# ax3.set_ylim(70, 80)
# plt.tight_layout()
#
# ax4.bar("YOLOv7-tiny", 12.8)
# ax4.bar("Our", 12.1)
# ax4.bar("YOLO4-tiny", 22.5)
# ax4.bar("SSD", 90.7)
# ax4.legend()
# ax4.set_xlabel("Model")
# ax4.set_ylabel("Params(M)")
# plt.tight_layout()
#
# plt.show()

